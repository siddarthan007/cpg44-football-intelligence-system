"""Near-real-time streaming pipeline with a live multi-window dashboard, Kalman
trajectory smoothing, Catapult-style load, and optional wearable fusion.

Single forward pass (no look-ahead) so it runs live:

    warmup (fit team colours) → per frame:
        detect(GPU) → track → team → homography → Kalman(pos/vel/accel)
        → possession/passes → Catapult load (metabolic power…) → fuse wearable
        → injury risk → recommendations → dashboard

Physical-load / injury analytics require metric mode (a pitch calibration), since
metabolic power is defined in metres; in pixel mode the system still does
detection, tracking, teams and possession. Wearable samples arrive async via a
:class:`SensorSource` and are time-aligned per frame.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from . import annotate as an
from .config import PipelineConfig
from .core import PLAYER, REFEREE, foot_points, centers
from .dashboard import Dashboard, DashboardState, compose_dashboard, render_dashboard_panel
from .device import resolve_device
from .filters import BallKalman, KalmanBank
from .heatmap import LiveHeatmap, radar_frame
from .metrics import MetricsEngine
from .reid import ReIDManager
from .shots import ShotDetector
from .substitution import SubstitutionAdvisor
from .tactical_engine import TacticalEngine
from .team_assign import TeamAssigner
from .tracker import Detector, SoccerTracker, pick_ball
from .view import CameraMotionEstimator, PitchHomographyTracker, ViewTransformer
from .sensors import FusionEngine, HeuristicInjuryModel, RecommendationEngine, SensorVideoSync
from .sensors.schema import InjuryRisk
from .sensors.source import SensorSource


class RealtimePipeline:
    def __init__(self, cfg: PipelineConfig, view: ViewTransformer,
                 sensor_source: Optional[SensorSource] = None,
                 roster_map: Optional[Dict[int, int]] = None,
                 roster_numbers: Optional[Dict[int, int]] = None,
                 injury_model=None, warmup: int = 40,
                 dashboard: bool = True, analytics_every: int = 10):
        self.cfg = cfg
        self.view = view
        self.metric = view.is_metric
        self.dev = resolve_device(cfg.device)
        print(f"[realtime] {self.dev.name} ({self.dev.device}) | "
              f"{'METRIC' if self.metric else 'PIXEL (no calibration)'} mode")

        self.detector = Detector(cfg.weights, imgsz=cfg.imgsz, device=self.dev.device,
                                 person_conf=cfg.conf, ball_conf=min(cfg.conf, 0.1),
                                 half=self.dev.half_ok)
        self.team = TeamAssigner()
        self.warmup = warmup

        sync = SensorVideoSync()
        if roster_map:
            sync.bind_many(roster_map)
        self.fusion = FusionEngine(sync)
        self.sensor_source = sensor_source
        self.injury_model = injury_model or HeuristicInjuryModel()
        self.recommender = RecommendationEngine()

        self.dash = Dashboard(windows=dashboard)
        self.live_heat = LiveHeatmap(cfg.pitch)
        self.analytics_every = analytics_every
        self._formation = {1: "", 2: ""}
        self._shots = {1: {"shots": 0, "xg": 0.0}, 2: {"shots": 0, "xg": 0.0}}

        # LSTM trajectory prediction (optional) — draws each player's predicted path
        self.traj_model, self._traj_K = None, 15
        traj_path = "runs/trajectory/traj_lstm.pt"
        if os.path.exists(traj_path):
            try:
                from .trajectory import load as _load_traj, K_IN
                self.traj_model = _load_traj(traj_path, device="cpu")
                self._traj_K = K_IN
                print("[realtime] loaded trajectory LSTM (draws predicted paths)")
            except Exception as e:
                print(f"[realtime] trajectory model unavailable: {e}")
        self._traj_hist: Dict[int, deque] = defaultdict(lambda: deque(maxlen=40))
        self._pred_paths: Dict[int, list] = {}         # image-pixel predicted paths
        self._pred_paths_pitch: Dict[int, list] = {}   # same, projected to pitch (radar)
        self.reid = ReIDManager()                      # persistent ids across occlusion
        self.subs = SubstitutionAdvisor()              # fatigue-based substitution watch
        self._sub_watch: list = []

        # jersey-number OCR → auto-bind wearers (optional; needs roster_numbers)
        self.autobinder = None
        if roster_numbers:
            from .jersey_ocr import JerseyReader, AutoBinder
            reader = JerseyReader(device=self.dev.device)
            if reader.available:
                self.autobinder = AutoBinder(reader, roster_numbers)

    # ------------------------------------------------------------------ #
    def _warmup(self, cap):
        colors = []
        for _ in range(self.warmup):
            ok, frame = cap.read()
            if not ok:
                break
            det = self.detector.detect(frame)
            for bbox in det.of_class(PLAYER).xyxy:
                c = self.team.shirt_color(frame, bbox)
                if c is not None:
                    colors.append(c)
        if len(colors) >= 2:
            self.team.fit(colors)
            print(f"[realtime] team model fit on {len(colors)} colour samples")
        else:
            print("[realtime] insufficient colour samples; team assignment disabled")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def run(self, video: str, out_path: Optional[str] = None, max_frames: int = 0,
            stats_out: Optional[str] = None):
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise FileNotFoundError(
                f"cannot open video source {video!r}. WSL has no webcam by default — "
                f"pass a video file path (e.g. demo/sample_match.mp4) instead of an index.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        dt = 1.0 / fps
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # larger lost buffer → ByteTrack itself recovers longer gaps; the ReID
        # layer then handles the very-long / out-of-view cases.
        tracker = SoccerTracker(frame_rate=int(round(fps)), lost_track_buffer=90)
        # metric mode: per-frame homography that follows the camera pan (replaces
        # the static homography + separate camera-motion pass — same optical-flow
        # cost, no FPS hit). pixel mode: keep the translation-only camera comp.
        pitch_tracker = PitchHomographyTracker(
            self.view.H, (H, W), pitch_length=self.cfg.pitch.length,
            pitch_width=self.cfg.pitch.width) if self.metric else None
        cam = None if self.metric else CameraMotionEstimator((H, W))
        engine = MetricsEngine(fps=fps, metric=self.metric,
                               possession_dist=2.0 if self.metric else 70.0,
                               max_speed_mps=self.cfg.max_speed_mps)
        tactic = TacticalEngine(self.cfg.pitch)
        kbank = KalmanBank(max_speed=self.cfg.max_speed_mps if self.metric else 1e9)
        ball_kf = BallKalman()
        shot_detector = ShotDetector(pitch_length=self.cfg.pitch.length,
                                     pitch_width=self.cfg.pitch.width)

        if self.sensor_source is not None:
            self.sensor_source.start()

        print("[realtime] warmup…")
        self._warmup(cap)

        writer = None   # created lazily once we know the composite size

        cum = np.array([0.0, 0.0])
        fi = 0
        last_ball_seen = -999          # frame index the ball was last actually detected
        last_ball_px = None            # last ball image centre (for the ROI zoom search)
        ball_miss = 0
        # live pacing: when the on-screen demo falls behind the video clock, skip
        # frames to stay real-time (no slow-motion). Recording-only runs process
        # every frame for maximum quality.
        pace_live = self.dash.windows
        t_start = time.time()
        frames_dropped = 0
        t_prev = time.time()
        fps_disp = 0.0
        injuries, recs = [], []
        paused = False
        while True:
            if not paused:
                # real-time pacing: if we're behind the video clock, drop frames
                dropped_now = 0
                if pace_live:
                    behind = (time.time() - t_start) - fi / fps
                    while behind > 2.0 / fps:
                        if not cap.grab():
                            break
                        fi += 1
                        dropped_now += 1
                        behind -= 1.0 / fps
                    frames_dropped += dropped_now
                dt = (1 + dropped_now) / fps       # true elapsed time between frames
                ok, frame = cap.read()
                if not ok or (max_frames and fi >= max_frames):
                    break
                # wall-clock capture time so vision aligns with the wearable's
                # epoch-stamped samples (both live); dt uses video frame spacing.
                vt = time.time()

                det = self.detector.detect(frame)
                tracked = tracker.update(det)
                ball_box = pick_ball(det)
                # ROI zoom search when the full-frame pass misses the small ball
                if ball_box is not None:
                    last_ball_px = centers(ball_box)[0]
                    ball_miss = 0
                elif last_ball_px is not None and ball_miss < 40:
                    roi = self.detector.detect_ball_in_roi(
                        frame, last_ball_px[0], last_ball_px[1], 70 + ball_miss * 10)
                    if roi is not None:
                        ball_box = roi
                        last_ball_px = centers(ball_box)[0]
                        ball_miss = 0
                    else:
                        ball_miss += 1
                else:
                    ball_miss += 1
                # one optical-flow pass: tracked homography (metric) OR camera comp.
                # Homography updated every 3rd frame (pan is smooth over 0.12 s) and
                # the held H is used to transform every frame — keeps FPS up.
                if self.metric:
                    if fi % 4 == 0:
                        pitch_tracker.update(frame)
                    def _xform(px):
                        return pitch_tracker.transform([px])[0]
                else:
                    dx, dy = cam.update(frame)
                    cum = cum + np.array([dx, dy])
                    def _xform(px):
                        return self.view.transform([np.asarray(px, float) - cum])[0]

                bound = self.fusion.sync._track2player
                players_xy, teams, vision_state = {}, {}, {}
                radar_by_team = {1: [], 2: []}

                # ---- pass 1: per-track jersey colour (every 5th frame) ----
                compute_color = (fi % 5 == 0)
                raw = []
                for i in range(len(tracked)):
                    bt = int(tracked.tracker_id[i])
                    xyxy = tracked.xyxy[i]
                    is_ref = int(tracked.class_id[i]) == REFEREE
                    col = None if is_ref else (self.team.shirt_color(frame, xyxy)
                                               if compute_color else None)
                    raw.append((bt, xyxy, is_ref, col))

                # ---- Re-ID: map ByteTrack ids → persistent stable ids ----
                reid_in = []
                for (bt, xyxy, is_ref, col) in raw:
                    if is_ref:
                        continue
                    ctr = ((float(xyxy[0]) + float(xyxy[2])) / 2,
                           (float(xyxy[1]) + float(xyxy[3])) / 2)
                    tr = (self.team.predict_team(col)
                          if col is not None and self.team._kmeans_team is not None else 0)
                    reid_in.append({"bt_id": bt, "center": ctr, "team": tr, "color": col})
                sid_map = self.reid.update(fi, reid_in)

                # jersey-OCR auto-bind wearers → keyed by STABLE id (stays locked
                # even if the player leaves view and ByteTrack re-numbers them)
                if self.autobinder is not None:
                    ppl = [(sid_map.get(bt, bt), xyxy)
                           for (bt, xyxy, is_ref, col) in raw if not is_ref]
                    self.autobinder.step(fi, frame, ppl, self.fusion.sync)

                # ---- pass 2: process by stable id ----
                for (bt, xyxy, is_ref, col) in raw:
                    if is_ref:
                        an.draw_player(frame, xyxy, 0, label="ref", is_ref=True)
                        continue
                    sid = sid_map.get(bt, bt)
                    foot_img = foot_points(xyxy)[0]
                    z = _xform(foot_img)
                    meas = None if np.isnan(z[0]) else (float(z[0]), float(z[1]))
                    if self.traj_model is not None:
                        self._traj_hist[sid].append((foot_img[0] / W, foot_img[1] / H))
                    kf = kbank.update(sid, meas, dt)
                    t = self.team.assign_from_color(sid, col)   # team voting keyed by stable id
                    tag = str(bound.get(sid, sid))              # roster player id if bound
                    an.draw_player(frame, xyxy, t, label=tag, is_ref=False,
                                   wearable=sid in bound)
                    pos = kf.position
                    players_xy[sid] = pos
                    teams[sid] = t
                    if not np.isnan(pos[0]):     # only fold valid (on-pitch) tracks
                        vision_state[sid] = (kf.velocity, kf.acceleration)
                        if t in (1, 2):
                            radar_by_team[t].append(pos)

                ball_c = None if ball_box is None else centers(ball_box)[0]
                ball_pitch = None
                if ball_c is not None:
                    bp = _xform(ball_c)
                    ball_pitch = None if np.isnan(bp[0]) else (float(bp[0]), float(bp[1]))
                an.draw_ball(frame, ball_c)

                if ball_pitch is not None:
                    last_ball_seen = fi
                ball_used = ball_pitch
                if self.metric:
                    ball_kf.step(ball_pitch, dt)
                    # coast the ball briefly through misses, but if it's been lost
                    # too long don't attribute phantom possession to a ghost ball
                    stale = (fi - last_ball_seen) > 15
                    ball_used = ball_kf.position if (ball_kf.initialized and not stale) else None

                # possession uses the ACTUALLY-OBSERVED ball (ball_pitch), not the
                # Kalman-coasted ball_used — coasting to the last-seen position
                # biases possession to whichever team last had it.
                engine.update(fi, players_xy, teams, ball_pitch)
                control_grid = None
                if self.metric:
                    team_pos = {1: {k: v for k, v in players_xy.items() if teams.get(k) == 1},
                                2: {k: v for k, v in players_xy.items() if teams.get(k) == 2}}
                    snap = tactic.update(team_pos, ball_used, engine.possessing_team,
                                         compute_control=(fi % self.analytics_every == 0))
                    control_grid = snap.control_grid
                    goal_x = {t: (self.cfg.pitch.length if tactic.attack_dir(t) > 0 else 0.0)
                              for t in (1, 2)}
                    shot_detector.update(fi, ball_used, ball_kf.velocity,
                                         engine.possessing_team, goal_x, engine._holder_id)
                    self.live_heat.add(radar_by_team[1] + radar_by_team[2])
                if self.sensor_source is not None:
                    self.fusion.ingest_sensors(self.sensor_source.drain())
                if self.metric:
                    self.fusion.step(dt, vt, vision_state)
                self._shots = shot_detector.summary()

                if fi % self.analytics_every == 0:
                    injuries, recs = self._analytics(engine, tactic)
                    if self.traj_model is not None:
                        self._predict_paths(W, H, set(teams.keys()),
                                            _xform if self.metric else None)
                if fi % 200 == 0:
                    kbank.prune()
                if fi % 80 == 0 and fi > 0:
                    self.team.refit()          # batch re-fit team centroids as data grows

                for tid, path in self._pred_paths.items():
                    an.draw_prediction(frame, path)
                an.draw_possession(frame, engine.possession_pct())
                # radar/heatmap are pitch-metre views → only meaningful with calibration
                radar = radar_frame(radar_by_team, ball_pitch, self.cfg.pitch,
                                    control_grid=control_grid,
                                    pred_paths=list(self._pred_paths_pitch.values())
                                    ) if self.metric else None
                heat = self.live_heat.render() if self.metric else None
                panel = self._panel(engine, tactic, injuries, recs, fps_disp, H, bound)
                # build the composite once — used for BOTH the live window and the mp4
                composite = (compose_dashboard(frame, panel, radar, heat)
                             if (out_path or self.dash.windows) else None)
                if out_path and composite is not None:
                    if writer is None:
                        ch, cw = composite.shape[:2]
                        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                                 fps, (cw, ch))
                    writer.write(composite)
                key = self.dash.show_composite(composite)

                fi += 1
                now = time.time()
                fps_disp = 0.9 * fps_disp + 0.1 * (1.0 / max(now - t_prev, 1e-6))
                t_prev = now
                if fi % 100 == 0:
                    print(f"[realtime] {fi} frames, {fps_disp:.1f} FPS")
            else:
                key = self.dash.wait()

            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused

        cap.release()
        if frames_dropped:
            print(f"[realtime] real-time pacing dropped {frames_dropped} frames "
                  f"({100 * frames_dropped / max(fi, 1):.0f}%) to stay live")
        if writer is not None:
            writer.release()
            print(f"[realtime] wrote dashboard video → {out_path}")
        if self.sensor_source is not None:
            self.sensor_source.stop()
        self.dash.close()

        if stats_out:
            self._write_stats(stats_out, engine, tactic, shot_detector, bound)
        return engine, tactic, self.fusion

    def _write_stats(self, path, engine, tactic, shot_detector, bound):
        import json
        poss = engine.possession_pct()
        stats = {
            "metric": self.metric,
            "possession_pct": {str(k): round(v, 1) for k, v in poss.items()},
            "passes": {"team1": sum(p.team == 1 for p in engine.passes),
                       "team2": sum(p.team == 2 for p in engine.passes)},
            "wearables_bound": {str(t): p for t, p in bound.items()},
        }
        # Always write a player table. Pixel mode has no metres/load, but the
        # browser dashboard still needs a roster to draw.
        pixel_players = {
            str(tid): {
                "team": s.get("team", 0),
                "distance_m": s.get("distance", 0.0),
                "top_speed_ms": s.get("top_speed", 0.0),
                "hsr_m": 0.0, "sprints": 0, "metabolic_power_avg_wkg": 0.0,
                "energy_kcal": 0.0, "wearable": tid in bound,
            }
            for tid, s in engine.speed_distance().items()
        }
        stats["players"] = pixel_players
        if not self.metric:
            stats["note"] = (
                "pixel mode (no calibration): distances are pixels, not metres. "
                "Load / injury / pitch sketch accuracy need a calibration YAML."
            )
        if self.metric:
            im = self.injury_model
            players, injuries = {}, {}
            for tid, f in self.fusion.load.all_features().items():
                players[str(tid)] = {
                    "team": engine.team_of.get(tid, 0), "distance_m": f.total_distance,
                    "top_speed_ms": f.top_speed, "hsr_m": f.hsr_distance,
                    "sprints": f.sprint_count, "metabolic_power_avg_wkg": f.metabolic_power_avg,
                    "energy_kcal": f.energy_kcal, "wearable": tid in bound,
                }
                r = im.predict(f)
                injuries[str(tid)] = {"risk": r.risk, "level": r.level, "factors": r.factors}
            stats["players"] = players
            stats["injury_risk"] = injuries
            stats["shots_xg"] = shot_detector.summary()
            stats["substitution_watch"] = [
                a.as_dict() for a in self.subs.advise(
                    {int(t): j["risk"] for t, j in injuries.items()})[:5]]
            try:
                stats["tactics"] = {f"team{t}": tactic.report(t, poss.get(t, 0.0)).as_dict()
                                    for t in (1, 2)}
            except Exception:
                pass
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)
        print(f"[realtime] wrote stats → {path}")

    def _predict_paths(self, W, H, alive: set, xform=None):
        """Batch-predict each live track's future path with the LSTM (one forward
        pass, on the analytics cadence). Cached in image pixels AND — via ``xform``
        (image→pitch homography) — projected to pitch metres for the radar overlay."""
        import torch
        seqs, ids = [], []
        for tid in list(self._traj_hist):
            if tid not in alive:
                self._traj_hist.pop(tid, None)      # prune gone tracks
                continue
            hist = self._traj_hist[tid]
            if len(hist) >= self._traj_K + 1:
                xy = np.asarray(hist, dtype=np.float32)
                vel = np.diff(xy[-(self._traj_K + 1):], axis=0)   # (K, 2) normalised
                seqs.append(vel)
                ids.append((tid, xy[-1]))
        self._pred_paths, self._pred_paths_pitch = {}, {}
        if not seqs:
            return
        with torch.no_grad():
            pv = self.traj_model(torch.from_numpy(np.stack(seqs))).numpy()  # (B,H,2)
        for k, (tid, last) in enumerate(ids):
            fut = last + np.cumsum(pv[k], axis=0)     # normalised future positions
            img = [(float(p[0] * W), float(p[1] * H)) for p in fut]
            self._pred_paths[tid] = img
            if xform is not None:                     # project predicted path to pitch
                pit = []
                for px in img:
                    q = xform(np.asarray(px, float))
                    if not np.isnan(q[0]):
                        pit.append((float(q[0]), float(q[1])))
                if len(pit) >= 2:
                    self._pred_paths_pitch[tid] = pit

    # ------------------------------------------------------------------ #
    def _analytics(self, engine: MetricsEngine, tactic: TacticalEngine):
        """Heavy per-N-frame analytics: injury + substitution + tactics + recs."""
        injuries, inj_map = [], {}
        recs = []
        if self.metric:
            feats = self.fusion.load.all_features()
            for tid, wf in feats.items():
                r = self.injury_model.predict(wf)
                injuries.append((tid, r.risk, r.level))
                inj_map[tid] = r
            injuries.sort(key=lambda z: -z[1])
            # substitution watch: snapshot loads, rank fatigue-declining players
            self.subs.update(time.time(), feats)
            self._sub_watch = self.subs.watchlist(
                {tid: r.risk for tid, r in inj_map.items()})
            for a in self._sub_watch[:2]:
                recs.append(f"SUB WATCH #{a.player_id}: priority {a.priority:.2f} — "
                            + "; ".join(a.reasons[:3]))
            poss = engine.possession_pct()
            try:
                reports = [tactic.report(1, poss.get(1, 0.0)),
                           tactic.report(2, poss.get(2, 0.0))]
                self._formation = {r.team: r.formation for r in reports}
            except Exception:
                reports = []
            for rec in self.recommender.evaluate(inj_map, self.fusion.load.all_features(),
                                                 tactical_reports=reports):
                recs.append(rec.message)
        return injuries, recs[:6]

    def _panel(self, engine, tactic: TacticalEngine, injuries, recs, fps_disp, H, bound):
        poss = engine.possession_pct()
        risk_of = {tid: (rk, lv) for (tid, rk, lv) in injuries}
        wvit = {}
        for tid, pid in bound.items():
            s = self.fusion.sync.latest(pid)
            if s is not None:
                wvit[int(pid)] = {"hr": s.hr, "spo2": s.spo2}
        state = DashboardState(fps=fps_disp, possession=poss, metric=self.metric,
                               recommendations=recs, tagged_count=len(bound),
                               wearable_vitals=wvit)
        if self.metric:
            snap = tactic.snapshot
            for t in (1, 2):
                state.team_stats[t] = {
                    "control": snap.control.get(t, 0.0),
                    "formation": self._formation.get(t, ""),
                    "width": snap.width.get(t, 0.0),
                    "line_height": snap.line_height.get(t, 0.0),
                    "pressing": float(snap.pressing.get(t, 0)),
                    "att_third": (snap.thirds.get(t, {}) or {}).get("att", 0),
                    "phase": snap.phase.get(t, "—"),
                    "xg": self._shots.get(t, {}).get("xg", 0.0),
                    "shots": self._shots.get(t, {}).get("shots", 0),
                }
            rows = []
            for tid, f in self.fusion.load.all_features().items():
                rk, lv = risk_of.get(tid, (0.0, "low"))
                rows.append({"tag": bound.get(tid, tid), "team": engine.team_of.get(tid, 0),
                             "distance": f.total_distance, "top_speed": f.top_speed,
                             "metabolic": f.metabolic_power_avg, "risk": rk, "level": lv,
                             "wearable": tid in bound})
            state.players = sorted(rows, key=lambda r: -r["distance"])
        else:
            sd = engine.speed_distance()
            state.players = sorted(
                ({"tag": tid, "team": s["team"], "distance": s["distance"],
                  "top_speed": s["top_speed"]} for tid, s in sd.items()),
                key=lambda r: -r["distance"])
        return render_dashboard_panel(state)   # fixed PANEL_H → compact composite


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    import argparse
    from .sensors import SimulatedSensorSource

    p = argparse.ArgumentParser(description="Near-real-time soccer analytics dashboard.")
    p.add_argument("--video", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--calibration", default=None, help="pitch calibration .yaml (metric mode)")
    p.add_argument("--out", default=None, help="composite dashboard mp4 output path")
    p.add_argument("--stats", default=None, help="write final stats.json to this path")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--no-window", action="store_true", help="headless (no GUI windows)")
    p.add_argument("--simulate-sensors", action="store_true",
                   help="attach a simulated wearable stream + auto-bind tracks 1..N")
    p.add_argument("--players", type=int, default=22, help="player count for sensor sim")
    p.add_argument("--wearable-endpoint", type=int, default=0, metavar="PORT",
                   help="run an HTTP/WS wearable ingest endpoint on this port")
    p.add_argument("--sensor-hub", default="", metavar="URL",
                   help="poll the ESP32 sensor hub, e.g. http://127.0.0.1:8081")
    p.add_argument("--esp32", default="", help="start the hub in-process and connect to this ESP32 IP")
    p.add_argument("--player-id", type=int, default=7, help="wearable roster id (must match --roster)")
    p.add_argument("--roster", default="", metavar="track:player,…",
                   help="manually bind vision track ids to wearable player ids")
    p.add_argument("--roster-numbers", default="", metavar="jersey:player,…",
                   help="jersey number → player id; enables OCR auto-binding of wearers")
    args = p.parse_args(argv)

    cfg = PipelineConfig(weights=args.weights, conf=args.conf, imgsz=args.imgsz,
                         device=args.device)
    if args.calibration:
        from .pitch import load_calibration
        view = load_calibration(args.calibration, cfg.pitch)
    else:
        view = ViewTransformer()

    def _parse_pairs(s):
        return {int(a): int(b) for a, b in
                (p.split(":") for p in s.split(",") if ":" in p)} if s else None

    roster = _parse_pairs(args.roster)
    roster_numbers = _parse_pairs(args.roster_numbers)

    source = None
    if args.esp32 or args.sensor_hub:
        from .sensors import HubSensorSource
        hub_url = args.sensor_hub or "http://127.0.0.1:8081"
        if args.esp32:
            import threading

            def _run_hub():
                import sys
                from .sensors import hub as hubmod
                sys.argv = [
                    "soccer_analytics.hub",
                    "--esp32", args.esp32,
                    "--http-port", "8081",
                    "--player-id", str(args.player_id),
                ]
                hubmod.main()

            threading.Thread(target=_run_hub, daemon=True).start()
            print("[realtime] sensor hub starting on http://127.0.0.1:8081/")
            time.sleep(1.5)
        source = HubSensorSource(hub_url, player_id=args.player_id)
        roster = roster or {args.player_id: args.player_id}
    elif args.wearable_endpoint:
        from .sensors import endpoint_source
        source = endpoint_source(port=args.wearable_endpoint)
    elif args.simulate_sensors:
        ids = list(range(1, args.players + 1))
        source = SimulatedSensorSource(ids, hz=10.0)
        roster = roster or {i: i for i in ids}

    video = int(args.video) if args.video.isdigit() else args.video
    rt = RealtimePipeline(cfg, view, sensor_source=source, roster_map=roster,
                          roster_numbers=roster_numbers, dashboard=not args.no_window)
    rt.run(video, out_path=args.out, max_frames=args.max_frames, stats_out=args.stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
