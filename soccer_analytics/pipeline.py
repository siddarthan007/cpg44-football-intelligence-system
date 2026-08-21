"""End-to-end pipeline: video → tracking → team/pitch → metrics → heatmaps →
tactics, with an optional annotated output video.

Design (two phases, so team colours are fit over the WHOLE match and the ball is
interpolated with full temporal context — both more robust than the per-frame,
first-frame-only approach in the reference repos):

  Phase 1  analyse each frame: detect + track people, pick the ball, estimate
           camera motion, cache per-player shirt colour + foot pixel.
  Between   fit the 2-team colour model over all samples; interpolate the ball.
  Post      per frame: vote team, camera-correct + homography → pitch metres,
           feed metrics + tactics.
  Phase 2  (optional --render) redraw annotations onto the video.

Only lightweight per-frame arrays are cached (never the images), so memory stays
small even for long clips.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import annotate as an
from . import heatmap as hm
from .catapult import LoadEngine
from .core import PLAYER, REFEREE, Detections, centers, foot_points
from .config import PipelineConfig, PitchConfig
from .device import resolve_device
from .filters import BallKalman, KalmanBank
from .metrics import MetricsEngine
from .reid import ReIDManager
from .shots import ShotDetector
from .sensors.injury import HeuristicInjuryModel
from .substitution import SubstitutionAdvisor
from .tactical_engine import TacticalEngine
from .team_assign import TeamAssigner
from .tracker import Detector, SoccerTracker, pick_ball, interpolate_ball
from .view import CameraMotionEstimator, ViewTransformer


@dataclass
class FrameRecord:
    players: List[tuple] = field(default_factory=list)  # (tid, xyxy, class_id, color)
    ball_px: Optional[np.ndarray] = None                # ball box [x1,y1,x2,y2]
    cam_offset: Tuple[float, float] = (0.0, 0.0)        # cumulative camera shift (px)


def _open_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, (h, w), n


def analyze(video: str, cfg: PipelineConfig, view: ViewTransformer,
            out_dir: str, max_frames: int = 0, render: bool = False) -> dict:

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap, fps, (H, W), n_total = _open_video(video)
    dev = resolve_device(cfg.device)
    print(f"[analyze] compute: {dev.name} ({dev.device})")
    detector = Detector(cfg.weights, imgsz=cfg.imgsz, device=dev.device,
                        person_conf=cfg.conf, ball_conf=min(cfg.conf, 0.1),
                        half=dev.half_ok)
    tracker = SoccerTracker(frame_rate=int(round(fps)))
    cam = CameraMotionEstimator((H, W))
    team = TeamAssigner()

    # ---------- Phase 1: analyse ----------
    records: List[FrameRecord] = []
    color_samples: List[np.ndarray] = []
    cum = np.array([0.0, 0.0])
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and fi >= max_frames):
            break
        det = detector.detect(frame)
        tracked = tracker.update(det)
        ball_box = pick_ball(det)

        dx, dy = cam.update(frame)
        cum = cum + np.array([dx, dy])

        rec = FrameRecord(ball_px=ball_box, cam_offset=(float(cum[0]), float(cum[1])))
        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i])
            xyxy = tracked.xyxy[i]
            cls = int(tracked.class_id[i])
            color = None
            if cls == PLAYER:
                color = team.shirt_color(frame, xyxy)
                if color is not None:
                    color_samples.append(color)
            rec.players.append((tid, xyxy, cls, color))
        records.append(rec)
        fi += 1
        if fi % 100 == 0:
            print(f"[analyze] {fi}/{n_total or '?'} frames")
    cap.release()
    n = len(records)
    print(f"[analyze] done: {n} frames @ {fps:.2f} fps, {len(color_samples)} colour samples")

    # ---------- fit team model + interpolate ball ----------
    if len(color_samples) >= 2:
        team.fit(color_samples)
        team_ok = True
    else:
        print("[warn] too few colour samples — team assignment disabled")
        team_ok = False

    ball_boxes = [r.ball_px for r in records]
    ball_boxes = interpolate_ball(ball_boxes)
    ball_centers = [None if b is None else centers(b)[0] for b in ball_boxes]

    # ---------- post: teams, pitch coords, metrics, tactics ----------
    metric = view.is_metric
    poss_dist = 2.0 if metric else 70.0
    engine = MetricsEngine(fps=fps, metric=metric, possession_dist=poss_dist,
                           max_speed_mps=cfg.max_speed_mps)
    tactic = TacticalEngine(cfg.pitch)

    def to_pitch(px_xy, offset):
        adj = np.array(px_xy, float) - np.array(offset, float)   # camera-correct
        return view.transform([adj])[0]

    # Kalman-smooth trajectories → accurate velocity/acceleration; Catapult load
    kbank = KalmanBank(max_speed=cfg.max_speed_mps if metric else 1e9)
    reid = ReIDManager()
    load = LoadEngine()
    subs = SubstitutionAdvisor()
    ball_kf = BallKalman()
    shots = ShotDetector(pitch_length=cfg.pitch.length, pitch_width=cfg.pitch.width)
    dt = 1.0 / fps if fps > 0 else 0.04
    running_possession: List[Dict[int, float]] = []
    per_frame_render: List[dict] = []

    for fidx, rec in enumerate(records):
        players_xy: Dict[int, Tuple[float, float]] = {}
        teams: Dict[int, int] = {}
        render_players = []
        radar_by_team: Dict[int, list] = {1: [], 2: []}

        prepared = []
        reid_input = []
        for (bt_id, xyxy, cls, color) in rec.players:
            is_ref = cls == REFEREE
            if is_ref or not team_ok:
                assigned_team = 0
            else:
                assigned_team = team.assign_from_color(bt_id, color)
            prepared.append((bt_id, xyxy, cls, color, assigned_team))
            if not is_ref:
                box_center = centers(np.asarray(xyxy).reshape(1, 4))[0]
                reid_input.append({
                    "bt_id": bt_id,
                    "center": (float(box_center[0]), float(box_center[1])),
                    "team": assigned_team,
                    "color": color,
                })
        stable_ids = reid.update(fidx, reid_input)

        for (bt_id, xyxy, cls, color, t) in prepared:
            is_ref = cls == REFEREE
            tid = bt_id if is_ref else stable_ids.get(bt_id, bt_id)
            foot = foot_points(xyxy)[0]
            pitch_xy = to_pitch(foot, rec.cam_offset)
            meas = None if np.isnan(pitch_xy[0]) else (float(pitch_xy[0]), float(pitch_xy[1]))
            kf = kbank.update(tid, meas, dt)
            if not is_ref:
                pos = kf.position
                players_xy[tid] = pos
                teams[tid] = t
                if metric and not np.isnan(pos[0]):
                    load.update(tid, dt, kf.velocity, kf.acceleration)
                if t in (1, 2) and not np.isnan(pos[0]):
                    radar_by_team[t].append(pos)
            render_players.append((xyxy, t, tid, is_ref))

        ball_c = ball_centers[fidx]
        ball_pitch = None
        if ball_c is not None:
            bp = to_pitch(ball_c, rec.cam_offset)
            ball_pitch = (float(bp[0]), float(bp[1]))

        # Kalman-smooth the ball → stable position + velocity for possession/shots
        ball_used = ball_pitch
        if metric:
            ball_kf.step(ball_pitch, dt)
            if ball_kf.initialized:
                ball_used = ball_kf.position

        engine.update(fidx, players_xy, teams, ball_used)
        if metric:
            tactic.update({1: {k: v for k, v in players_xy.items() if teams.get(k) == 1},
                           2: {k: v for k, v in players_xy.items() if teams.get(k) == 2}},
                          ball_used, engine.possessing_team,
                          compute_control=(fidx % 10 == 0))
            goal_x = {t: (cfg.pitch.length if tactic.attack_dir(t) > 0 else 0.0)
                      for t in (1, 2)}
            shots.update(fidx, ball_kf.position if ball_kf.initialized else None,
                         ball_kf.velocity, engine.possessing_team, goal_x,
                         engine._holder_id)
        running_possession.append(engine.possession_pct())
        if metric and fidx % 50 == 0:
            subs.update(fidx / fps, load.all_features())   # fatigue snapshots
        if fidx % 200 == 0:
            kbank.prune()
        if render:
            per_frame_render.append({"players": render_players, "ball": ball_c,
                                     "radar": radar_by_team, "ball_pitch": ball_pitch})

    # ---------- outputs ----------
    possession = engine.possession_pct()
    maximum_players_in_frame = max(
        (sum(int(player[2]) == PLAYER for player in record.players) for record in records),
        default=0,
    )
    stats = {
        "video": video, "frames": n, "fps": fps, "metric": metric,
        "provenance": {
            "kind": "vision_analysis",
            "generated_at": time.time(),
            "input_video": str(Path(video).resolve()),
            "detector_weights": str(Path(cfg.weights).resolve()),
            "wearable_source": "none",
            "synthetic_telemetry": False,
        },
        "tracking": {
            "tracker": "ByteTrack",
            "stable_ids": reid._next - 1,
            "maximum_simultaneous_players": maximum_players_in_frame,
            "reidentifications": reid.reids,
            "raw_ball_detection_rate": round(
                sum(position is not None for position in [r.ball_px for r in records]) / max(n, 1), 4
            ),
        },
        "possession_pct": {str(k): round(v, 1) for k, v in possession.items()},
        "passes": {"team1": sum(p.team == 1 for p in engine.passes),
                   "team2": sum(p.team == 2 for p in engine.passes)},
    }
    if metric:
        # accurate Catapult-style per-player load (metabolic power, zones, …)
        indicator_model = HeuristicInjuryModel()
        feats = load.all_features()
        players_out, indicator_out = {}, {}
        for tid, f in feats.items():
            players_out[str(tid)] = {
                "team": engine.team_of.get(tid, 0),
                "distance_m": f.total_distance, "hsr_m": f.hsr_distance,
                "sprints": f.sprint_count, "accel_efforts": f.accel_efforts,
                "decel_efforts": f.decel_efforts, "top_speed_ms": f.top_speed,
                "metabolic_power_avg_wkg": f.metabolic_power_avg,
                "high_metabolic_m": f.high_metabolic_distance,
                "energy_kcal": f.energy_kcal, "player_load": f.player_load,
                "distance_by_zone": f.distance_by_zone,
            }
            indicator = indicator_model.predict(f)
            indicator_out[str(tid)] = {
                "score": indicator.risk,
                "severity": indicator.level,
                "factors": indicator.factors,
                "medical_prediction": False,
            }
        stats["players"] = players_out
        stats["load_indicators"] = indicator_out
        stats["shots_xg"] = shots.summary()
        stats["substitution_watch"] = [
            a.as_dict() for a in subs.advise(
                {int(t): j["score"] for t, j in indicator_out.items()})[:5]]
        r1 = tactic.report(1, possession.get(1, 0.0))
        r2 = tactic.report(2, possession.get(2, 0.0))
        stats["tactics"] = {
            f"team{report.team}": report.as_dict()
            for report in (r1, r2) if report.evidence_ready
        }
        speed_cap_fraction = sum(
            abs(player["top_speed_ms"] - cfg.max_speed_mps) < 0.02
            for player in players_out.values()
        ) / max(len(players_out), 1)
        fragmentation_ratio = (reid._next - 1) / max(maximum_players_in_frame, 1)
        warnings = []
        if not players_out:
            warnings.append("no confirmed player tracks were available")
        if not r1.evidence_ready or not r2.evidence_ready:
            warnings.append("tactical advice withheld because team evidence was insufficient")
        if fragmentation_ratio > 1.5:
            warnings.append("stable identity count exceeds the simultaneous-player baseline")
        if speed_cap_fraction > 0.2:
            warnings.append("many tracks reached the configured speed cap; review homography")
        stats["quality"] = {
            "status": "review" if warnings else "usable",
            "identity_fragmentation_ratio": round(fragmentation_ratio, 3),
            "speed_cap_fraction": round(speed_cap_fraction, 3),
            "warnings": warnings,
        }
    else:
        stats["players"] = {str(tid): s for tid, s in engine.speed_distance().items()}
        stats["note"] = "pixel mode (no calibration): distances/speeds in pixels; " \
                        "load and metabolic power need a pitch calibration."

    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[out] wrote {out/'stats.json'}")

    # heatmaps (need metric pitch coords)
    if metric:
        _write_heatmaps(engine, cfg.pitch, out)

    # optional annotated video
    if render:
        _render_video(video, per_frame_render, running_possession, cfg, out, fps)

    return stats


def _write_heatmaps(engine: MetricsEngine, pc: PitchConfig, out: Path):
    team_pts: Dict[int, list] = {1: [], 2: []}
    for tid, seq in engine.positions.items():
        t = engine.team_of.get(tid, 0)
        if t in (1, 2):
            team_pts[t].extend([(x, y) for (_, x, y) in seq])
    for t in (1, 2):
        if team_pts[t]:
            hm.generate_heatmap(team_pts[t], pc, str(out / f"heatmap_team{t}.png"),
                                title=f"Team {t} occupancy")
    print(f"[out] wrote heatmaps to {out}")


def _render_video(video, per_frame, running_possession, cfg, out: Path, fps):
    cap, _, (H, W), _ = _open_video(video)
    writer = cv2.VideoWriter(str(out / "annotated.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    fidx = 0
    while fidx < len(per_frame):
        ok, frame = cap.read()
        if not ok:
            break
        rf = per_frame[fidx]
        for (xyxy, t, tid, is_ref) in rf["players"]:
            an.draw_player(frame, xyxy, t, label=str(tid), is_ref=is_ref)
        an.draw_ball(frame, rf["ball"])
        an.draw_possession(frame, running_possession[fidx])
        radar = hm.radar_frame(rf["radar"], rf["ball_pitch"], cfg.pitch)
        an.overlay_radar(frame, radar)
        writer.write(frame)
        fidx += 1
    cap.release()
    writer.release()
    print(f"[out] wrote {out/'annotated.mp4'}")


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run the soccer analytics vision pipeline.")
    p.add_argument("--video", required=True)
    p.add_argument("--weights", required=True, help="trained YOLOv8 best.pt")
    p.add_argument("--out", default="analytics_out")
    p.add_argument("--calibration", default=None,
                   help="pitch calibration .yaml (from soccer_analytics.calibrate). "
                        "Without it, metrics stay in pixels and tactics/heatmaps are skipped.")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--render", action="store_true", help="write annotated.mp4")
    args = p.parse_args(argv)

    cfg = PipelineConfig(weights=args.weights, conf=args.conf, imgsz=args.imgsz,
                         device=args.device)
    if args.calibration:
        from .pitch import load_calibration
        view = load_calibration(args.calibration, cfg.pitch)
        print(f"[calib] loaded homography from {args.calibration} (metric mode)")
    else:
        view = ViewTransformer()  # identity, pixel mode
        print("[calib] no calibration — pixel mode (no meters/tactics/heatmaps)")

    analyze(args.video, cfg, view, args.out, max_frames=args.max_frames, render=args.render)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
