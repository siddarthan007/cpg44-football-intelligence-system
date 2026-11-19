"""Near-real-time streaming pipeline with a live multi-window dashboard, Kalman
trajectory smoothing, Catapult-style load, and optional wearable fusion.

Single forward pass (no look-ahead) so it runs live:

    warmup (fit team colours) to per frame:
        detect(GPU) to track to team to homography to Kalman(pos/vel/accel)
        to possession/passes to Catapult load (metabolic power…) to fuse wearable
        to explainable load review to recommendations to dashboard

Physical-load / injury analytics require metric mode (a pitch calibration), since
metabolic power is defined in metres; in pixel mode the system still does
detection, tracking, teams and possession. Wearable samples arrive async via a
:class:`SensorSource` and are time-aligned per frame.
"""

from __future__ import annotations

import os
import json
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Optional

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
from .live_publish import LiveSnapshotPublisher
from .profiling import StageProfiler
from .readiness import ReadinessEngine
from .coach_engine import CoachEngine, Sample as CoachSample, VestSeries
from .pass_prediction import pass_options
from .reid import ReIDManager
from .shots import ShotDetector
from .substitution import SubstitutionAdvisor
from .tactical_engine import TacticalEngine
from .team_assign import TeamAssigner
from .tracker import Detector, SoccerTracker, pick_ball
from .view import CameraMotionEstimator, PitchHomographyTracker, StaticPitch, ViewTransformer
from .lens import undistort_frame
from .sensors import FusionEngine, HeuristicInjuryModel, RecommendationEngine, SensorVideoSync
from .sensors.source import SensorSource


class LatestFrameCapture:
    """Camera reader that keeps one frame so slow inference cannot build latency."""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open camera or stream source {source!r}")
        self.buffer_request_applied = bool(self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1))
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.cap.release()
            raise RuntimeError(f"camera or stream source {source!r} returned no frame")
        self._condition = threading.Condition()
        self._frame = frame
        self._arrival_epoch = time.time()
        self._arrival_mono = time.monotonic()
        self.last_frame_epoch = self._arrival_epoch
        self.last_frame_mono = self._arrival_mono
        self.overwritten_frames = 0
        self._sequence = 1
        self._delivered = 0
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                with self._condition:
                    self._running = False
                    self._condition.notify_all()
                return
            with self._condition:
                self.overwritten_frames += int(self._sequence > self._delivered)
                self._frame = frame
                self._arrival_epoch = time.time()
                self._arrival_mono = time.monotonic()
                self._sequence += 1
                self._condition.notify_all()

    def isOpened(self) -> bool:
        return self.cap.isOpened() and self._running

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._frame.shape[1])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._frame.shape[0])
        return float(self.cap.get(prop))

    def read(self):
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > self._delivered or not self._running,
                timeout=2.0,
            )
            if self._sequence <= self._delivered:
                return False, None
            self._delivered = self._sequence
            self.last_frame_epoch = self._arrival_epoch
            self.last_frame_mono = self._arrival_mono
            return True, self._frame

    def release(self) -> None:
        self._running = False
        self.cap.release()
        with self._condition:
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


class RealtimePipeline:
    def __init__(self, cfg: PipelineConfig, view: ViewTransformer,
                 sensor_source: Optional[SensorSource] = None,
                 roster_map: Optional[Dict[int, int]] = None,
                 roster_numbers: Optional[Dict[int, int]] = None,
                 injury_model=None, warmup: int = 40,
                 dashboard: bool = True, analytics_every: int = 10,
                 snapshot_every: int = 2, latency_budget_ms: int = 240,
                 product_api: str = "", camera_id: str = ""):
        self.cfg = cfg
        self.view = view
        self.metric = view.is_metric
        self.dev = resolve_device(cfg.device)
        print(f"[realtime] {self.dev.name} ({self.dev.device}) | "
              f"{'METRIC' if self.metric else 'PIXEL (no calibration)'} mode")

        self.detector = Detector(cfg.weights, imgsz=cfg.imgsz, device=self.dev.device,
                                 person_conf=cfg.conf, ball_conf=min(cfg.conf, 0.1),
                                 half=self.dev.half_ok, track_low_conf=cfg.track_low_conf)
        self.team = TeamAssigner()
        self.warmup = warmup

        sync = SensorVideoSync()
        if roster_map:
            sync.bind_many(roster_map)
        # vests without a manual binding are matched to their wearer by motion
        from .sensors.association import LiveAssociator
        self.associator = LiveAssociator()
        self._manual_players = set((roster_map or {}).values())
        self.vest_links: Dict[int, dict] = {}
        self.fusion = FusionEngine(sync)
        self.sensor_source = sensor_source
        self.injury_model = injury_model or HeuristicInjuryModel()
        self.injury_model_name = (
            f"outcome_{getattr(self.injury_model, '_backend', 'model')}"
            if self.injury_model.__class__.__name__ == "InjuryRiskModel"
            else "transparent_load_review"
        )
        self.recommender = RecommendationEngine()

        self.dash = Dashboard(windows=dashboard)
        self.live_heat = LiveHeatmap(cfg.pitch)
        self._position_quality = {}
        self._calibration_quality = {"valid": self.metric}
        self._team_calibration_warning = None
        self.analytics_every = analytics_every
        self.snapshot_every = max(1, int(snapshot_every))
        self.latency_budget_s = max(0.08, min(float(latency_budget_ms) / 1000.0, 1.0))
        self._formation = {1: "", 2: ""}
        self._shots = {1: {"shots": 0, "xg": 0.0}, 2: {"shots": 0, "xg": 0.0}}
        self.publisher = LiveSnapshotPublisher(product_api)
        self.camera_id = camera_id or None
        self.loop_count = 0
        self.profiler = StageProfiler()
        self.readiness = ReadinessEngine()
        # Live coach engine: rolling 10-minute window of 5 Hz camera samples and
        # quality-gated vest heart rate per roster-bound track.
        self.coach = CoachEngine(cfg.pitch.length, cfg.pitch.width)
        self._coach_samples: deque = deque(maxlen=3000)
        self._vest_rows: Dict[int, deque] = defaultdict(lambda: deque(maxlen=900))
        self._coach_cards: list = []
        self._coach_last_t = -1e9

        # LSTM trajectory prediction (optional), draws each player's predicted path
        self.traj_model, self._traj_K = None, 15
        traj_path = "runs/trajectory/traj_lstm.pt"
        if cfg.predict_trajectories and os.path.exists(traj_path):
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

        # jersey-number OCR to auto-bind wearers (optional; needs roster_numbers)
        self.autobinder = None
        if roster_numbers:
            from .jersey_ocr import JerseyReader, AutoBinder
            reader = JerseyReader(device=self.dev.device)
            if reader.available:
                self.autobinder = AutoBinder(reader, roster_numbers)

    # ------------------------------------------------------------------ #
    def _warmup(self, cap, rewind: bool = True):
        self._load_team_seeds()
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
            # Validate if calibrated seeds actually match the observed kit colors in this video
            if self.team._seeds is not None:
                matches = sum(1 for c in colors if self.team._predict_conf(c)[1])
                match_rate = matches / max(1, len(colors))
                if match_rate < 0.25:
                    self._team_calibration_warning = "Calibrated kit colours do not match this footage; review team calibration."
                    print(f"[realtime] WARNING: Calibrated seeds matched only {matches}/{len(colors)} ({match_rate:.0%}) samples in this video.")
                    print("[realtime] Keeping calibrated team labels; unmatched kits will abstain. Review kit colours.")
                else:
                    print(f"[realtime] Calibrated kit seeds confirmed ({matches}/{len(colors)} matching samples)")
            print(f"[realtime] team model fit on {len(colors)} colour samples")
        else:
            print("[realtime] insufficient colour samples; team assignment disabled")
        if rewind:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _load_team_seeds(self) -> None:
        """Anchor team clusters to kit colours calibrated on the review desk.

        ``configs/teams.json`` is written by the dashboard tagging page in RGB;
        the assigner works in OpenCV CIELAB, so convert once here. With seeds
        the live Team 1 / Team 2 tags match what the operator calibrated
        instead of depending on an arbitrary KMeans label order."""
        path = Path(__file__).resolve().parents[1] / "configs" / "teams.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not data.get("calibrated"):
            return
        import cv2 as _cv2
        seeds = {}
        for key, team in (("team_1", 1), ("team_2", 2)):
            rgb = (data.get(key) or {}).get("primary_color_rgb")
            if isinstance(rgb, list) and len(rgb) == 3 and all(
                isinstance(v, (int, float)) and 0 <= v <= 255 for v in rgb
            ):
                lab = _cv2.cvtColor(
                    np.asarray([[rgb]], dtype=np.uint8), _cv2.COLOR_RGB2LAB
                )[0, 0].astype(float)
                seeds[team] = lab.tolist()
        if len(seeds) == 2:
            self.team.seed_with_lab(seeds)
            print("[realtime] team model seeded from calibrated kit colours")

    def run(self, video: str, out_path: Optional[str] = None, max_frames: int = 0,
            stats_out: Optional[str] = None, fused_out: Optional[str] = None,
            loop_file: bool = False, video_start_epoch: Optional[float] = None):
        self._capture = self._writer = self._fused_file = None
        try:
            return self._run(video, out_path, max_frames, stats_out, fused_out,
                             loop_file, video_start_epoch)
        finally:
            self.publisher.close()
            if self._capture is not None:
                self._capture.release()
            if self._writer is not None:
                self._writer.release()
            if self._fused_file is not None:
                self._fused_file.close()
            if self.sensor_source is not None:
                self.sensor_source.stop()
            self.dash.close()

    def _run(self, video: str, out_path: Optional[str] = None, max_frames: int = 0,
             stats_out: Optional[str] = None, fused_out: Optional[str] = None,
             loop_file: bool = False, video_start_epoch: Optional[float] = None):
        file_playback = isinstance(video, str) and Path(video).is_file()
        if video_start_epoch is not None and not np.isfinite(video_start_epoch):
            raise ValueError("video_start_epoch must be finite")
        self._wearable_time_supported = not file_playback or video_start_epoch is not None
        self._video_timestamp = None
        if isinstance(video, str) and video.startswith("phone://"):
            from .phone_camera import PhoneCamera
            cap = PhoneCamera(video)
        else:
            cap = cv2.VideoCapture(video) if file_playback else LatestFrameCapture(video)
        self._capture = cap
        if not cap.isOpened():
            raise FileNotFoundError(
                f"cannot open video source {video!r}. WSL has no webcam by default, "
                f"pass a video file path (e.g. demo/sample_match.mp4) instead of an index.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        dt = 1.0 / fps
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)

        # larger lost buffer to ByteTrack itself recovers longer gaps; the ReID
        # layer then handles the very-long / out-of-view cases.
        tracker = SoccerTracker(frame_rate=int(round(fps)), lost_track_buffer=self.cfg.track_buffer,
                                track_activation_threshold=self.cfg.conf,
                                backend=self.cfg.tracker, track_low_conf=self.cfg.track_low_conf)
        self.reid.frame_rate = fps
        # metric mode: per-frame homography that follows the camera pan (replaces
        # the static homography + separate camera-motion pass, same optical-flow
        # cost, no FPS hit). pixel mode: keep the translation-only camera comp.
        fixed = getattr(self, "camera_mode", "auto") == "fixed"
        if self.metric and fixed:
            pitch_tracker = StaticPitch(self.view.H, self.cfg.pitch.length, self.cfg.pitch.width)
        else:
            pitch_tracker = PitchHomographyTracker(
                self.view.H, (H, W), pitch_length=self.cfg.pitch.length,
                pitch_width=self.cfg.pitch.width) if self.metric else None
        cam = None if self.metric else CameraMotionEstimator((H, W))
        engine = MetricsEngine(fps=fps, metric=self.metric,
                               possession_dist=2.0 if self.metric else 70.0,
                               max_speed_mps=self.cfg.max_speed_mps,
                               history_limit=max(300, round(fps * 120)))
        engine.pitch_width = self.cfg.pitch.width
        engine.pitch_length = self.cfg.pitch.length
        tactic = TacticalEngine(self.cfg.pitch)
        kbank = KalmanBank(max_speed=self.cfg.max_speed_mps if self.metric else 1e9)
        ball_kf = BallKalman()
        shot_detector = ShotDetector(pitch_length=self.cfg.pitch.length,
                                     pitch_width=self.cfg.pitch.width)

        if self.sensor_source is not None:
            self.sensor_source.start()

        print("[realtime] warmup…")
        self._warmup(cap, rewind=file_playback)

        writer = None   # created lazily once we know the composite size
        fused_fh = None
        if fused_out:
            fused_path = Path(fused_out)
            fused_path.parent.mkdir(parents=True, exist_ok=True)
            fused_fh = fused_path.open("a", encoding="utf-8")
            self._fused_file = fused_fh
            print(f"[realtime] recording aligned fusion snapshots to {fused_path}")

        cum = np.array([0.0, 0.0])
        fi = 0
        last_ball_seen = -999          # frame index the ball was last actually detected
        last_ball_px = None            # last ball image centre (for the ROI zoom search)
        ball_miss = 0
        # A prerecorded file shown in either the OpenCV or React live view follows
        # its source clock. If inference falls behind, frames are skipped instead
        # of letting the dashboard become an increasingly delayed slow-motion view.
        self.source_kind = "recorded_file" if file_playback else "live_camera"
        pace_live = file_playback and (self.dash.windows or bool(self.publisher.url))
        t_start = time.time()
        frames_dropped = 0
        t_prev = time.time()
        last_capture_time = None
        fps_disp = 0.0
        injuries, recs = [], []
        paused = False
        while True:
            if not paused:
                # real-time pacing: if we're behind the video clock, drop frames
                dropped_now = 0
                if pace_live:
                    target_elapsed = fi / fps
                    elapsed = time.time() - t_start
                    if elapsed < target_elapsed:
                        time.sleep(target_elapsed - elapsed)
                    behind = (time.time() - t_start) - fi / fps
                    while behind > self.latency_budget_s:
                        if not cap.grab():
                            break
                        fi += 1
                        dropped_now += 1
                        behind -= 1.0 / fps
                    frames_dropped += dropped_now
                dt = (1 + dropped_now) / fps       # true elapsed time between frames
                if max_frames and fi >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok and loop_file and file_playback:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if ok:
                        self.loop_count += 1
                if not ok:
                    break
                frame = undistort_frame(frame, getattr(self, "lens_k1", 0.0))
                # wall-clock capture time so vision aligns with the wearable's
                # epoch-stamped samples (both live); dt uses video frame spacing.
                vt = (video_start_epoch + fi / fps if video_start_epoch is not None and self.loop_count == 0
                      else None) if file_playback else cap.last_frame_epoch
                self._video_timestamp = vt
                capture_time = time.monotonic() if file_playback else cap.last_frame_mono
                if not file_playback and last_capture_time is not None:
                    dt = max(1e-3, capture_time - last_capture_time)
                last_capture_time = capture_time

                self.profiler.start()
                det = self.detector.detect(frame)
                self.profiler.mark("detection")
                tracked = tracker.update(det, frame)
                self.profiler.mark("association")
                ball_box = pick_ball(det)
                # ROI zoom search when the full-frame pass misses the small ball
                if ball_box is not None:
                    last_ball_px = centers(ball_box)[0]
                    ball_miss = 0
                elif last_ball_px is not None and ball_miss < 40 and ball_miss % 3 == 0:
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
                # Homography updates every sixth frame. The held mapping still
                # transforms every frame while the camera movement stays smooth.
                if self.metric:
                    pitch_tracker.update(frame, det.of_class(PLAYER, REFEREE).xyxy)
                    self._calibration_quality = pitch_tracker.quality()
                    def _xform(px):
                        return pitch_tracker.transform([px])[0]
                else:
                    dx, dy = cam.update(frame)
                    cum = cum + np.array([dx, dy])
                    def _xform(px):
                        return self.view.transform([np.asarray(px, float) - cum])[0]

                self.profiler.mark("ball_and_camera_mapping")
                bound = self.fusion.sync._track2player
                players_xy, teams, vision_state = {}, {}, {}
                radar_by_team = {1: [], 2: []}

                # ---- pass 1: per-track jersey colour (every 8th frame) ----
                compute_color = (fi % 8 == 0)
                raw = []
                self._position_quality = {}
                for i in range(len(tracked)):
                    bt = int(tracked.tracker_id[i])
                    xyxy = tracked.xyxy[i]
                    is_ref = int(tracked.class_id[i]) == REFEREE
                    col = None if is_ref else (self.team.shirt_color(frame, xyxy)
                                               if compute_color or bt not in self.reid.t2s else None)
                    raw.append((bt, xyxy, is_ref, col))

                # ---- Re-ID: map ByteTrack ids to persistent stable ids ----
                reid_in = []
                for (bt, xyxy, is_ref, col) in raw:
                    if is_ref:
                        continue
                    ctr = ((float(xyxy[0]) + float(xyxy[2])) / 2,
                           (float(xyxy[1]) + float(xyxy[3])) / 2)
                    tr = 0
                    if col is not None and (self.team._kmeans_team is not None or self.team._seeds is not None):
                        candidate, confident = self.team._predict_conf(col)
                        tr = candidate if confident else 0
                    projected = _xform(foot_points(xyxy)[0]) if self.metric else None
                    pitch = tuple(projected) if projected is not None and np.isfinite(projected).all() else None
                    reid_in.append({"bt_id": bt, "center": ctr, "team": tr, "color": col, "pitch": pitch})
                sid_map = self.reid.update(fi, reid_in)

                # jersey-OCR auto-bind wearers to keyed by STABLE id (stays locked
                # even if the player leaves view and ByteTrack re-numbers them)
                if self.autobinder is not None:
                    ppl = [(sid_map.get(bt, bt), xyxy, self.team._final.get(sid_map.get(bt, bt), 0))
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
                    meas = tuple(map(float, z)) if np.isfinite(z).all() else None
                    if self.traj_model is not None:
                        self._traj_hist[sid].append((foot_img[0] / W, foot_img[1] / H))
                    kf = kbank.update(sid, meas, dt, time_s=fi / fps if file_playback else capture_time)
                    t = self.team.assign_from_color(sid, col)   # team voting keyed by stable id
                    tag = str(bound.get(sid, sid))              # roster player id if bound
                    an.draw_player(frame, xyxy, t, label=tag, is_ref=False,
                                   wearable=sid in bound)
                    pos = kf.position if kf.measurement_accepted else (np.nan, np.nan)
                    self._position_quality[sid] = (
                        "unprojectable" if not self.metric else "observed" if kf.measurement_accepted
                        else "rejected" if meas is not None else "unprojectable"
                    )
                    players_xy[sid] = pos
                    teams[sid] = t
                    if not np.isnan(pos[0]):     # only fold valid (on-pitch) tracks
                        vision_state[sid] = (kf.velocity, kf.acceleration)
                        if t in (1, 2):
                            radar_by_team[t].append(pos)
                kbank.prune(players_xy)

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
                # Kalman-coasted ball_used, coasting to the last-seen position
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
                    self.live_heat.add([p for p in players_xy.values() if np.isfinite(p).all()], dt=dt)
                self.profiler.mark("identity_position_and_tactics")
                if self.sensor_source is not None:
                    batch = self.sensor_source.drain()
                    for smp in batch:
                        if smp.accel is not None and smp.player_id not in self._manual_players:
                            self.associator.add_vest(smp.player_id, smp.t, smp.accel)
                    self.fusion.ingest_sensors(batch)
                    if vt is not None:
                        for sid, (vel, _acc) in vision_state.items():
                            self.associator.add_track(sid, vt, float(np.hypot(*vel)))
                        for pid, tid in self.associator.step(vt).items():
                            self.fusion.sync.bind(tid, pid)
                            m = self.associator.last[pid]
                            self.vest_links[pid] = {"track": tid, "score": m.score, "margin": m.margin, "lag_s": m.lag_s}
                            print(f"[vest] vest {pid} is worn by track {tid} (r={m.score:.2f}, clock lag {m.lag_s:+.1f} s)", flush=True)
                if self.metric:
                    self.fusion.step(dt, vt, vision_state)
                self._shots = shot_detector.summary()
                self.profiler.mark("sensor_fusion")

                if fi % self.analytics_every == 0:
                    injuries, recs = self._analytics(engine, tactic)
                    if self.traj_model is not None:
                        self._predict_paths(W, H, set(teams.keys()),
                                            _xform if self.metric else None)
                if fi % self.snapshot_every == 0 and (self.publisher.url or fused_fh is not None):
                    live_snapshot = self._live_snapshot(
                        fi, fps, fps_disp, engine, tactic, players_xy, teams,
                        vision_state, ball_pitch, injuries, recs, bound,
                    )
                    self.publisher.publish(live_snapshot)
                    if fused_fh is not None:
                        fused_fh.write(json.dumps(live_snapshot, allow_nan=False) + "\n")
                        fused_fh.flush()
                if fi % 200 == 0:
                    kbank.prune()
                if fi % 80 == 0 and fi > 0:
                    self.team.refit()          # batch re-fit team centroids as data grows
                    # Seed alignment keeps cluster order stable across refits,
                    # so per-track votes stay valid and tags do not flicker.

                for tid, path in self._pred_paths.items():
                    an.draw_prediction(frame, path)
                an.draw_possession(frame, engine.possession_pct())
                self.publisher.publish_frame(frame)
                self.profiler.mark("analytics_and_publish")
                composite = None
                if out_path or self.dash.windows:
                    # Radar, heatmap and the OpenCV side panel are not needed by the
                    # React dashboard. Keep them off the headless per-frame path.
                    radar = radar_frame(
                        radar_by_team,
                        ball_pitch,
                        self.cfg.pitch,
                        control_grid=control_grid,
                        pred_paths=list(self._pred_paths_pitch.values()),
                    ) if self.metric else None
                    heat = self.live_heat.render() if self.metric else None
                    panel = self._panel(engine, tactic, injuries, recs, fps_disp, H, bound)
                    composite = compose_dashboard(frame, panel, radar, heat)
                if out_path and composite is not None:
                    if writer is None:
                        ch, cw = composite.shape[:2]
                        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                                 fps, (cw, ch))
                        self._writer = writer
                    writer.write(composite)
                key = self.dash.show_composite(composite)
                self.profiler.mark("render_and_output")

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
        self.publisher.close()
        if frames_dropped:
            print(f"[realtime] real-time pacing dropped {frames_dropped} frames "
                  f"({100 * frames_dropped / max(fi, 1):.0f}%) to stay live")
        if writer is not None:
            writer.release()
            print(f"[realtime] wrote dashboard video to {out_path}")
        if fused_fh is not None:
            fused_fh.close()
        if self.sensor_source is not None:
            self.sensor_source.stop()
        self.dash.close()

        if stats_out:
            self._write_stats(stats_out, engine, tactic, shot_detector, bound)
        return engine, tactic, self.fusion

    def _live_snapshot(self, frame_index, source_fps, processing_fps, engine, tactic,
                       players_xy, teams, vision_state, ball_pitch, injuries, recs, bound):
        """Serialize only measurements produced by the current processing frame."""
        totals = engine.speed_distance()
        indicator_of = {int(tid): (float(score), level) for tid, score, level in injuries}
        load_features = self.fusion.load.all_features() if self.metric else {}
        def finite(value):
            try:
                number = float(value)
                return number if np.isfinite(number) else None
            except (TypeError, ValueError):
                return None

        rows = []
        for tid, position in players_xy.items():
            velocity = vision_state.get(tid, ((0.0, 0.0), None))[0]
            speed = float(np.linalg.norm(velocity)) if tid in vision_state and self.metric else None
            aggregate = totals.get(tid, {})
            load = load_features.get(tid)
            wearable_id = bound.get(tid)
            wearable = self.fusion.aligned.get(tid)
            alignment = self.fusion.alignment.get(tid, {})
            indicator_score, indicator_level = indicator_of.get(tid, (None, None))
            rows.append({
                "global_player_id": f"TRACK_{int(tid)}",
                "track_id": int(tid),
                "player_id": int(wearable_id) if wearable_id is not None else None,
                "jersey": self.autobinder.votes.confident_number(tid) if self.autobinder is not None else None,
                "team": int(teams.get(tid, 0)),
                "team_id": f"TEAM_{int(teams.get(tid, 0))}",
                "position_status": getattr(self, "_position_quality", {}).get(tid, "unknown"),
                "identity_status": "roster_bound" if wearable_id is not None else "track_only",
                "x": float(position[0]) if self.metric and np.isfinite(position[0]) else None,
                "y": float(position[1]) if self.metric and np.isfinite(position[1]) else None,
                "speed_mps": speed,
                "top_speed_mps": float(aggregate.get("top_speed", 0.0)) if self.metric else None,
                "distance_m": float(aggregate.get("distance", 0.0)) if self.metric else None,
                "wearable": wearable is not None,
                "wearable_metrics": ({
                    "hr": wearable.hr,
                    "spo2": wearable.spo2,
                    "signal_quality": wearable.signal_quality,
                    "clock_synced": wearable.clock_synced,
                    "alignment_offset_ms": alignment.get("offset_ms"),
                    "transport_delay_ms": wearable.transport_delay_ms,
                    "sample_age_ms": wearable.sample_age_ms,
                    "imu_calibrated": wearable.imu_calibrated,
                    "acceleration_g": (
                        float(np.linalg.norm(wearable.accel))
                        if wearable.accel is not None else None
                    ),
                    "gps": list(wearable.gps) if wearable.gps is not None else None,
                    "player_load_imu": (
                        load.player_load_imu
                        if load is not None and np.isfinite(load.player_load_imu) else None
                    ),
                    "timestamp": wearable.t,
                } if wearable is not None else None),
                "load": ({
                    "hsr_m": finite(load.hsr_distance),
                    "sprints": load.sprint_count,
                    "accel_efforts": load.accel_efforts,
                    "decel_efforts": load.decel_efforts,
                    "player_load": finite(load.player_load),
                    "player_load_imu": finite(load.player_load_imu),
                    "metabolic_power_wkg": finite(load.metabolic_power_avg),
                    "high_metabolic_distance_m": finite(load.high_metabolic_distance),
                    "energy_kcal": finite(load.energy_kcal),
                    "avg_hr": finite(load.avg_hr),
                    "hr_drift": finite(load.hr_drift),
                    "min_spo2": finite(load.min_spo2),
                    "acwr": finite(load.acwr),
                } if load is not None else None),
                "load_indicator": {
                    "score": indicator_score,
                    "severity": indicator_level,
                    "model": self.injury_model_name,
                    "evidence_ready": indicator_score is not None,
                    "medical_prediction": False,
                },
                "readiness": self.readiness.update(
                    tid, getattr(self, "_video_timestamp", None) or frame_index / max(source_fps, 1),
                    speed, wearable, player_id=wearable_id,
                    position_ok=getattr(self, "_position_quality", {}).get(tid) == "observed",
                    offset_ms=alignment.get("offset_ms"),
                ),
            })
        possession = engine.possession_pct()
        passes = {
            "team1": sum(event.team == 1 for event in engine.passes),
            "team2": sum(event.team == 2 for event in engine.passes),
        }
        # Recent measured events for the dashboard timeline: confirmed passes and
        # throw-ins, newest first. Each row points at its frame so review can
        # find the exact moment.
        recent_events = [
            {
                "frame": int(event.frame),
                "type": "pass",
                "team": int(event.team),
                "from_track": int(event.from_id),
                "to_track": int(event.to_id),
            }
            for event in engine.passes[-8:]
        ]
        recent_events += [
            {
                "frame": int(row["frame"]),
                "type": "throw_in",
                "team": int(row["team"]),
                "x": float(row["entry_x"]),
                "y": float(row["entry_y"]),
                "side": str(row["side"]),
            }
            for row in engine.throw_ins[-4:]
        ]
        recent_events.sort(key=lambda item: item["frame"], reverse=True)
        recent_events = recent_events[:10]
        tactics = {}
        open_passes = []
        if self.metric:
            for team_id in (1, 2):
                try:
                    report = tactic.report(team_id, possession.get(team_id, 0.0))
                    if report.evidence_ready:
                        tactics[f"team{team_id}"] = report.as_dict()
                except Exception:
                    continue
            carrier = getattr(engine, "_holder_id", None)
            carrier_team = teams.get(carrier)
            if carrier is not None and carrier_team in (1, 2) and carrier in players_xy:
                teammates = {
                    tid: xy for tid, xy in players_xy.items()
                    if tid != carrier and teams.get(tid) == carrier_team
                }
                opponents = [
                    xy for tid, xy in players_xy.items()
                    if teams.get(tid) in (1, 2) and teams.get(tid) != carrier_team
                ]
                if len(teammates) >= 2 and len(opponents) >= 3:
                    goal_x = self.cfg.pitch.length if tactic.attack_dir(carrier_team) > 0 else 0.0
                    open_passes = [
                        {
                            "carrier": int(carrier),
                            "receiver": int(option.receiver),
                            "score": option.score,
                            "openness": option.openness,
                            "lane": option.lane,
                            "progress": option.progress,
                            "x": option.xy[0],
                            "y": option.xy[1],
                            "model": "geometric_option_score",
                        }
                        for option in pass_options(
                            players_xy[carrier], teammates, opponents, goal_x,
                            self.cfg.pitch.length,
                        )[:3]
                    ]
        warnings = [] if self.metric else [
            "Pitch is not calibrated; positions, speed and load are withheld."
        ]
        if not rows:
            warnings.append("No confirmed player tracks are available in this frame.")
        if self.loop_count:
            warnings.append(
                "The demo clip has repeated; cumulative values include repeated footage."
            )
        assigned_per_team = {
            team_id: sum(row.get("team") == team_id for row in rows)
            for team_id in (1, 2)
        }
        if self.metric and any(count < 5 for count in assigned_per_team.values()):
            warnings.append(
                "Tactical recommendations require at least five measured players per team."
            )
        coach_cards = self._coach_step(frame_index, source_fps, players_xy, teams, vision_state,
                                       ball_pitch, engine, bound)
        return {
            "timestamp": time.time(),
            "coach_insights": coach_cards,
            "performance": {"stages": self.profiler.summary(),
                            "publisher_replaced": dict(self.publisher.replaced),
                            "publisher_sent": dict(self.publisher.sent),
                            "sensor_queue_dropped": getattr(self.sensor_source, "dropped_samples", 0)},
            "video_timestamp_epoch": getattr(self, "_video_timestamp", None),
            "source_kind": getattr(self, "source_kind", "live_stream"),
            "camera_id": self.camera_id,
            "frame_index": int(frame_index),
            "loop_count": int(self.loop_count),
            "source_fps": float(source_fps),
            "processing_fps": float(processing_fps),
            "metric": self.metric,
            "pitch": {"kind": self.cfg.pitch.kind, "length_m": float(self.cfg.pitch.length),
                      "width_m": float(self.cfg.pitch.width)},
            "players": rows,
            "ball": ({"x": float(ball_pitch[0]), "y": float(ball_pitch[1])}
                     if ball_pitch is not None and self.metric else None),
            "possession_pct": {str(key): float(value) for key, value in possession.items()},
            "passes": passes,
            "events": recent_events,
            "throw_ins": len(engine.throw_ins),
            "heatmap": self.live_heat.grid_snapshot() if self.metric else None,
            "shots_xg": self._shots,
            "tactics": tactics,
            "pass_options": open_passes,
            "recommendations": list(recs),
            "data_quality": {
                "status": "usable" if self.metric and self._calibration_quality.get("valid")
                          and not self._team_calibration_warning else "review",
                "metric_calibration": self.metric,
                "calibration": self._calibration_quality,
                "positioned_players": int(sum(np.isfinite(p).all() for p in players_xy.values())) if self.metric else 0,
                "team_assigned_players": sum(t in (1, 2) for t in teams.values()),
                "roster_bound_players": sum(tid in bound for tid in players_xy),
                "wearable_time_supported": getattr(self, "_wearable_time_supported", False) and not self.loop_count,
                "reidentifications": self.reid.reids,
                "ambiguous_reidentifications": self.reid.ambiguous,
                "simultaneous_players": len(rows),
                "ball_observed": ball_pitch is not None,
                "warnings": (["Pitch calibration is required for positions, speed and heatmaps."]
                             if not self.metric else ["Pitch mapping lost visual support; position pins are withheld."]
                             if not self._calibration_quality.get("valid") else [])
                            + ([self._team_calibration_warning] if self._team_calibration_warning else [])
                            + (["Wearable fusion withheld: recorded footage needs a video-start epoch anchor."]
                               if self.sensor_source is not None and not getattr(self, "_wearable_time_supported", False) else [])
                            + (["Repeated footage: cumulative metrics include replayed observations."] if self.loop_count else []),
            },
        }

    def _coach_step(self, frame_index, fps, players_xy, teams, vision_state, ball_pitch, engine, bound):
        """Feed the rolling coach engine and re-run it every 10 s of footage."""
        if not self.metric:
            return []
        t = frame_index / max(fps, 1.0)
        last = self._coach_samples[-1].t if self._coach_samples else -1.0
        if t - last >= 0.2:
            row = {}
            for tid, (x, y) in players_xy.items():
                if np.isfinite([x, y]).all():
                    v = vision_state.get(tid, ((np.nan, np.nan), None))[0]
                    row[int(tid)] = (float(x), float(y), int(teams.get(tid, 0)), float(np.hypot(*v)))
            ball = (float(ball_pitch[0]), float(ball_pitch[1]), 0.0) if ball_pitch is not None else None
            self._coach_samples.append(CoachSample(t, row, ball, int(engine._cur_team) if ball else 0,
                                                   engine._holder_id))
            for tid in players_xy:
                w = self.fusion.aligned.get(tid)
                if tid in bound and w is not None and w.hr is not None:
                    self._vest_rows[int(tid)].append((t, float(w.hr), float(w.signal_quality or 0.0)))
        if t - self._coach_last_t >= 10.0 and len(self._coach_samples) >= 50:
            self._coach_last_t = t
            vests = {tid: VestSeries(np.asarray([r[0] for r in rows]), np.asarray([r[1] for r in rows]),
                                     np.asarray([r[2] for r in rows]))
                     for tid, rows in self._vest_rows.items() if len(rows) >= 30}
            try:
                self._coach_cards = [c.as_dict() for c in self.coach.analyse(list(self._coach_samples), vests)][:5]
            except Exception as exc:          # never let advice break the live loop
                print(f"[realtime] coach engine skipped: {exc}")
        return self._coach_cards

    def _write_stats(self, path, engine, tactic, shot_detector, bound):
        import json
        poss = engine.possession_pct()
        stats = {
            "metric": self.metric,
            "loop_count": int(self.loop_count),
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
                "distance_m": s.get("distance", 0.0) if self.metric else None,
                "top_speed_ms": s.get("top_speed", 0.0) if self.metric else None,
                "distance_px": s.get("distance", 0.0) if not self.metric else None,
                "top_speed_px_s": s.get("top_speed", 0.0) if not self.metric else None,
                "hsr_m": None, "sprints": None, "metabolic_power_avg_wkg": None,
                "energy_kcal": None, "wearable": tid in bound,
            }
            for tid, s in engine.speed_distance().items()
        }
        stats["players"] = pixel_players
        if not self.metric:
            stats["note"] = (
                "pixel mode (no calibration): distances are pixels, not metres. "
                "Load indicators and pitch geometry need a calibration YAML."
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
            stats["load_indicators"] = {
                key: {
                    "score": value["risk"],
                    "severity": value["level"],
                    "factors": value["factors"],
                    "medical_prediction": False,
                }
                for key, value in injuries.items()
            }
            stats["shots_xg"] = shot_detector.summary()
            stats["substitution_watch"] = [
                a.as_dict() for a in self.subs.advise(
                    {int(t): j["risk"] for t, j in injuries.items()})[:5]]
            try:
                reports = [tactic.report(t, poss.get(t, 0.0)) for t in (1, 2)]
                stats["tactics"] = {
                    f"team{report.team}": report.as_dict()
                    for report in reports if report.evidence_ready
                }
            except Exception:
                pass
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)
        print(f"[realtime] wrote stats to {path}")

    def _predict_paths(self, W, H, alive: set, xform=None):
        """Batch-predict each live track's future path with the LSTM (one forward
        pass, on the analytics cadence). Cached in image pixels AND, via ``xform``
        (image to pitch homography), projected to pitch metres for the radar overlay."""
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
                if wf.duration_s < 30.0:
                    continue
                r = self.injury_model.predict(wf)
                injuries.append((tid, r.risk, r.level))
                inj_map[tid] = r
            injuries.sort(key=lambda z: -z[1])
            # substitution watch: snapshot loads, rank fatigue-declining players
            play_time_s = max((feature.duration_s for feature in feats.values()), default=0.0)
            self.subs.update(play_time_s, feats)
            self._sub_watch = self.subs.watchlist(
                {tid: r.risk for tid, r in inj_map.items()})
            for a in self._sub_watch[:2]:
                recs.append(f"SUB WATCH #{a.player_id}: priority {a.priority:.2f}: "
                            + "; ".join(a.reasons[:3]))
            poss = engine.possession_pct()
            try:
                reports = [
                    report for report in (
                        tactic.report(1, poss.get(1, 0.0)),
                        tactic.report(2, poss.get(2, 0.0)),
                    ) if report.evidence_ready
                ]
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
                report = tactic.report(t, poss.get(t, 0.0))
                if not report.evidence_ready:
                    continue
                state.team_stats[t] = {
                    "control": snap.control.get(t, 0.0),
                    "formation": self._formation.get(t, ""),
                    "width": snap.width.get(t, 0.0),
                    "line_height": snap.line_height.get(t, 0.0),
                    "pressing": float(snap.pressing.get(t, 0)),
                    "att_third": (snap.thirds.get(t, {}) or {}).get("att", 0),
                    "phase": snap.phase.get(t, " - "),
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
        return render_dashboard_panel(state)   # fixed PANEL_H to compact composite


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Near-real-time soccer analytics dashboard.")
    p.add_argument("--video", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--calibration", default=None, help="pitch calibration .yaml (metric mode)")
    p.add_argument("--out", default=None, help="composite dashboard mp4 output path")
    p.add_argument("--stats", default=None, help="write final stats.json to this path")
    p.add_argument("--fused-out", default=None,
                   help="append clock-aligned vision/wearable snapshots as NDJSON")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--tracker", choices=("imm", "bytetrack", "botsort"), default="imm",
                   help="imm = IMM-EIoU + camera compensation + football ReID (measured best)")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--loop", action="store_true", help="repeat a recorded source until stopped")
    p.add_argument("--video-start-epoch", type=float, default=None,
                   help="capture epoch of recorded frame zero; required to align recorded footage with wearable logs")
    p.add_argument("--analytics-every", type=int, default=10,
                   help="run heavier tactical/load updates every N processed frames")
    p.add_argument("--snapshot-every", type=int, default=2,
                   help="publish React dashboard data every N processed frames")
    p.add_argument("--latency-budget-ms", type=int, default=240,
                   help="maximum recorded-playback catch-up window before dropping frames")
    p.add_argument("--no-window", action="store_true", help="headless (no GUI windows)")
    p.add_argument("--sensor-hub", default="", metavar="URL",
                   help="poll the local relay-backed sensor processor")
    p.add_argument("--player-id", type=int, default=7, help="wearable roster id (must match --roster)")
    p.add_argument("--roster", default="", metavar="track:player,…",
                   help="manually bind vision track ids to wearable player ids")
    p.add_argument("--roster-numbers", default="", metavar="jersey:player,…",
                   help="jersey:player or team/jersey:player pairs; use team-qualified keys when numbers repeat")
    p.add_argument(
        "--product-api",
        default=os.environ.get("CPG44_PRODUCT_API", ""),
        metavar="URL",
        help="publish measured snapshots to the dashboard API, e.g. http://127.0.0.1:8000",
    )
    p.add_argument("--camera-id", default="", help="registered camera id for source health")
    p.add_argument(
        "--strain-model",
        default=os.environ.get("CPG44_STRAIN_MODEL", "models/strain_xgboost.pkl"),
        help="outcome-labelled workload model; transparent review is used until it exists",
    )
    args = p.parse_args(argv)

    cfg = PipelineConfig(weights=args.weights, conf=args.conf, imgsz=args.imgsz,
                         device=args.device, tracker=args.tracker)
    lens_k1, camera_mode = 0.0, "auto"
    if args.calibration:
        from .config import PitchConfig
        from .pitch import calibration_meta, load_calibration
        meta = calibration_meta(args.calibration)
        cfg.pitch = PitchConfig.for_kind(meta["pitch_kind"], meta["pitch_length"], meta["pitch_width"])
        view = load_calibration(args.calibration, cfg.pitch)
        lens_k1, camera_mode = meta["k1"], meta["camera"]
    else:
        view = ViewTransformer()

    def _parse_pairs(s):
        return {int(a): int(b) for a, b in
                (p.split(":") for p in s.split(",") if ":" in p)} if s else None

    def _parse_jerseys(s):
        if not s:
            return None
        result = {}
        for pair in s.split(","):
            number, player = pair.split(":")
            key = tuple(map(int, number.split("/"))) if "/" in number else int(number)
            if isinstance(key, tuple) and (len(key) != 2 or key[0] not in (1, 2)):
                raise ValueError("jersey map uses team/number:player, with team 1 or 2")
            result[key] = int(player)
        return result

    roster = _parse_pairs(args.roster)
    roster_numbers = _parse_jerseys(args.roster_numbers)

    source = None
    if args.sensor_hub:
        from .sensors import HubSensorSource
        source = HubSensorSource(args.sensor_hub, player_id=args.player_id)
    injury_model = None
    if args.strain_model and Path(args.strain_model).is_file():
        from .sensors import InjuryRiskModel
        injury_model = InjuryRiskModel.load(args.strain_model)
        print(f"[realtime] loaded outcome-labelled workload model: {args.strain_model}")
    video = int(args.video) if args.video.isdigit() else args.video
    rt = RealtimePipeline(cfg, view, sensor_source=source, roster_map=roster,
                          roster_numbers=roster_numbers, dashboard=not args.no_window,
                          injury_model=injury_model,
                          analytics_every=max(1, args.analytics_every),
                          snapshot_every=max(1, args.snapshot_every),
                          latency_budget_ms=args.latency_budget_ms,
                          product_api=args.product_api, camera_id=args.camera_id)
    rt.lens_k1, rt.camera_mode = lens_k1, camera_mode
    if cfg.pitch.kind == "futsal":                     # hard court: no grass running cost
        from .catapult import TERRAIN_KT
        rt.fusion.load.kt = TERRAIN_KT["court"]
    rt.run(video, out_path=args.out, max_frames=args.max_frames,
           stats_out=args.stats, fused_out=args.fused_out, loop_file=args.loop,
           video_start_epoch=args.video_start_epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
