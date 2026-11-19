"""Detection + multi-object tracking (YOLOv8 + ByteTrack via supervision), GPU-first.

Robustness choices distilled from the reference repos:
- low ``ball_conf`` (0.1) so the small, fast, motion-blurred ball is still
  detected on amateur footage;
- goalkeeper class optionally remapped to ``player`` so a rarely-seen class does
  not destabilise team assignment / tracking;
- ByteTrack (uses low-score boxes for association) for ID stability under blur;
- FP16 inference on the GPU for throughput on the 8 GB target card.
"""

from __future__ import annotations

from typing import Optional
import warnings

import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO

from .core import BALL, GOALKEEPER, PLAYER, REFEREE, Detections


class Detector:
    def __init__(self, weights: str, imgsz: int = 1280, device: str = "",
                 person_conf: float = 0.3, ball_conf: float = 0.1,
                 remap_goalkeeper: bool = True, half: bool = False,
                 track_low_conf: float = 0.1):
        self.model = YOLO(weights)
        self.class_map = canonical_class_map(self.model.names)
        self.imgsz = imgsz
        self.device = device or None
        self.person_conf = person_conf
        self.ball_conf = ball_conf
        self.remap_goalkeeper = remap_goalkeeper
        self.half = bool(half)
        self.track_low_conf = min(float(track_low_conf), float(person_conf))

    def _precision_args(self) -> dict:
        # Current Ultralytics uses quantize=16 for FP16 inference. Omitting the
        # option on CPU leaves the model at FP32 without a deprecation warning.
        return {"quantize": 16} if self.half else {}

    def detect(self, frame) -> Detections:
        # predict at the lower of the two thresholds, then filter per-class so the
        # ball can survive a much lower confidence than the people.
        res = self.model.predict(frame, imgsz=self.imgsz, device=self.device,
                                 conf=min(self.track_low_conf, self.ball_conf),
                                 verbose=False, **self._precision_args())[0]
        if res.boxes is None or len(res.boxes) == 0:
            return Detections.empty()
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = self._canonical_ids(res.boxes.cls.cpu().numpy().astype(int))
        conf = res.boxes.conf.cpu().numpy()

        # ByteTrack must receive weak people boxes to recover established tracks.
        # New-track activation is gated separately inside SoccerTracker.
        keep = np.where(cls == BALL, conf >= self.ball_conf, conf >= self.track_low_conf)
        keep &= np.isfinite(xyxy).all(axis=1) & np.isfinite(conf)
        keep &= (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
        keep &= cls >= 0
        xyxy, cls, conf = xyxy[keep], cls[keep], conf[keep]

        if self.remap_goalkeeper:
            cls = np.where(cls == GOALKEEPER, PLAYER, cls)
        return Detections(xyxy, cls, conf)

    def _canonical_ids(self, ids):
        return np.asarray([self.class_map.get(int(value), -1) for value in ids], int)

    def detect_ball_in_roi(self, frame, cx, cy, radius, up: int = 2):
        """Search an enlarged crop after a full-frame ball miss.

        Return the selected ball box in full-frame coordinates. ROI recall and
        false positives still need evaluation on the target camera footage.
        """
        import cv2
        Hh, Ww = frame.shape[:2]
        x1, y1 = max(0, int(cx - radius)), max(0, int(cy - radius))
        x2, y2 = min(Ww, int(cx + radius)), min(Hh, int(cy + radius))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        crop = cv2.resize(frame[y1:y2, x1:x2], None, fx=up, fy=up,
                          interpolation=cv2.INTER_LINEAR)
        # NOTE 1: do NOT pass classes=[BALL], ultralytics persists that filter on
        # the shared predictor (next full-frame detect() would lose all players).
        # NOTE 2: use the SAME imgsz as the main pass, TensorRT engines have a
        # fixed input size and crash on any other value.
        try:
            res = self.model.predict(crop, imgsz=self.imgsz,
                                     device=self.device, conf=self.ball_conf,
                                     verbose=False, **self._precision_args())[0]
        except Exception:
            return None                       # ROI search is best-effort only
        if res.boxes is None or len(res.boxes) == 0:
            return None
        cls = self._canonical_ids(res.boxes.cls.cpu().numpy().astype(int))
        ball_m = cls == BALL
        if not ball_m.any():
            return None
        b = res.boxes.xyxy.cpu().numpy()[ball_m]
        c = res.boxes.conf.cpu().numpy()[ball_m]
        bx = b[int(np.argmax(c))]
        return np.array([x1 + bx[0] / up, y1 + bx[1] / up,
                         x1 + bx[2] / up, y1 + bx[3] / up])


def canonical_class_map(names) -> dict[int, int]:
    """Resolve checkpoint-specific class order before any football logic uses it."""
    aliases = {"ball": BALL, "sports_ball": BALL, "football": BALL,
               "player": PLAYER, "person": PLAYER, "goalkeeper": GOALKEEPER,
               "goal_keeper": GOALKEEPER, "referee": REFEREE}
    entries = names.items() if isinstance(names, dict) else enumerate(names)
    result = {int(key): aliases.get(str(name).strip().lower().replace(" ", "_"), -1)
              for key, name in entries}
    if BALL not in result.values() or PLAYER not in result.values():
        raise ValueError("detector checkpoint must name player/person and ball/sports ball classes")
    return result


class SoccerTracker:
    """Assigns persistent ``tracker_id`` to people (players + referees). The ball
    is NOT tracked (single object), the pipeline picks the top-confidence ball
    and interpolates gaps separately."""

    def __init__(self, track_activation_threshold: float = 0.25,
                 lost_track_buffer: int = 50, minimum_matching_threshold: float = 0.8,
                 minimum_consecutive_frames: int = 2, frame_rate: int = 25,
                 backend: str = "bytetrack", track_low_conf: float = 0.1):
        self.backend = backend.removesuffix(".yaml")
        if self.backend not in {"bytetrack", "botsort", "imm"}:
            raise ValueError("tracker must be bytetrack, botsort or imm")
        if self.backend == "imm":
            # Measured best on SoccerNet (docs/TRACKING_BENCHMARK.md): expansion
            # IoU + masked camera-motion compensation + football ReID veto.
            from .imm_tracker import CameraMotion, IMMTracker
            from .reid_model import ReIDEncoder
            self._reid = ReIDEncoder()
            params = (dict(app_veto=0.65, app_weight=0.5) if self._reid.available
                      else dict(app_veto=0.55, app_weight=0.25))
            self._cmc = CameraMotion()
            self._frame_no = 0
            self.reid_every = 2          # appearance every 2nd frame; geometry carries the frames between
            self._imm = {role: IMMTracker(high_thresh=max(track_activation_threshold, 0.3) + 0.2,
                                          low_thresh=track_low_conf, track_buffer=lost_track_buffer,
                                          use_imm=False, **params)
                         for role in (PLAYER, REFEREE)}
        self._trackers = {}
        self._ids = {}
        self._next_id = 1
        for role in (() if self.backend == "imm" else (PLAYER, REFEREE)):
            self._trackers[role] = self._make_tracker(
                track_activation_threshold, lost_track_buffer,
                minimum_matching_threshold, minimum_consecutive_frames,
                frame_rate, track_low_conf,
            )
        self.bytetrack = self._trackers.get(PLAYER)

    def _make_tracker(self, activation, buffer, matching, consecutive, fps, low):
        if self.backend == "botsort":
            from types import SimpleNamespace
            from ultralytics.trackers.bot_sort import BOTSORT
            return BOTSORT(SimpleNamespace(
                track_high_thresh=activation, track_low_thresh=low,
                new_track_thresh=activation, track_buffer=buffer,
                match_thresh=matching, fuse_score=True, gmc_method="sparseOptFlow",
                proximity_thresh=0.5, appearance_thresh=0.8,
                with_reid=False, model="auto",
            ))
        # Supervision 0.29 retains ByteTrack while directing future releases to
        # an external tracker package. requirements.txt bounds this adapter below
        # 0.30; suppress only that known transition warning, never tracker errors.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The `ByteTrack` was deprecated.*",
                category=FutureWarning,
            )
            return sv.ByteTrack(
                track_activation_threshold=activation, lost_track_buffer=buffer,
                minimum_matching_threshold=matching,
                minimum_consecutive_frames=consecutive, frame_rate=fps,
            )

    def _update_imm(self, det: Detections, frame) -> Detections:
        from .imm_tracker import appearance_embedding
        people = det.of_class(PLAYER, REFEREE)
        warp = self._cmc.update(frame, people.xyxy) if frame is not None else None
        self._frame_no += 1
        use_app = self._frame_no % self.reid_every == 0
        boxes, roles, scores, ids = [], [], [], []
        for role, tr in self._imm.items():
            d = det.of_class(role)
            emb = None
            if frame is not None and len(d) and use_app:
                emb = self._reid(frame, d.xyxy) if self._reid.available else appearance_embedding(frame, d.xyxy)
            for local_id, box, score in tr.update(d.xyxy, d.confidence, emb, warp):
                key = (role, int(local_id))
                if key not in self._ids:
                    self._ids[key] = self._next_id
                    self._next_id += 1
                boxes.append(box); roles.append(role); scores.append(score); ids.append(self._ids[key])
        return Detections(np.asarray(boxes, float).reshape(-1, 4), np.asarray(roles, int),
                          np.asarray(scores, float), np.asarray(ids, int))

    def update(self, det: Detections, frame=None) -> Detections:
        """Track the people in ``det``; returns people-only Detections with ids."""
        if self.backend == "imm":
            return self._update_imm(det, frame)
        if self.backend == "botsort" and frame is None:
            raise ValueError("BoT-SORT requires the source frame for camera motion compensation")
        boxes, roles, scores, ids = [], [], [], []
        # Role isolation prevents a referee box from inheriting a player's ID.
        for role, tracker in self._trackers.items():
            people = det.of_class(role)
            if self.backend == "botsort":
                from ultralytics.engine.results import Boxes
                raw = np.column_stack((people.xyxy, people.confidence, people.class_id))
                tracked = tracker.update(Boxes(raw.astype(np.float32), frame.shape[:2]), frame)
                rows = [(row[:4], row[5], int(row[4])) for row in tracked]
            else:
                sv_det = sv.Detections(xyxy=people.xyxy.astype(float),
                                      confidence=people.confidence.astype(float),
                                      class_id=people.class_id.astype(int))
                tracked = tracker.update_with_detections(sv_det)
                rows = [] if tracked.tracker_id is None else zip(
                    tracked.xyxy, tracked.confidence, tracked.tracker_id)
            for box, score, local_id in rows:
                key = (role, int(local_id))
                if key not in self._ids:
                    self._ids[key] = self._next_id
                    self._next_id += 1
                boxes.append(box)
                roles.append(role)
                scores.append(score)
                ids.append(self._ids[key])
        return Detections(np.asarray(boxes, float).reshape(-1, 4),
                          np.asarray(roles, int), np.asarray(scores, float),
                          np.asarray(ids, int))


def pick_ball(det: Detections) -> Optional[np.ndarray]:
    """Return the highest-confidence ball box [x1,y1,x2,y2], or None."""
    balls = det.of_class(BALL)
    if len(balls) == 0:
        return None
    return balls.xyxy[int(np.argmax(balls.confidence))]


def interpolate_ball(positions: list) -> list:
    """Fill missing ball boxes across frames via linear interpolation + back/fwd
    fill. ``positions`` is a list (len = n_frames) of [x1,y1,x2,y2] or None."""
    n = len(positions)
    arr = np.full((n, 4), np.nan)
    for i, p in enumerate(positions):
        if p is not None:
            arr[i] = p
    df = pd.DataFrame(arr, columns=["x1", "y1", "x2", "y2"])
    df = df.interpolate(limit_direction="both").bfill().ffill()
    arr = df.to_numpy()
    return [None if np.isnan(arr[i]).any() else arr[i] for i in range(n)]
