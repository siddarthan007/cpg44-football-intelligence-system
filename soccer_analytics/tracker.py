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
                 remap_goalkeeper: bool = True, half: bool = False):
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.device = device or None
        self.person_conf = person_conf
        self.ball_conf = ball_conf
        self.remap_goalkeeper = remap_goalkeeper
        self.half = bool(half)

    def _precision_args(self) -> dict:
        # Current Ultralytics uses quantize=16 for FP16 inference. Omitting the
        # option on CPU leaves the model at FP32 without a deprecation warning.
        return {"quantize": 16} if self.half else {}

    def detect(self, frame) -> Detections:
        # predict at the lower of the two thresholds, then filter per-class so the
        # ball can survive a much lower confidence than the people.
        res = self.model.predict(frame, imgsz=self.imgsz, device=self.device,
                                 conf=min(self.person_conf, self.ball_conf),
                                 verbose=False, **self._precision_args())[0]
        if res.boxes is None or len(res.boxes) == 0:
            return Detections.empty()
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()

        keep = np.where(cls == BALL, conf >= self.ball_conf, conf >= self.person_conf)
        xyxy, cls, conf = xyxy[keep], cls[keep], conf[keep]

        if self.remap_goalkeeper:
            cls = np.where(cls == GOALKEEPER, PLAYER, cls)
        return Detections(xyxy, cls, conf)

    def detect_ball_in_roi(self, frame, cx, cy, radius, up: int = 2):
        """Zoomed ball search: crop a window around (cx, cy), upscale ``up``×, and
        re-run the detector for the ball class only. The ~12 px ball gets 2-4× the
        pixels, so recall jumps. Cheap (one small inference), so run it only when
        the full-frame pass misses the ball. Returns a full-frame [x1,y1,x2,y2]."""
        import cv2
        Hh, Ww = frame.shape[:2]
        x1, y1 = max(0, int(cx - radius)), max(0, int(cy - radius))
        x2, y2 = min(Ww, int(cx + radius)), min(Hh, int(cy + radius))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        crop = cv2.resize(frame[y1:y2, x1:x2], None, fx=up, fy=up,
                          interpolation=cv2.INTER_LINEAR)
        # NOTE 1: do NOT pass classes=[BALL] — ultralytics persists that filter on
        # the shared predictor (next full-frame detect() would lose all players).
        # NOTE 2: use the SAME imgsz as the main pass — TensorRT engines have a
        # fixed input size and crash on any other value.
        try:
            res = self.model.predict(crop, imgsz=self.imgsz,
                                     device=self.device, conf=self.ball_conf,
                                     verbose=False, **self._precision_args())[0]
        except Exception:
            return None                       # ROI search is best-effort only
        if res.boxes is None or len(res.boxes) == 0:
            return None
        cls = res.boxes.cls.cpu().numpy().astype(int)
        ball_m = cls == BALL
        if not ball_m.any():
            return None
        b = res.boxes.xyxy.cpu().numpy()[ball_m]
        c = res.boxes.conf.cpu().numpy()[ball_m]
        bx = b[int(np.argmax(c))]
        return np.array([x1 + bx[0] / up, y1 + bx[1] / up,
                         x1 + bx[2] / up, y1 + bx[3] / up])


class SoccerTracker:
    """Assigns persistent ``tracker_id`` to people (players + referees). The ball
    is NOT tracked (single object) — the pipeline picks the top-confidence ball
    and interpolates gaps separately."""

    def __init__(self, track_activation_threshold: float = 0.25,
                 lost_track_buffer: int = 50, minimum_matching_threshold: float = 0.8,
                 minimum_consecutive_frames: int = 2, frame_rate: int = 25):
        # Supervision 0.29 retains ByteTrack while directing future releases to
        # an external tracker package. requirements.txt bounds this adapter below
        # 0.30; suppress only that known transition warning, never tracker errors.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The `ByteTrack` was deprecated.*",
                category=FutureWarning,
            )
            try:
                self.bytetrack = sv.ByteTrack(
                    track_activation_threshold=track_activation_threshold,
                    lost_track_buffer=lost_track_buffer,
                    minimum_matching_threshold=minimum_matching_threshold,
                    minimum_consecutive_frames=minimum_consecutive_frames,
                    frame_rate=frame_rate,
                )
            except TypeError:
                # older supervision API used different kwarg names
                self.bytetrack = sv.ByteTrack(frame_rate=frame_rate)

    def update(self, det: Detections) -> Detections:
        """Track the people in ``det``; returns people-only Detections with ids."""
        people = det.of_class(PLAYER, REFEREE)
        sv_det = sv.Detections(xyxy=people.xyxy.astype(float),
                               confidence=people.confidence.astype(float),
                               class_id=people.class_id.astype(int))
        tracked = self.bytetrack.update_with_detections(sv_det)
        tid = tracked.tracker_id if tracked.tracker_id is not None else np.arange(len(tracked))
        return Detections(np.asarray(tracked.xyxy, float),
                          np.asarray(tracked.class_id, int),
                          np.asarray(tracked.confidence, float),
                          np.asarray(tid, int))


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
