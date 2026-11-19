"""Lens curvature for wide phone/action cameras on campus pitches.

Wide lenses bend straight pitch lines (barrel distortion). A pitch homography
assumes a pinhole camera, so without correction positions near the image
edges are pulled towards the centre by metres. One radial coefficient covers
typical phone wide modes (OpenCV Brown model with only k1):

    x_d = x_u * (1 + k1 * r^2),  r in units of the focal length,

with the focal length taken as the image width and the principal point at the
image centre. ``fit_k1`` estimates k1 jointly with the homography from the
landmarks clicked during calibration (Zhang-style planar calibration reduced
to one distortion term): for each candidate k1 the clicked points are
undistorted, a homography is fitted, and the k1 with the lowest reprojection
error wins, but only when it beats the pinhole fit clearly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence, Tuple

import cv2
import numpy as np


def camera_matrix(w: int, h: int) -> np.ndarray:
    return np.array([[float(w), 0, w / 2.0], [0, float(w), h / 2.0], [0, 0, 1.0]])


@lru_cache(maxsize=8)
def _maps(w: int, h: int, k1: float):
    K = camera_matrix(w, h)
    return cv2.initUndistortRectifyMap(K, np.array([k1, 0, 0, 0], float), None, K, (w, h), cv2.CV_16SC2)


def undistort_frame(frame: np.ndarray, k1: float) -> np.ndarray:
    """Remap a frame to a pinhole view (identity when k1 == 0)."""
    if not k1:
        return frame
    h, w = frame.shape[:2]
    m1, m2 = _maps(w, h, round(float(k1), 4))
    return cv2.remap(frame, m1, m2, cv2.INTER_LINEAR)


def undistort_points(pts: Sequence[Sequence[float]], k1: float, w: int, h: int) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if not k1 or not len(pts):
        return pts
    K = camera_matrix(w, h)
    return cv2.undistortPoints(pts.reshape(-1, 1, 2), K, np.array([k1, 0, 0, 0.]), P=K).reshape(-1, 2)


def _fit_rms(img_pts: np.ndarray, pitch_pts: np.ndarray) -> Tuple[float, np.ndarray]:
    H, _ = cv2.findHomography(pitch_pts, img_pts, 0)
    if H is None:
        return np.inf, None
    proj = cv2.perspectiveTransform(pitch_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((proj - img_pts) ** 2, axis=1)))), H


def loo_errors(img: np.ndarray, pit: np.ndarray) -> np.ndarray:
    """Per-landmark leave-one-out error (px): fit on the others, predict this one."""
    errs = []
    for i in range(len(img)):
        m = np.ones(len(img), bool); m[i] = False
        H, _ = cv2.findHomography(pit[m], img[m], 0)
        if H is None:
            errs.append(np.inf); continue
        p = cv2.perspectiveTransform(pit[i:i + 1].reshape(-1, 1, 2), H).reshape(2)
        errs.append(float(np.hypot(*(p - img[i]))))
    return np.asarray(errs)


def _loo_rms(img: np.ndarray, pit: np.ndarray) -> float:
    """Leave-one-out reprojection RMS: fit on all but one landmark, predict it."""
    errs = []
    for i in range(len(img)):
        m = np.ones(len(img), bool); m[i] = False
        H, _ = cv2.findHomography(pit[m], img[m], 0)
        if H is None:
            return np.inf
        p = cv2.perspectiveTransform(pit[i:i + 1].reshape(-1, 1, 2), H).reshape(2)
        errs.append(float(np.sum((p - img[i]) ** 2)))
    return float(np.sqrt(np.mean(errs)))


def fit_k1(img_pts: Sequence[Sequence[float]], pitch_pts: Sequence[Sequence[float]], w: int, h: int,
           lo: float = -0.5, hi: float = 0.2, steps: int = 71) -> Tuple[float, dict]:
    """Joint k1 + homography from >= 8 well-spread landmark correspondences.

    The curvature is chosen by *leave-one-out* error, not the fitting error:
    with few clicks a lens term can absorb click mistakes and still fit them,
    while predicting unseen landmarks worse. A curvature is reported only if
    it predicts held-out clicks clearly better than a pinhole camera.
    """
    img = np.asarray(img_pts, np.float64).reshape(-1, 2)
    pit = np.asarray(pitch_pts, np.float64).reshape(-1, 2)
    if len(img) < 8:
        return 0.0, {"reason": "lens curve needs at least 8 landmarks"}
    spread = np.ptp(img, axis=0) / np.array([w, h])
    if spread[0] < 0.5 or spread[1] < 0.2:
        return 0.0, {"reason": "landmarks too close together to judge the lens"}
    base = _loo_rms(img, pit)
    best_k, best = 0.0, base
    for k in np.linspace(lo, hi, steps):
        rms = _loo_rms(undistort_points(img, k, w, h), pit)
        if rms < best:
            best_k, best = float(k), rms
    if best > 0.75 * base or base - best < 1.0:
        return 0.0, {"loo_px_pinhole": round(base, 2), "loo_px_best": round(best, 2)}
    return round(best_k, 3), {"loo_px_pinhole": round(base, 2), "loo_px_best": round(best, 2)}
