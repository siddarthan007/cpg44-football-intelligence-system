"""Perspective (homography) transform: image pixels → pitch metres, plus
camera-motion compensation for panning broadcast/handheld footage.

The homography solver is pure NumPy (normalised DLT), so pitch-coordinate maths
is testable without OpenCV. OpenCV is only needed for the optical-flow camera
estimator, and is imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Homography (normalised DLT — no OpenCV needed)
# --------------------------------------------------------------------------- #
def _normalise(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation: translate to centroid, scale to mean dist √2."""
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2) / d if d > 1e-12 else 1.0
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]], dtype=float)
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    return (T @ pts_h.T).T[:, :2], T


def fit_homography(src: Sequence[Sequence[float]], dst: Sequence[Sequence[float]]) -> np.ndarray:
    """Least-squares homography H (3×3) mapping src→dst. Needs ≥4 correspondences.

    Solved via normalised DLT + SVD, so it is robust and OpenCV-free.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError("need ≥4 matching src/dst points")

    src_n, Ts = _normalise(src)
    dst_n, Td = _normalise(dst)

    A = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.asarray(A)
    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts     # denormalise
    return H / H[2, 2]


def fit_homography_lines(src: Sequence[Sequence[float]], dst: Sequence[Sequence[float]],
                         constraints: Sequence[tuple], iters: int = 400,
                         init_src=None, init_dst=None) -> np.ndarray:
    """Homography from point correspondences PLUS geometric constraints.

    ``constraints`` = [(img_xy, kind, params), ...]:
      * kind 'x' / 'y' — the image point lies on the pitch line  axis == params
        (e.g. far touchline: ('y', 0.0); halfway line: ('x', 52.5)).
      * kind 'circle'  — the image point lies ON a pitch circle;
        params = (cx, cy, r) (e.g. the centre circle (52.5, 34, 9.15)).

    The exact position along the line/circle is unknown, so we solve by
    alternating projection: start from the point-only fit, synthesise each
    constrained point's pitch position by snapping the current estimate onto the
    line (fix the constrained axis) or circle (radial projection), refit, repeat.

    Why it matters: hand-picked "circle top/bottom" clicks are the ELLIPSE's
    extremes, which are not the circle's cardinal points under perspective — a
    classic bias that wrecks the vertical scale; and point landmarks confined to
    a small region make the frame edges extrapolate into nonsense. Lines that
    span the frame + the full circle-ellipse correspondence anchor the fit
    everywhere. This mirrors line/conic-based pitch calibration from the
    sports-vision literature in a simple, dependency-free form.
    """
    src = np.asarray(src, dtype=float).reshape(-1, 2)
    dst = np.asarray(dst, dtype=float).reshape(-1, 2)
    if init_src is not None and len(init_src):
        # extra approximate correspondences used ONLY to bootstrap the first fit
        H = fit_homography(np.vstack([src, np.asarray(init_src, float).reshape(-1, 2)]),
                           np.vstack([dst, np.asarray(init_dst, float).reshape(-1, 2)]))
    else:
        H = fit_homography(src, dst)
    if not constraints:
        return H
    c_img = np.asarray([c[0] for c in constraints], dtype=float)
    for _ in range(iters):
        proj = apply_homography(H, c_img)
        synth = proj.copy()
        for k, (_, kind, params) in enumerate(constraints):
            if kind == "y":
                synth[k, 1] = float(params)
            elif kind == "x":
                synth[k, 0] = float(params)
            elif kind == "circle":
                cx, cy, r = params
                v = proj[k] - np.array([cx, cy])
                n = np.linalg.norm(v)
                if n < 1e-9:
                    v, n = np.array([r, 0.0]), r
                synth[k] = np.array([cx, cy]) + v * (r / n)   # radial snap onto circle
        H = fit_homography(np.vstack([src, c_img]), np.vstack([dst, synth]))
    return H


def apply_homography(H: np.ndarray, pts: Sequence[Sequence[float]]) -> np.ndarray:
    """Apply H to an (N,2) array of points → (N,2)."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    ones = np.ones((len(pts), 1))
    hom = np.hstack([pts, ones]) @ H.T
    w = hom[:, 2:3]
    w[np.abs(w) < 1e-12] = 1e-12
    return hom[:, :2] / w


@dataclass
class ViewTransformer:
    """Maps image points to pitch metres via a fitted homography.

    Build from ≥4 (image_px → pitch_m) correspondences — from a keypoint model
    (see :mod:`soccer_analytics.pitch`) or the manual calibration tool. If no
    homography is set, :meth:`transform` returns points unchanged and
    :attr:`is_metric` is False (downstream metrics stay in pixels + get flagged).
    """

    H: Optional[np.ndarray] = None
    # optional pitch bounds (metres) to reject wildly out-of-pitch projections
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    clip_margin: float = 5.0

    @property
    def is_metric(self) -> bool:
        return self.H is not None

    @classmethod
    def from_points(cls, image_pts, pitch_pts, **kw) -> "ViewTransformer":
        return cls(H=fit_homography(image_pts, pitch_pts), **kw)

    def transform(self, pts: Sequence[Sequence[float]]) -> np.ndarray:
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        if self.H is None:
            return pts
        out = apply_homography(self.H, pts)
        # NaN-out points that land far outside the pitch (bad detections)
        m = self.clip_margin
        bad = (
            (out[:, 0] < -m) | (out[:, 0] > self.pitch_length + m)
            | (out[:, 1] < -m) | (out[:, 1] > self.pitch_width + m)
        )
        out[bad] = np.nan
        return out


# --------------------------------------------------------------------------- #
# Camera-motion compensation (optical flow; OpenCV lazy)
# --------------------------------------------------------------------------- #
class CameraMotionEstimator:
    """Estimate per-frame camera translation (px) via Lucas–Kanade optical flow
    on static regions (sidelines / stands), so player speeds aren't corrupted by
    panning. Mask regions are derived from frame size — NOT hardcoded to one clip.
    """

    def __init__(self, frame_shape: Tuple[int, int], edge_frac: float = 0.12,
                 min_shift: float = 1.0, scale: float = 0.5):
        # optical flow runs on a downscaled gray frame — 4× cheaper at scale 0.5,
        # and camera translation just scales back by 1/scale. goodFeaturesToTrack on
        # the full 1080p frame was the dominant per-frame cost.
        self.scale = scale
        h, w = int(frame_shape[0] * scale), int(frame_shape[1] * scale)
        band = max(1, int(w * edge_frac))
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:, :band] = 1
        mask[:, w - band:] = 1
        self.feature_mask = mask
        self.min_shift = min_shift
        self.prev_gray = None
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feat_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=3, blockSize=7)

    def update(self, frame) -> Tuple[float, float]:
        """Return per-frame (dx, dy) camera shift in full-resolution pixels."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.scale != 1.0:
            gray = cv2.resize(gray, (self.feature_mask.shape[1], self.feature_mask.shape[0]),
                              interpolation=cv2.INTER_AREA)
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0, 0.0
        p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=self.feature_mask, **self.feat_params)
        if p0 is None:
            self.prev_gray = gray
            return 0.0, 0.0
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self.lk_params)
        self.prev_gray = gray
        if p1 is None or st is None:
            return 0.0, 0.0
        good_new = p1[st.flatten() == 1]
        good_old = p0[st.flatten() == 1]
        if len(good_new) == 0:
            return 0.0, 0.0
        shift = (good_new - good_old).reshape(-1, 2)
        dx, dy = np.median(shift, axis=0) / self.scale     # back to full-res pixels
        if np.hypot(dx, dy) < self.min_shift:
            return 0.0, 0.0
        return float(dx), float(dy)


class PitchHomographyTracker:
    """Per-frame image→pitch homography for a MOVING (panning/zooming) camera.

    Anchored to an initial calibration ``H0`` (frame-0 image px → pitch metres),
    it estimates the frame-to-frame homography each step from optical-flow feature
    correspondences (RANSAC, so moving players are rejected as outliers) and chains
    it, so the pitch mapping follows the camera instead of drifting like a static
    homography. This is what makes speeds/distances accurate on broadcast pans
    without a pitch-keypoint model. (Chained estimates accumulate slow drift over
    very long clips — fine for match segments; a keypoint model would re-anchor.)
    """

    def __init__(self, H0: np.ndarray, frame_shape, scale: float = 0.4,
                 pitch_length: float = 105.0, pitch_width: float = 68.0,
                 clip_margin: float = 5.0, reanchor_every: int = 20):
        self.H = H0.astype(float)          # current image → pitch
        self.H0 = H0.astype(float)         # absolute frame-0 reference → pitch
        self.scale = scale
        self.prev_gray = None
        self.pitch_length, self.pitch_width, self.clip_margin = (
            pitch_length, pitch_width, clip_margin)
        # kept light (half-res, ~150 corners) so the per-frame cost ≈ the camera
        # optical-flow it replaces — no net FPS hit vs the static-homography path.
        self.feat_params = dict(maxCorners=150, qualityLevel=0.02,
                                minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03))
        self._S = np.diag([scale, scale, 1.0])
        self._Sinv = np.diag([1 / scale, 1 / scale, 1.0])
        # multi-keyframe re-anchoring: keyframes are (kp, des, H_at_keyframe).
        # As the camera pans into new territory a new keyframe is stored, so a
        # later return to ANY previously-seen view re-anchors there — this bounds
        # the chained-flow drift that otherwise smears wing/sideline play toward
        # the calibration centre.
        self.reanchor_every = reanchor_every
        self._orb = cv2.ORB_create(700)
        self._keyframes: list = []
        self._upd = 0
        self.reanchors = 0                 # successful absolute re-anchors
        self.keyframes_added = 0

    is_metric = True

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.scale != 1.0:
            gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale,
                              interpolation=cv2.INTER_AREA)
        if self.prev_gray is None:
            self.prev_gray = gray                     # frame 0 = first keyframe
            kp, des = self._orb.detectAndCompute(gray, None)
            if des is not None:
                self._keyframes.append((kp, des, self.H.copy()))
                self.keyframes_added += 1
            return

        # ---- frame-to-frame chaining (cheap, every call) ----
        p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feat_params)
        if p0 is not None and len(p0) >= 8:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self.lk_params)
            if p1 is not None and st is not None:
                m = st.flatten() == 1
                cur, prv = p1[m].reshape(-1, 2), p0[m].reshape(-1, 2)
                if len(cur) >= 8:
                    H_cur_prev, _ = cv2.findHomography(cur, prv, cv2.RANSAC, 3.0)
                    if H_cur_prev is not None:
                        self.H = self.H @ (self._Sinv @ H_cur_prev @ self._S)
        self.prev_gray = gray

        # ---- absolute re-anchor to frame 0 (bounds drift while views overlap) ----
        self._upd += 1
        if self.reanchor_every and self._upd % self.reanchor_every == 0:
            self._reanchor(gray)

    def _reanchor(self, gray):
        if not self._keyframes:
            return
        kp, des = self._orb.detectAndCompute(gray, None)
        if des is None or len(kp) < 30:
            return
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        # try the most recent keyframes first (camera usually near where it was)
        best = None
        for kf_kp, kf_des, kf_H in reversed(self._keyframes[-6:]):
            knn = bf.knnMatch(des, kf_des, k=2)
            good = [m for m, n in (p for p in knn if len(p) == 2)
                    if m.distance < 0.75 * n.distance]
            if len(good) < 20:
                continue
            cur = np.float32([kp[m.queryIdx].pt for m in good])
            ref = np.float32([kf_kp[m.trainIdx].pt for m in good])
            H_cur_ref, mask = cv2.findHomography(cur, ref, cv2.RANSAC, 3.0)
            if H_cur_ref is None or mask is None:
                continue
            inl = int(mask.sum())
            if inl >= 15 and (best is None or inl > best[0]):
                best = (inl, H_cur_ref, kf_H)
                if inl > 60:                    # excellent match — stop searching
                    break
        if best is not None:
            inl, H_cur_ref, kf_H = best
            # absolute against that keyframe's pitch mapping — drift resets to the
            # keyframe's accuracy instead of accumulating per frame
            self.H = kf_H @ (self._Sinv @ H_cur_ref @ self._S)
            self.reanchors += 1
            if inl < 35:                        # weak overlap → camera in new area:
                self._keyframes.append((kp, des, self.H.copy()))
                self.keyframes_added += 1
                if len(self._keyframes) > 12:   # bound memory
                    del self._keyframes[1]      # keep frame-0 anchor forever
        else:
            # no keyframe overlaps at all → brand-new view; store as keyframe
            self._keyframes.append((kp, des, self.H.copy()))
            self.keyframes_added += 1
            if len(self._keyframes) > 12:
                del self._keyframes[1]

    def transform(self, pts):
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        out = apply_homography(self.H, pts)
        m = self.clip_margin
        bad = ((out[:, 0] < -m) | (out[:, 0] > self.pitch_length + m)
               | (out[:, 1] < -m) | (out[:, 1] > self.pitch_width + m))
        out[bad] = np.nan
        return out
