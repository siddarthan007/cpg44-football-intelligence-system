"""Perspective (homography) transform: image pixels to pitch metres, plus
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
# Homography (normalised DLT, no OpenCV needed)
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
    """Least-squares homography H (3×3) mapping src to dst. Needs ≥4 correspondences.

    Solved via normalised DLT + SVD, so it is robust and OpenCV-free.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError("need ≥4 matching src/dst points")
    if src.shape != (len(src), 2) or dst.shape != src.shape or not (
        np.isfinite(src).all() and np.isfinite(dst).all()
    ):
        raise ValueError("homography points must be finite Nx2 arrays")
    for points in (src, dst):
        if len(np.unique(points, axis=0)) < 4 or np.linalg.matrix_rank(
            np.column_stack((points, np.ones(len(points))))
        ) < 3:
            raise ValueError("homography needs four distinct, non-collinear points")

    src_n, Ts = _normalise(src)
    dst_n, Td = _normalise(dst)

    A = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.asarray(A)
    if np.linalg.matrix_rank(A) < 8:
        raise ValueError("degenerate homography correspondences")
    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts     # denormalise
    scale = H[2, 2] if abs(H[2, 2]) > 1e-12 else np.linalg.norm(H)
    return H / scale


def fit_homography_lines(src: Sequence[Sequence[float]], dst: Sequence[Sequence[float]],
                         constraints: Sequence[tuple], iters: int = 400,
                         init_src=None, init_dst=None) -> np.ndarray:
    """Homography from point correspondences PLUS geometric constraints.

    ``constraints`` = [(img_xy, kind, params), ...]:
      * kind 'x' / 'y', the image point lies on the pitch line  axis == params
        (e.g. far touchline: ('y', 0.0); halfway line: ('x', 52.5)).
      * kind 'circle' , the image point lies ON a pitch circle;
        params = (cx, cy, r) (e.g. the centre circle (52.5, 34, 9.15)).

    The exact position along a line/circle is unknown, so each constrained
    point contributes ONE residual axis: distance to the target line, or the
    radial error against the circle. Hard correspondences contribute their full
    reprojection error. All residuals are minimised together by Levenberg-
    Marquardt from the point-only bootstrap. Minimising true geometric errors
    (instead of refitting against synthetic targets snapped from the current
    estimate) cannot lock in a bootstrap bias: a wrong solution leaves real
    residual that the solver can still descend.

    Why it matters: hand-picked "circle top/bottom" clicks are the ELLIPSE's
    extremes, which are not the circle's cardinal points under perspective, a
    classic bias that wrecks the vertical scale; and point landmarks confined to
    a small region make the frame edges extrapolate into nonsense. Lines that
    span the frame + the full circle-ellipse correspondence anchor the fit
    everywhere. This mirrors line/conic-based pitch calibration from the
    sports-vision literature in a simple, dependency-light form.
    """
    src = np.asarray(src, dtype=float).reshape(-1, 2)
    dst = np.asarray(dst, dtype=float).reshape(-1, 2)
    if init_src is not None and len(init_src):
        # extra approximate correspondences used ONLY to start the solver in the
        # right basin; their bias is washed out by the exact hard points below
        H = fit_homography(np.vstack([src, np.asarray(init_src, float).reshape(-1, 2)]),
                           np.vstack([dst, np.asarray(init_dst, float).reshape(-1, 2)]))
    else:
        H = fit_homography(src, dst)
    if not constraints:
        return H

    c_img = np.asarray([c[0] for c in constraints], dtype=float)
    kinds = [c[1] for c in constraints]
    params = [c[2] for c in constraints]

    def residuals(h8: np.ndarray) -> np.ndarray:
        Hc = np.append(h8, 1.0).reshape(3, 3)
        parts: List[np.ndarray] = []
        if len(src):
            parts.append((apply_homography(Hc, src) - dst).ravel())
        proj = apply_homography(Hc, c_img)
        for k, kind in enumerate(kinds):
            px, py = proj[k]
            if kind == "y":
                parts.append(np.array([py - float(params[k])]))
            elif kind == "x":
                parts.append(np.array([px - float(params[k])]))
            elif kind == "circle":
                cx, cy, r = params[k]
                parts.append(np.array([float(np.hypot(px - cx, py - cy)) - r]))
        return np.concatenate(parts) if parts else np.zeros(1)

    from scipy.optimize import least_squares

    h0 = (H / H[2, 2]).ravel()[:8]
    sol = least_squares(residuals, h0, method="lm", max_nfev=max(200, 4 * int(iters)))
    H = np.append(sol.x, 1.0).reshape(3, 3)
    return H / H[2, 2]


def apply_homography(H: np.ndarray, pts: Sequence[Sequence[float]]) -> np.ndarray:
    """Apply H to an (N,2) array of points to (N,2)."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    ones = np.ones((len(pts), 1))
    hom = np.hstack([pts, ones]) @ H.T
    w = hom[:, 2:3]
    valid = np.isfinite(hom).all(axis=1) & (np.abs(w[:, 0]) >= 1e-12)
    out = np.full((len(pts), 2), np.nan)
    out[valid] = hom[valid, :2] / w[valid]
    return out


@dataclass
class ViewTransformer:
    """Maps image points to pitch metres via a fitted homography.

    Build from ≥4 (image_px to pitch_m) correspondences, from a keypoint model
    (see :mod:`soccer_analytics.pitch`) or the manual calibration tool. If no
    homography is set, :meth:`transform` returns points unchanged and
    :attr:`is_metric` is False (downstream metrics stay in pixels + get flagged).
    """

    H: Optional[np.ndarray] = None
    # optional pitch bounds (metres) to reject wildly out-of-pitch projections
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    clip_margin: float = 5.0
    calibrated: bool = True

    @property
    def is_metric(self) -> bool:
        return self.H is not None and self.calibrated

    @classmethod
    def from_field_span(cls, frame_shape=(1080, 1920), pc=None) -> "ViewTransformer":
        """Illustrative preview only. Assumed image corners are not calibration."""
        if pc is None:
            from .config import PitchConfig
            pc = PitchConfig()
        H, W = frame_shape[:2]
        src = np.array([
            [0.10 * W, 0.22 * H],  # Top-left (far sideline)
            [0.90 * W, 0.22 * H],  # Top-right (far sideline)
            [0.98 * W, 0.94 * H],  # Bottom-right (near sideline)
            [0.02 * W, 0.94 * H],  # Bottom-left (near sideline)
        ], dtype=float)
        dst = np.array([
            [4.0, 4.0],                         # Top-left (metres)
            [pc.length - 4.0, 4.0],              # Top-right (metres)
            [pc.length - 4.0, pc.width - 4.0],   # Bottom-right (metres)
            [4.0, pc.width - 4.0],              # Bottom-left (metres)
        ], dtype=float)
        return cls(H=fit_homography(src, dst), pitch_length=pc.length,
                   pitch_width=pc.width, calibrated=False)

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
    """Estimate per-frame camera translation (px) via Lucas, Kanade optical flow
    on static regions (sidelines / stands), so player speeds aren't corrupted by
    panning. Mask regions are derived from frame size, NOT hardcoded to one clip.
    """

    def __init__(self, frame_shape: Tuple[int, int], edge_frac: float = 0.12,
                 min_shift: float = 1.0, scale: float = 0.5):
        # optical flow runs on a downscaled gray frame, 4× cheaper at scale 0.5,
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
    """Per-frame image to pitch homography for a MOVING (panning/zooming) camera.

    Anchored to an initial calibration ``H0`` (frame-0 image px to pitch metres),
    it estimates the frame-to-frame homography each step from optical-flow feature
    correspondences (RANSAC, so moving players are rejected as outliers) and chains
    it, so the pitch mapping follows the camera instead of drifting like a static
    homography. Accuracy depends on the initial calibration and feature support;
    chained estimates can drift. Unsupported updates withhold projected points.
    """

    def __init__(self, H0: np.ndarray, frame_shape, scale: float = 0.4,
                 pitch_length: float = 105.0, pitch_width: float = 68.0,
                 clip_margin: float = 5.0, reanchor_every: int = 10):
        self.H = H0.astype(float)          # current image to pitch
        self.H0 = H0.astype(float)         # absolute frame-0 reference to pitch
        self.scale = scale
        self.prev_gray = None
        self.pitch_length, self.pitch_width, self.clip_margin = (
            pitch_length, pitch_width, clip_margin)
        # kept light (half-res, ~150 corners) so the per-frame cost ≈ the camera
        # optical-flow it replaces, no net FPS hit vs the static-homography path.
        self.feat_params = dict(maxCorners=150, qualityLevel=0.02,
                                minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03))
        self._S = np.diag([scale, scale, 1.0])
        self._Sinv = np.diag([1 / scale, 1 / scale, 1.0])
        # multi-keyframe re-anchoring: keyframes are (kp, des, H_at_keyframe).
        # As the camera pans into new territory a new keyframe is stored, so a
        # later return to ANY previously-seen view re-anchors there, this bounds
        # the chained-flow drift that otherwise smears wing/sideline play toward
        # the calibration centre.
        self.reanchor_every = reanchor_every
        self._orb = cv2.ORB_create(700)
        self._keyframes: list = []
        self._upd = 0
        self.reanchors = 0                 # successful absolute re-anchors
        self.keyframes_added = 0
        self.last_anchor_inliers = 0       # inlier count of the most recent anchor
        self.valid = True
        self.last_flow_inliers = 0
        self.last_flow_ratio = 0.0
        self.failed_updates = 0
        self.prev_mask = None

    is_metric = True

    def reset(self) -> None:
        """Back to the calibrated view, e.g. when a looping clip restarts:
        chaining across the cut would carry the last frame's pan into frame 0."""
        self.H = self.H0.copy()
        self.prev_gray = self.prev_mask = None
        self._keyframes.clear()
        self._upd = 0
        self.valid = True

    def update(self, frame, exclude_boxes=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.scale != 1.0:
            gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale,
                              interpolation=cv2.INTER_AREA)
        mask = np.full(gray.shape, 255, np.uint8)
        for box in ([] if exclude_boxes is None else exclude_boxes):
            x1, y1, x2, y2 = np.asarray(box) * self.scale
            margin = max(3, int(0.1 * (x2 - x1)))
            cv2.rectangle(mask, (int(x1) - margin, int(y1) - margin),
                          (int(x2) + margin, int(y2) + margin), 0, -1)
        if self.prev_gray is None:
            self.prev_gray = gray                     # frame 0 = first keyframe
            self.prev_mask = mask
            kp, des = self._orb.detectAndCompute(gray, mask)
            if des is not None:
                self._keyframes.append((kp, des, self.H.copy()))
                self.keyframes_added += 1
            return

        # ---- frame-to-frame chaining (cheap, every call) ----
        chain_valid = False
        self.last_flow_inliers = 0
        self.last_flow_ratio = 0.0
        p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=self.prev_mask, **self.feat_params)
        if p0 is not None and len(p0) >= 8:
            p1, st, flow_error = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self.lk_params)
            if p1 is not None and st is not None:
                m = st.flatten() == 1
                if flow_error is not None:
                    m &= flow_error.ravel() < 30.0
                back, back_status, _ = cv2.calcOpticalFlowPyrLK(
                    gray, self.prev_gray, p1, None, **self.lk_params)
                if back is None or back_status is None:
                    m[:] = False
                else:
                    m &= back_status.ravel() == 1
                    m &= np.linalg.norm((p0 - back).reshape(-1, 2), axis=1) < 1.0
                cur, prv = p1[m].reshape(-1, 2), p0[m].reshape(-1, 2)
                if len(cur) >= 8:
                    H_cur_prev, inliers = cv2.findHomography(cur, prv, cv2.RANSAC, 3.0)
                    # A low-inlier flow estimate is noise (blur, occlusion, zoom
                    # past the feature set). Chaining it corrupts the mapping and
                    # drags positions toward the calibration centre, so skip it
                    # and let the next anchor fix the view instead.
                    self.last_flow_inliers = int(inliers.sum()) if inliers is not None else 0
                    self.last_flow_ratio = self.last_flow_inliers / len(cur)
                    coverage = (cv2.contourArea(cv2.convexHull(cur[inliers.ravel() == 1]))
                                / gray.size if inliers is not None and self.last_flow_inliers >= 3 else 0)
                    if self.valid and H_cur_prev is not None and np.isfinite(H_cur_prev).all() \
                            and self.last_flow_inliers >= 12 and self.last_flow_ratio >= 0.5 \
                            and coverage >= 0.03:
                        self.H = self.H @ (self._Sinv @ H_cur_prev @ self._S)
                        chain_valid = True
        self.valid = chain_valid
        if not chain_valid:
            self.failed_updates += 1
        self.prev_gray = gray
        self.prev_mask = mask

        # ---- absolute re-anchor to frame 0 (bounds drift while views overlap) ----
        self._upd += 1
        if not self.valid or (self.reanchor_every and self._upd % self.reanchor_every == 0):
            self._reanchor(gray, mask)

    def _reanchor(self, gray, feature_mask=None):
        if not self._keyframes:
            return
        kp, des = self._orb.detectAndCompute(gray, feature_mask)
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
            coverage = cv2.contourArea(cv2.convexHull(cur[mask.ravel() == 1])) / gray.size if inl >= 3 else 0
            if inl >= 15 and inl / len(good) >= 0.5 and coverage >= 0.03 \
                    and np.isfinite(H_cur_ref).all() and (best is None or inl > best[0]):
                best = (inl, H_cur_ref, kf_H)
                if inl > 60:                    # excellent match, stop searching
                    break
        if best is not None:
            inl, H_cur_ref, kf_H = best
            # absolute against that keyframe's pitch mapping, drift resets to the
            # keyframe's accuracy instead of accumulating per frame
            self.H = kf_H @ (self._Sinv @ H_cur_ref @ self._S)
            self.valid = True
            self.reanchors += 1
            self.last_anchor_inliers = inl
            if inl < 35:                        # weak overlap to camera in new area:
                self._keyframes.append((kp, des, self.H.copy()))
                self.keyframes_added += 1
                if len(self._keyframes) > 12:   # bound memory
                    del self._keyframes[1]      # keep frame-0 anchor forever
        elif self.valid:
            # no keyframe overlaps at all to brand-new view; store as keyframe
            self._keyframes.append((kp, des, self.H.copy()))
            self.keyframes_added += 1
            if len(self._keyframes) > 12:
                del self._keyframes[1]

    def transform(self, pts):
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        if not self.valid:
            return np.full_like(pts, np.nan)
        out = apply_homography(self.H, pts)
        m = self.clip_margin
        bad = ((out[:, 0] < -m) | (out[:, 0] > self.pitch_length + m)
               | (out[:, 1] < -m) | (out[:, 1] > self.pitch_width + m))
        out[bad] = np.nan
        return out

    def quality(self) -> dict:
        return {"valid": self.valid, "flow_inliers": self.last_flow_inliers,
                "flow_inlier_ratio": round(self.last_flow_ratio, 3),
                "failed_updates": self.failed_updates, "reanchors": self.reanchors}


class StaticPitch:
    """Image -> pitch mapping for a fixed camera (tripod or mast).

    Same interface as :class:`PitchHomographyTracker`, but nothing is
    propagated: a fixed camera's homography does not change, so per-frame
    optical flow would only add drift and ~10 ms per frame. Every
    ``refine_every`` frames the mapping may be re-snapped to the painted lines,
    accepted only for small, clearly better corrections (knocks, sway).
    """

    is_metric = True

    def __init__(self, H0: np.ndarray, pitch_length: float = 105.0, pitch_width: float = 68.0,
                 clip_margin: float = 5.0, refine_every: int = 50):
        self.H = np.asarray(H0, float).copy()
        self.pitch_length, self.pitch_width, self.clip_margin = pitch_length, pitch_width, clip_margin
        self.refine_every = refine_every
        self.valid = True
        self.refinements = 0
        self._n = 0

    def update(self, frame, exclude_boxes=None):
        self._n += 1
        if self.refine_every and self._n % self.refine_every == 0:
            from .config import PitchConfig
            from .line_refine import refine
            Hn, info = refine(frame, self.H, PitchConfig(length=self.pitch_length, width=self.pitch_width), max_shift_px=15.0)
            if info.get("accepted"):
                self.H = Hn
                self.refinements += 1

    def transform(self, pts):
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        out = apply_homography(self.H, pts)
        m = self.clip_margin
        bad = ((out[:, 0] < -m) | (out[:, 0] > self.pitch_length + m)
               | (out[:, 1] < -m) | (out[:, 1] > self.pitch_width + m))
        out[bad] = np.nan
        return out

    def quality(self) -> dict:
        return {"valid": True, "mode": "fixed", "line_refinements": self.refinements}
