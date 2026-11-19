"""Snap a propagated pitch homography back onto the painted lines.

Optical-flow propagation (``view.PitchHomographyTracker``) keeps calibration
through camera pans but drifts. Painted lines are visible in most frames, so
each refinement:

1. finds line pixels with a white top-hat on the value channel, restricted to
   the grass (robust to shade and floodlights);
2. builds a distance transform of that line mask;
3. samples the pitch template lines in metres and projects them with the
   current homography;
4. minimises the robust (Huber, capped) distance of the projected samples to
   the nearest observed line pixel over the 8 homography parameters
   (chamfer/distance-transform model alignment, cf. Sha et al. CVPR 2020 and
   Homayounfar et al. CVPR 2017);
5. accepts the update only if the share of samples within 3 px rises and the
   correction is small, otherwise keeps the input (never makes it worse by
   design).

Works on a downscaled frame; typical cost 20-60 ms, so it runs every few frames.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from .config import PitchConfig


def line_mask(frame: np.ndarray, scale: float = 0.5) -> np.ndarray:
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    grass = (hsv[..., 0] >= 30) & (hsv[..., 0] <= 90) & (hsv[..., 1] >= 40) & (hsv[..., 2] >= 35)
    ridge = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    white = ((ridge >= 22) & (hsv[..., 1] <= 110)).astype(np.uint8)
    near_grass = cv2.dilate(grass.astype(np.uint8), np.ones((9, 9), np.uint8))
    return white & near_grass


def template_points(pc: PitchConfig, step: float = 1.0) -> np.ndarray:
    L, W, cy = pc.length, pc.width, pc.width / 2
    segs = [((0, 0), (L, 0)), ((0, W), (L, W)), ((0, 0), (0, W)), ((L, 0), (L, W)), ((L / 2, 0), (L / 2, W))]
    for x0, d in ((0.0, 1.0), (L, -1.0)):
        for depth, half in ((pc.penalty_box_length, pc.penalty_box_width / 2), (pc.goal_box_length, pc.goal_box_width / 2)):
            segs += [((x0, cy - half), (x0 + d * depth, cy - half)),
                     ((x0 + d * depth, cy - half), (x0 + d * depth, cy + half)),
                     ((x0, cy + half), (x0 + d * depth, cy + half))]
    pts = []
    for a, b in segs:
        n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / step))
        pts.append(np.linspace(a, b, n))
    ang = np.linspace(0, 2 * np.pi, int(2 * np.pi * pc.centre_circle_radius / step))
    pts.append(np.column_stack((L / 2 + pc.centre_circle_radius * np.cos(ang), cy + pc.centre_circle_radius * np.sin(ang))))
    return np.vstack(pts)


def _project(Hp2i: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h = np.column_stack((pts, np.ones(len(pts)))) @ Hp2i.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = h[:, :2] / h[:, 2:3]
    uv[h[:, 2] <= 1e-9] = np.nan
    return uv


def refine(frame: np.ndarray, H_img2pitch: np.ndarray, pc: Optional[PitchConfig] = None,
           scale: float = 0.5, max_shift_px: float = 120.0) -> Tuple[np.ndarray, dict]:
    """Return (refined image->pitch homography, diagnostics)."""
    pc = pc or PitchConfig()
    mask = line_mask(frame, scale)
    info = {"accepted": False}
    if mask.sum() < 200:
        info["reason"] = "too few line pixels"
        return H_img2pitch, info
    dt = cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
    h, w = dt.shape
    S = np.diag([scale, scale, 1.0])
    Hp2i = S @ np.linalg.inv(H_img2pitch)              # pitch -> small image
    Hp2i = Hp2i / Hp2i[2, 2]
    tpl = template_points(pc)
    uv0 = _project(Hp2i, tpl)
    vis = np.isfinite(uv0).all(1) & (uv0[:, 0] >= 2) & (uv0[:, 0] < w - 2) & (uv0[:, 1] >= 2) & (uv0[:, 1] < h - 2)
    if vis.sum() < 40:
        info["reason"] = "too few template lines in view"
        return H_img2pitch, info
    tpl = tpl[vis]

    def sample(uv):
        bad = ~np.isfinite(uv).all(1)
        uv = np.where(bad[:, None], -1.0, uv)
        x = np.clip(uv[:, 0], 0, w - 1.001)
        y = np.clip(uv[:, 1], 0, h - 1.001)
        x0, y0 = x.astype(int), y.astype(int)
        fx, fy = x - x0, y - y0
        d = (dt[y0, x0] * (1 - fx) * (1 - fy) + dt[y0, x0 + 1] * fx * (1 - fy)
             + dt[y0 + 1, x0] * (1 - fx) * fy + dt[y0 + 1, x0 + 1] * fx * fy)
        out = bad | (uv[:, 0] < 0) | (uv[:, 0] >= w) | (uv[:, 1] < 0) | (uv[:, 1] >= h)
        d[out] = 15.0
        return np.minimum(d, 15.0)

    def res(p):
        Hc = np.append(p, 1.0).reshape(3, 3)
        return sample(_project(Hc, tpl))

    p0 = Hp2i.reshape(-1)[:8]
    before = res(p0)
    sol = least_squares(res, p0, loss="huber", f_scale=2.0, x_scale="jac", max_nfev=200)
    after = res(sol.x)
    Hn = np.append(sol.x, 1.0).reshape(3, 3)
    shift = np.nanmax(np.linalg.norm(_project(Hn, tpl) - _project(Hp2i, tpl), axis=1)) / scale
    inl_b, inl_a = float((before <= 1.5).mean()), float((after <= 1.5).mean())
    info.update(inliers_before=round(inl_b, 3), inliers_after=round(inl_a, 3),
                median_px_before=round(float(np.median(before)) / scale, 2),
                median_px_after=round(float(np.median(after)) / scale, 2), max_shift_px=round(float(shift), 1))
    if inl_a >= inl_b + 0.03 and np.median(after) <= 0.7 * np.median(before) and shift <= max_shift_px:
        Hi2p = np.linalg.inv(np.linalg.inv(S) @ Hn)
        info["accepted"] = True
        return Hi2p / Hi2p[2, 2], info
    return H_img2pitch, info
