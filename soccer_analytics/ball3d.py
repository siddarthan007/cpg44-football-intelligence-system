"""Physics-based 3D ball trajectory from one calibrated camera.

A ground-plane homography maps the *ground* to the image. A ball in the air is
not on the ground, so projecting its pixel through the homography puts it far
behind its real position (the ball "shadow" on the pitch). This module follows
the approach benchmarked by Grad, Czajkowski & Varashylau, "Physics-Based 3D
Ball Trajectory Reconstruction from Monocular Soccer Video" (CVPR-W CVsports
2026, https://lukaszgrad.github.io/soccer-ball3d/): split the 2D ball track at
contacts, then fit each arc with a physical flight model by minimising image
reprojection error.

Steps
-----
1. ``camera_from_homography``: recover a pinhole camera ``P = K [r1 r2 r3 t]``
   from the pitch->image homography, assuming square pixels and a centred
   principal point (Zhang 2000 / Hartley & Zisserman §8.8). Valid for oblique
   views; a near-fronto-parallel camera gives no focal constraint and returns
   ``None`` (the caller then keeps the honest 2D result).
2. ``segment_track``: cut the track at gaps and at sharp changes of image
   velocity (kicks, headers, bounces): the "pivot" points of the paper.
3. ``fit_segment``: compare a *ground roll* model (z = 0, constant
   deceleration) with a *ballistic* model (gravity, optional linear drag
   k ~ 0.1 1/s for a football at typical speeds). The aerial model is used only when it
   explains the pixels clearly better and stays above the ground; otherwise
   the ball is reported on the ground. This is model selection, not a guess.

Outputs per frame: pitch X, Y (m), height Z (m), speed (m/s) and per segment:
kind, launch speed, peak height, reprojection RMSE (px). Shot-level metrics
(speed, distance and angle to goal) come from the segment containing a shot.

Accuracy depends on calibration quality and frame rate; Grad et al. report
~0.9-1.5 m mean 3D error for their best monocular models on real matches.
Treat heights as estimates with that order of uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

G = 9.81


@dataclass
class Camera:
    K: np.ndarray
    R: np.ndarray            # columns r1, r2, r3 (r3 points to +height)
    t: np.ndarray
    image_size: Tuple[int, int]

    @property
    def P(self) -> np.ndarray:
        return self.K @ np.column_stack((self.R, self.t))

    @property
    def centre(self) -> np.ndarray:
        return -np.linalg.solve(self.R, self.t)

    def project(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float).reshape(-1, 3)
        h = np.column_stack((X, np.ones(len(X)))) @ self.P.T
        return h[:, :2] / h[:, 2:3]


def camera_from_homography(H_pitch_to_img: np.ndarray, image_size: Tuple[int, int]) -> Optional[Camera]:
    """Pinhole camera from a pitch-plane homography (metres -> pixels).

    ``image_size`` is (width, height). Returns None when the focal length is
    not observable or the solution is implausible.
    """
    Gm = np.asarray(H_pitch_to_img, float)
    if Gm.shape != (3, 3) or not np.isfinite(Gm).all():
        return None
    w, h = image_size
    cx, cy = w / 2.0, h / 2.0
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    Gc = T @ Gm
    g1, g2 = Gc[:, 0], Gc[:, 1]
    cands = []
    d1 = g1[2] * g2[2]
    if abs(d1) > 1e-12:
        cands.append(-(g1[0] * g2[0] + g1[1] * g2[1]) / d1)
    d2 = g1[2] ** 2 - g2[2] ** 2
    if abs(d2) > 1e-12:
        cands.append(-(g1[0] ** 2 + g1[1] ** 2 - g2[0] ** 2 - g2[1] ** 2) / d2)
    cands = [c for c in cands if np.isfinite(c) and c > 0]
    if not cands:
        return None
    f = float(np.sqrt(np.median(cands)))
    if not (0.2 * w <= f <= 20 * w):
        return None
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    M = np.linalg.solve(K, Gm)
    lam = 1.0 / np.linalg.norm(M[:, 0])
    if M[2, 2] * lam < 0:                    # pitch must be in front of the camera
        lam = -lam
    r1, r2, tvec = M[:, 0] * lam, M[:, 1] * lam, M[:, 2] * lam
    # nearest rotation to [r1 r2 r1xr2]
    U, _, Vt = np.linalg.svd(np.column_stack((r1, r2, np.cross(r1, r2))))
    R = U @ Vt
    r1, r2 = R[:, 0], R[:, 1]
    r3 = np.cross(r1, r2)
    cam = Camera(K, np.column_stack((r1, r2, r3)), tvec, (w, h))
    # choose the height axis so the camera stands above the pitch
    if cam.centre[2] < 0:
        cam = Camera(K, np.column_stack((r1, r2, -r3)), tvec, (w, h))
    if cam.centre[2] < 1.0:
        return None
    return _refine_camera(cam, Gm)


def _refine_camera(cam: Camera, H_pitch_to_img: np.ndarray) -> Camera:
    """Make the camera agree with the homography on the ground plane.

    The closed form above orthogonalises [r1 r2] after the fact, which moves
    every ground point when the homography is not exactly physical (clicked
    landmarks, line refinement, mild lens bend). On broadcast clips that shift
    was 35 to 105 px, several metres on the pitch. Here focal length, rotation
    and translation are refined so the camera reprojects the visible pitch the
    way the homography does (Zhang 2000, step 2: maximum-likelihood refinement).
    """
    from scipy.spatial.transform import Rotation
    w, h = cam.image_size
    xs, ys = np.meshgrid(np.linspace(-5, 110, 24), np.linspace(-5, 73, 16))
    ground = np.column_stack((xs.ravel(), ys.ravel(), np.ones(xs.size)))
    img = ground @ np.asarray(H_pitch_to_img, float).T
    front = img[:, 2] * np.sign(img[:, 2].mean()) > 1e-9
    uv = img[:, :2] / img[:, 2:3]
    keep = front & (uv[:, 0] > -0.1 * w) & (uv[:, 0] < 1.1 * w) & (uv[:, 1] > -0.1 * h) & (uv[:, 1] < 1.1 * h)
    if keep.sum() < 8:
        return cam
    X = np.column_stack((ground[keep, :2], np.zeros(keep.sum())))
    target = uv[keep]
    R0, K0 = cam.R, cam.K

    def build(p):
        K = K0.copy()
        K[0, 0] = K[1, 1] = K0[0, 0] * np.exp(p[0])
        return Camera(K, R0 @ Rotation.from_rotvec(p[1:4]).as_matrix(), p[4:7], (w, h))

    def res(p):
        return (build(p).project(X) - target).ravel()

    p0 = np.r_[0.0, 0.0, 0.0, 0.0, cam.t]
    sol = least_squares(res, p0, loss="soft_l1", f_scale=5.0)
    out = build(sol.x)
    before = np.median(np.linalg.norm((cam.project(X) - target), axis=1))
    after = np.median(np.linalg.norm((out.project(X) - target), axis=1))
    if not np.isfinite(after) or after >= before or out.centre[2] < 1.0:
        return cam
    out.ground_residual_px = float(after)
    return out


# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    start: int
    end: int
    kind: str                       # "air" | "ground" | "unresolved"
    rmse_px: float
    launch_speed_mps: Optional[float]
    peak_height_m: Optional[float]
    params: List[float] = field(default_factory=list)


@dataclass
class Ball3DResult:
    frames: List[int]
    xyz: np.ndarray                 # (n,3) metres, NaN where unresolved
    speed_mps: np.ndarray           # (n,)
    segments: List[Segment]
    camera_focal_px: float
    camera_height_m: float

    def as_dict(self) -> dict:
        return {
            "frames": [int(f) for f in self.frames],
            "xyz": [[None if not np.isfinite(v) else round(float(v), 3) for v in row] for row in self.xyz],
            "speed_mps": [None if not np.isfinite(v) else round(float(v), 2) for v in self.speed_mps],
            "segments": [asdict(s) for s in self.segments],
            "camera": {"focal_px": round(self.camera_focal_px, 1), "height_m": round(self.camera_height_m, 2)},
            "method": "homography camera + per-arc ballistic/ground model selection (Grad et al. 2026)",
        }


def _ballistic(p0, v0, t, drag: float = 0.0):
    """Closed-form flight with linear drag k (1/s); k=0 is a parabola."""
    t = np.asarray(t, float)[:, None]
    a = np.array([0.0, 0.0, -G])
    if drag <= 1e-6:
        return p0 + v0 * t + 0.5 * a * t * t
    e = np.exp(-drag * t)
    return p0 + (v0 - a / drag) * (1 - e) / drag + a * t / drag


def _ground(p0, v0, decel, t):
    """Rolling ball on z = 0 with constant deceleration until it stops."""
    t = np.asarray(t, float)
    sp = float(np.linalg.norm(v0))
    if sp < 1e-6:
        return np.repeat(np.r_[p0, 0.0][None], len(t), 0)
    d = v0 / sp
    t_stop = sp / max(decel, 1e-6)
    tt = np.minimum(t, t_stop)
    s = sp * tt - 0.5 * decel * tt * tt
    xy = p0[None] + s[:, None] * d[None]
    return np.column_stack((xy, np.zeros(len(t))))


def ptz_cameras(Hs: np.ndarray, image_size: Tuple[int, int]) -> List[Optional[Camera]]:
    """Cameras for a pan-tilt-zoom shot: one position for the whole clip.

    Recovered one frame at a time, a broadcast camera's position wanders by
    tens of metres (SNMOT-074: from 63 to 94 m behind the touchline) because
    each homography carries its own error. The camera stands on one tripod,
    so its position is taken as the robust median over frames whose
    homography a pinhole camera explains well, and each frame keeps only
    its rotation and zoom, refitted with that position fixed (the usual
    pan-tilt-zoom model for sports cameras)."""
    from scipy.spatial.transform import Rotation
    w, h = image_size
    raw = []
    for Hf in Hs:
        Hinv = np.linalg.inv(Hf)
        raw.append(camera_from_homography(Hinv / Hinv[2, 2], image_size))
    good = [c for c in raw if c is not None and getattr(c, "ground_residual_px", 99) < 6.0]
    if len(good) < 3:
        good = [c for c in raw if c is not None]
    if not good:
        return raw
    C = np.median(np.stack([c.centre for c in good]), axis=0)
    xs, ys = np.meshgrid(np.linspace(-5, 110, 24), np.linspace(-5, 73, 16))
    ground = np.column_stack((xs.ravel(), ys.ravel(), np.zeros(xs.size)))
    out: List[Optional[Camera]] = []
    cache: dict = {}
    prev = None
    for Hf, c0 in zip(Hs, raw):
        key = np.round(Hf, 9).tobytes()
        if key in cache:
            out.append(cache[key]); continue
        G = np.linalg.inv(Hf)
        img = np.column_stack((ground[:, :2], np.ones(len(ground)))) @ G.T
        front = img[:, 2] * np.sign(np.median(img[:, 2])) > 1e-9
        uv = img[:, :2] / img[:, 2:3]
        keep = front & (uv[:, 0] > -0.1 * w) & (uv[:, 0] < 1.1 * w) & (uv[:, 1] > -0.1 * h) & (uv[:, 1] < 1.1 * h)
        start = c0 or prev or good[0]
        if keep.sum() < 8:
            out.append(start); cache[key] = start; continue
        X, target, R0, f0 = ground[keep], uv[keep], start.R, start.K[0, 0]

        def build(p):
            K = np.array([[f0 * np.exp(p[0]), 0, w / 2], [0, f0 * np.exp(p[0]), h / 2], [0, 0, 1.0]])
            R = R0 @ Rotation.from_rotvec(p[1:4]).as_matrix()
            return Camera(K, R, _t_from_centre(R, C), (w, h))

        def res(p):
            return (build(p).project(X) - target).ravel()
        sol = least_squares(res, np.zeros(4), loss="soft_l1", f_scale=5.0)
        cam = build(sol.x)
        cam.ground_residual_px = float(np.median(np.linalg.norm((cam.project(X) - target).reshape(-1, 2), axis=1)))
        out.append(cam); cache[key] = cam; prev = cam
    return out


def _t_from_centre(R: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Translation of a camera with columns R = [r1 r2 r3] centred at C.

    ``Camera.P = K [R | t]`` maps world X to K (R X + t) with R as stored, so the
    centre satisfies R C + t = 0."""
    return -R @ C


class CameraSet:
    """One camera and one ground mapping per observation.

    A panning camera has a different pose in every frame. Projecting a
    candidate 3D path through each frame's own camera is exact, where warping
    all pixels into one reference frame is only right for a pure rotation
    with a perfect calibration."""

    def __init__(self, Ps: np.ndarray, Hi2p: np.ndarray):
        self.P = np.asarray(Ps, float)
        self.Hi2p = np.asarray(Hi2p, float)
        self.Hp2i = np.linalg.inv(self.Hi2p)

    def sub(self, a: int, b: int) -> "CameraSet":
        out = CameraSet.__new__(CameraSet)
        out.P, out.Hi2p, out.Hp2i = self.P[a:b], self.Hi2p[a:b], self.Hp2i[a:b]
        return out

    def project(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float).reshape(-1, 3)
        h = np.einsum("nij,nj->ni", self.P, np.column_stack((X, np.ones(len(X)))))
        return h[:, :2] / h[:, 2:3]

    def ground_to_image(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        h = np.einsum("nij,nj->ni", self.Hp2i, np.column_stack((xy, np.ones(len(xy)))))
        return h[:, :2] / h[:, 2:3]

    def image_to_ground(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, float).reshape(-1, 2)
        h = np.einsum("nij,nj->ni", self.Hi2p, np.column_stack((uv, np.ones(len(uv)))))
        return h[:, :2] / h[:, 2:3]


def _reproj(cam, X, uv):
    return (cam.project(X) - uv).reshape(-1)


def fit_segment(cam: "CameraSet", t: np.ndarray, uv: np.ndarray, H_img_to_pitch=None,
                allow_drag: bool = True, air_gain: float = 0.7,
                min_height: float = 0.35) -> Tuple[str, np.ndarray, float, list]:
    """Return (kind, xyz per sample, rmse px, params). ``cam`` holds one camera
    and ground mapping per sample."""
    t = np.asarray(t, float) - t[0]
    uv = np.asarray(uv, float)
    ground_xy = cam.image_to_ground(uv)
    ok = np.isfinite(ground_xy).all(1)
    if ok.sum() < 3:
        return "unresolved", np.full((len(t), 3), np.nan), np.inf, []
    # ---- ground model: initial guess from the homography projection
    A = np.column_stack((np.ones(ok.sum()), t[ok]))
    coef, *_ = np.linalg.lstsq(A, ground_xy[ok], rcond=None)
    g0 = np.r_[coef[0], coef[1], 1.0]

    # A rolling ball lies on the plane the homography describes, so it is
    # imaged through the homography itself: the same map that places the
    # players, so ball and feet agree even when the homography is not exactly
    # a physical camera. Residuals stay in pixels, where the noise lives.
    def rg(p):
        xy = _ground(p[:2], p[2:4], abs(p[4]), t)[:, :2]
        return (cam.ground_to_image(xy) - uv).ravel()
    sol_g = least_squares(rg, g0, loss="soft_l1", f_scale=3.0)
    rmse_g = float(np.sqrt(np.mean(sol_g.fun ** 2)))
    xyz_g = _ground(sol_g.x[:2], sol_g.x[2:4], abs(sol_g.x[4]), t)
    # From one camera, a ball coming towards the lens and a ball rising look
    # alike over a short stretch; a flight fitted to a few frames invents
    # height (measured: 6-frame arcs with 12-14 m apexes on SNMOT-074). An arc
    # needs at least 0.4 s of observations.
    if t[-1] - t[0] < 0.4 or len(t) < 10:
        return "ground", xyz_g, rmse_g, sol_g.x.tolist()

    # ---- aerial model: linear DLT initialisation of p0, v0 for a parabola
    rows, rhs = [], []
    for i, (ti, (u, v)) in enumerate(zip(t, uv)):
        P = cam.P[i]
        grav = np.array([0, 0, -0.5 * G * ti * ti, 0.0])
        for img, Prow in ((u, P[0]), (v, P[1])):
            c = img * P[2] - Prow                      # c . [X 1] = 0
            rows.append(np.r_[c[:3], c[:3] * ti])
            rhs.append(-(c[3] + c[:3] @ grav[:3]))
    sol, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
    a0 = np.r_[sol, 0.1 if allow_drag else 0.0]

    def ra(p):
        X = _ballistic(p[:3], p[3:6], t, abs(p[6]) if allow_drag else 0.0)
        res = _reproj(cam, X, uv)
        below = np.minimum(X[:, 2], 0.0) * 50.0         # ball cannot go underground
        return np.r_[res, below]
    sol_a = least_squares(ra, a0, loss="soft_l1", f_scale=3.0)
    Xa = _ballistic(sol_a.x[:3], sol_a.x[3:6], t, abs(sol_a.x[6]) if allow_drag else 0.0)
    rmse_a = float(np.sqrt(np.mean(sol_a.fun[: 2 * len(t)] ** 2)))
    peak = float(np.nanmax(Xa[:, 2]))
    speed0 = float(np.linalg.norm(sol_a.x[3:6]))
    # 36 m/s launch covers a hard shot or a long punt; the apex must be one the
    # ball can reach in the time it is seen (vz^2 / 2g above the lowest point)
    vz_max = float(np.max(np.abs(np.gradient(Xa[:, 2], t)))) if len(t) > 2 else 0.0
    plausible = (peak >= min_height and Xa[:, 2].min() > -0.3 and peak < 25 and speed0 < 36
                 and peak - Xa[:, 2].min() <= vz_max ** 2 / (2 * G) + 0.5)
    # the flight must explain the pixels clearly better, in absolute terms too
    if plausible and rmse_a < air_gain * rmse_g and rmse_g - rmse_a > 1.0:
        return "air", Xa, rmse_a, sol_a.x.tolist()
    if float(np.linalg.norm(sol_g.x[2:4])) > 36.0:
        # no ball rolls this fast: an airborne ball whose flight this camera
        # cannot resolve; better unknown than a ball racing over the grass
        return "unresolved", np.full((len(t), 3), np.nan), rmse_g, []
    return "ground", xyz_g, rmse_g, sol_g.x.tolist()


def segment_track(frames: Sequence[int], uv: np.ndarray, max_gap: int = 6,
                  turn_deg: float = 35.0, speed_jump: float = 0.6, min_len: int = 4) -> List[Tuple[int, int]]:
    """Split a pixel track into arcs at gaps and sharp velocity changes."""
    f = np.asarray(frames)
    uv = np.asarray(uv, float)
    if len(f) < 2:
        return [(0, len(f))] if len(f) else []
    cuts = {0, len(f)}
    for i in range(1, len(f)):
        if f[i] - f[i - 1] > max_gap:
            cuts.add(i)
    v = np.gradient(uv, f.astype(float), axis=0)
    # smoothed velocity before/after each point
    for i in range(2, len(f) - 2):
        a = v[max(0, i - 3):i].mean(0)
        b = v[i + 1:i + 4].mean(0)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1.0 and nb < 1.0:
            continue
        cos = float(a @ b / max(na * nb, 1e-9))
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        jump = abs(nb - na) / max(na, nb)
        if ang > turn_deg or jump > speed_jump:
            cuts.add(i)
    cuts = sorted(cuts)
    segs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a >= 1:
            segs.append((a, b))
    # merge tiny segments into neighbours
    merged = []
    for s in segs:
        if merged and (s[1] - s[0] < min_len) and f[s[0]] - f[merged[-1][1] - 1] <= max_gap:
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(s)
    return merged


def reconstruct(frames: Sequence[int], uv: np.ndarray, H_img_to_pitch: np.ndarray,
                image_size: Tuple[int, int], fps: float,
                max_rmse_px: float = 4.0) -> Optional[Ball3DResult]:
    """Full 3D reconstruction for one ball track (pixels -> metres).

    ``H_img_to_pitch`` is one homography (fixed camera) or one per
    observation, shape (n, 3, 3), for a camera that pans or zooms."""
    Hs = np.asarray(H_img_to_pitch, float)
    if Hs.ndim == 2:
        Hs = np.repeat(Hs[None], len(frames), axis=0)
    moving = len({np.round(Hf, 6).tobytes() for Hf in Hs}) > 1
    if moving:
        cams_list = ptz_cameras(Hs, image_size)
    else:
        Hinv = np.linalg.inv(Hs[0])
        c = camera_from_homography(Hinv / Hinv[2, 2], image_size)
        cams_list = [c] * len(Hs)
    Ps = [None if c is None else c.P for c in cams_list]
    good = [P for P in Ps if P is not None]
    if not good:
        return None
    first = next(c for c in cams_list if c is not None)
    # a frame whose camera cannot be recovered borrows the nearest one that can
    last = good[0]
    for i, P in enumerate(Ps):
        Ps[i] = last = P if P is not None else last
    cam = CameraSet(np.stack(Ps), Hs)
    frames = np.asarray(frames, int)
    uv = np.asarray(uv, float)
    xyz = np.full((len(frames), 3), np.nan)
    segs: List[Segment] = []

    def fit(a, b):
        return fit_segment(cam.sub(a, b), frames[a:b] / fps, uv[a:b])

    def total_err(parts):
        tot = 0.0
        for a, b in parts:
            _, _, rmse, _ = fit(a, b)
            tot += (rmse if np.isfinite(rmse) else 1e3) ** 2 * (b - a)
        return tot

    pieces = segment_track(frames, uv)
    # Refine each pivot within +-3 samples: the velocity test localises a
    # kick/bounce only to within its smoothing window.
    for k in range(1, len(pieces)):
        (a0, c), (_, b1) = pieces[k - 1], pieces[k]
        if frames[c] - frames[c - 1] > 6:
            continue                                  # a real gap, not a pivot
        best = (total_err([(a0, c), (c, b1)]), c)
        for cc in range(max(a0 + 3, c - 3), min(b1 - 3, c + 4)):
            if cc != c:
                best = min(best, (total_err([(a0, cc), (cc, b1)]), cc))
        pieces[k - 1], pieces[k] = (a0, best[1]), (best[1], b1)
    # Recursively split arcs that one flight model cannot explain.
    stack, final = list(pieces), []
    while stack:
        a, b = stack.pop(0)
        kind, X, rmse, params = fit(a, b)
        if rmse > max_rmse_px and b - a >= 12:
            res = np.linalg.norm(cam.sub(a, b).project(X) - uv[a:b], axis=1) if np.isfinite(X).all() else None
            cut = a + (int(np.argmax(res[3:-3])) + 3 if res is not None else (b - a) // 2)
            stack[:0] = [(a, cut), (cut, b)]
            continue
        final.append((a, b, kind, X, rmse, params))
    # Merge neighbours that one flight model explains as well as two: a high
    # ball's curved image path can trip the velocity test several times.
    final.sort(key=lambda r: r[0])
    merged = True
    while merged and len(final) > 1:
        merged = False
        for k in range(len(final) - 1):
            a, m = final[k][0], final[k][1]
            b = final[k + 1][1]
            if frames[m] - frames[m - 1] > 6:
                continue
            kind, X, rmse, params = fit(a, b)
            worst = max(final[k][4], final[k + 1][4])
            if np.isfinite(rmse) and rmse <= max_rmse_px and rmse <= 1.25 * worst + 0.5:
                final[k:k + 2] = [(a, b, kind, X, rmse, params)]
                merged = True
                break
    for a, b, kind, X, rmse, params in final:
        xyz[a:b] = X
        launch = peak = None
        if kind == "air":
            launch = float(np.linalg.norm(params[3:6]))
            peak = float(np.nanmax(X[:, 2]))
        elif kind == "ground" and params:
            launch = float(np.linalg.norm(params[2:4]))
            peak = 0.0
        segs.append(Segment(int(frames[a]), int(frames[b - 1]), kind, float(rmse), launch, peak,
                            [float(v) for v in params]))
    speed = np.full(len(frames), np.nan)
    for i in range(1, len(frames)):
        dt = (frames[i] - frames[i - 1]) / fps
        if dt > 0 and np.isfinite(xyz[i]).all() and np.isfinite(xyz[i - 1]).all() and dt < 0.5:
            speed[i] = np.linalg.norm(xyz[i] - xyz[i - 1]) / dt
    return Ball3DResult(list(frames), xyz, speed, segs, float(first.K[0, 0]), float(first.centre[2]))


def shot_metrics(xyz: np.ndarray, speed: np.ndarray, goal_x: float, pitch_width: float = 68.0,
                 defenders_xy: Optional[np.ndarray] = None) -> dict:
    """Replayfield-style goal breakdown for a shot starting at xyz[0]."""
    p = np.asarray(xyz, float)[0]
    gy = pitch_width / 2.0
    dist = float(np.hypot(goal_x - p[0], gy - p[1]))
    half = 7.32 / 2
    a1 = np.arctan2(gy - half - p[1], abs(goal_x - p[0]))
    a2 = np.arctan2(gy + half - p[1], abs(goal_x - p[0]))
    sp = np.asarray(speed, float)
    in_way = None
    if defenders_xy is not None and len(defenders_xy):
        # players inside the triangle formed by the ball and the two posts
        tri = np.array([p[:2], [goal_x, gy - half], [goal_x, gy + half]])
        def inside(q):
            d = np.cross(tri[1] - tri[0], q - tri[0]), np.cross(tri[2] - tri[1], q - tri[1]), \
                np.cross(tri[0] - tri[2], q - tri[2])
            return (all(x >= 0 for x in d) or all(x <= 0 for x in d))
        in_way = int(sum(inside(np.asarray(q, float)) for q in defenders_xy))
    return {
        # upper quartile of the first frames: a single noisy step must not set the speed
        "shot_speed_kmh": None if not np.isfinite(sp[:6]).any() else round(float(np.nanpercentile(sp[:6], 75)) * 3.6, 1),
        "distance_to_goal_m": round(dist, 1),
        "goal_angle_deg": round(float(np.degrees(abs(a2 - a1))), 1),
        "players_in_the_way": in_way,
        "peak_height_m": None if not np.isfinite(p[2]) else round(float(np.nanmax(np.asarray(xyz)[:, 2])), 2),
    }
