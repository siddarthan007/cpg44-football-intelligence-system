"""IMM-EIoU tracker for football players (training-free, CPU, NumPy).

This tracker combines four published ideas that address the specific failure
modes of football footage, each of which can be switched off for ablation:

1. **Interacting Multiple Model (IMM) Kalman filter** (Blom & Bar-Shalom 1988;
   used for sports MOT in ARC-SORT, Adari 2026). Two constant-velocity modes
   share the box state ``[cx, cy, w, h, vx, vy, vw, vh]``: a *steady* mode with
   BoT-SORT's process noise and a *manoeuvre* mode with ~4x velocity noise. A
   winger who cuts or stops moves probability mass into the manoeuvre mode, so
   the predicted covariance widens exactly when linear extrapolation fails.
2. **Uncertainty-adaptive Expansion IoU** (Deep-EIoU, Huang et al. WACVW 2024;
   continuous variant as in ARC-SORT). Track boxes are enlarged by
   ``e = clip(e0 + lam * sqrt(tr P_xy) / sqrt(w h), e0, e_max)`` before IoU,
   so fast, uncertain or long-lost tracks get a larger search region while
   steady tracks keep tight gating.
3. **Two-stage BYTE association** (ByteTrack, Zhang et al. ECCV 2022): weak
   detections are kept for a second pass that rescues occluded/blurred players.
4. **Camera motion compensation** with player boxes masked out (BoT-SORT GMC,
   Aharon et al. 2022; masking as recommended in McByte++, 2026), estimated as a
   similarity transform from sparse optical flow on a downscaled frame.

Optional appearance (``emb``) is a colour-histogram embedding that acts as a
veto against matching across very different kits; it is *not* a trained ReID
model and is only used to break geometric ties.

Benchmark it with ``scripts/benchmark_tracking.py`` on cached detections before
promoting it over ByteTrack; the numbers live in docs/TRACKING_BENCHMARK.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# --------------------------------------------------------------------------- #
# geometry helpers


def xyxy_to_cxcywh(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, float).reshape(-1, 4)
    return np.column_stack(((b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                            b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]))


def cxcywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, float).reshape(-1, 4)
    return np.column_stack((b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                            b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2))


def expanded_iou(a_cxcywh: np.ndarray, ea: np.ndarray,
                 b_cxcywh: np.ndarray, eb: float) -> np.ndarray:
    """IoU after enlarging each box by its own expansion factor (Deep-EIoU)."""
    if len(a_cxcywh) == 0 or len(b_cxcywh) == 0:
        return np.zeros((len(a_cxcywh), len(b_cxcywh)))
    a = a_cxcywh.copy()
    b = b_cxcywh.copy()
    a[:, 2:] *= (1 + 2 * np.asarray(ea, float))[:, None]
    b[:, 2:] *= 1 + 2 * eb
    A, B = cxcywh_to_xyxy(a), cxcywh_to_xyxy(b)
    iw = np.clip(np.minimum(A[:, None, 2], B[None, :, 2]) - np.maximum(A[:, None, 0], B[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(A[:, None, 3], B[None, :, 3]) - np.maximum(A[:, None, 1], B[None, :, 1]), 0, None)
    inter = iw * ih
    area_a = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    area_b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    union = area_a[:, None] + area_b[None] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


# --------------------------------------------------------------------------- #
# camera motion (optional, needs OpenCV)


class CameraMotion:
    """Similarity transform between consecutive frames from masked sparse flow.

    Players are masked so their own running does not masquerade as a camera pan.
    Returns a 2x3 matrix mapping previous-frame pixels to current-frame pixels.
    """

    def __init__(self, scale: float = 0.25, max_corners: int = 400):
        self.scale = scale
        self.max_corners = max_corners
        self._prev = None
        self._prev_pts = None

    def update(self, frame, boxes_xyxy: Optional[np.ndarray] = None) -> np.ndarray:
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        small = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        mask = np.full(small.shape, 255, np.uint8)
        if boxes_xyxy is not None and len(boxes_xyxy):
            for x1, y1, x2, y2 in (np.asarray(boxes_xyxy) * self.scale).astype(int):
                mask[max(0, y1 - 2):max(0, y2 + 2), max(0, x1 - 2):max(0, x2 + 2)] = 0
        warp = np.eye(2, 3)
        if self._prev is not None and self._prev_pts is not None and len(self._prev_pts) >= 8:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(self._prev, small, self._prev_pts, None,
                                                      winSize=(15, 15), maxLevel=3)
            ok = status.reshape(-1) == 1
            if ok.sum() >= 8:
                m, inliers = cv2.estimateAffinePartial2D(self._prev_pts[ok], nxt[ok],
                                                         method=cv2.RANSAC,
                                                         ransacReprojThreshold=1.5)
                if m is not None and inliers is not None and inliers.sum() >= 8:
                    warp = m.copy()
                    warp[:, 2] /= self.scale
        self._prev = small
        self._prev_pts = cv2.goodFeaturesToTrack(small, self.max_corners, 0.01, 6, mask=mask)
        return warp


def appearance_embedding(frame, boxes_xyxy: np.ndarray) -> np.ndarray:
    """64-d Hellinger-normalised HSV histogram of shirt and shorts, grass masked.

    Cheap kit/shorts descriptor; separates teams and referees, weakly separates
    team-mates. Use only as a veto or tie-breaker.
    """
    import cv2
    boxes = np.asarray(boxes_xyxy, float).reshape(-1, 4)
    out = np.zeros((len(boxes), 64), np.float32)
    H, W = frame.shape[:2]
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(W, x2)), int(min(H, y2))
        w, h = x2 - x1, y2 - y1
        if w < 6 or h < 12:
            continue
        feats = []
        for top, bottom in ((0.15, 0.50), (0.50, 0.75)):
            patch = frame[y1 + int(top * h):y1 + int(bottom * h), x1 + int(0.2 * w):x2 - int(0.2 * w)]
            if patch.size == 0:
                feats.append(np.zeros(32, np.float32))
                continue
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            grass = (hsv[..., 0] >= 35) & (hsv[..., 0] <= 85) & (hsv[..., 1] >= 60)
            m = (~grass).astype(np.uint8)
            hist = cv2.calcHist([hsv], [0, 1], m, [8, 4], [0, 180, 0, 256]).reshape(-1)
            # low-saturation pixels (white/black kits) are hue-less; add value info
            feats.append(hist.astype(np.float32))
        v = np.sqrt(np.concatenate(feats))
        n = np.linalg.norm(v)
        if n > 0:
            out[i] = v / n
    return out


# --------------------------------------------------------------------------- #
# vectorised IMM Kalman bank


@dataclass
class IMMConfig:
    std_pos: float = 1 / 20          # BoT-SORT relative noise weights
    std_vel: float = 1 / 160
    manoeuvre_scale: float = 4.0     # velocity noise multiplier of mode 2
    stay_prob: Tuple[float, float] = (0.95, 0.85)


class _Track:
    __slots__ = ("id", "x", "P", "mu", "emb", "hits", "age_lost", "state", "last_frame",
                 "start_frame", "score", "det_box")

    def __init__(self, tid, x, P, mu, emb, frame, score, det_box):
        self.id = tid
        self.x = x          # (M, 8)
        self.P = P          # (M, 8, 8)
        self.mu = mu        # (M,)
        self.emb = emb
        self.hits = 1
        self.age_lost = 0
        self.state = "new"  # new -> tracked -> lost -> removed
        self.last_frame = frame
        self.start_frame = frame
        self.score = score
        self.det_box = det_box


class IMMTracker:
    """Online multi-object tracker. ``update`` returns confirmed tracks for the frame.

    Parameters mirror ByteTrack where they mean the same thing.
    """

    def __init__(self, high_thresh: float = 0.5, low_thresh: float = 0.1,
                 new_track_thresh: float = 0.6, match_thresh: float = 0.8,
                 second_match_thresh: float = 0.5, unconfirmed_thresh: float = 0.7,
                 track_buffer: int = 60, e0: float = 0.3, e_lambda: float = 0.5,
                 e_max: float = 0.9, det_expand: float = 0.3, use_imm: bool = True,
                 use_appearance: bool = True, app_veto: float = 0.55, app_weight: float = 0.25,
                 output: str = "detection", cfg: IMMConfig | None = None):
        self.high, self.low = high_thresh, low_thresh
        self.new_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.second_thresh = second_match_thresh
        self.unconfirmed_thresh = unconfirmed_thresh
        self.track_buffer = track_buffer
        self.e0, self.e_lambda, self.e_max, self.det_expand = e0, e_lambda, e_max, det_expand
        self.use_imm = use_imm
        self.use_app = use_appearance
        self.app_veto, self.app_weight = app_veto, app_weight
        self.output = output
        self.cfg = cfg or IMMConfig()
        self.tracks: List[_Track] = []
        self.frame = 0
        self._next_id = 1
        M = 2 if use_imm else 1
        self.M = M
        p0, p1 = self.cfg.stay_prob
        self.TPM = np.array([[p0, 1 - p0], [1 - p1, p1]]) if M == 2 else np.ones((1, 1))
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.Hm = np.eye(4, 8)
        # Diagnostics for the review queue: (frame, id_a, id_b) close calls.
        self.ambiguous_pairs: List[Tuple[int, int, int]] = []

    # ---------------- Kalman primitives (batched over tracks) ---------------- #
    def _Q(self, wh: np.ndarray) -> np.ndarray:
        """Process noise per track and mode: (N, M, 8, 8)."""
        N = len(wh)
        sp, sv = self.cfg.std_pos, self.cfg.std_vel
        base = np.column_stack((sp * wh[:, 0], sp * wh[:, 1], sp * wh[:, 0], sp * wh[:, 1],
                                sv * wh[:, 0], sv * wh[:, 1], sv * wh[:, 0], sv * wh[:, 1]))
        Q = np.zeros((N, self.M, 8, 8))
        for m in range(self.M):
            s = base.copy()
            if m == 1:
                s[:, 4:6] *= self.cfg.manoeuvre_scale
                s[:, 0:2] *= 1.5
            Q[:, m] = np.einsum("ni,ij->nij", s ** 2, np.eye(8))
        return Q

    def _R(self, wh: np.ndarray) -> np.ndarray:
        sp = self.cfg.std_pos
        s = np.column_stack((sp * wh[:, 0], sp * wh[:, 1], sp * wh[:, 0], sp * wh[:, 1]))
        return np.einsum("ni,ij->nij", s ** 2, np.eye(4))

    def _init_state(self, z: np.ndarray):
        w, h = z[2], z[3]
        sp, sv = self.cfg.std_pos, self.cfg.std_vel
        x = np.zeros(8)
        x[:4] = z
        std = np.array([2 * sp * w, 2 * sp * h, 2 * sp * w, 2 * sp * h,
                        10 * sv * w, 10 * sv * h, 10 * sv * w, 10 * sv * h])
        P = np.diag(std ** 2)
        mu = np.array([0.8, 0.2]) if self.M == 2 else np.ones(1)
        return np.repeat(x[None], self.M, 0), np.repeat(P[None], self.M, 0), mu

    def _predict(self, tracks: List[_Track]) -> None:
        if not tracks:
            return
        X = np.stack([t.x for t in tracks])            # (N,M,8)
        P = np.stack([t.P for t in tracks])            # (N,M,8,8)
        mu = np.stack([t.mu for t in tracks])          # (N,M)
        for i, t in enumerate(tracks):                 # lost tracks: freeze size drift
            if t.state != "tracked":
                X[i, :, 6:8] = 0
        if self.M == 2:
            c = mu @ self.TPM                          # predicted mode probs (N,M)
            wmix = (self.TPM[None] * mu[:, :, None]) / np.maximum(c[:, None, :], 1e-12)  # (N,i,j)
            X0 = np.einsum("nij,nik->njk", wmix, X)
            d = X[:, :, None, :] - X0[:, None, :, :]   # (N,i,j,8)
            P0 = np.einsum("nij,nikl->njkl", wmix, P) + \
                np.einsum("nij,nijk,nijl->njkl", wmix, d, d)
        else:
            c, X0, P0 = mu, X, P
        wh = np.maximum(np.einsum("nm,nmk->nk", c, X0)[:, 2:4], 1.0)
        Xp = np.einsum("ij,nmj->nmi", self.F, X0)
        Pp = np.einsum("ij,nmjk,lk->nmil", self.F, P0, self.F) + self._Q(wh)
        for i, t in enumerate(tracks):
            t.x, t.P, t.mu = Xp[i], Pp[i], c[i]

    @staticmethod
    def _combined(t: _Track) -> Tuple[np.ndarray, np.ndarray]:
        x = t.mu @ t.x
        d = t.x - x
        P = np.einsum("m,mij->ij", t.mu, t.P) + np.einsum("m,mi,mj->ij", t.mu, d, d)
        return x, P

    def _update_track(self, t: _Track, z: np.ndarray) -> None:
        R = self._R(np.maximum(z[None, 2:4], 1.0))[0]
        lik = np.zeros(self.M)
        for m in range(self.M):
            x, P = t.x[m], t.P[m]
            y = z - self.Hm @ x
            S = self.Hm @ P @ self.Hm.T + R
            Sinv = np.linalg.inv(S)
            K = P @ self.Hm.T @ Sinv
            t.x[m] = x + K @ y
            I_KH = np.eye(8) - K @ self.Hm
            t.P[m] = I_KH @ P @ I_KH.T + K @ R @ K.T
            maha = float(y @ Sinv @ y)
            lik[m] = np.exp(-0.5 * min(maha, 200.0)) / np.sqrt(max(np.linalg.det(2 * np.pi * S), 1e-300))
        if self.M == 2:
            mu = t.mu * lik
            s = mu.sum()
            t.mu = mu / s if s > 1e-300 and np.isfinite(s) else np.array([0.5, 0.5])
            t.mu = np.clip(t.mu, 1e-4, 1.0)
            t.mu /= t.mu.sum()

    def _apply_warp(self, warp: np.ndarray) -> None:
        if warp is None or np.allclose(warp, np.eye(2, 3)):
            return
        A = warp[:, :2]
        tvec = warp[:, 2]
        scale = float(np.sqrt(abs(np.linalg.det(A))))
        T = np.zeros((8, 8))
        T[0:2, 0:2] = A
        T[4:6, 4:6] = A
        T[2, 2] = T[3, 3] = T[6, 6] = T[7, 7] = scale
        for t in self.tracks:
            t.x = np.einsum("ij,mj->mi", T, t.x)
            t.x[:, 0:2] += tvec
            t.P = np.einsum("ij,mjk,lk->mil", T, t.P, T)

    # ---------------- association helpers ---------------- #
    def _boxes_and_expansion(self, tracks: List[_Track]):
        if not tracks:
            return np.zeros((0, 4)), np.zeros(0)
        boxes, exp = [], []
        for t in tracks:
            x, P = self._combined(t)
            w, h = max(x[2], 1.0), max(x[3], 1.0)
            unc = np.sqrt(max(P[0, 0] + P[1, 1], 0.0)) / np.sqrt(w * h)
            boxes.append(x[:4])
            exp.append(np.clip(self.e0 + self.e_lambda * unc, self.e0, self.e_max))
        return np.asarray(boxes), np.asarray(exp)

    def _cost(self, tracks, det_cxcywh, det_scores, det_emb, fuse: bool):
        tb, te = self._boxes_and_expansion(tracks)
        iou = expanded_iou(tb, te, det_cxcywh, self.det_expand)
        sim = iou * det_scores[None] if fuse else iou
        cost = 1.0 - sim
        if self.use_app and det_emb is not None and len(tracks) and len(det_cxcywh):
            temb = np.stack([t.emb if t.emb is not None else np.zeros(det_emb.shape[1]) for t in tracks])
            valid_t = np.linalg.norm(temb, axis=1) > 0
            valid_d = np.linalg.norm(det_emb, axis=1) > 0
            app = 1.0 - temb @ det_emb.T
            both = valid_t[:, None] & valid_d[None]
            cost = np.where(both, (1 - self.app_weight) * cost + self.app_weight * app, cost)
            cost = np.where(both & (app > self.app_veto), 1.0, cost)
        cost = np.where(iou <= 0, 1.0, cost)
        return cost

    def _assign(self, cost: np.ndarray, thresh: float):
        if cost.size == 0:
            return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
        rows, cols = linear_sum_assignment(cost)
        matches = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= thresh]
        mr = {r for r, _ in matches}
        mc = {c for _, c in matches}
        # Close-call diagnostics: two tracks competing for the same detection.
        for r, c in matches:
            col = np.sort(cost[:, c])
            if len(col) > 1 and col[1] - col[0] < 0.05 and col[1] <= thresh:
                other = int(np.argsort(cost[:, c])[1])
                self.ambiguous_pairs.append((self.frame, r, other))
        return matches, [r for r in range(cost.shape[0]) if r not in mr], \
            [c for c in range(cost.shape[1]) if c not in mc]

    # ---------------- main step ---------------- #
    def update(self, xyxy: np.ndarray, scores: np.ndarray, emb: Optional[np.ndarray] = None,
               warp: Optional[np.ndarray] = None) -> List[Tuple[int, np.ndarray, float]]:
        self.frame += 1
        xyxy = np.asarray(xyxy, float).reshape(-1, 4)
        scores = np.asarray(scores, float).reshape(-1)
        if emb is not None:
            emb = np.asarray(emb, float).reshape(len(xyxy), -1)
        keep = scores >= self.low
        xyxy, scores = xyxy[keep], scores[keep]
        emb = emb[keep] if emb is not None else None
        dets = xyxy_to_cxcywh(xyxy)
        hi = scores >= self.high
        lo = ~hi

        self.tracks = [t for t in self.tracks if t.state != "removed"]
        self._predict(self.tracks)
        self._apply_warp(warp)

        tracked = [t for t in self.tracks if t.state == "tracked"]
        lost = [t for t in self.tracks if t.state == "lost"]
        new = [t for t in self.tracks if t.state == "new"]
        pool = tracked + lost

        hi_idx = np.nonzero(hi)[0]
        lo_idx = np.nonzero(lo)[0]

        # Stage 1: confident detections vs tracked + lost tracks
        cost = self._cost(pool, dets[hi_idx], scores[hi_idx],
                          emb[hi_idx] if emb is not None else None, fuse=True)
        m1, um_t, um_d = self._assign(cost, self.match_thresh)
        for r, c in m1:
            self._activate(pool[r], hi_idx[c], dets, scores, xyxy, emb)

        # Stage 2: weak detections vs still-unmatched *tracked* tracks
        remain = [pool[r] for r in um_t if pool[r].state == "tracked"]
        cost2 = self._cost(remain, dets[lo_idx], scores[lo_idx], None, fuse=False)
        m2, um_t2, _ = self._assign(cost2, self.second_thresh)
        for r, c in m2:
            self._activate(remain[r], lo_idx[c], dets, scores, xyxy, None)
        for r in um_t2:
            remain[r].state = "lost"
        for r in um_t:
            if pool[r].state == "lost":
                pool[r].age_lost += 1

        # Stage 3: unconfirmed tracks vs remaining confident detections
        rest = [hi_idx[c] for c in um_d]
        cost3 = self._cost(new, dets[rest], scores[rest],
                           emb[rest] if emb is not None else None, fuse=True)
        m3, um_new, um_d3 = self._assign(cost3, self.unconfirmed_thresh)
        for r, c in m3:
            self._activate(new[r], rest[c], dets, scores, xyxy, emb)
        for r in um_new:
            new[r].state = "removed"

        # Births
        for c in um_d3:
            j = rest[c]
            if scores[j] < self.new_thresh:
                continue
            x, P, mu = self._init_state(dets[j])
            t = _Track(self._next_id, x, P, mu, None if emb is None else emb[j].copy(),
                       self.frame, scores[j], xyxy[j].copy())
            self._next_id += 1
            if self.frame == 1:
                t.state = "tracked"
            self.tracks.append(t)

        for t in self.tracks:
            if t.state == "lost":
                t.age_lost = self.frame - t.last_frame
                if t.age_lost > self.track_buffer:
                    t.state = "removed"

        out = []
        for t in self.tracks:
            if t.state == "tracked" and t.last_frame == self.frame:
                if self.output == "detection":
                    box = t.det_box
                else:
                    box = cxcywh_to_xyxy(self._combined(t)[0][None, :4])[0]
                out.append((t.id, box, t.score))
        return out

    def _activate(self, t: _Track, j: int, dets, scores, xyxy, emb) -> None:
        self._update_track(t, dets[j])
        t.hits += 1
        t.last_frame = self.frame
        t.age_lost = 0
        t.score = float(scores[j])
        t.det_box = xyxy[j].copy()
        t.state = "tracked"
        if emb is not None and np.linalg.norm(emb[j]) > 0:
            # EMA appearance bank, updated only on confident matches
            if t.emb is None or np.linalg.norm(t.emb) == 0:
                t.emb = emb[j].copy()
            else:
                v = 0.9 * t.emb + 0.1 * emb[j]
                t.emb = v / max(np.linalg.norm(v), 1e-9)
