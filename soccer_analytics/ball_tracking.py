"""Ball trajectory selection and gap filling.

The ball is the hardest object in football video: it is small, blurred, often
occluded, and detectors fire on heads, socks, penalty spots and advertising.
Taking the single highest-confidence ball box per frame (the previous
behaviour) therefore both misses low-confidence true balls and accepts isolated
false positives, and the old interpolation filled *any* gap, including
back/forward-filling the first and last sightings over the whole clip.

``select_ball_path`` instead solves a small dynamic programme over every ball
candidate in every frame (TrackNet-style temporal consistency, cf. Huang et al.
2019; the "ball seen" metric of Replayfield):

* a candidate scores ``conf - tau`` so weak boxes only survive when they are
  supported by neighbours;
* a chain is a sequence of candidates linked within ``max_skip`` frames whose
  implied speed stays below a physical bound; each new chain pays ``birth``;
* the optimum picks at most one ball per frame and maximises total score.

Gaps *inside* a chain (the ball was occluded or missed but the trajectory on
both sides agrees) are filled by quadratic interpolation up to
``max_interp`` frames and flagged as interpolated. Gaps between chains are left
empty: an honest "ball not seen".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class BallPoint:
    frame: int
    x: float
    y: float
    conf: float
    interpolated: bool
    chain: int


def select_ball_path(candidates: Dict[int, np.ndarray], n_frames: int, fps: float = 25.0,
                     drop_static: bool = True, **kw) -> List[BallPoint]:
    """Ball path with spare-ball rejection.

    Grounds often have a second, motionless ball (by a goal or the touchline).
    It forms a perfectly consistent chain, so a single DP pass can prefer it.
    A chain that never moves during its whole life (>= 2 s, < 15 px drift) is
    treated as a spare ball when moving candidates exist elsewhere: its
    location is removed from the candidates and the path is solved again.
    """
    path = _select(candidates, n_frames, **kw)
    if not drop_static or not path:
        return path
    chains: Dict[int, List[BallPoint]] = {}
    for p in path:
        if not p.interpolated:
            chains.setdefault(p.chain, []).append(p)
    spots = []
    for pts in chains.values():
        if len(pts) < 2 * fps:
            continue
        xy = np.asarray([[p.x, p.y] for p in pts])
        med = np.median(xy, axis=0)
        if np.max(np.hypot(*(xy - med).T)) >= 15.0:
            continue
        # a spare ball sits there for most of the clip; a match ball awaiting
        # a restart does not
        present = sum(1 for c in candidates.values()
                      if np.any(np.hypot(np.asarray(c).reshape(-1, 3)[:, 0] - med[0], np.asarray(c).reshape(-1, 3)[:, 1] - med[1]) < 20))
        if present >= 0.6 * n_frames:
            spots.append(med)
    if not spots:
        return path
    def far(row):
        return all(np.hypot(row[0] - s[0], row[1] - s[1]) > 20 for s in spots)
    pruned = {f: np.asarray([r for r in np.asarray(c).reshape(-1, 3) if far(r)]).reshape(-1, 3) for f, c in candidates.items()}
    moving = sum(len(c) for c in pruned.values())
    if moving < fps:                                 # nothing else to follow: keep the original
        return path
    return _select(pruned, n_frames, **kw)


def _select(candidates: Dict[int, np.ndarray], n_frames: int,
                     tau: float = 0.2, birth: float = 0.6, max_skip: int = 12,
                     vmax_px: float = 70.0, skip_cost: float = 0.02,
                     motion_weight: float = 0.25, max_interp: int = 12) -> List[BallPoint]:
    """Pick a temporally consistent ball path.

    ``candidates[f]`` is an (n,3) array of (cx, cy, conf) image-pixel ball
    candidates for frame ``f`` (0-based). ``vmax_px`` is the largest plausible
    ball displacement per frame in pixels (about 70 px at 1080p broadcast for a
    hard shot; scale with resolution).
    """
    nodes: List[Tuple[int, float, float, float]] = []
    by_frame: Dict[int, List[int]] = {}
    for f in range(n_frames):
        c = candidates.get(f)
        if c is None or len(c) == 0:
            continue
        for row in np.asarray(c, float).reshape(-1, 3):
            if not np.isfinite(row).all():
                continue
            by_frame.setdefault(f, []).append(len(nodes))
            nodes.append((f, float(row[0]), float(row[1]), float(row[2])))
    if not nodes:
        return []

    best = np.full(len(nodes), -np.inf)
    prev = np.full(len(nodes), -1)
    ended = np.zeros(n_frames + 1)             # best total of chains finished before f
    ended_arg: List[Optional[int]] = [None] * (n_frames + 1)
    for f in range(n_frames):
        ended[f + 1] = ended[f]
        ended_arg[f + 1] = ended_arg[f]
        for i in by_frame.get(f, []):
            _, x, y, conf = nodes[i]
            score = conf - tau
            start = ended[f] - birth + score
            b, p = start, -2 - (ended_arg[f] if ended_arg[f] is not None else -1)
            # -2 - k encodes "new chain after node k" (k=-1 -> nothing before)
            for g in range(max(0, f - max_skip), f):
                for j in by_frame.get(g, []):
                    if not np.isfinite(best[j]):
                        continue
                    k = f - g
                    d = np.hypot(x - nodes[j][1], y - nodes[j][2])
                    if d > vmax_px * k:
                        continue
                    v = best[j] + score - skip_cost * (k - 1) - motion_weight * (d / (vmax_px * k)) ** 2
                    if v > b:
                        b, p = v, j
            best[i], prev[i] = b, p
            if b > ended[f + 1]:
                ended[f + 1] = b
                ended_arg[f + 1] = i

    # backtrack
    picked: List[Tuple[int, int]] = []          # (node, chain id)
    i = ended_arg[n_frames]
    chain = 0
    while i is not None and i >= 0:
        picked.append((i, chain))
        p = int(prev[i])
        if p <= -2:                             # chain start
            chain += 1
            k = -2 - p
            i = k if k >= 0 else None
        else:
            i = p
    picked.reverse()
    nchains = chain
    pts = [BallPoint(nodes[i][0], nodes[i][1], nodes[i][2], nodes[i][3], False,
                     nchains - c) for i, c in picked]
    return fill_gaps(pts, max_interp)


def fill_gaps(points: Sequence[BallPoint], max_interp: int = 12) -> List[BallPoint]:
    """Quadratic interpolation of short gaps *within* a chain only."""
    out: List[BallPoint] = []
    pts = list(points)
    for k, p in enumerate(pts):
        out.append(p)
        if k + 1 >= len(pts):
            continue
        q = pts[k + 1]
        gap = q.frame - p.frame
        if q.chain != p.chain or gap <= 1 or gap > max_interp:
            continue
        # neighbours on both sides of the gap for a local quadratic fit
        lo = [r for r in pts[max(0, k - 3):k + 1] if r.chain == p.chain]
        hi = [r for r in pts[k + 1:k + 5] if r.chain == p.chain]
        ref = lo + hi
        t = np.asarray([r.frame for r in ref], float)
        xy = np.asarray([[r.x, r.y] for r in ref], float)
        deg = 2 if len(ref) >= 4 else 1
        coef = np.polyfit(t - p.frame, xy, deg)
        for f in range(p.frame + 1, q.frame):
            v = np.polyval(coef[:, 0], f - p.frame), np.polyval(coef[:, 1], f - p.frame)
            out.append(BallPoint(f, float(v[0]), float(v[1]), 0.0, True, p.chain))
    return out


def ball_path_metrics(path: Sequence[BallPoint], gt: Dict[int, Tuple[float, float]],
                      n_frames: int, radius: float = 12.0) -> dict:
    """Precision/recall of per-frame ball positions against ground-truth centres."""
    pred = {p.frame: p for p in path}
    tp = fp = 0
    tp_obs = tp_interp = fp_interp = 0
    for f, p in pred.items():
        g = gt.get(f)
        ok = g is not None and np.hypot(p.x - g[0], p.y - g[1]) <= radius
        tp += ok
        fp += not ok
        if p.interpolated:
            tp_interp += ok
            fp_interp += not ok
        else:
            tp_obs += ok
    n_gt = len(gt)
    return {
        "gt_frames": n_gt, "predicted_frames": len(pred), "frames": n_frames,
        "precision": tp / max(1, tp + fp), "recall": tp / max(1, n_gt),
        "f1": 2 * tp / max(1, 2 * tp + fp + (n_gt - tp)),
        "interpolated_frames": sum(p.interpolated for p in pred.values()),
        "interpolated_precision": tp_interp / max(1, tp_interp + fp_interp),
    }
