"""Offline tracklet linking and gap filling for recorded matches.

Online trackers must decide identity frame by frame. A recorded match can be
revisited with full context, the strategy behind GTA (Sun et al., ACCV-W 2024)
and the offline merge stage of ARC-SORT/McByte++, and Replayfield's
"forward-backward" post-processing. This module:

1. Builds tracklets from per-frame tracker output.
2. Links a tracklet that ends to one that starts later when motion
   extrapolation, box size and (optionally) appearance agree, never allowing
   two linked pieces to overlap in time. Linking is a min-cost assignment over
   endpoint pairs, repeated until no admissible link remains.
3. Optionally caps the number of simultaneous identities per role/team
   (closed-set cardinality, as Replayfield does with A1..A5/B1..B5 slots).
4. Linearly interpolates short gaps inside a linked identity. Interpolated boxes
   are flagged so analytics can keep them out of *observed* occupancy.

Every link decision is returned with its cost so the review queue can show the
uncertain ones to a coach ("Check" items) instead of silently merging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Tracklet:
    tid: int
    frames: np.ndarray            # (n,) sorted ints
    boxes: np.ndarray             # (n,4) xyxy
    emb: Optional[np.ndarray] = None   # mean appearance, L2 normalised
    team: int = 0
    _ep: dict = field(default_factory=dict, repr=False)

    @property
    def start(self) -> int:
        return int(self.frames[0])

    @property
    def end(self) -> int:
        return int(self.frames[-1])

    def endpoint(self, tail: bool, k: int = 10) -> Tuple[np.ndarray, np.ndarray, float]:
        """(position, velocity px/frame, mean height) from the last/first k frames."""
        key = (tail, k)
        if key not in self._ep:
            self._ep[key] = self._endpoint(tail, k)
        return self._ep[key]

    def _endpoint(self, tail: bool, k: int):
        sel = slice(-k, None) if tail else slice(0, k)
        f = self.frames[sel].astype(float)
        b = self.boxes[sel]
        c = np.column_stack(((b[:, 0] + b[:, 2]) / 2, b[:, 3]))   # foot point
        h = float(np.mean(b[:, 3] - b[:, 1]))
        if len(f) >= 3 and f[-1] > f[0]:
            A = np.column_stack((f - f.mean(), np.ones_like(f)))
            coef, *_ = np.linalg.lstsq(A, c, rcond=None)
            vel = coef[0]
            pos = coef[1] + coef[0] * ((f[-1] if tail else f[0]) - f.mean())
        else:
            vel = np.zeros(2)
            pos = c[-1] if tail else c[0]
        return pos, vel, h


@dataclass
class LinkDecision:
    from_id: int
    to_id: int
    gap_frames: int
    cost: float
    motion: float
    appearance: Optional[float]
    uncertain: bool


@dataclass
class LinkResult:
    frames: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]  # frame -> (ids, xyxy, interpolated)
    identity_of: Dict[int, int]                                    # tracklet id -> final id
    links: List[LinkDecision] = field(default_factory=list)


def tracklets_from_frames(per_frame: Mapping[int, List[Tuple[int, np.ndarray]]],
                          emb: Optional[Mapping[int, Dict[int, np.ndarray]]] = None,
                          teams: Optional[Mapping[int, int]] = None) -> Dict[int, Tracklet]:
    rows: Dict[int, list] = {}
    for f, items in per_frame.items():
        for tid, box in items:
            rows.setdefault(int(tid), []).append((int(f), np.asarray(box, float)))
    out = {}
    for tid, items in rows.items():
        items.sort(key=lambda r: r[0])
        frames = np.asarray([r[0] for r in items])
        boxes = np.stack([r[1] for r in items])
        e = None
        if emb is not None and tid in emb and emb[tid]:
            vecs = np.stack(list(emb[tid].values()))
            vecs = vecs[np.linalg.norm(vecs, axis=1) > 0]
            if len(vecs):
                m = vecs.mean(0)
                e = m / max(np.linalg.norm(m), 1e-9)
        out[tid] = Tracklet(tid, frames, boxes, e, int((teams or {}).get(tid, 0)))
    return out


class TrackletLinker:
    def __init__(self, max_gap: int = 120, motion_gate: float = 1.6,
                 app_gate: float = 0.45, app_weight: float = 0.5, min_len: int = 1,
                 max_extrapolate: int = 25, interpolate_gap: int = 30,
                 uncertain_margin: float = 0.25, max_identities: Optional[int] = None):
        self.max_gap = max_gap
        self.motion_gate = motion_gate
        self.app_gate = app_gate
        self.app_weight = app_weight
        self.min_len = min_len
        self.max_extrapolate = max_extrapolate
        self.interpolate_gap = interpolate_gap
        self.uncertain_margin = uncertain_margin
        self.max_identities = max_identities

    # -------------------------------------------------------------- #
    def _pair_cost(self, a: Tracklet, b: Tracklet) -> Tuple[float, float, Optional[float]]:
        gap = b.start - a.end
        if gap <= 0 or gap > self.max_gap:
            return np.inf, np.inf, None
        if a.team and b.team and a.team != b.team:
            return np.inf, np.inf, None
        pa, va, ha = a.endpoint(tail=True)
        pb, vb, hb = b.endpoint(tail=False)
        if min(ha, hb) <= 1 or max(ha, hb) / min(ha, hb) > 1.8:
            return np.inf, np.inf, None
        steps = min(gap, self.max_extrapolate)
        fwd = pa + va * steps
        bwd = pb - vb * steps
        # Allowed radius grows with the gap (a footballer covers ~0.3 body
        # heights per frame at most); normalised so 1.0 = edge of the gate.
        h = 0.5 * (ha + hb)
        radius = h * (0.6 + 0.06 * gap)
        d = min(np.linalg.norm(fwd - pb), np.linalg.norm(bwd - pa),
                np.linalg.norm(0.5 * (fwd + bwd) - 0.5 * (pa + pb)) + 0.5 * np.linalg.norm(pb - pa))
        motion = float(d / radius)
        if motion > self.motion_gate:
            return np.inf, motion, None
        app = None
        if a.emb is not None and b.emb is not None:
            app = float(1.0 - a.emb @ b.emb)
            if app > self.app_gate:
                return np.inf, motion, app
        cost = motion + (self.app_weight * app / self.app_gate if app is not None else 0.3)
        return float(cost), motion, app

    def link(self, tracklets: Dict[int, Tracklet]) -> LinkResult:
        parent = {tid: tid for tid in tracklets}

        def root(t):
            while parent[t] != t:
                parent[t] = parent[parent[t]]
                t = parent[t]
            return t

        # chains keyed by root: (start, end, last tracklet, first tracklet, frames set)
        chains = {tid: {"first": tid, "last": tid, "frames": set(map(int, t.frames))}
                  for tid, t in tracklets.items() if len(t.frames) >= self.min_len}
        decisions: List[LinkDecision] = []
        for _ in range(8):
            roots = list(chains)
            if len(roots) < 2:
                break
            tails = [tracklets[chains[r]["last"]] for r in roots]
            heads = [tracklets[chains[r]["first"]] for r in roots]
            n = len(roots)
            cost = np.full((n, n), np.inf)
            parts = {}
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    c, m, ap = self._pair_cost(tails[i], heads[j])
                    if np.isfinite(c) and not (chains[roots[i]]["frames"] & chains[roots[j]]["frames"]):
                        cost[i, j] = c
                        parts[(i, j)] = (m, ap)
            if not np.isfinite(cost).any():
                break
            big = np.where(np.isfinite(cost), cost, 1e6)
            rows, cols = linear_sum_assignment(big)
            changed = False
            for i, j in zip(rows, cols):
                if big[i, j] >= 1e6:
                    continue
                ri, rj = roots[i], roots[j]
                if ri not in chains or rj not in chains or ri == rj:
                    continue
                if chains[ri]["frames"] & chains[rj]["frames"]:
                    continue
                row = np.sort(cost[i][np.isfinite(cost[i])])
                col = np.sort(cost[:, j][np.isfinite(cost[:, j])])
                second = min(row[1] if len(row) > 1 else np.inf, col[1] if len(col) > 1 else np.inf)
                m, ap = parts[(i, j)]
                decisions.append(LinkDecision(
                    chains[ri]["last"], chains[rj]["first"],
                    tracklets[chains[rj]["first"]].start - tracklets[chains[ri]["last"]].end,
                    float(cost[i, j]), float(m), ap,
                    uncertain=bool(second - cost[i, j] < self.uncertain_margin)))
                parent[rj] = ri
                chains[ri]["last"] = chains[rj]["last"]
                chains[ri]["frames"] |= chains[rj]["frames"]
                del chains[rj]
                changed = True
            if not changed:
                break

        identity = {tid: root(tid) for tid in tracklets}
        # compact final ids in order of first appearance
        order = sorted({identity[t] for t in tracklets},
                       key=lambda r: min(tracklets[t].start for t in tracklets if identity[t] == r))
        final = {r: k + 1 for k, r in enumerate(order)}
        identity = {t: final[r] for t, r in identity.items()}
        return LinkResult(self._materialise(tracklets, identity), identity, decisions)

    def _materialise(self, tracklets, identity):
        per_id: Dict[int, list] = {}
        for tid, t in tracklets.items():
            per_id.setdefault(identity[tid], []).append(t)
        frames: Dict[int, list] = {}
        for fid, parts in per_id.items():
            fr = np.concatenate([p.frames for p in parts])
            bx = np.concatenate([p.boxes for p in parts])
            order = np.argsort(fr, kind="stable")
            fr, bx = fr[order], bx[order]
            _, uniq = np.unique(fr, return_index=True)
            fr, bx = fr[uniq], bx[uniq]
            for k in range(len(fr)):
                frames.setdefault(int(fr[k]), []).append((fid, bx[k], False))
                if k + 1 < len(fr):
                    gap = int(fr[k + 1] - fr[k])
                    if 1 < gap <= self.interpolate_gap:
                        for s in range(1, gap):
                            a = s / gap
                            frames.setdefault(int(fr[k] + s), []).append(
                                (fid, (1 - a) * bx[k] + a * bx[k + 1], True))
        out = {}
        for f, items in frames.items():
            out[f] = (np.asarray([i[0] for i in items], int),
                      np.asarray([i[1] for i in items], float).reshape(-1, 4),
                      np.asarray([i[2] for i in items], bool))
        return out
