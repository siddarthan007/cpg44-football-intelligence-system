"""Multi-object tracking metrics: HOTA, IDF1 and CLEAR-MOT (MOTA, IDSW).

A dependency-free NumPy/SciPy implementation that follows the published
definitions so tracker changes can be measured on SoccerNet-Tracking style
ground truth without installing TrackEval:

* HOTA, DetA, AssA, LocA: Luiten et al., "HOTA: A Higher Order Metric for
  Evaluating Multi-Object Tracking", IJCV 2021. The matching procedure mirrors
  TrackEval's reference implementation (global alignment score weighting of the
  per-frame Hungarian assignment, alpha thresholds 0.05..0.95).
* IDF1: Ristani et al., ECCV-W 2016 (global ID bipartite matching at IoU 0.5).
* MOTA / IDSW: Bernardin & Stiefelhagen 2008, with TrackEval's "keep previous
  match when still valid" rule.

Inputs are per-frame dictionaries ``{frame: (ids[N], boxes_xywh[N,4])}``. For
the official challenge numbers use the SoccerNet toolkit; this module exists so
every local experiment is reproducible and scored the same way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

Frames = Mapping[int, Tuple[np.ndarray, np.ndarray]]
ALPHAS = np.arange(0.05, 0.99, 0.05)
EPS = np.finfo(float).eps


def box_iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / np.maximum(union, EPS), 0.0)


def read_mot(path: str | Path, keep_ids: Iterable[int] | None = None) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Read a MOTChallenge text file into ``{frame: (ids, xywh)}``."""
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    keep = None if keep_ids is None else set(int(v) for v in keep_ids)
    out: Dict[int, list] = {}
    for row in rows:
        if keep is not None and int(row[1]) not in keep:
            continue
        out.setdefault(int(row[0]), []).append(row[1:6])
    return {f: (np.asarray(v)[:, 0].astype(int), np.asarray(v)[:, 1:5]) for f, v in out.items()}


def write_mot(path: str | Path, frames: Frames) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for frame in sorted(frames):
            ids, boxes = frames[frame]
            for tid, (x, y, w, h) in zip(ids, boxes):
                fh.write(f"{frame},{int(tid)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")


def _remap(frames: Frames):
    ids = sorted({int(i) for ids, _ in frames.values() for i in ids})
    return {v: k for k, v in enumerate(ids)}


def evaluate(gt: Frames, pred: Frames) -> dict:
    """Return HOTA/DetA/AssA/LocA, IDF1/IDP/IDR, MOTA/IDSW/FP/FN for one sequence."""
    frames = sorted(set(gt) | set(pred))
    gmap, pmap = _remap(gt), _remap(pred)
    ng, npr = len(gmap), len(pmap)
    empty = (np.zeros(0, int), np.zeros((0, 4)))

    seq = []
    for f in frames:
        gi, gb = gt.get(f, empty)
        pi, pb = pred.get(f, empty)
        g = np.asarray([gmap[int(i)] for i in gi], int)
        p = np.asarray([pmap[int(i)] for i in pi], int)
        seq.append((g, p, box_iou_xywh(gb, pb)))

    n_gt_dets = sum(len(g) for g, _, _ in seq)
    n_pr_dets = sum(len(p) for _, p, _ in seq)
    result = {"frames": len(frames), "gt_dets": n_gt_dets, "pred_dets": n_pr_dets,
              "gt_ids": ng, "pred_ids": npr}
    if n_gt_dets == 0 or n_pr_dets == 0:
        result.update(HOTA=0.0, DetA=0.0, AssA=0.0, LocA=0.0, IDF1=0.0, IDP=0.0, IDR=0.0,
                      MOTA=0.0 if n_gt_dets else 1.0, IDSW=0, FP=n_pr_dets, FN=n_gt_dets)
        return result

    # ---------------- HOTA (TrackEval procedure) ---------------- #
    potential = np.zeros((ng, npr))
    gt_count = np.zeros((ng, 1))
    pr_count = np.zeros((1, npr))
    for g, p, sim in seq:
        if len(g) and len(p):
            denom = sim.sum(0)[None] + sim.sum(1)[:, None] - sim
            sim_iou = np.zeros_like(sim)
            mask = denom > EPS
            sim_iou[mask] = sim[mask] / denom[mask]
            potential[g[:, None], p[None]] += sim_iou
        gt_count[g] += 1
        pr_count[0, p] += 1
    global_align = potential / np.maximum(gt_count + pr_count - potential, EPS)

    na = len(ALPHAS)
    tp, fn, fp, loc = np.zeros(na), np.zeros(na), np.zeros(na), np.zeros(na)
    matches = np.zeros((na, ng, npr))
    for g, p, sim in seq:
        if len(g) == 0:
            fp += len(p)
            continue
        if len(p) == 0:
            fn += len(g)
            continue
        score = global_align[g[:, None], p[None]] * sim
        rows, cols = linear_sum_assignment(-score)
        for a, alpha in enumerate(ALPHAS):
            ok = sim[rows, cols] >= alpha - EPS
            r, c = rows[ok], cols[ok]
            n = len(r)
            tp[a] += n
            fn[a] += len(g) - n
            fp[a] += len(p) - n
            if n:
                loc[a] += sim[r, c].sum()
                matches[a, g[r], p[c]] += 1
    ass = np.zeros(na)
    for a in range(na):
        m = matches[a]
        ass_a = m / np.maximum(gt_count + pr_count - m, 1)
        ass[a] = (m * ass_a).sum() / max(1.0, tp[a])
    det = tp / np.maximum(1.0, tp + fn + fp)
    loca = np.maximum(EPS, loc) / np.maximum(EPS, tp)
    hota = np.sqrt(det * ass)

    # ---------------- IDF1 ---------------- #
    id_overlap = np.zeros((ng, npr))
    for g, p, sim in seq:
        if len(g) and len(p):
            gg, pp = np.nonzero(sim >= 0.5 - EPS)
            np.add.at(id_overlap, (g[gg], p[pp]), 1)
    fn_cost = gt_count[:, 0][:, None] - id_overlap      # misses if paired
    big = np.zeros((ng + npr, ng + npr))
    big[:ng, :npr] = fn_cost + (pr_count[0][None] - id_overlap)
    big[:ng, npr:] = 1e10
    big[ng:, :npr] = 1e10
    np.fill_diagonal(big[:ng, npr:], gt_count[:, 0])
    np.fill_diagonal(big[ng:, :npr], pr_count[0])
    rows, cols = linear_sum_assignment(big)
    idtp = sum(id_overlap[r, c] for r, c in zip(rows, cols) if r < ng and c < npr)
    idfn = n_gt_dets - idtp
    idfp = n_pr_dets - idtp

    # ---------------- CLEAR MOT ---------------- #
    prev = np.full(ng, -1)
    prev_any = np.full(ng, -1)
    c_tp = c_fp = c_fn = idsw = 0
    for g, p, sim in seq:
        if len(g) == 0:
            c_fp += len(p)
            continue
        if len(p) == 0:
            c_fn += len(g)
            continue
        score = sim.copy()
        keep_prev = (prev[g][:, None] == p[None]) & (sim >= 0.5 - EPS)
        score = np.where(sim >= 0.5 - EPS, score, 0) + 1000 * keep_prev
        rows, cols = linear_sum_assignment(-score)
        ok = sim[rows, cols] >= 0.5 - EPS
        r, c = rows[ok], cols[ok]
        mg, mp = g[r], p[c]
        switched = (prev_any[mg] != mp) & (prev_any[mg] != -1)
        idsw += int(switched.sum())
        prev_any[mg] = mp
        prev[:] = -1
        prev[mg] = mp
        c_tp += len(r)
        c_fn += len(g) - len(r)
        c_fp += len(p) - len(r)

    result.update(
        HOTA=float(hota.mean()), DetA=float(det.mean()), AssA=float(ass.mean()),
        LocA=float(loca.mean()), HOTA_per_alpha=[round(float(v), 4) for v in hota],
        IDF1=float(idtp / max(1.0, idtp + 0.5 * idfp + 0.5 * idfn)),
        IDP=float(idtp / max(1.0, idtp + idfp)), IDR=float(idtp / max(1.0, idtp + idfn)),
        MOTA=float(1.0 - (c_fn + c_fp + idsw) / max(1, n_gt_dets)),
        IDSW=int(idsw), FP=int(c_fp), FN=int(c_fn),
    )
    return result


def combine(results: Iterable[dict]) -> dict:
    """Detection-weighted average over sequences (HOTA family averaged per sequence
    as SoccerNet reports per-sequence means; counts summed)."""
    rs = [r for r in results if r]
    if not rs:
        return {}
    out = {k: float(np.mean([r[k] for r in rs])) for k in ("HOTA", "DetA", "AssA", "LocA", "IDF1", "MOTA")}
    for k in ("IDSW", "FP", "FN", "gt_dets", "pred_dets", "gt_ids", "pred_ids", "frames"):
        out[k] = int(sum(r.get(k, 0) for r in rs))
    out["sequences"] = len(rs)
    return out
