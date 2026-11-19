"""Which tracked player is wearing the vest, and how far apart are the clocks.

The camera sees every player but not who wears a vest; the vest knows its own
motion but not where it is in the image. The two describe the same body, so
the vest's movement intensity rises and falls with the speed of exactly one
track. Matching those two signals by normalised cross-correlation identifies
the wearer and, from the lag of the best match, the residual offset between
the vest clock and the video clock (the same idea as PEM-ID, Teixeira et al.
2010, and the wearable/camera association of Cabrera-Quiros et al. 2018).

Signals
-------
* vest: body acceleration with gravity removed (0.5 Hz high-pass per axis),
  magnitude, RMS over 1 s. Foot-strike impacts grow with running speed, so
  this envelope follows how fast the player is moving.
* track: speed from the video, smoothed over 1 s. A short burst of
  acceleration is visible in both.
* optional GNSS: when the vest has a fix, the distance between its position
  on the pitch and each track is a far stronger cue and is combined in.

Both are resampled to 5 Hz. The correlation is computed over the overlap at
lags of -max_lag..+max_lag; tracks seen for less than ``min_overlap_s`` are
not candidates, because short fragments correlate by chance. Several vests
are assigned jointly (Hungarian) so two vests never claim one player.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, sosfiltfilt

FS = 5.0


def imu_envelope(t: np.ndarray, accel: np.ndarray, fs_in: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
    """(t at 5 Hz, RMS of gravity-free acceleration over 1 s)."""
    t = np.asarray(t, float)
    a = np.asarray(accel, float)
    if len(t) < fs_in * 4:
        return np.zeros(0), np.zeros(0)
    grid = np.arange(t[0], t[-1], 1.0 / fs_in)
    a = np.column_stack([np.interp(grid, t, a[:, k]) for k in range(3)])
    sos = butter(2, 0.5, "highpass", fs=fs_in, output="sos")
    dyn = np.linalg.norm(sosfiltfilt(sos, a, axis=0), axis=1)
    w = int(fs_in)
    rms = np.sqrt(np.convolve(dyn ** 2, np.ones(w) / w, mode="same"))
    out_t = np.arange(grid[0], grid[-1], 1.0 / FS)
    return out_t, np.interp(out_t, grid, rms)


def track_envelope(t: np.ndarray, speed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, float)
    sp = np.asarray(speed, float)
    ok = np.isfinite(sp)
    if ok.sum() < FS * 4:
        return np.zeros(0), np.zeros(0)
    out_t = np.arange(t[ok][0], t[ok][-1], 1.0 / FS)
    v = np.interp(out_t, t[ok], sp[ok])
    # samples more than 0.6 s from any observation are missing (track left the view)
    obs = t[ok]
    k = np.clip(np.searchsorted(obs, out_t), 1, len(obs) - 1)
    gap = np.minimum(np.abs(out_t - obs[k - 1]), np.abs(obs[k] - out_t))
    v[gap > 0.6] = np.nan
    k = int(FS)
    vv = np.where(np.isfinite(v), v, 0.0)
    cnt = np.convolve(np.isfinite(v).astype(float), np.ones(k), mode="same")
    sm = np.convolve(vv, np.ones(k), mode="same") / np.maximum(cnt, 1)
    sm[~np.isfinite(v)] = np.nan
    return out_t, sm


def _ncc(a: np.ndarray, b: np.ndarray) -> Tuple[float, int]:
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < 10:
        return float("nan"), n
    x, y = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    d = float(np.sqrt((x @ x) * (y @ y)))
    return (float(x @ y / d) if d > 0 else float("nan")), n


@dataclass
class Match:
    vest: int
    track: Optional[int]
    score: float
    lag_s: float
    margin: float
    overlap_s: float
    confident: bool
    candidates: List[Tuple[int, float]]


def associate(vests: Dict[int, Tuple[np.ndarray, np.ndarray]],
              tracks: Dict[int, Tuple[np.ndarray, np.ndarray]],
              max_lag_s: float = 2.0, min_overlap_s: float = 30.0,
              gps: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
              track_xy: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
              min_score: float = 0.35, min_margin: float = 0.08) -> Dict[int, Match]:
    """``vests[v] = (t, envelope)`` and ``tracks[k] = (t, envelope)`` on one
    clock (seconds). Optional ``gps[v] = (t, xy)`` and ``track_xy[k] = (t, xy)``
    in pitch metres add a position term."""
    lags = np.arange(-max_lag_s, max_lag_s + 1e-9, 1.0 / FS)
    vids, tids = list(vests), [k for k, (t, e) in tracks.items() if np.isfinite(e).sum() / FS >= min_overlap_s]
    if not vids or not tids:
        return {v: Match(v, None, float("nan"), 0.0, 0.0, 0.0, False, []) for v in vids}
    S = np.full((len(vids), len(tids)), -1.0)
    L = np.zeros_like(S)
    O = np.zeros_like(S)
    for i, v in enumerate(vids):
        tv, ev = vests[v]
        for j, k in enumerate(tids):
            tk, ek = tracks[k]
            best = (-1.0, 0.0, 0)
            for lag in lags:
                # vest sample at time t + lag corresponds to video time t
                e_v = np.interp(tk, tv - lag, ev, left=np.nan, right=np.nan)
                r, n = _ncc(e_v, ek)
                if np.isfinite(r) and n / FS >= min_overlap_s and r > best[0]:
                    best = (r, lag, n)
            score = best[0]
            if gps and track_xy and v in gps and k in track_xy and best[2]:
                gt, gxy = gps[v]
                kt, kxy = track_xy[k]
                gx = np.column_stack([np.interp(kt, gt - best[1], gxy[:, c], left=np.nan, right=np.nan) for c in (0, 1)])
                d = np.linalg.norm(gx - kxy, axis=1)
                d = d[np.isfinite(d)]
                if len(d) >= FS * 10:
                    score = 0.5 * score + 0.5 * float(np.exp(-np.median(d) / 6.0))   # 6 m: GNSS + pin error
            S[i, j], L[i, j], O[i, j] = score, best[1], best[2] / FS
    rows, cols = linear_sum_assignment(-S)
    out: Dict[int, Match] = {}
    for i, v in enumerate(vids):
        order = np.argsort(-S[i])
        cands = [(int(tids[j]), round(float(S[i, j]), 3)) for j in order[:4]]
        if i in rows:
            j = int(cols[list(rows).index(i)])
            others = [S[i, jj] for jj in range(len(tids)) if jj != j]
            margin = float(S[i, j] - max(others)) if others else 1.0
            conf = bool(S[i, j] >= min_score and margin >= min_margin)
            out[v] = Match(v, int(tids[j]), round(float(S[i, j]), 3), round(float(L[i, j]), 2), round(margin, 3),
                           round(float(O[i, j]), 1), conf, cands)
        else:
            out[v] = Match(v, None, float("nan"), 0.0, 0.0, 0.0, False, cands)
    return out


class LiveAssociator:
    """Rolling vest-to-track matching for the live pipeline.

    Keeps the last ``window_s`` of vest acceleration and track speed on the
    host clock, re-runs :func:`associate` every ``every_s`` and reports a binding
    only after ``confirm`` consecutive confident runs agree, so a single lucky
    correlation never moves a vest to another player.
    """

    def __init__(self, window_s: float = 90.0, every_s: float = 10.0, confirm: int = 2,
                 min_overlap_s: float = 30.0):
        from collections import defaultdict, deque
        self.window_s, self.every_s, self.confirm, self.min_overlap_s = window_s, every_s, confirm, min_overlap_s
        self.vest = defaultdict(lambda: deque(maxlen=int(window_s * 60)))     # pid -> (t, ax, ay, az) m/s^2
        self.track = defaultdict(lambda: deque(maxlen=int(window_s * 30)))    # tid -> (t, speed)
        self._last_run = 0.0
        self._streak: Dict[int, Tuple[Optional[int], int]] = {}
        self.last: Dict[int, Match] = {}

    def add_vest(self, player_id: int, t: float, accel_g) -> None:
        self.vest[int(player_id)].append((t, *(9.80665 * np.asarray(accel_g, float))))

    def add_track(self, track_id: int, t: float, speed: float) -> None:
        if np.isfinite(speed):
            self.track[int(track_id)].append((t, float(speed)))

    def step(self, now: float) -> Dict[int, int]:
        """Returns {player_id: track_id} for bindings confirmed on this run."""
        if now - self._last_run < self.every_s:
            return {}
        self._last_run = now
        lo = now - self.window_s
        vests = {}
        for pid, rows in self.vest.items():
            a = np.asarray([r for r in rows if lo <= r[0] <= now])
            if len(a) >= 50 * 20:
                vests[pid] = imu_envelope(a[:, 0], a[:, 1:4])
        tracks = {}
        for tid, rows in self.track.items():
            a = np.asarray([r for r in rows if lo <= r[0] <= now])
            if len(a) >= FS * 10:
                tracks[tid] = track_envelope(a[:, 0], a[:, 1])
        if not vests or not tracks:
            return {}
        self.last = associate(vests, tracks, min_overlap_s=self.min_overlap_s)
        out = {}
        for pid, m in self.last.items():
            prev, n = self._streak.get(pid, (None, 0))
            if m.confident and m.track == prev:
                n += 1
            elif m.confident:
                prev, n = m.track, 1
            else:
                prev, n = None, 0
            self._streak[pid] = (prev, n)
            if n == self.confirm:
                out[pid] = int(m.track)
        return out
