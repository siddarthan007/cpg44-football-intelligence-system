"""Tactical analysis - formation, team shape, and rule-based recommendations.

None of the reference repos do this; it is the project's tactical-decision module
(CPG44 Objective 3). Everything here is pure NumPy over **pitch-metre** positions,
so it is unit-testable without the CV stack.

Conventions: positions are (x, y) in metres, x along the 105 m length, y along
the 68 m width. ``attack_dir`` is +1 if the team attacks toward larger x, else -1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# geometry (NumPy only)
# --------------------------------------------------------------------------- #
def convex_hull_area(points: np.ndarray) -> float:
    """Area (m²) of the convex hull of 2-D points via Andrew's monotone chain."""
    pts = np.asarray([p for p in points if not np.isnan(p).any()], dtype=float)
    if len(pts) < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    x, y = hull[:, 0], hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def team_shape(points: np.ndarray) -> dict:
    """Instantaneous team-shape descriptors from one frame's player positions."""
    pts = np.asarray([p for p in points if not np.isnan(p).any()], dtype=float)
    if len(pts) == 0:
        return {"centroid": (np.nan, np.nan), "width": 0.0, "length": 0.0,
                "hull_area": 0.0, "compactness": 0.0}
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    width = float(pts[:, 1].max() - pts[:, 1].min())      # spread across the pitch
    length = float(pts[:, 0].max() - pts[:, 0].min())     # depth goal-to-goal
    area = convex_hull_area(pts)
    # compactness = mean distance of players to centroid (smaller = tighter block)
    compact = float(np.mean(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)))
    return {"centroid": (float(cx), float(cy)), "width": width, "length": length,
            "hull_area": area, "compactness": compact}


def estimate_formation(player_mean_x: List[float], player_mean_y: List[float],
                       attack_dir: int = 1, n_outfield: int = 10) -> str:
    """Rough formation string (e.g. "4-3-3") from players' average positions.

    Excludes the deepest player (assumed goalkeeper), orients by ``attack_dir``,
    then splits the remaining outfield players into 3 lines (defence/mid/attack)
    by their depth coordinate.
    """
    xs = np.asarray(player_mean_x, dtype=float)
    ys = np.asarray(player_mean_y, dtype=float)
    ok = ~np.isnan(xs)
    xs, ys = xs[ok], ys[ok]
    if len(xs) < 4:
        return "n/a"
    depth = xs * attack_dir            # larger = further forward
    order = np.argsort(depth)          # deepest first
    # drop the single deepest as GK, keep up to n_outfield
    outfield = order[1:1 + n_outfield]
    d = depth[outfield]
    if len(d) < 3:
        return "n/a"
    # split into 3 lines by depth using simple 1-D k-means (3 centroids)
    lines = _kmeans_1d(d, k=3)
    counts = [int((lines == i).sum()) for i in range(3)]
    # order lines back-to-front by centroid depth
    centroids = [d[lines == i].mean() if counts[i] else -1e9 for i in range(3)]
    front_to_back = np.argsort(centroids)   # ascending depth = def→att
    ordered = [counts[i] for i in front_to_back]
    ordered = [c for c in ordered if c > 0]
    return "-".join(str(c) for c in ordered)


def _kmeans_1d(x: np.ndarray, k: int = 3, iters: int = 25) -> np.ndarray:
    """Tiny deterministic 1-D k-means (quantile-seeded) - no sklearn needed."""
    x = np.asarray(x, dtype=float)
    if len(x) <= k:
        return np.arange(len(x)) % k
    centers = np.quantile(x, np.linspace(0.15, 0.85, k))
    labels = np.zeros(len(x), int)
    for _ in range(iters):
        labels = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
        new = np.array([x[labels == i].mean() if (labels == i).any() else centers[i]
                        for i in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    return labels


# --------------------------------------------------------------------------- #
# match-level tactical report + recommendations
# --------------------------------------------------------------------------- #
@dataclass
class TacticalReport:
    team: int
    formation: str
    avg_width: float
    avg_length: float
    avg_compactness: float
    avg_line_height: float
    possession_pct: float
    recommendations: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "team": self.team, "formation": self.formation,
            "avg_width_m": round(self.avg_width, 1),
            "avg_length_m": round(self.avg_length, 1),
            "avg_compactness_m": round(self.avg_compactness, 1),
            "avg_line_height_m": round(self.avg_line_height, 1),
            "possession_pct": round(self.possession_pct, 1),
            "recommendations": self.recommendations,
        }


class TacticalAnalyzer:
    """Aggregates per-frame team positions into a match tactical report."""

    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        self.L, self.W = pitch_length, pitch_width
        # team -> list of frames, each a list of (x,y)
        self._frames: Dict[int, List[List[Tuple[float, float]]]] = {1: [], 2: []}
        # team -> {track_id: [x...]} and [y...] for mean-position formation
        self._px: Dict[int, Dict[int, List[float]]] = {1: {}, 2: {}}
        self._py: Dict[int, Dict[int, List[float]]] = {1: {}, 2: {}}

    def update(self, teams_positions: Dict[int, Dict[int, Tuple[float, float]]]):
        """``teams_positions`` = {team: {track_id: (x,y)}} for one frame."""
        for team in (1, 2):
            tp = teams_positions.get(team, {})
            self._frames[team].append(list(tp.values()))
            for tid, (x, y) in tp.items():
                if x is None or np.isnan(x):
                    continue
                self._px[team].setdefault(tid, []).append(x)
                self._py[team].setdefault(tid, []).append(y)

    def _attack_dir(self, team: int) -> int:
        """+1 if the team's mean x is in the left half (attacks toward +x)."""
        allx = [np.mean(v) for v in self._px[team].values() if v]
        if not allx:
            return 1
        return 1 if np.mean(allx) < self.L / 2 else -1

    def report(self, team: int, possession_pct: float,
               opponent: Optional["TacticalReport"] = None) -> TacticalReport:
        frames = self._frames[team]
        shapes = [team_shape(np.asarray(f)) for f in frames if f]
        width = float(np.nanmean([s["width"] for s in shapes])) if shapes else 0.0
        length = float(np.nanmean([s["length"] for s in shapes])) if shapes else 0.0
        compact = float(np.nanmean([s["compactness"] for s in shapes])) if shapes else 0.0

        adir = self._attack_dir(team)
        mean_x = {tid: float(np.mean(v)) for tid, v in self._px[team].items() if v}
        mean_y = {tid: float(np.mean(self._py[team][tid])) for tid in mean_x}
        formation = estimate_formation(list(mean_x.values()), list(mean_y.values()), adir)

        # defensive line height = mean depth of the 4 deepest outfield players,
        # expressed as % of pitch length from own goal (0 = own goal, 100 = opp goal)
        line_height = self._line_height(mean_x, adir)

        rep = TacticalReport(team, formation, width, length, compact, line_height,
                             possession_pct)
        rep.recommendations = self._recommend(rep)
        return rep

    def _line_height(self, mean_x: Dict[int, float], adir: int) -> float:
        if not mean_x:
            return 0.0
        depth = np.sort(np.array(list(mean_x.values())) * adir)  # ascending
        deepest = depth[1:5] if len(depth) > 5 else depth[:max(1, len(depth) // 2)]
        # metres from own goal: adir=+1 → 0 + x = x; adir=-1 → L + (-x) = L - x.
        # (deepest already equals x*adir, so add it directly — do NOT re-apply adir.)
        base = 0 if adir > 0 else self.L
        return float(np.mean(np.abs(base + deepest)))

    def _recommend(self, r: TacticalReport) -> List[str]:
        recs: List[str] = []
        if r.avg_width < 0.35 * self.W:
            recs.append("Team is narrow (avg width {:.0f} m) - use wide players / "
                        "switch play to stretch the opponent.".format(r.avg_width))
        if r.avg_width > 0.75 * self.W:
            recs.append("Team is over-stretched horizontally - risk of central gaps; "
                        "tuck full-backs in when out of possession.")
        if r.avg_compactness > 0.30 * self.L:
            recs.append("Block is loose (compactness {:.0f} m) - compress lines "
                        "vertically to cut passing lanes.".format(r.avg_compactness))
        if r.avg_line_height > 0.60 * self.L:
            recs.append("High defensive line ({:.0f} m up the pitch) - vulnerable to "
                        "balls in behind; add cover or drop the line.".format(r.avg_line_height))
        if r.avg_line_height < 0.25 * self.L:
            recs.append("Very deep line - inviting pressure; step up to compress space "
                        "and support attacks.")
        if r.possession_pct < 40:
            recs.append("Low possession ({:.0f}%) - improve build-up / press higher to "
                        "win the ball back sooner.".format(r.possession_pct))
        if not recs:
            recs.append("Shape within normal ranges - maintain current structure.")
        return recs
