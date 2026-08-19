"""Rich tactical analysis engine (CPG44 Objective 3).

Goes beyond formation strings: per-frame spatial algorithms plus time-aggregated
team metrics that drive the recommendation engine.

Algorithms
----------
* **Pitch control (Voronoi):** every pitch cell is assigned to the nearest player;
  the share of the pitch each team is nearest to = space control %. A coarse grid
  keeps it real-time. Also returns the control grid for the dashboard overlay.
* **Team shape:** centroid, width, depth, compactness, defensive-line height,
  block length — instantaneous and time-averaged.
* **Pressing intensity:** opponents within a radius of the ball / ball carrier.
* **Numerical superiority by thirds:** player counts per third per team.
* **Phase detection:** in-possession / out-of-possession / transition.

Pure NumPy over pitch-metre coordinates — unit-testable.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import PitchConfig
from .tactics import estimate_formation, team_shape


# --------------------------------------------------------------------------- #
# spatial algorithms
# --------------------------------------------------------------------------- #
def pitch_control(team_positions: Dict[int, List[Tuple[float, float]]],
                  pc: PitchConfig, cell: float = 3.0):
    """Voronoi space control. Returns ({team: pct}, grid) where grid[gy,gx] ∈
    {0,1,2} is the controlling team (0 = none). ``cell`` = grid resolution (m)."""
    gx = max(1, int(pc.length / cell))
    gy = max(1, int(pc.width / cell))
    xs = (np.arange(gx) + 0.5) * pc.length / gx
    ys = (np.arange(gy) + 0.5) * pc.width / gy
    cells = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)   # (gx*gy, 2)

    pts, labels = [], []
    for team, ps in team_positions.items():
        for (x, y) in ps:
            if x is not None and not np.isnan(x):
                pts.append((x, y)); labels.append(team)
    if not pts:
        return {1: 0.0, 2: 0.0}, np.zeros((gy, gx), int)
    pts = np.asarray(pts); labels = np.asarray(labels)
    d2 = ((cells[:, None, :] - pts[None, :, :]) ** 2).sum(-1)       # (cells, players)
    nearest = labels[np.argmin(d2, axis=1)]
    grid = nearest.reshape(gy, gx)
    total = grid.size
    return ({1: round(100.0 * (grid == 1).sum() / total, 1),
             2: round(100.0 * (grid == 2).sum() / total, 1)}, grid)


def pressing_intensity(defenders: List[Tuple[float, float]],
                       ball: Optional[Tuple[float, float]], radius: float = 8.0) -> int:
    """Number of defenders within ``radius`` of the ball (pressing count)."""
    if ball is None or ball[0] is None or np.isnan(ball[0]):
        return 0
    n = 0
    for (x, y) in defenders:
        if x is not None and not np.isnan(x) and np.hypot(x - ball[0], y - ball[1]) <= radius:
            n += 1
    return n


def thirds_count(positions: List[Tuple[float, float]], pc: PitchConfig,
                 attack_dir: int) -> Dict[str, int]:
    """Players in defensive / middle / attacking third (relative to attack_dir)."""
    third = pc.length / 3.0
    out = {"def": 0, "mid": 0, "att": 0}
    for (x, y) in positions:
        if x is None or np.isnan(x):
            continue
        xx = x if attack_dir > 0 else pc.length - x     # orient so attack = +x
        out["def" if xx < third else "mid" if xx < 2 * third else "att"] += 1
    return out


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #
@dataclass
class TacticalSnapshot:
    """Current-frame tactical state, for the live dashboard."""
    control: Dict[int, float] = field(default_factory=lambda: {1: 0.0, 2: 0.0})
    control_grid: Optional[np.ndarray] = None
    pressing: Dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0})
    centroid: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    width: Dict[int, float] = field(default_factory=dict)
    depth: Dict[int, float] = field(default_factory=dict)
    line_height: Dict[int, float] = field(default_factory=dict)
    thirds: Dict[int, Dict[str, int]] = field(default_factory=dict)
    phase: Dict[int, str] = field(default_factory=dict)


@dataclass
class TeamTacticalReport:
    team: int
    formation: str
    avg_control: float
    avg_width: float
    avg_depth: float
    avg_compactness: float
    avg_line_height: float
    avg_pressing: float
    avg_surface_area: float          # convex-hull area, m² (collective dynamics)
    possession_pct: float
    time_in_att_third_pct: float
    recommendations: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {k: (round(v, 1) if isinstance(v, float) else v)
             for k, v in self.__dict__.items() if k != "recommendations"}
        d["recommendations"] = self.recommendations
        return d


class TacticalEngine:
    def __init__(self, pc: PitchConfig, avg_window: int = 250):
        self.pc = pc
        self._acc: Dict[int, Dict[str, list]] = {1: defaultdict(list), 2: defaultdict(list)}
        self._px: Dict[int, Dict[int, List[float]]] = {1: {}, 2: {}}
        self._py: Dict[int, Dict[int, List[float]]] = {1: {}, 2: {}}
        self._phase_hist = deque(maxlen=15)
        self._attack_dir_latched: Dict[int, int] = {}
        self._nframes = 0
        self.snapshot = TacticalSnapshot()
        self._last_control = {1: 0.0, 2: 0.0}   # carried across non-compute frames
        self._last_grid = None

    def attack_dir(self, team: int) -> int:
        """Attacking direction (+1 → goal at x=L). LATCHED from the early game,
        when teams are in their own halves (kickoff/build-up), so it does not
        invert once a team pushes up and camps in the opponent's half. Assumes the
        clip starts before an end-swap; reset the engine at halftime for a new one."""
        if team in self._attack_dir_latched:
            return self._attack_dir_latched[team]
        allx = [np.mean(v) for v in self._px[team].values() if v]
        if not allx:
            return 1 if team == 1 else -1        # fallback until data arrives
        d = 1 if np.mean(allx) < self.pc.length / 2 else -1
        if self._nframes >= 50:                  # enough early frames → latch
            self._attack_dir_latched[team] = d
        return d

    def update(self, team_positions: Dict[int, Dict[int, Tuple[float, float]]],
               ball: Optional[Tuple[float, float]], possession_team: int,
               compute_control: bool = True) -> TacticalSnapshot:
        """One frame. ``team_positions`` = {team: {track_id: (x,y)}}."""
        self._nframes += 1
        snap = TacticalSnapshot()
        pos_lists = {t: list(team_positions.get(t, {}).values()) for t in (1, 2)}

        for t in (1, 2):
            for tid, (x, y) in team_positions.get(t, {}).items():
                if x is None or np.isnan(x):
                    continue
                self._px[t].setdefault(tid, []).append(x)
                self._py[t].setdefault(tid, []).append(y)
            sh = team_shape(np.asarray(pos_lists[t])) if pos_lists[t] else None
            if sh:
                snap.centroid[t] = sh["centroid"]
                snap.width[t] = sh["width"]; snap.depth[t] = sh["length"]
                self._acc[t]["width"].append(sh["width"])
                self._acc[t]["depth"].append(sh["length"])
                self._acc[t]["compact"].append(sh["compactness"])
                # collective-dynamics (Teixeira [7]): stretch index = mean spread
                # from centroid; surface area = convex-hull area of the team.
                self._acc[t]["surface"].append(sh["hull_area"])
                adir = self.attack_dir(t)
                lh = self._line_height(pos_lists[t], adir)
                snap.line_height[t] = lh
                self._acc[t]["line"].append(lh)
                snap.thirds[t] = thirds_count(pos_lists[t], self.pc, adir)
                self._acc[t]["att_third"].append(snap.thirds[t]["att"])

        # pressing: defenders of the NON-possessing team near the ball
        for t in (1, 2):
            opp = 2 if t == 1 else 1
            if possession_team == opp:      # t is defending
                snap.pressing[t] = pressing_intensity(pos_lists[t], ball)
                self._acc[t]["press"].append(snap.pressing[t])

        # pitch control (coarse Voronoi) — computed on a cadence to save cost, but
        # carried forward on the in-between frames so the live display doesn't strobe
        if compute_control:
            ctrl, grid = pitch_control(pos_lists, self.pc)
            self._last_control, self._last_grid = ctrl, grid
            self._acc[1]["control"].append(ctrl[1])
            self._acc[2]["control"].append(ctrl[2])
        snap.control = dict(self._last_control)
        snap.control_grid = self._last_grid

        # phase
        self._phase_hist.append(possession_team)
        for t in (1, 2):
            snap.phase[t] = self._phase(t)

        self.snapshot = snap
        return snap

    def _phase(self, team: int) -> str:
        recent = list(self._phase_hist)
        if not recent:
            return "—"
        cur = recent[-1]
        changed = len(set(recent[-5:])) > 1
        if cur == team:
            return "transition→attack" if changed else "in possession"
        if cur and cur != team:
            return "transition→defend" if changed else "out of possession"
        return "loose"

    def _line_height(self, positions, adir: int) -> float:
        xs = np.array([p[0] for p in positions if p[0] is not None and not np.isnan(p[0])])
        if len(xs) == 0:
            return 0.0
        depth = np.sort(xs * adir)
        deepest = depth[1:5] if len(depth) > 5 else depth[: max(1, len(depth) // 2)]
        # metres from own goal: deepest already = x*adir, so add directly (adir=+1
        # → x; adir=-1 → L - x). Re-applying adir here double-counts the sign.
        base = 0 if adir > 0 else self.pc.length
        return float(np.mean(np.abs(base + deepest)))

    # ------------------------------------------------------------------ #
    def report(self, team: int, possession_pct: float) -> TeamTacticalReport:
        a = self._acc[team]
        mean = lambda k: float(np.mean(a[k])) if a[k] else 0.0
        mean_x = {tid: float(np.mean(v)) for tid, v in self._px[team].items() if v}
        mean_y = {tid: float(np.mean(self._py[team][tid])) for tid in mean_x}
        formation = estimate_formation(list(mean_x.values()), list(mean_y.values()),
                                       self.attack_dir(team))
        n_frames = max(len(a["att_third"]), 1)
        att_pct = 100.0 * sum(1 for c in a["att_third"] if c >= 3) / n_frames
        rep = TeamTacticalReport(
            team=team, formation=formation,
            avg_control=mean("control"), avg_width=mean("width"),
            avg_depth=mean("depth"), avg_compactness=mean("compact"),
            avg_line_height=mean("line"), avg_pressing=mean("press"),
            avg_surface_area=mean("surface"),
            possession_pct=possession_pct, time_in_att_third_pct=att_pct,
        )
        rep.recommendations = self._recommend(rep)
        return rep

    def _recommend(self, r: TeamTacticalReport) -> List[str]:
        L, W = self.pc.length, self.pc.width
        recs: List[str] = []
        if r.avg_control and r.avg_control < 40:
            recs.append(f"Low space control ({r.avg_control:.0f}%) — push the block "
                        f"higher and press to reclaim territory.")
        if r.avg_width < 0.35 * W:
            recs.append(f"Narrow ({r.avg_width:.0f} m) — use the width to stretch the "
                        f"opponent and open central lanes.")
        if r.avg_width > 0.78 * W:
            recs.append("Over-stretched — tuck full-backs in when defending to protect "
                        "the centre.")
        if r.avg_compactness > 0.30 * L:
            recs.append(f"Loose block (compactness {r.avg_compactness:.0f} m) — compress "
                        f"lines vertically to deny space between them.")
        if r.avg_line_height > 0.62 * L:
            recs.append(f"High line ({r.avg_line_height:.0f} m) — vulnerable in behind; "
                        f"add cover or time the press better.")
        if r.avg_pressing and r.avg_pressing < 1.0 and r.possession_pct < 45:
            recs.append("Passive out of possession — commit more players to pressing "
                        "the ball to force turnovers higher up.")
        if r.time_in_att_third_pct < 20 and r.possession_pct > 50:
            recs.append("Possession without penetration — progress the ball into the "
                        "attacking third earlier (verticality).")
        if not recs:
            recs.append("Shape and control within normal ranges — hold structure.")
        return recs
