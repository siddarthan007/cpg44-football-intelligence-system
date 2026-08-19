"""Match metrics: distance run, speed, ball possession, passes.

All spatial maths uses **pitch coordinates in metres** when a homography is
available (``metric=True``); otherwise it falls back to pixel space and the
results are flagged as pixel-based (see ``MetricsEngine.metric``).

Robustness vs. the reference repos:
- ``fps`` is taken from the video, never hardcoded to 24/25;
- per-frame speeds are clamped to a plausible max (amateur-footage jitter);
- possession uses a hysteresis counter so a one-frame nearest-player flip does
  not steal possession.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Pass:
    frame: int
    team: int
    from_id: int
    to_id: int
    start: Tuple[float, float]
    end: Tuple[float, float]


class MetricsEngine:
    def __init__(self, fps: float, metric: bool = True,
                 possession_dist: float = 2.0,       # metres (≈70 px if pixel mode)
                 possession_hysteresis: int = 12,     # frames before possession flips
                 max_speed_mps: float = 12.0):
        self.fps = float(fps)
        self.metric = metric
        self.possession_dist = possession_dist
        self.hysteresis = possession_hysteresis
        self.max_speed_mps = max_speed_mps

        # track_id -> list of (frame_idx, x, y)  (x,y may be nan)
        self.positions: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
        self.team_of: Dict[int, int] = {}

        # possession state
        self.team_frames = {1: 0, 2: 0}
        self._cur_team = 0
        self._cand_team = 0
        self._cand_count = 0
        self._holder_id = None

        # passes
        self.passes: List[Pass] = []
        self._confirmed_holder = None
        self._confirmed_team = 0
        self._pass_cand_holder = None   # debounce: candidate new ball holder
        self._pass_cand_count = 0       # (distinct from possession's _cand_count)
        self.pass_confirm_frames = 3    # frames a new player must hold before it counts

    # ------------------------------------------------------------------ #
    def update(self, frame_idx: int,
               players: Dict[int, Tuple[float, float]],
               teams: Dict[int, int],
               ball: Optional[Tuple[float, float]]):
        """One frame. ``players`` = {track_id: (x,y)}, ``teams`` = {track_id: team},
        ``ball`` = (x,y) or None. Coordinates in metres (or px if not metric)."""
        for tid, (x, y) in players.items():
            self.positions[tid].append((frame_idx, x, y))
            if teams.get(tid):
                self.team_of[tid] = teams[tid]

        holder = self._nearest_player(players, ball)
        self._update_possession(holder, teams)
        self._update_passes(frame_idx, holder, teams, players, ball)

    def _nearest_player(self, players, ball) -> Optional[int]:
        if ball is None or not players:
            return None
        bx, by = ball
        best, bestd = None, self.possession_dist
        for tid, (x, y) in players.items():
            if x is None or np.isnan(x):
                continue
            d = np.hypot(x - bx, y - by)
            if d < bestd:
                best, bestd = tid, d
        return best

    def _update_possession(self, holder, teams):
        team = teams.get(holder, 0) if holder is not None else 0
        # hysteresis: a new team must hold the ball for `hysteresis` frames to flip
        if team and team != self._cur_team:
            if team == self._cand_team:
                self._cand_count += 1
            else:
                self._cand_team, self._cand_count = team, 1
            if self._cand_count >= self.hysteresis:
                self._cur_team = team
                self._cand_count = 0
        elif team == self._cur_team:
            self._cand_count = 0
        self._holder_id = holder
        # possession % = fraction of gated frames each team's player is ACTUALLY
        # nearest the ball (not the sticky _cur_team, which locks to one side and
        # never releases for a contested ball). _cur_team still drives the holder /
        # pass state machine above.
        if team in (1, 2):
            self.team_frames[team] += 1

    def _update_passes(self, frame_idx, holder, teams, players, ball):
        """Log a pass when confirmed control transfers between two players of the
        same team. Turnovers (different team) reset the passer."""
        if holder is None:
            return
        team = teams.get(holder, 0)
        if self._confirmed_holder is None:
            self._confirmed_holder, self._confirmed_team = holder, team
            return
        if holder == self._confirmed_holder:
            self._pass_cand_holder, self._pass_cand_count = None, 0
            return
        # a different player is nearest — debounce before treating it as a transfer
        if holder == self._pass_cand_holder:
            self._pass_cand_count += 1
        else:
            self._pass_cand_holder, self._pass_cand_count = holder, 1
        if self._pass_cand_count < self.pass_confirm_frames:
            return
        # confirmed control transfer
        if team == self._confirmed_team and team in (1, 2):
            if self._confirmed_holder in players and holder in players:
                self.passes.append(Pass(
                    frame=frame_idx, team=team,
                    from_id=self._confirmed_holder, to_id=holder,
                    start=players[self._confirmed_holder], end=players[holder],
                ))
        self._confirmed_holder, self._confirmed_team = holder, team
        self._pass_cand_holder, self._pass_cand_count = None, 0

    # ------------------------------------------------------------------ #
    @property
    def possessing_team(self) -> int:
        """Team (1/2) currently deemed in possession, 0 if none yet."""
        return self._cur_team

    def possession_pct(self) -> Dict[int, float]:
        total = self.team_frames[1] + self.team_frames[2]
        if total == 0:
            return {1: 0.0, 2: 0.0}
        return {t: 100.0 * self.team_frames[t] / total for t in (1, 2)}

    def speed_distance(self, window: int = 5) -> Dict[int, dict]:
        """Per track_id: total distance (m or px) and average speed.

        Distance is summed over sliding windows of ``window`` frames using the
        pitch positions; per-window speed above ``max_speed_mps`` is treated as a
        tracking glitch and that step is dropped from the distance total.
        """
        out: Dict[int, dict] = {}
        for tid, seq in self.positions.items():
            pts = [(f, x, y) for (f, x, y) in seq if x is not None and not np.isnan(x)]
            if len(pts) < 2:
                out[tid] = {"distance": 0.0, "avg_speed": 0.0, "top_speed": 0.0,
                            "team": self.team_of.get(tid, 0), "frames": len(seq)}
                continue
            total = 0.0
            speeds = []
            for i in range(0, len(pts) - window, window):
                f0, x0, y0 = pts[i]
                f1, x1, y1 = pts[i + window]
                # real elapsed time from frame indices — occlusion gaps span more
                # than `window` frames, so a fixed window/fps would inflate speed
                dt = (f1 - f0) / self.fps if self.fps > 0 else 0.0
                d = float(np.hypot(x1 - x0, y1 - y0))
                spd = d / dt if dt > 0 else 0.0
                if self.metric and spd > self.max_speed_mps:
                    continue          # implausible → drop as jitter
                total += d
                speeds.append(spd)
            out[tid] = {
                "distance": round(total, 2),
                "avg_speed": round(float(np.mean(speeds)) if speeds else 0.0, 2),
                "top_speed": round(float(np.max(speeds)) if speeds else 0.0, 2),
                "team": self.team_of.get(tid, 0),
                "frames": len(seq),
            }
        return out
