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

from collections import defaultdict, deque
from dataclasses import dataclass
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
                 max_speed_mps: float = 12.0, history_limit: Optional[int] = None):
        self.fps = float(fps)
        self.metric = metric
        self.possession_dist = possession_dist
        self.hysteresis = possession_hysteresis
        self.max_speed_mps = max_speed_mps
        self.pitch_width = 68.0   # throw-in touchline bounds (metres)
        self.pitch_length = 105.0

        # track_id -> list of (frame_idx, x, y)  (x,y may be nan)
        self.positions = defaultdict(list if history_limit is None else lambda: deque(maxlen=history_limit))
        self._distance = {}
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
        # A pass needs the ball to travel: two teammates standing over a loose
        # ball swap "nearest player" without any pass being played.
        self.min_pass_travel = 3.0 if metric else 100.0
        self._release_ball = None       # where the ball was when the passer last had it

        # restarts: ball fully leaving the pitch over a touchline and coming back
        # is recorded as a throw-in. Coordinates are pitch metres.
        self.throw_ins: List[dict] = []
        self._ball_in_play = False
        self._last_ball_in = None       # last observed in-play ball position
        self._exit_count = 0
        self._entry_count = 0
        self._last_exit = None
        self._last_restart_frame = -1e9

    # ------------------------------------------------------------------ #
    def update(self, frame_idx: int,
               players: Dict[int, Tuple[float, float]],
               teams: Dict[int, int],
               ball: Optional[Tuple[float, float]]):
        """One frame. ``players`` = {track_id: (x,y)}, ``teams`` = {track_id: team},
        ``ball`` = (x,y) or None. Coordinates in metres (or px if not metric)."""
        for tid, (x, y) in players.items():
            self.positions[tid].append((frame_idx, x, y))
            acc = self._distance.setdefault(tid, {
                "frames": 0, "valid": 0, "anchor": None,
                "distance": 0.0, "speed_sum": 0.0, "steps": 0, "top": 0.0,
            })
            acc["frames"] += 1
            if x is not None and y is not None and np.isfinite([x, y]).all():
                if acc["valid"] % 5 == 0:
                    if acc["anchor"] is not None:
                        f0, x0, y0 = acc["anchor"]
                        dt = (frame_idx - f0) / self.fps if self.fps > 0 else 0.0
                        distance = float(np.hypot(x - x0, y - y0))
                        speed = distance / dt if dt > 0 else 0.0
                        if dt > 0 and (not self.metric or speed <= self.max_speed_mps):
                            acc["distance"] += distance
                            acc["speed_sum"] += speed
                            acc["steps"] += 1
                            acc["top"] = max(acc["top"], speed)
                    acc["anchor"] = (frame_idx, x, y)
                acc["valid"] += 1
            if teams.get(tid):
                self.team_of[tid] = teams[tid]

        holder = self._nearest_player(players, ball)
        self._update_possession(holder, teams)
        self._update_passes(frame_idx, holder, teams, players, ball)
        self._update_throw_ins(frame_idx, ball, teams, players)

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
            self._release_ball = ball
            return
        if holder == self._confirmed_holder:
            self._pass_cand_holder, self._pass_cand_count = None, 0
            if ball is not None:
                self._release_ball = ball
            return
        # a different player is nearest, debounce before treating it as a transfer
        if holder == self._pass_cand_holder:
            self._pass_cand_count += 1
        else:
            self._pass_cand_holder, self._pass_cand_count = holder, 1
        if self._pass_cand_count < self.pass_confirm_frames:
            return
        # confirmed control transfer
        if team == self._confirmed_team and team in (1, 2):
            rel = self._release_ball
            travel = (float(np.hypot(ball[0] - rel[0], ball[1] - rel[1]))
                      if rel is not None and ball is not None else 0.0)
            if self._confirmed_holder in players and holder in players and travel >= self.min_pass_travel:
                self.passes.append(Pass(
                    frame=frame_idx, team=team,
                    from_id=self._confirmed_holder, to_id=holder,
                    start=(float(rel[0]), float(rel[1])), end=(float(ball[0]), float(ball[1])),
                ))
        self._confirmed_holder, self._confirmed_team = holder, team
        self._release_ball = ball
        self._pass_cand_holder, self._pass_cand_count = None, 0

    def _update_throw_ins(self, frame_idx, ball, teams, players) -> None:
        """Record a throw-in when the observed ball leaves over a touchline and
        later re-enters. Only actually-observed ball positions are used, so a
        Kalman coast through an occlusion cannot invent a restart."""
        if not self.metric or ball is None or not np.isfinite(ball).all():
            self._exit_count = self._entry_count = 0
            return
        margin = 1.0  # metres of tolerance for calibration error at the lines
        outside = ball[1] < -margin or ball[1] > self.pitch_width + margin
        if outside:
            self._entry_count = 0
            self._exit_count += 1
            if self._exit_count >= 3 and self._ball_in_play:
                self._ball_in_play = False
                self._last_exit = ((frame_idx, float(ball[0]), float(ball[1]))
                                   if 0 <= ball[0] <= self.pitch_length else None)
            return
        self._exit_count = 0
        self._entry_count += 1
        if not self._ball_in_play and self._entry_count >= 3:
            # first in-play observation after an exit (or at stream start)
            if self._last_exit is not None:
                exit_f, exit_x, exit_y = self._last_exit
                boundary = 0.0 if exit_y < 0 else self.pitch_width
                plausible = (self.fps <= frame_idx - exit_f <= 30 * self.fps
                             and frame_idx - self._last_restart_frame >= 3 * self.fps
                             and abs(float(ball[0]) - exit_x) <= 8.0
                             and abs(float(ball[1]) - boundary) <= 4.0)
                if not plausible:
                    self._last_exit = None
                    self._ball_in_play = True
                    return
                team = 0
                holder = self._nearest_player(players, ball)
                if holder is not None:
                    team = teams.get(holder, 0)
                self.throw_ins.append({
                    "frame": frame_idx,
                    "gap_frames": frame_idx - exit_f,
                    "exit_x": round(exit_x, 2),
                    "exit_y": round(exit_y, 2),
                    "entry_x": round(float(ball[0]), 2),
                    "entry_y": round(float(ball[1]), 2),
                    "team": team,   # team of the player nearest the returning ball
                    "side": "far" if exit_y < 0 else "near",
                })
                self.throw_ins = self.throw_ins[-200:]
                self._last_restart_frame = frame_idx
            self._ball_in_play = True
            self._last_exit = None
        self._last_ball_in = (float(ball[0]), float(ball[1]))

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
        if window == 5:
            return {tid: {
                "distance": round(acc["distance"], 2),
                "avg_speed": round(acc["speed_sum"] / max(acc["steps"], 1), 2),
                "top_speed": round(acc["top"], 2),
                "team": self.team_of.get(tid, 0), "frames": acc["frames"],
            } for tid, acc in self._distance.items()}
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
                # real elapsed time from frame indices, occlusion gaps span more
                # than `window` frames, so a fixed window/fps would inflate speed
                dt = (f1 - f0) / self.fps if self.fps > 0 else 0.0
                d = float(np.hypot(x1 - x0, y1 - y0))
                spd = d / dt if dt > 0 else 0.0
                if self.metric and spd > self.max_speed_mps:
                    continue          # implausible to drop as jitter
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
