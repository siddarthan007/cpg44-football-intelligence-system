"""Expected-goals (xG) model + shot detection.

**xG geometry** (standard): a shot's quality is driven by the distance to goal and
the angle the goal mouth subtends from the shot location. ``xg_from_location``
computes both and maps them through a logistic model. Default coefficients give
sensible, monotonic open-play values (closer + wider angle → higher xG); call
``XGModel.fit`` on labelled shot data to calibrate for your competition.

**Shot detection** uses the Kalman-filtered ball: a shot is flagged when the ball
accelerates past a speed threshold while travelling toward the attacking goal.
This is heuristic (no shot-event labels), so treat shot counts / xG sums as
indicative; it improves directly once real event data is available.

Pure NumPy + optional scikit-learn — the geometry and default model are testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

GOAL_WIDTH = 7.32   # metres


def shot_geometry(x: float, y: float, goal_x: float, pitch_width: float = 68.0
                  ) -> Tuple[float, float]:
    """Return (distance_m, angle_rad) from shot point (x,y) to the goal at
    ``goal_x`` (0 or pitch length), centred on the pitch width."""
    gy = pitch_width / 2.0
    dx = abs(goal_x - x)
    dy = y - gy
    distance = math.hypot(dx, dy)
    # angle subtended by the goal mouth (radians); larger = better shooting angle
    denom = dx * dx + dy * dy - (GOAL_WIDTH / 2.0) ** 2
    angle = math.atan2(GOAL_WIDTH * dx, denom)
    if angle < 0:
        angle += math.pi
    return distance, angle


@dataclass
class XGModel:
    """Logistic xG on [angle, distance]. Defaults are an approximate open-play
    model; fit() calibrates on real shots."""
    b0: float = 0.2
    b_angle: float = 1.2
    b_distance: float = -0.10
    _clf: object = None

    def xg(self, x: float, y: float, goal_x: float, pitch_width: float = 68.0) -> float:
        dist, ang = shot_geometry(x, y, goal_x, pitch_width)
        if self._clf is not None:
            p = self._clf.predict_proba([[ang, dist]])[0, 1]
            return float(p)
        logit = self.b0 + self.b_angle * ang + self.b_distance * dist
        return float(1.0 / (1.0 + math.exp(-logit)))

    def fit(self, shots_xy_goal: List[Tuple[float, float, float]], made: List[int],
            pitch_width: float = 68.0):
        """Calibrate on labelled shots. ``shots_xy_goal`` = [(x,y,goal_x),...],
        ``made`` = 1 if goal else 0."""
        from sklearn.linear_model import LogisticRegression
        X = np.array([shot_geometry(x, y, g, pitch_width) for (x, y, g) in shots_xy_goal])
        X = X[:, ::-1]     # -> [angle, distance] order
        self._clf = LogisticRegression(max_iter=1000).fit(X, np.asarray(made, int))
        return self


@dataclass
class Shot:
    frame: int
    team: int
    x: float
    y: float
    xg: float
    player: Optional[int] = None


class ShotDetector:
    """Flags shots from the Kalman-filtered ball kinematics and assigns xG."""

    def __init__(self, xg_model: Optional[XGModel] = None,
                 shot_speed: float = 6.0, cone_deg: float = 40.0,
                 pitch_length: float = 105.0, pitch_width: float = 68.0):
        self.xg_model = xg_model or XGModel()
        self.shot_speed = shot_speed
        self.cos_cone = math.cos(math.radians(cone_deg))
        self.L, self.Wd = pitch_length, pitch_width
        self.shots: List[Shot] = []
        self._in_shot = False

    def update(self, frame_idx: int, ball_pos: Optional[Tuple[float, float]],
               ball_vel: Optional[Tuple[float, float]], possessing_team: int,
               team_goal_x: Dict[int, float], nearest_player: Optional[int] = None):
        """One frame. ``team_goal_x`` = {team: attacking goal x (0 or L)}."""
        if ball_pos is None or ball_vel is None or possessing_team not in (1, 2):
            return
        bx, by = ball_pos
        vx, vy = ball_vel
        if bx is None or np.isnan(bx):
            return
        speed = math.hypot(vx, vy)
        if speed < self.shot_speed:
            self._in_shot = False
            return
        goal_x = team_goal_x.get(possessing_team)
        if goal_x is None:
            return
        # is the ball travelling toward the attacking goal (within a cone)?
        gx, gy = goal_x, self.Wd / 2.0
        dgx, dgy = gx - bx, gy - by
        dg = math.hypot(dgx, dgy)
        if dg < 1e-6 or speed < 1e-6:
            return
        cos_dir = (vx * dgx + vy * dgy) / (speed * dg)
        # _in_shot marks "currently inside one continuous shot trajectory"; it must
        # clear whenever the ball stops satisfying the shot condition (off-cone or
        # out of the attacking half), else a deflected rebound/follow-up whose speed
        # never drops below the threshold would be silently dropped.
        attacking_half = (bx > self.L / 2) if goal_x > self.L / 2 else (bx < self.L / 2)
        if cos_dir < self.cos_cone or not attacking_half:
            self._in_shot = False
            return
        if not self._in_shot:
            xg = self.xg_model.xg(bx, by, goal_x, self.Wd)
            self.shots.append(Shot(frame_idx, possessing_team, float(bx), float(by),
                                   round(xg, 3), nearest_player))
            self._in_shot = True

    def summary(self) -> Dict[int, dict]:
        out = {1: {"shots": 0, "xg": 0.0}, 2: {"shots": 0, "xg": 0.0}}
        for s in self.shots:
            out[s.team]["shots"] += 1
            out[s.team]["xg"] = round(out[s.team]["xg"] + s.xg, 3)
        return out
