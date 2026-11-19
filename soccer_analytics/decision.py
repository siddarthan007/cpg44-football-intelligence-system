"""On-ball decision support from tracking: pitch control, pass options, shot value.

Pitch control follows the time-to-intercept idea of Spearman (2018, "Beyond
Expected Goals") and Fernández & Bornn (2018, "Wide Open Spaces"): a player
reaches a point after a reaction time plus the travel time from where his
current velocity carries him, at a maximum running speed; control at that
point is a logistic function of the difference between the two teams'
earliest arrivals. Parameters are the published defaults (reaction 0.7 s,
max speed 5 m/s for the model, temporal uncertainty 0.45 s).

Pass options for the ball carrier combine
  P(arrive) = P(no opponent intercepts the ball in flight)
              x P(team controls the target point)
where an opponent can intercept when he reaches some point on the lane
before the ball does (ball speed ~15 m/s along the ground), and
  value = threat(target) - threat(origin)
with ``threat`` a shot-threat surface from the logistic xG model (distance and
angle to goal, Caley/Pollard style). Shoot-or-pass compares the carrier's xG
with the best pass's expected value. Everything is computed from observed
positions only; players outside the camera view are unknown, so options near
the edge of the camera footprint are marked lower-confidence by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

REACTION_S = 0.7
VMAX = 5.0          # m/s, model running speed (Spearman uses ~5 m/s)
SIGMA_T = 0.45      # s, temporal uncertainty of arrival
BALL_SPEED = 15.0   # m/s, typical ground pass


def arrival_times(players: np.ndarray, vels: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """(n_players, n_points) earliest arrival time in seconds."""
    if len(players) == 0:
        return np.full((0, len(pts)), np.inf)
    start = players + vels * REACTION_S                     # where momentum carries them
    d = np.linalg.norm(pts[None, :, :] - start[:, None, :], axis=2)
    return REACTION_S + d / VMAX


def control_probability(t_att: np.ndarray, t_def: np.ndarray) -> np.ndarray:
    """P(attacking team controls) from each side's earliest arrival."""
    ta = t_att.min(0) if len(t_att) else np.full(t_def.shape[1], np.inf)
    td = t_def.min(0) if len(t_def) else np.full(t_att.shape[1], np.inf)
    diff = np.clip(td - ta, -10, 10)
    return 1.0 / (1.0 + np.exp(-np.pi / np.sqrt(3) * diff / SIGMA_T))


def pitch_control_grid(att: np.ndarray, att_v: np.ndarray, dfn: np.ndarray, dfn_v: np.ndarray,
                       length: float, width: float, nx: int = 32, ny: int = 20) -> np.ndarray:
    xs = (np.arange(nx) + 0.5) * length / nx
    ys = (np.arange(ny) + 0.5) * width / ny
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack((gx.ravel(), gy.ravel()))
    pc = control_probability(arrival_times(att, att_v, pts), arrival_times(dfn, dfn_v, pts))
    return pc.reshape(ny, nx)


def xg(x: float, y: float, goal_x: float, width: float, goal_w: float = 7.32) -> float:
    """Logistic shot model on distance and goal-mouth angle (open play)."""
    gy = width / 2.0
    dx, dy = abs(goal_x - x), abs(gy - y)
    dist = float(np.hypot(dx, dy))
    a1 = np.arctan2(gy - goal_w / 2 - y, dx + 1e-6)
    a2 = np.arctan2(gy + goal_w / 2 - y, dx + 1e-6)
    angle = abs(a2 - a1)
    z = -1.1 - 0.105 * dist + 1.6 * angle
    return float(1.0 / (1.0 + np.exp(-z)))


def threat(x: float, y: float, goal_x: float, width: float, goal_w: float = 7.32) -> float:
    """Value of having the ball at (x, y): shot value, smoothed outward."""
    return xg(x, y, goal_x, width, goal_w)


@dataclass
class PassOption:
    to: int
    x: float
    y: float
    p_arrive: float
    value: float          # threat gained if it arrives
    ev: float             # p_arrive * value - (1 - p_arrive) * loss_risk
    distance: float


def pass_options(carrier: Tuple[float, float], teammates: Dict[int, Tuple[float, float, float, float]],
                 opponents: Sequence[Tuple[float, float, float, float]], goal_x: float, width: float,
                 length: float, goal_w: float = 7.32) -> List[PassOption]:
    """Rank passes from the carrier. Players as (x, y, vx, vy)."""
    cx, cy = carrier
    opp = np.asarray(opponents, float).reshape(-1, 4)
    mates = {k: v for k, v in teammates.items()}
    att_xy = np.asarray([v[:2] for v in mates.values()], float).reshape(-1, 2)
    att_v = np.asarray([v[2:] for v in mates.values()], float).reshape(-1, 2)
    t0 = threat(cx, cy, goal_x, width, goal_w)
    out = []
    for pid, (tx, ty, vx, vy) in mates.items():
        # receive where the runner will be when the ball arrives (lead the pass)
        dist0 = float(np.hypot(tx - cx, ty - cy))
        if dist0 < 3 or dist0 > 45:
            continue
        flight = dist0 / BALL_SPEED
        rx, ry = tx + vx * flight * 0.6, ty + vy * flight * 0.6
        if not (0 <= rx <= length and 0 <= ry <= width):
            continue
        dist = float(np.hypot(rx - cx, ry - cy))
        # interception along the lane
        n = max(4, int(dist / 2))
        s = np.linspace(0.15, 1.0, n)
        lane = np.column_stack((cx + (rx - cx) * s, cy + (ry - cy) * s))
        ball_t = s * dist / BALL_SPEED
        if len(opp):
            t_opp = arrival_times(opp[:, :2], opp[:, 2:], lane).min(0)
            p_cut = 1.0 / (1.0 + np.exp(-np.pi / np.sqrt(3) * (ball_t - t_opp) / SIGMA_T))
            p_safe_lane = float(np.prod(1 - 0.35 * p_cut))
        else:
            p_safe_lane = 1.0
        pc = control_probability(arrival_times(att_xy, att_v, np.array([[rx, ry]])),
                                 arrival_times(opp[:, :2], opp[:, 2:], np.array([[rx, ry]])) if len(opp)
                                 else np.full((1, 1), np.inf))[0]
        p = float(np.clip(p_safe_lane * pc, 0.0, 1.0))
        value = threat(rx, ry, goal_x, width, goal_w) - t0
        loss = 0.04 + 0.08 * max(0.0, 1 - abs(goal_x - cx) / length)   # a turnover nearer our goal costs more
        out.append(PassOption(int(pid), round(rx, 2), round(ry, 2), round(p, 3), round(value, 4),
                              round(p * value - (1 - p) * loss, 4), round(dist, 1)))
    out.sort(key=lambda o: -o.ev)
    return out


def decide(carrier: Tuple[float, float], options: List[PassOption], goal_x: float, width: float,
           pressure: int, goal_w: float = 7.32) -> dict:
    """Shoot, pass forward, keep, or reset, with the numbers behind it."""
    shot = xg(carrier[0], carrier[1], goal_x, width, goal_w)
    best = options[0] if options else None
    safe = [o for o in options if o.p_arrive >= 0.8]
    if shot >= 0.12 and (best is None or shot >= best.ev + 0.02):
        return {"call": "shoot", "xg": round(shot, 3)}
    if best is not None and best.ev > 0.005 and best.p_arrive >= 0.55:
        return {"call": "pass", "to": best.to, "p": best.p_arrive, "gain": best.value, "xg": round(shot, 3)}
    if pressure >= 2 and safe:
        s = max(safe, key=lambda o: o.p_arrive)
        return {"call": "recycle", "to": s.to, "p": s.p_arrive, "xg": round(shot, 3)}
    return {"call": "carry", "xg": round(shot, 3)}
