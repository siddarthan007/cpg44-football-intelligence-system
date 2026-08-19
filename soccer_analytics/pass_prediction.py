"""Pass-receiver prediction (CPG44 ref [8] Honda et al.).

Honda et al. predict the pass receiver by fusing video with player trajectories
(3D-CNN + LSTM + Transformer). This is the interpretable geometric counterpart:
for the ball carrier, each teammate is scored as a pass option from spatial
features — receiver **openness** (space from nearest defender), passing-**lane
clearness** (no defender intercepting the straight lane), **progressiveness**
(territory gained toward goal), and a plausible **range**. The trajectory LSTM
(:mod:`soccer_analytics.trajectory`) can extend this to where players *will* be.

Pure NumPy over pitch-metre coordinates — unit-testable. A logistic model can be
fitted on labelled completed/failed passes later to calibrate the weighting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def _lane_clearness(carrier: np.ndarray, receiver: np.ndarray,
                    opponents: np.ndarray, corridor: float = 2.5) -> float:
    """0..1: how clear the straight passing lane is. 1 = no defender within
    ``corridor`` metres of the segment (between carrier and receiver)."""
    seg = receiver - carrier
    L = np.linalg.norm(seg)
    if L < 1e-6 or len(opponents) == 0:
        return 1.0
    u = seg / L
    rel = opponents - carrier
    t = rel @ u                                   # projection along the lane
    on = (t > 0) & (t < L)                        # defenders between the two
    if not on.any():
        return 1.0
    perp = np.linalg.norm(rel[on] - np.outer(t[on], u), axis=1)
    nearest = perp.min()
    return float(np.clip(nearest / corridor, 0.0, 1.0))


@dataclass
class PassOption:
    receiver: int
    score: float
    openness: float
    lane: float
    progress: float
    xy: Tuple[float, float]


def pass_options(carrier_xy, teammates: Dict[int, Tuple[float, float]],
                 opponents: List[Tuple[float, float]], attack_goal_x: float,
                 pitch_length: float = 105.0, min_range: float = 3.0,
                 max_range: float = 45.0) -> List[PassOption]:
    """Score every teammate as a pass option for the ball carrier. Returns options
    sorted best-first."""
    carrier = np.asarray(carrier_xy, float)
    opp = np.asarray([o for o in opponents if not np.isnan(o[0])], float).reshape(-1, 2)
    opts: List[PassOption] = []
    for rid, xy in teammates.items():
        r = np.asarray(xy, float)
        if np.isnan(r[0]):
            continue
        dist = float(np.linalg.norm(r - carrier))
        if dist < min_range or dist > max_range:
            continue
        # openness: distance from the receiver to the nearest defender (capped)
        open_m = 10.0 if len(opp) == 0 else float(np.linalg.norm(opp - r, axis=1).min())
        openness = float(np.clip(open_m / 8.0, 0.0, 1.0))
        lane = _lane_clearness(carrier, r, opp)
        # progressiveness: fraction of remaining distance to goal gained
        gain = (abs(attack_goal_x - carrier[0]) - abs(attack_goal_x - r[0]))
        progress = float(np.clip(gain / pitch_length + 0.5, 0.0, 1.0))
        score = float(0.4 * lane + 0.35 * openness + 0.25 * progress)
        opts.append(PassOption(rid, round(score, 3), round(openness, 2),
                               round(lane, 2), round(progress, 2),
                               (float(r[0]), float(r[1]))))
    opts.sort(key=lambda o: -o.score)
    return opts


def best_receiver(carrier_xy, teammates, opponents, attack_goal_x,
                  pitch_length: float = 105.0) -> Optional[PassOption]:
    opts = pass_options(carrier_xy, teammates, opponents, attack_goal_x, pitch_length)
    return opts[0] if opts else None
