"""Substitution prediction (CPG44 Objective 3 — tactical decision module).

Predicts WHICH player should be substituted and with what urgency, from the
time-course of their physical output. Sports-science grounding: the strongest
on-pitch fatigue signals are DECLINING RATES of high-intensity work — a tiring
player's total distance keeps rising, but their high-speed-running rate, sprint
frequency and work rate fall between the early and recent phases of play, while
cardiac drift rises (same output at higher heart rate). Coaches substitute on
exactly these cues; this module quantifies them.

Design
------
:class:`SubstitutionAdvisor` consumes periodic snapshots of each player's
accumulated :class:`WorkloadFeatures` (already produced by the Catapult load
engine every analytics tick). From the snapshot series it computes early-window
vs recent-window RATES (m/min HSR, sprints/min, m/min work rate, W/kg metabolic
power), turning cumulative counters into fatigue trajectories:

    fatigue index  = weighted decline of {HSR rate, work rate, sprint rate,
                     metabolic power} + cardiac drift (wearable, when present)
    sub priority   = fatigue index blended with injury risk and minutes played

Every factor is exposed in the output (interpretable — a coach can see WHY).
Like the injury model, the weighting is a defensible heuristic baseline that can
be replaced by a model fitted on labelled substitution events (`fit()` hook).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .loadtypes import WorkloadFeatures


@dataclass
class SubAdvice:
    player_id: int
    priority: float                    # 0..1 — substitution urgency
    fatigue: float                     # 0..1 — physical-decline component
    injury_risk: float                 # 0..1 — from the injury model
    minutes: float
    reasons: List[str] = field(default_factory=list)
    factors: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"player_id": self.player_id, "priority": round(self.priority, 3),
                "fatigue": round(self.fatigue, 3), "injury_risk": round(self.injury_risk, 3),
                "minutes": round(self.minutes, 1), "reasons": self.reasons,
                "factors": {k: round(v, 3) for k, v in self.factors.items()}}


class SubstitutionAdvisor:
    def __init__(self, min_minutes: float = 3.0, min_snapshots: int = 6,
                 early_frac: float = 0.35, recent_frac: float = 0.30):
        # snapshots: player -> list of (t_seconds, dist, hsr, sprints, energy_kcal)
        self._snaps: Dict[int, list] = defaultdict(list)
        self._last_feats: Dict[int, WorkloadFeatures] = {}
        self.min_minutes = min_minutes          # need enough play to judge decline
        self.min_snapshots = min_snapshots
        self.early_frac = early_frac
        self.recent_frac = recent_frac

    # ------------------------------------------------------------------ #
    def update(self, t: float, features: Dict[int, WorkloadFeatures]):
        """Record a snapshot of every player's cumulative load at time ``t`` (s).
        Call on the analytics cadence (~every few seconds of play)."""
        for pid, f in features.items():
            self._snaps[pid].append((t, f.total_distance, f.hsr_distance,
                                     float(f.sprint_count), f.energy_kcal))
            self._last_feats[pid] = f

    @staticmethod
    def _rate(snaps: list, i0: int, i1: int, col: int) -> float:
        """Average rate of a cumulative column between snapshots i0..i1, per min."""
        t0, t1 = snaps[i0][0], snaps[i1][0]
        if t1 - t0 < 1e-6:
            return 0.0
        return (snaps[i1][col] - snaps[i0][col]) / (t1 - t0) * 60.0

    def _decline(self, early: float, recent: float) -> float:
        """0..1 relative decline of a rate (0 = maintained, 1 = collapsed)."""
        if early <= 1e-6:
            return 0.0
        return float(np.clip((early - recent) / early, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    def advise(self, injury_risk: Optional[Dict[int, float]] = None
               ) -> List[SubAdvice]:
        """Ranked substitution candidates (highest priority first)."""
        injury_risk = injury_risk or {}
        out: List[SubAdvice] = []
        for pid, snaps in self._snaps.items():
            if len(snaps) < self.min_snapshots:
                continue
            dur_min = (snaps[-1][0] - snaps[0][0]) / 60.0
            if dur_min < self.min_minutes:
                continue
            n = len(snaps)
            e1 = max(1, int(n * self.early_frac))            # early window end
            r0 = max(e1, int(n * (1.0 - self.recent_frac)))  # recent window start
            if r0 >= n - 1:
                continue

            factors = {}
            # rate declines: distance (work rate), HSR, sprints, energy (metabolic)
            for name, col, w in (("work_rate", 1, 0.25), ("hsr_rate", 2, 0.35),
                                 ("sprint_rate", 3, 0.20), ("energy_rate", 4, 0.20)):
                early = self._rate(snaps, 0, e1, col)
                recent = self._rate(snaps, r0, n - 1, col)
                factors[name + "_decline"] = self._decline(early, recent) * w

            fatigue = float(np.clip(sum(factors.values()) / 1.0, 0.0, 1.0))

            f = self._last_feats.get(pid)
            # cardiac drift (wearable): same output at rising HR = internal fatigue
            drift_boost = 0.0
            if f is not None and not np.isnan(f.hr_drift) and f.hr_drift > 8:
                drift_boost = float(np.clip((f.hr_drift - 8) / 25.0, 0, 1)) * 0.25
                factors["cardiac_drift"] = drift_boost
                fatigue = float(np.clip(fatigue + drift_boost, 0, 1))

            risk = float(injury_risk.get(pid, 0.0))
            minutes_norm = float(np.clip(dur_min / 90.0, 0, 1))
            priority = float(np.clip(0.55 * fatigue + 0.35 * risk + 0.10 * minutes_norm,
                                     0, 1))

            reasons = []
            if factors.get("hsr_rate_decline", 0) > 0.15:
                reasons.append(f"high-speed running rate down "
                               f"{factors['hsr_rate_decline'] / 0.35 * 100:.0f}%")
            if factors.get("work_rate_decline", 0) > 0.10:
                reasons.append(f"work rate down {factors['work_rate_decline'] / 0.25 * 100:.0f}%")
            if factors.get("sprint_rate_decline", 0) > 0.10:
                reasons.append("sprint frequency dropping")
            if drift_boost > 0.05:
                reasons.append(f"cardiac drift +{f.hr_drift:.0f} bpm")
            if risk >= 0.5:
                reasons.append(f"injury risk {risk:.2f}")
            if not reasons:
                reasons.append("output maintained")

            out.append(SubAdvice(pid, priority, fatigue, risk, dur_min,
                                 reasons, factors))
        out.sort(key=lambda a: -a.priority)
        return out

    def watchlist(self, injury_risk=None, threshold: float = 0.45,
                  top: int = 3) -> List[SubAdvice]:
        """The players a coach should be considering — priority ≥ threshold."""
        return [a for a in self.advise(injury_risk)[:top] if a.priority >= threshold]
