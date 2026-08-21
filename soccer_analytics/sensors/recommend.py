"""Recommendation engine (CPG44 Objective 3).

Combines three signals into actionable coaching recommendations:
  * tactical shape (from :mod:`soccer_analytics.tactics`),
  * per-player explainable load-review score,
  * per-player performance decline (speed drop-off vs the player's own baseline).

Output categories mirror the report: **substitution**, **load redistribution**,
and **tactical/formation adjustment**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .schema import InjuryRisk, WorkloadFeatures


@dataclass
class Recommendation:
    category: str           # "substitution" | "load" | "tactical"
    priority: str           # "high" | "medium" | "low"
    player_id: Optional[int]
    message: str

    def as_dict(self) -> dict:
        return {"category": self.category, "priority": self.priority,
                "player_id": self.player_id, "message": self.message}


class RecommendationEngine:
    def __init__(self, high_risk: float = 0.66, fatigue_speed_drop: float = 0.25):
        self.high_risk = high_risk
        self.fatigue_speed_drop = fatigue_speed_drop

    def evaluate(self,
                 injury: Dict[int, InjuryRisk],
                 workload: Dict[int, WorkloadFeatures],
                 baseline_top_speed: Optional[Dict[int, float]] = None,
                 tactical_reports: Optional[list] = None) -> List[Recommendation]:
        recs: List[Recommendation] = []
        baseline_top_speed = baseline_top_speed or {}

        for pid, risk in injury.items():
            if risk.risk >= self.high_risk:
                top = ", ".join(f"{k}" for k in risk.factors) or "elevated load"
                recs.append(Recommendation(
                    "substitution", "high", pid,
                    f"Player {pid}: high load-review score ({risk.risk:.2f}; {top}). "
                    f"Review the player and consider reduced minutes."))
            elif risk.level == "moderate":
                recs.append(Recommendation(
                    "load", "medium", pid,
                    f"Player {pid}: moderate load-review score ({risk.risk:.2f}). "
                    f"Monitor and redistribute high-intensity actions."))

        # performance decline via top-speed drop vs baseline
        for pid, wf in workload.items():
            base = baseline_top_speed.get(pid)
            if base and wf.top_speed and (base - wf.top_speed) / base >= self.fatigue_speed_drop:
                recs.append(Recommendation(
                    "load", "medium", pid,
                    f"Player {pid}: top speed down {100*(base-wf.top_speed)/base:.0f}% "
                    f"vs baseline — fatigue; rotate or rest."))

        # tactical adjustments straight from the tactical reports
        for rep in (tactical_reports or []):
            for msg in getattr(rep, "recommendations", []):
                recs.append(Recommendation("tactical", "low",
                                           None, f"Team {rep.team}: {msg}"))
        return recs
