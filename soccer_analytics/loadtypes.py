"""Neutral load/injury data types shared by :mod:`soccer_analytics.catapult`
(which produces them) and :mod:`soccer_analytics.sensors` (which consumes them).

Kept in its own module — with no dependency on either package — so the two can
reference these types without an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class WorkloadFeatures:
    """Accumulated per-player load features — the injury model's input vector.

    Catapult-aligned: external load comes from vision (distance, speed zones, HSR,
    sprints, accel/decel efforts, metabolic power/energy) and — when the wearable
    is present — internal load (HR/SpO2) and true IMU PlayerLoad augment it."""
    player_id: int
    total_distance: float = 0.0           # m
    hsr_distance: float = 0.0             # high-speed-running distance, m (>5.5 m/s)
    sprint_count: int = 0                 # >7 m/s efforts
    accel_efforts: int = 0                # high accelerations (>2 m/s²)
    decel_efforts: int = 0                # high decelerations (<-2 m/s²)
    player_load: float = 0.0              # vision dynamic load proxy (accel-based)
    player_load_imu: float = float("nan") # true Catapult PlayerLoad from IMU (if worn)
    metabolic_power_avg: float = 0.0      # W/kg (di Prampero from speed+accel)
    metabolic_power_peak: float = 0.0     # W/kg
    high_metabolic_distance: float = 0.0  # m spent above 20 W/kg
    energy_kcal: float = 0.0              # estimated energy expenditure
    top_speed: float = 0.0               # m/s
    distance_by_zone: Dict[str, float] = field(default_factory=dict)
    avg_hr: float = float("nan")
    max_hr: float = float("nan")
    hr_drift: float = 0.0                 # late-vs-early HR rise (fatigue proxy)
    min_spo2: float = float("nan")
    acwr: float = float("nan")            # acute:chronic workload ratio
    duration_s: float = 0.0

    def to_vector(self) -> Dict[str, float]:
        return {
            "total_distance": self.total_distance,
            "hsr_distance": self.hsr_distance,
            "sprint_count": float(self.sprint_count),
            "accel_efforts": float(self.accel_efforts),
            "decel_efforts": float(self.decel_efforts),
            "player_load": self.player_load,
            "metabolic_power_avg": self.metabolic_power_avg,
            "high_metabolic_distance": self.high_metabolic_distance,
            "energy_kcal": self.energy_kcal,
            "top_speed": self.top_speed,
            "avg_hr": self.avg_hr,
            "hr_drift": self.hr_drift,
            "min_spo2": self.min_spo2,
            "acwr": self.acwr,
        }


@dataclass
class InjuryRisk:
    player_id: int
    risk: float                           # 0..1
    level: str                            # "low" | "moderate" | "high"
    factors: Dict[str, float] = field(default_factory=dict)  # contributing factors
