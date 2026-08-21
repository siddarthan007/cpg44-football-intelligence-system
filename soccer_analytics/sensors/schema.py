"""Typed records exchanged across the fusion layer.

Every sample carries a ``t`` (epoch seconds, float) so the wearable and video
streams can be time-aligned regardless of their independent sample rates — the
foundation for near-real-time multimodal sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# load/injury types live in a neutral module to avoid a catapult<->sensors cycle;
# re-exported here so existing `from .schema import WorkloadFeatures` keeps working.
from ..loadtypes import WorkloadFeatures, InjuryRisk


@dataclass
class SensorSample:
    """One reading from a player's wearable. Fields mirror the report's hardware:
    GY-87 10-DoF (accel/gyro/mag/baro) + NEO-6M GPS + MAX30102 pulse-oximeter."""
    player_id: int
    t: float                              # epoch seconds
    hr: Optional[float] = None            # heart rate, bpm (MAX30102)
    spo2: Optional[float] = None          # blood oxygen, % (MAX30102)
    accel: Optional[Tuple[float, float, float]] = None   # ax, ay, az in g (MPU6050)
    gyro: Optional[Tuple[float, float, float]] = None    # deg/s (MPU6050)
    mag: Optional[Tuple[float, float, float]] = None     # uT (HMC5883L)
    altitude: Optional[float] = None                     # m (BMP180)
    gps: Optional[Tuple[float, float]] = None            # lat, lon (NEO-6M)
    source: str = "unknown"
    signal_quality: Optional[float] = None               # 0..1 after source gating

    @classmethod
    def from_json(cls, d: dict) -> "SensorSample":
        def tup(x):
            return tuple(float(v) for v in x) if x is not None else None
        return cls(
            player_id=int(d["player_id"]), t=float(d["t"]),
            hr=d.get("hr"), spo2=d.get("spo2"),
            accel=tup(d.get("accel")), gyro=tup(d.get("gyro")), mag=tup(d.get("mag")),
            altitude=d.get("altitude"), gps=tup(d.get("gps")),
            source=d.get("source", "json"),
            signal_quality=d.get("signal_quality"),
        )


@dataclass
class VisionSample:
    """Per-player vision-derived state at a video timestamp (external load)."""
    track_id: int
    t: float
    x: float                              # pitch metres
    y: float
    speed: float = 0.0                    # m/s (instantaneous)


@dataclass
class FusedSample:
    """Vision + wearable merged for one player at one instant."""
    player_id: int
    t: float
    x: float = float("nan")
    y: float = float("nan")
    speed: float = 0.0
    hr: Optional[float] = None
    spo2: Optional[float] = None
    accel_mag: Optional[float] = None
    matched: bool = False                 # True if a wearable sample was aligned


# WorkloadFeatures and InjuryRisk are defined in soccer_analytics.loadtypes and
# re-exported above.
