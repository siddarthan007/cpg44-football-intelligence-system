"""Fuse the vision and wearable streams into per-player Catapult-style load.

The vision pipeline supplies, per frame, each track's **Kalman-smoothed velocity
and acceleration** (not raw positions — accuracy matters for metabolic power).
This maps tracks to players, attaches the time-aligned wearable reading (HR /
SpO2 / IMU), and feeds the :class:`LoadEngine`. Everything downstream (injury,
recommendations) reads from ``load``. Vision-only operation is first-class: with
no wearable bound, all external-load metrics still compute; only HR/SpO2 and true
IMU PlayerLoad stay empty.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from ..catapult import LoadEngine
from .sync import SensorVideoSync


class FusionEngine:
    def __init__(self, sync: Optional[SensorVideoSync] = None, mass_kg: float = 75.0):
        self.sync = sync or SensorVideoSync()
        self.load = LoadEngine(mass_kg=mass_kg)

    def step(self, dt: float, video_t: float,
             vision_state: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]]):
        """``vision_state`` = {track_id: ((vx,vy), (ax,ay))} in m/s and m/s².

        External (vision) load is computed for EVERY track — so a demo with only
        1-2 wearables still gets full per-player performance for all players. For
        tracks bound to a wearable player, the time-aligned HR/SpO2/IMU are also
        attached (internal load + true IMU PlayerLoad). Load is keyed by track id."""
        for track_id, (vel, acc) in vision_state.items():
            hr = spo2 = imu = None
            pid = self.sync.player_of(track_id)
            if pid is not None:
                s = self.sync.sample_at(pid, video_t)
                if s is not None:
                    hr, spo2, imu = s.hr, s.spo2, s.accel
            self.load.update(track_id, dt, vel, acc, hr=hr, spo2=spo2, imu_accel=imu)

    def ingest_sensors(self, samples):
        self.sync.ingest(samples)
