"""Fuse the vision and wearable streams into per-player Catapult-style load.

The vision pipeline supplies, per frame, each track's **Kalman-smoothed velocity
and acceleration** (not raw positions, accuracy matters for metabolic power).
This maps tracks to players, attaches the time-aligned wearable reading (HR /
IMU), and feeds the :class:`LoadEngine`. Everything downstream (injury,
recommendations) reads from ``load``. Vision-only operation is first-class: with
no wearable bound, all external-load metrics still compute; only heart rate and true
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
        self.aligned = {}
        self.alignment = {}
        self._last_load_sample = {}
        self._last_hr_sample = {}

    def step(self, dt: float, video_t: Optional[float],
             vision_state: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]]):
        """``vision_state`` = {track_id: ((vx,vy), (ax,ay))} in m/s and m/s².

        External (vision) load is computed for EVERY track, so a demo with only
        1-2 wearables still gets full per-player performance for all players. For
        tracks bound to a wearable player, the time-aligned heart rate and IMU are also
        attached (internal load + true IMU PlayerLoad). Load is keyed by track id."""
        self.aligned.clear()
        self.alignment.clear()
        for track_id, (vel, acc) in vision_state.items():
            hr = imu = None
            pid = self.sync.player_of(track_id)
            self.aligned.pop(track_id, None)
            if pid is not None and video_t is not None:
                s, offset = self.sync.sample_at_with_offset(pid, video_t)
                clock_ok = s is not None and s.clock_synced is not False
                if s is not None and clock_ok:
                    hr, imu = s.hr, s.accel
                    self.aligned[track_id] = s
                self.alignment[track_id] = {
                    "matched": bool(s is not None and clock_ok),
                    "offset_ms": None if offset is None else round(offset * 1000.0, 1),
                    "clock_synced": None if s is None else s.clock_synced,
                    "signal_quality": None if s is None else s.signal_quality,
                }
            elif pid is not None:
                self.alignment[track_id] = {"matched": False, "reason": "video_source_time_unavailable"}
            sensor = self.aligned.get(track_id)
            # A 25 FPS camera must not turn one 1 Hz pulse observation into 25
            # independent physiological measurements.
            stamp = None if sensor is None else (pid, sensor.t)
            if stamp is None or stamp == self._last_load_sample.get(track_id):
                imu = None
            else:
                self._last_load_sample[track_id] = stamp
            hr_stamp = None if sensor is None else (pid, sensor.hr_timestamp or sensor.t)
            if hr_stamp is None or hr_stamp == self._last_hr_sample.get(track_id):
                hr = None
            else:
                self._last_hr_sample[track_id] = hr_stamp
            self.load.update(track_id, dt, vel, acc, hr=hr, imu_accel=imu)

    def ingest_sensors(self, samples):
        self.sync.ingest(samples)
