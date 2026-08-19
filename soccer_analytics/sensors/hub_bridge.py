"""Map the ESP32 sensor hub into the vision fusion layer.

The firmware (`firmware/wearable_stream`) streams raw IMU / PPG / GPS over TCP.
`soccer_analytics.sensors.hub` turns that into processed vitals. This module is
the last hop: a :class:`SensorSource` the realtime pipeline can drain each frame,
plus the Capstone join-key (`match_id + global_player_id + timestamp`).
"""

from __future__ import annotations

import json
import math
import time
import urllib.request
from typing import Any, Dict, Optional

from .schema import SensorSample
from .source import SensorSource

G = 9.80665


def _triple(values) -> Optional[tuple]:
    if not values or len(values) != 3:
        return None
    try:
        return tuple(float(v) for v in values)
    except (TypeError, ValueError):
        return None


def snapshot_to_sample(state: Dict[str, Any], player_id: int) -> Optional[SensorSample]:
    """Convert one hub `/api/latest` snapshot into a fusion :class:`SensorSample`.

    Heart rate / SpO2 prefer the 15-second stable estimate (motion-gated). IMU
    acceleration is converted from m/s² to g so :class:`LoadEngine` PlayerLoad
    matches the Catapult convention used elsewhere in the pipeline.
    """
    if not state:
        return None

    vitals = state.get("vitals") or {}
    stable = vitals.get("stable_15s") or {}
    rolling = vitals.get("rolling") or {}
    imu = state.get("imu") or {}
    gps = state.get("gps") or {}

    hr = spo2 = None
    if stable.get("valid"):
        hr, spo2 = stable.get("bpm"), stable.get("spo2_estimate_pct")
    elif rolling.get("valid"):
        hr, spo2 = rolling.get("bpm"), rolling.get("spo2_estimate_pct")
    else:
        hr, spo2 = rolling.get("bpm"), rolling.get("spo2_estimate_pct")

    accel_g = None
    body = imu.get("accel_body_mps2")
    if body:
        accel_g = tuple(float(v) / G for v in body)

    gyro_dps = None
    gyro = imu.get("gyro_body_rads") or imu.get("gyro_corrected_rads")
    if gyro:
        gyro_dps = tuple(math.degrees(float(v)) for v in gyro)

    gps_ll = None
    if gps.get("fix") and gps.get("lat") is not None and gps.get("lon") is not None:
        gps_ll = (float(gps["lat"]), float(gps["lon"]))

    t = time.time()
    ts = imu.get("timestamp") or {}
    unix_ns = ts.get("host_unix_ns") or (state.get("server") or {}).get("host_unix_ns")
    if unix_ns:
        t = float(unix_ns) / 1e9

    connected = bool((state.get("device") or {}).get("connected"))
    if not connected and not imu.get("valid") and hr is None and gps_ll is None:
        return None

    return SensorSample(
        player_id=int(player_id),
        t=t,
        hr=None if hr is None else float(hr),
        spo2=None if spo2 is None else float(spo2),
        accel=_triple(accel_g),
        gyro=_triple(gyro_dps),
        altitude=None if gps.get("alt_m") is None else float(gps["alt_m"]),
        gps=gps_ll,
        source="esp32-hub",
    )


def sample_to_observation(sample: SensorSample, match_id: str = "live") -> Dict[str, Any]:
    """Capstone wearable contract: join CV later on match + player + time."""
    accel_mag = None
    if sample.accel:
        accel_mag = math.sqrt(sum(v * v for v in sample.accel))
    return {
        "match_id": match_id,
        "global_player_id": f"PLAYER_{int(sample.player_id)}",
        "timestamp": float(sample.t),
        "source": "wearable",
        "metrics": {
            "heart_rate_bpm": sample.hr,
            "spo2_pct": sample.spo2,
            "acceleration_g": accel_mag,
            "gps": list(sample.gps) if sample.gps else None,
            "transport": sample.source,
        },
    }


class HubSensorSource(SensorSource):
    """Poll the sensor-hub HTTP API (same machine or another WSL/Windows process)."""

    def __init__(self, url: str = "http://127.0.0.1:8081", player_id: int = 7,
                 hz: float = 10.0):
        super().__init__()
        self.url = url.rstrip("/")
        self.player_id = int(player_id)
        self.hz = float(hz)

    def _fetch(self) -> Optional[dict]:
        req = urllib.request.Request(
            f"{self.url}/api/latest",
            headers={"Accept": "application/json", "Cache-Control": "no-store"},
        )
        try:
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    def _produce(self):
        dt = 1.0 / max(self.hz, 1.0)
        while not self._stop.is_set():
            state = self._fetch()
            sample = snapshot_to_sample(state or {}, self.player_id)
            if sample is not None:
                yield sample
            time.sleep(dt)
