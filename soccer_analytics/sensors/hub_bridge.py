"""Map the ESP32 sensor hub into the vision fusion layer.

The firmware (`firmware/wearable_stream`) sends raw IMU / PPG / GPS batches to
the public relay. `soccer_analytics.sensors.hub` consumes the relay and turns
that data into processed vitals. This module is
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
    live_vit = vitals.get("live") or {}
    imu = state.get("imu") or {}
    gps = state.get("gps") or {}

    hr = spo2 = quality = None
    pulse_window = {}
    if stable.get("valid"):
        hr, spo2 = stable.get("bpm"), stable.get("spo2_estimate_pct")
        quality = stable.get("quality")
        pulse_window = stable
    elif rolling.get("valid"):
        hr, spo2 = rolling.get("bpm"), rolling.get("spo2_estimate_pct")
        quality = rolling.get("quality")
        pulse_window = rolling
    elif live_vit.get("valid") and not live_vit.get("stale"):
        hr, spo2 = live_vit.get("bpm"), live_vit.get("spo2_estimate_pct")
        quality = live_vit.get("quality")
        pulse_window = live_vit
    if quality is not None and float(quality) < 0.35:
        hr = spo2 = None
    pulse_ns = pulse_window.get("window_end_unix_ns")
    if pulse_ns is None and pulse_window.get("window_end_host_mono_ns"):
        server = state.get("server") or {}
        if server.get("host_mono_ns") and server.get("host_unix_ns"):
            pulse_ns = (server["host_unix_ns"] - server["host_mono_ns"]
                        + pulse_window["window_end_host_mono_ns"])
    server_epoch = (state.get("server") or {}).get("host_unix_ns")
    if pulse_ns is not None and server_epoch is not None:
        if not 0 <= (server_epoch - pulse_ns) / 1e9 <= 2.0:
            hr = spo2 = None

    accel_g = None
    ts = imu.get("timestamp") or {}
    unix_ns = ts.get("host_unix_ns")
    server_unix_ns = (state.get("server") or {}).get("host_unix_ns")
    imu_age_s = None
    if unix_ns and server_unix_ns:
        imu_age_s = max(0.0, (float(server_unix_ns) - float(unix_ns)) / 1e9)
    body = imu.get("accel_body_mps2")
    if imu.get("valid") and body and (imu_age_s is None or imu_age_s <= 2.0):
        accel_g = tuple(float(v) / G for v in body)

    gyro_dps = None
    gyro = imu.get("gyro_body_rads") or imu.get("gyro_corrected_rads")
    if gyro:
        gyro_dps = tuple(math.degrees(float(v)) for v in gyro)

    gps_ll = None
    hdop = gps.get("hdop")
    gps_quality_ok = hdop is None or float(hdop) <= 5.0
    if gps.get("fix") and gps_quality_ok and gps.get("lat") is not None and gps.get("lon") is not None:
        gps_ll = (float(gps["lat"]), float(gps["lon"]))

    kinematics = state.get("kinematics") or {}
    speed_val = (state.get("gps") or {}).get("speed_mps") if (state.get("gps") or {}).get("fix") else kinematics.get("speed_mps")
    if speed_val is None:
        speed_val = 0.0

    t = time.time()
    unix_ns = unix_ns or server_unix_ns
    if unix_ns:
        t = float(unix_ns) / 1e9

    connected = bool((state.get("device") or {}).get("connected"))
    if not connected and not imu.get("valid") and hr is None and gps_ll is None:
        return None

    sample = SensorSample(
        player_id=int(player_id),
        t=t,
        hr=None if hr is None else float(hr),
        spo2=None if spo2 is None else float(spo2),
        accel=_triple(accel_g),
        gyro=_triple(gyro_dps),
        altitude=None if gps.get("alt_m") is None else float(gps["alt_m"]),
        gps=gps_ll,
        source="esp32-hub",
        signal_quality=None if quality is None else float(quality),
        clock_synced=bool(ts.get("clock_synced")) if ts else False,
        transport_delay_ms=(
            None if ts.get("transport_delay_ms") is None
            else float(ts["transport_delay_ms"])
        ),
        sample_age_ms=None if imu_age_s is None else imu_age_s * 1000.0,
        imu_calibrated=bool(imu.get("calibrated_gyro_bias") and imu.get("accel_scale_calibrated")),
        hr_timestamp=None if pulse_ns is None else pulse_ns / 1e9,
    )
    sample.speed = speed_val
    return sample


def sample_to_observation(sample: SensorSample, match_id: str = "live") -> Dict[str, Any]:
    """Capstone wearable contract: join CV later on match + player + time."""
    accel_mag = None
    if sample.accel:
        accel_mag = round(math.sqrt(sum(v * v for v in sample.accel)), 2)
    load = 0.0
    if accel_mag is not None:
        dynamic_g = abs(accel_mag - 1.0)
        if dynamic_g > 0.07:
            load = round(dynamic_g * 10.0, 2)
        else:
            load = 0.0
    speed_val = getattr(sample, "speed_mps", None)
    if speed_val is None and getattr(sample, "speed", None) is not None:
        speed_val = sample.speed
    if speed_val is None or load == 0.0:
        speed_val = 0.0

    return {
        "match_id": match_id,
        "global_player_id": f"PLAYER_{int(sample.player_id)}",
        "player_id": int(sample.player_id),
        "timestamp": float(sample.t),
        "source": "wearable",
        "hr": sample.hr,
        "spo2": sample.spo2,
        "playerLoad": load,
        "speed": speed_val,
        "metrics": {
            "hr": sample.hr,
            "heart_rate_bpm": sample.hr,
            "spo2": sample.spo2,
            "spo2_pct": sample.spo2,
            "player_load": load,
            "player_load_imu": load,
            "playerLoad": load,
            "acceleration_g": accel_mag,
            "speed_mps": speed_val,
            "speed": speed_val,
            "gps": list(sample.gps) if sample.gps else None,
            "signal_quality": sample.signal_quality if sample.signal_quality is not None else 0.88,
            "clock_synced": sample.clock_synced,
            "transport_delay_ms": sample.transport_delay_ms,
            "sample_age_ms": sample.sample_age_ms,
            "imu_calibrated": sample.imu_calibrated,
            "transport": sample.source,
        },
    }


class HubSensorSource(SensorSource):
    """Consume the local processor's NDJSON stream without a poll round trip."""

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
        last_timestamp = None
        while not self._stop.is_set():
            try:
                request = urllib.request.Request(
                    f"{self.url}/api/stream", headers={"Accept": "application/x-ndjson"},
                )
                with urllib.request.urlopen(request, timeout=1.0) as response:
                    for line in response:
                        if self._stop.is_set():
                            return
                        if len(line) > 256 * 1024:
                            raise ValueError("oversized sensor snapshot")
                        sample = snapshot_to_sample(json.loads(line), self.player_id)
                        if sample is not None and sample.t != last_timestamp:
                            last_timestamp = sample.t
                            yield sample
            except (OSError, ValueError):
                self._stop.wait(0.25)
