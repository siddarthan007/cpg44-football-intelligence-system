"""Load a recorded vest log for match review.

The log (``wearable_raw.ndjson`` written by the hub) is replayed through the
same :class:`TelemetryEngine` that runs live, so offline and live heart rate,
calibration and PlayerLoad come from one implementation. Times are returned
in seconds on the vest's SNTP clock (unix), ready to be shifted onto the video
clock and refined by :mod:`soccer_analytics.sensors.association`.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class VestLog:
    player_id: Optional[int]
    imu_t: np.ndarray            # unix seconds
    accel: np.ndarray            # (n,3) m/s^2, calibrated
    hr_t: np.ndarray
    hr: np.ndarray               # bpm, NaN where the quality gate failed
    hr_quality: np.ndarray
    load_t: np.ndarray
    player_load: np.ndarray      # cumulative
    gps_t: np.ndarray
    gps_latlon: np.ndarray       # (n,2), NaN without fix
    calibration: dict


def load(path: str, vitals_every_s: float = 1.0) -> VestLog:
    from .hub import TelemetryEngine

    async def run():
        eng = TelemetryEngine()
        imu_t, acc, hr_t, hr, q, lt, pl, gt, gl = [], [], [], [], [], [], [], [], []
        last_v = None
        pid = None
        for line in Path(path).read_text().splitlines():
            try:
                p = json.loads(line)
            except ValueError:
                continue
            pid = pid or p.get("player_id")
            for k in ("host_mono_ns", "host_unix_ns", "clock_synced", "transport_delay_ms"):
                p.pop(k, None)
            src = p.get("source_unix_ns")
            if src is None:
                continue
            # replay on the vest's own clock: receive time = acquisition time
            await eng.ingest(p, int(src), int(src))
            t = int(src) / 1e9
            kind = p.get("t")
            if kind == "imu":
                imu_t.append(t)
                acc.append(eng.state["imu"]["accel_body_mps2"])
                lt.append(t)
                pl.append(eng.player_load_total)
            elif kind == "ecg" and (last_v is None or t - last_v >= vitals_every_s):
                last_v = t
                r = eng.ecg.estimate(now=t)
                ok = bool(r.get("valid")) and r.get("bpm") is not None
                hr_t.append(t)
                hr.append(float(r["bpm"]) if ok else np.nan)
                q.append(float(r.get("quality") or 0.0) if ok else 0.0)
            elif kind == "gps":
                gt.append(t)
                gl.append([p.get("lat", np.nan) if p.get("fix") else np.nan,
                           p.get("lon", np.nan) if p.get("fix") else np.nan])
        cal = {"gyro_bias_rads": [float(x) for x in eng.gyro_cal.bias], "accel_calibrated": eng.accel_cal.calibrated}
        f = lambda x, d=1: np.asarray(x, float).reshape(-1, d) if d > 1 else np.asarray(x, float)
        return VestLog(pid, f(imu_t), f(acc, 3), f(hr_t), f(hr), f(q), f(lt), f(pl), f(gt), f(gl, 2), cal)

    return asyncio.run(run())
