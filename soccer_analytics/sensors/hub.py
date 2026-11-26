#!/usr/bin/env python3
"""
ESP32-S3 wearable sensor processor.

- Subscribes to the authenticated VPS relay. It never opens a LAN connection
  to an ESP32.
- Maps the firmware's SNTP-anchored sample time into the host monotonic clock
  for later CV fusion.
- Preserves raw packets with acquisition timestamps.
- Heart rate and heart-rate variability from the AD8232 chest ECG (beat
  detection in ``ecg.py``), with electrode-contact and signal-quality gates.
- Processes MPU6050 acceleration/gyro, auto-calibrates gyro bias while still,
  estimates 6-DoF orientation, removes gravity, computes jerk and short-term
  delta-v. Absolute speed comes from GPS, not long-term IMU integration.
- Exposes latest JSON plus processed and raw streaming HTTP endpoints.
- Optionally records synchronized NDJSON for later CV/sensor fusion.
"""

import argparse
import asyncio
import json
import ipaddress
import math
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode, urlparse, urlunparse

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from websockets.asyncio.client import connect

from soccer_analytics.sensors.ecg import ECGProcessor
from soccer_analytics.sensors.calibration import AccelAutoCal, GpsKalman, GyroBias, battery_state

G = 9.80665
HR_STABLE_SECONDS = 12.0       # the steadier heart-rate figure: median of valid ECG estimates


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _cardinal(deg):
    if deg is None or not math.isfinite(float(deg)):
        return None
    names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return names[int((float(deg) % 360) / 45 + 0.5) % 8]


def mono_to_unix_ns(mono_ns: int) -> int:
    # Approximate wall-clock mapping at the moment of conversion.
    return int(mono_ns + (time.time_ns() - time.perf_counter_ns()))


# ============================================================================
# Clock synchronization
# ============================================================================

@dataclass
class SyncPoint:
    device_us: float
    host_ns: float
    rtt_ns: float


class RelayClockOffset:
    """Offset of this machine's wall clock against the relay (NTP-style).

    Each probe is one request: t0 local send, ts server time, t1 local receive.
    offset = ts - (t0 + t1) / 2, with uncertainty (t1 - t0) / 2. Queueing only
    lengthens the round trip, so the estimate is taken from the shortest
    round trips of the recent window (the NTP clock filter idea). Measured on
    the recorded sessions, this laptop's clock ran 30 to 770 ms behind the
    relay, which made raw vest-to-video alignment wrong by that much.
    """

    def __init__(self, window: int = 24):
        self.samples = deque(maxlen=window)        # (rtt_ns, offset_ns, at_unix_ns)

    def add(self, t0_ns: int, server_ns: int, t1_ns: int) -> None:
        rtt = t1_ns - t0_ns
        if rtt <= 0 or rtt > 3_000_000_000:
            return
        self.samples.append((rtt, server_ns - (t0_ns + t1_ns) // 2, t1_ns))

    @property
    def valid(self) -> bool:
        return bool(self.samples)

    def estimate(self):
        if not self.samples:
            return None
        best = sorted(self.samples)[: max(1, len(self.samples) // 4)]
        offs = sorted(o for _, o, _ in best)
        return {"offset_ns": int(offs[len(offs) // 2]), "uncertainty_ns": int(best[0][0] // 2),
                "samples": len(self.samples)}

    def to_host_unix_ns(self, relay_clock_ns: int) -> int:
        est = self.estimate()
        return int(relay_clock_ns - est["offset_ns"]) if est else int(relay_clock_ns)


async def relay_clock_loop(engine, relay_url, every_s: float = 30.0, probes: int = 6):
    """Keep ``engine.relay_clock`` fresh: a burst of probes every ``every_s``."""
    import urllib.request
    url = relay_url.rstrip("/") + "/api/v1/time"

    # Cloudflare answers Python's default "Python-urllib" agent with 403, which
    # left the offset unmeasured; a named client gets through.
    request = urllib.request.Request(url, headers={"User-Agent": "cpg44-sensor-hub/1.0", "Accept": "application/json"})

    def probe():
        t0 = time.time_ns()
        with urllib.request.urlopen(request, timeout=3) as r:
            body = json.loads(r.read())
        return t0, int(body["t_ns"]), time.time_ns()

    while True:
        for _ in range(probes):
            try:
                t0, ts, t1 = await asyncio.to_thread(probe)
                engine.relay_clock.add(t0, ts, t1)
            except Exception:
                pass
            await asyncio.sleep(0.2)
        est = engine.relay_clock.estimate()
        if est:
            async with engine.lock:
                engine.state["relay_clock_offset"] = {
                    "offset_ms": round(est["offset_ns"] / 1e6, 1),
                    "uncertainty_ms": round(est["uncertainty_ns"] / 1e6, 1), "samples": est["samples"]}
        await asyncio.sleep(every_s)


class ClockMapper:
    def __init__(self):
        self.points = deque(maxlen=80)
        self.slope = 1000.0  # host ns / device us
        self.offset = None
        self.last_rtt_ns = None
        self.best_rtt_ns = None
        self.ppm = None

    def reset(self):
        self.points.clear()
        self.slope = 1000.0
        self.offset = None
        self.last_rtt_ns = None
        self.best_rtt_ns = None
        self.ppm = None

    def add_exchange(self, t1_ns: int, t2_us: int, t3_us: int, t4_ns: int):
        device_processing_ns = max(0, (t3_us - t2_us) * 1000)
        rtt_ns = max(0, (t4_ns - t1_ns) - device_processing_ns)
        device_mid_us = (t2_us + t3_us) / 2.0
        host_mid_ns = (t1_ns + t4_ns) / 2.0

        self.points.append(SyncPoint(device_mid_us, host_mid_ns, float(rtt_ns)))
        self.last_rtt_ns = rtt_ns
        self.best_rtt_ns = rtt_ns if self.best_rtt_ns is None else min(self.best_rtt_ns, rtt_ns)
        self._fit()

    def _fit(self):
        if not self.points:
            return

        pts = list(self.points)
        # Network queueing only makes RTT larger; fit from the lowest-latency half.
        pts.sort(key=lambda p: p.rtt_ns)
        keep = max(1, min(len(pts), max(4, len(pts) // 2)))
        pts = pts[:keep]

        if len(pts) == 1:
            p = pts[0]
            self.slope = 1000.0
            self.offset = p.host_ns - self.slope * p.device_us
            self.ppm = 0.0
            return

        x = np.asarray([p.device_us for p in pts], dtype=np.float64)
        y = np.asarray([p.host_ns for p in pts], dtype=np.float64)
        x0 = float(np.mean(x))
        y0 = float(np.mean(y))
        dx = x - x0
        dy = y - y0
        denom = float(np.dot(dx, dx))
        if denom <= 0:
            return

        slope = float(np.dot(dx, dy) / denom)
        # Device clock drift should be small; reject absurd fits caused by jitter.
        if 995.0 <= slope <= 1005.0:
            self.slope = slope
            self.offset = y0 - slope * x0
            self.ppm = (slope / 1000.0 - 1.0) * 1e6

    @property
    def valid(self):
        return self.offset is not None

    def map_ns(self, device_us: int):
        if self.offset is None:
            return None
        return int(self.offset + self.slope * float(device_us))

    def status(self):
        return {
            "valid": self.valid,
            "samples": len(self.points),
            "slope_ns_per_us": self.slope,
            "drift_ppm": self.ppm,
            "last_rtt_ms": None if self.last_rtt_ns is None else self.last_rtt_ns / 1e6,
            "best_rtt_ms": None if self.best_rtt_ns is None else self.best_rtt_ns / 1e6,
        }


# ============================================================================
# 6-DoF Mahony filter (roll/pitch stabilized by gravity, yaw remains relative)
# ============================================================================

class Mahony6D:
    def __init__(self, kp=1.6):
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.kp = float(kp)

    def reset(self):
        self.q[:] = [1.0, 0.0, 0.0, 0.0]

    def update(self, gyro_rads, accel_mps2, dt):
        if not (0.001 <= dt <= 0.05):
            return self.q.copy()

        q0, q1, q2, q3 = self.q
        a = np.asarray(accel_mps2, dtype=np.float64)
        norm = float(np.linalg.norm(a))

        if norm > 1e-9:
            ax, ay, az = a / norm
            vx = 2.0 * (q1 * q3 - q0 * q2)
            vy = 2.0 * (q0 * q1 + q2 * q3)
            vz = q0*q0 - q1*q1 - q2*q2 + q3*q3
            ex = ay * vz - az * vy
            ey = az * vx - ax * vz
            ez = ax * vy - ay * vx
        else:
            ex = ey = ez = 0.0

        wx, wy, wz = np.asarray(gyro_rads, dtype=np.float64) + self.kp * np.array([ex, ey, ez])
        qdot = 0.5 * np.array([
            -q1*wx - q2*wy - q3*wz,
             q0*wx + q2*wz - q3*wy,
             q0*wy - q1*wz + q3*wx,
             q0*wz + q1*wy - q2*wx,
        ])

        q = self.q + qdot * dt
        q /= max(float(np.linalg.norm(q)), 1e-12)
        self.q = q
        return q.copy()


def q_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def q_euler_deg(q):
    w, x, y, z = q
    roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp = clamp(2*(w*y - z*x), -1.0, 1.0)
    pitch = math.asin(sinp)
    yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return {
        "roll": math.degrees(roll),
        "pitch": math.degrees(pitch),
        "yaw_relative": math.degrees(yaw),
    }


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371009.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*r*math.asin(min(1.0, math.sqrt(a)))


# ============================================================================
# Broadcast raw synchronized events to downstream subscribers
# ============================================================================

class Broadcast:
    def __init__(self):
        self.subscribers = set()

    def subscribe(self):
        q = asyncio.Queue(maxsize=1000)
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q):
        self.subscribers.discard(q)

    def publish(self, item):
        for q in tuple(self.subscribers):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(item)
            except asyncio.QueueFull:
                pass


# ============================================================================
# Telemetry engine
# ============================================================================

class TelemetryEngine:
    def __init__(self, record_path=None, calibration_path=None):
        self.lock = asyncio.Lock()
        self.clock = ClockMapper()
        self.relay_clock = RelayClockOffset()
        self.raw_broadcast = Broadcast()
        self.imu_history = deque(maxlen=6000)
        self.stable_vitals = deque(maxlen=40)
        self.filter = Mahony6D()
        self.ecg = ECGProcessor()
        self.gyro_cal = GyroBias(fs=50.0)
        self.accel_cal = AccelAutoCal()
        self.gps_kf = GpsKalman()
        self.battery_hist = deque(maxlen=1000)          # 30 min of 2 s status packets
        self.player_load_total = 0.0
        self.player_load_window = deque(maxlen=3000)          # (host_s, cumulative PL)
        self._last_accel_g = None
        self._still_accel = deque(maxlen=50)
        self.calibration_path = calibration_path
        self._rates = {}

        self.current_boot_id = None
        self.last_imu_device_us = None
        self.gyro_bias = np.zeros(3, dtype=np.float64)
        self.accel_bias = np.zeros(3, dtype=np.float64)
        self.accel_scale = np.ones(3, dtype=np.float64)
        self.accel_scale_calibrated = False
        self.auto_gyro_samples = []
        self.auto_gyro_calibrated = False
        self.linear_lp = np.zeros(3, dtype=np.float64)
        self.previous_linear = np.zeros(3, dtype=np.float64)
        self.short_delta_v = np.zeros(3, dtype=np.float64)

        self.prev_gps = None
        self.distance_total_m = 0.0
        self.gps_speed_accel = None

        if calibration_path:
            cfg = json.loads(Path(calibration_path).read_text())
            self.gyro_bias = np.asarray(cfg.get("gyro_bias_rads", [0, 0, 0]), dtype=np.float64)
            self.accel_bias = np.asarray(cfg.get("accel_bias_mps2", [0, 0, 0]), dtype=np.float64)
            self.accel_scale = np.asarray(cfg.get("accel_scale", [1, 1, 1]), dtype=np.float64)
            self.auto_gyro_calibrated = True
            self.accel_scale_calibrated = "accel_scale" in cfg
            self.gyro_cal.bias, self.gyro_cal.calibrated = self.gyro_bias.copy(), True
            if self.accel_scale_calibrated:
                self.accel_cal.bias, self.accel_cal.scale, self.accel_cal.calibrated = \
                    self.accel_bias.copy(), self.accel_scale.copy(), True

        self.record_fp = open(record_path, "a", buffering=1) if record_path else None

        self.state = {
            "device": {"connected": False, "ip": None, "boot_id": None, "last_packet_mono_ns": None},
            "clock": self.clock.status(),
            "imu": {"valid": False},
            "gps": {"rx": False, "fix": False},
            "navigation": {"speed_mps": None, "speed_source": None, "course_deg": None},
            "vitals": {
                "rolling": {"valid": False, "reason": "collecting"},
                "live": {"valid": False, "bpm": None, "reason": "collecting", "source": "ecg"},
                "stable_15s": {"valid": False, "reason": "collecting"},
                "medical_calibrated": False,
            },
            "ecg": {"lead_off": None, "fs": None},
            "player_id": None,
            "match_id": "live",
        }

    def _record(self, packet):
        if self.record_fp:
            self.record_fp.write(json.dumps(packet, separators=(",", ":")) + "\n")

    def _handle_boot(self, boot_id):
        if boot_id is None:
            return
        if self.current_boot_id is None:
            self.current_boot_id = boot_id
        elif boot_id != self.current_boot_id:
            self.current_boot_id = boot_id
            self.clock.reset()
            self.ecg = ECGProcessor()
            self.imu_history.clear()
            self.stable_vitals.clear()
            self.last_imu_device_us = None
            self.filter.reset()

    def map_packet_timestamp(self, packet, rx_mono_ns, rx_unix_ns):
        source_unix_ns = packet.get("source_unix_ns")
        if source_unix_ns is not None:
            # The vest stamps samples with SNTP time; this machine's clock can
            # be off by hundreds of ms (WSL after sleep), so map the stamp
            # through the measured offset to the relay before comparing it
            # with video frames stamped on this machine.
            source_unix_ns = self.relay_clock.to_host_unix_ns(int(source_unix_ns))
            raw_diff_ns = rx_unix_ns - source_unix_ns
            packet["rx_host_mono_ns"] = int(rx_mono_ns)
            packet["rx_host_unix_ns"] = int(rx_unix_ns)
            packet["host_mono_ns"] = int(rx_mono_ns - raw_diff_ns)
            packet["host_unix_ns"] = source_unix_ns
            clock = packet.get("relay_clock") or {}
            est = self.relay_clock.estimate()
            slack = est["uncertainty_ns"] if est else 0
            packet["clock_synced"] = bool(clock.get("valid") and raw_diff_ns >= -slack)
            packet["clock_offset_ms"] = round(est["offset_ns"] / 1e6, 1) if est else None
            packet["transport_delay_ms"] = round(raw_diff_ns / 1e6, 3)
            return packet

        device_us = packet.get("device_us")
        mapped_ns = self.clock.map_ns(int(device_us)) if device_us is not None else None
        packet["rx_host_mono_ns"] = int(rx_mono_ns)
        packet["rx_host_unix_ns"] = int(rx_unix_ns)
        packet["host_mono_ns"] = int(mapped_ns if mapped_ns is not None else rx_mono_ns)
        packet["host_unix_ns"] = mono_to_unix_ns(packet["host_mono_ns"])
        packet["clock_synced"] = bool(mapped_ns is not None)
        if mapped_ns is not None:
            packet["transport_delay_ms"] = round((rx_mono_ns - mapped_ns) / 1e6, 3)
        else:
            packet["transport_delay_ms"] = None
        return packet

    async def ingest(self, packet, rx_mono_ns, rx_unix_ns):
        self._handle_boot(packet.get("boot_id"))
        packet = self.map_packet_timestamp(packet, rx_mono_ns, rx_unix_ns)

        async with self.lock:
            self.state["device"]["connected"] = True
            self.state["device"]["last_packet_mono_ns"] = rx_mono_ns
            if packet.get("boot_id") is not None:
                self.state["device"]["boot_id"] = packet.get("boot_id")
            if packet.get("relay_seq") is not None:
                self.state["device"]["last_relay_seq"] = packet.get("relay_seq")
            if packet.get("player_id") is not None:
                self.state["player_id"] = int(packet["player_id"])
            if packet.get("match_id"):
                self.state["match_id"] = str(packet["match_id"])

            t = packet.get("t")
            if t in ("imu", "gps", "ecg"):
                self._count_rate(t, int(packet["host_mono_ns"]))
            if t == "hello":
                self.state["device"].update({k: v for k, v in packet.items() if k not in {"t"}})
            elif t == "imu":
                self._process_imu(packet)
            elif t == "ecg":
                self.ecg.add_block(int(packet["host_mono_ns"]) / 1e9, float(packet.get("fs") or 250),
                                   packet.get("v") or [], bool(packet.get("lead_off")))
                self.state["ecg"] = {"lead_off": bool(packet.get("lead_off")), "fs": packet.get("fs")}
            elif t == "gps":
                self._process_gps(packet)
            elif t == "status":
                self.state["device"].update({
                    "rssi_dbm": packet.get("rssi_dbm"),
                    "heap": packet.get("heap"),
                    "mpu6050": packet.get("mpu6050"),
                    "magnetometer": packet.get("magnetometer"),
                    "barometer": packet.get("barometer"),
                    "ecg": packet.get("ecg"),
                    "gps_rx": packet.get("gps_rx"),
                    "dropped_samples": packet.get("dropped_samples"),
                    "queued_samples": packet.get("queued_samples"),
                    "imu_rate_hz": packet.get("imu_rate_hz"),
                    "imu_accel_range_g": packet.get("imu_accel_range_g"),
                    "imu_gyro_range_dps": packet.get("imu_gyro_range_dps"),
                    "ecg_rate_hz": packet.get("ecg_rate_hz"),
                    "clock_anchor_age_ms": packet.get("clock_anchor_age_ms"),
                    "gps_rate_hz": packet.get("gps_rate_hz"),
                    "battery": battery_state(self.battery_hist, rx_mono_ns / 1e9, packet.get("battery_pct"),
                                             packet.get("battery_v")),
                })

            relay_clock = packet.get("relay_clock")
            self.state["clock"] = dict(relay_clock) if isinstance(relay_clock, dict) else self.clock.status()

        self._record(packet)
        self.raw_broadcast.publish(packet)

    def _count_rate(self, kind, acquired_ns):
        """Delivered samples per second over the last 10 s of acquisition time.

        Measured on acquisition time, not arrival time, because the vest sends
        in batches; the shortfall against the configured rate is the share of
        samples lost before they reached this machine."""
        q = self._rates.setdefault(kind, deque(maxlen=1200))
        q.append(acquired_ns)
        while q and acquired_ns - q[0] > 10_000_000_000:
            q.popleft()
        if len(q) >= 3 and q[-1] > q[0]:
            self.state["device"].setdefault("rates_hz", {})[kind] = round((len(q) - 1) / ((q[-1] - q[0]) / 1e9), 1)

    def _player_load_rate(self):
        w = self.player_load_window
        if len(w) < 2 or w[-1][0] - w[0][0] < 10:
            return None
        t1, p1 = w[-1]
        k = next((i for i, (t, _) in enumerate(w) if t1 - t <= 60), 0)
        t0, p0 = w[k]
        return round((p1 - p0) / max(t1 - t0, 1e-6) * 60, 2)

    def _save_calibration(self):
        if not self.calibration_path:
            return
        try:
            path = Path(self.calibration_path)
            cfg = json.loads(path.read_text()) if path.is_file() else {}
            cfg.update({"gyro_bias_rads": [float(x) for x in self.gyro_cal.bias]})
            if self.accel_cal.calibrated:
                cfg.update({"accel_bias_mps2": [float(x) for x in self.accel_cal.bias],
                            "accel_scale": [float(x) for x in self.accel_cal.scale],
                            "accel_fit_residual_g": self.accel_cal.residual_g})
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cfg, indent=2))
            tmp.replace(path)
        except OSError:
            pass

    def _process_imu(self, p):
        a_raw = np.asarray(p["a"], dtype=np.float64)
        g_raw = np.asarray(p["g"], dtype=np.float64)
        device_us = int(p["device_us"])

        if self.last_imu_device_us is None:
            dt = 0.02                                      # 50 Hz IMU
        else:
            dt = (device_us - self.last_imu_device_us) / 1e6
            if not (0.002 <= dt <= 0.1):
                dt = 0.02
        self.last_imu_device_us = device_us

        # Gyro bias from genuinely still windows only (the old code kept
        # sampling once it had started, so motion leaked into the bias), and
        # accelerometer offset/scale from still poses in different orientations.
        self._still_accel.append(a_raw.copy())
        accel = self.accel_cal.apply(a_raw)
        # Until the accelerometer's own scale is fitted, "gravity only" is judged
        # loosely (a 4 % scale error put this vest at 9.4 m/s2 and no window ever
        # counted as still, so neither sensor could calibrate).
        if self.gyro_cal.update(g_raw, accel, accel_tolerance=0.35 if self.accel_cal.calibrated else 1.5):
            if self.accel_cal.add_still_mean(np.mean(np.asarray(self._still_accel), axis=0)):
                self._save_calibration()
        self.gyro_bias = self.gyro_cal.bias
        self.auto_gyro_calibrated = self.gyro_cal.calibrated
        self.accel_bias, self.accel_scale = self.accel_cal.bias, self.accel_cal.scale
        self.accel_scale_calibrated = self.accel_cal.calibrated
        gyro = g_raw - self.gyro_bias
        accel_g = accel / G
        if self._last_accel_g is not None:
            self.player_load_total += float(np.linalg.norm(accel_g - self._last_accel_g)) / 100.0
        self._last_accel_g = accel_g
        self.player_load_window.append((int(p["host_mono_ns"]) / 1e9, self.player_load_total))

        q = self.filter.update(gyro, accel, dt)
        R = q_rotation_matrix(q)
        accel_world = R @ accel
        linear_world = accel_world - np.array([0.0, 0.0, G])

        # Gentle LPF for the public motion estimate; raw values are still preserved.
        alpha = 0.22
        self.linear_lp = (1.0 - alpha) * self.linear_lp + alpha * linear_world
        linear_mag = float(np.linalg.norm(self.linear_lp))
        gyro_mag = float(np.linalg.norm(gyro))
        dyn_mag = abs(float(np.linalg.norm(accel)) - G)
        jerk = (self.linear_lp - self.previous_linear) / max(dt, 1e-3)
        self.previous_linear = self.linear_lp.copy()

        stationary = linear_mag < 0.60 and gyro_mag < 0.25

        row = {
            "host_mono_ns": int(p["host_mono_ns"]),
            "dynamic_accel_mag": dyn_mag,
            "gyro_mag": gyro_mag,
            "linear_world": self.linear_lp.copy(),
            "accel_body": accel.copy(),
        }
        self.imu_history.append(row)

        self.state["imu"] = {
            "valid": True,
            "timestamp": {
                "device_us": device_us,
                "host_mono_ns": int(p["host_mono_ns"]),
                "host_unix_ns": int(p.get("rx_host_unix_ns") or p["host_unix_ns"]),
                "clock_synced": bool(p["clock_synced"]),
                "transport_delay_ms": p.get("transport_delay_ms"),
            },
            "calibrated_gyro_bias": self.auto_gyro_calibrated,
            "accel_body_mps2": [float(x) for x in accel],
            "gyro_body_rads": [float(x) for x in gyro],
            "accel_magnitude_mps2": float(np.linalg.norm(accel)),
            "dynamic_accel_magnitude_mps2": dyn_mag,
            "linear_accel_world_mps2": [float(x) for x in self.linear_lp],
            "linear_accel_magnitude_mps2": linear_mag,
            "player_load": round(self.player_load_total, 2),
            "player_load_per_min": self._player_load_rate(),
            "accel_calibrated": bool(self.accel_cal.calibrated),
            "accel_cal_poses": len(self.accel_cal.poses),
            "acceleration_g": round(float(np.linalg.norm(accel)) / G, 2),
            "jerk_world_mps3": [float(x) for x in jerk],
            "orientation_deg": q_euler_deg(q),
            "stationary": bool(stationary),
            "temperature_c": p.get("temp_c"),
            "gyro_bias_rads": [float(x) for x in self.gyro_bias],
            "yaw_note": "relative only; MPU6050 has no magnetometer",
        }

    def _process_gps(self, p):
        out = dict(p)
        fix = bool(p.get("fix"))
        speed = p.get("speed_mps")
        course = p.get("course_deg")
        now_ns = int(p["host_mono_ns"])

        kf = None
        if fix and p.get("lat") is not None and p.get("lon") is not None:
            kf = self.gps_kf.update(now_ns / 1e9, float(p["lat"]), float(p["lon"]),
                                    None if speed is None else float(speed),
                                    None if course is None else float(course),
                                    None if p.get("hdop") is None else float(p["hdop"]), p.get("sat"))
        if kf is not None:
            self.distance_total_m = kf["distance_m"]
            out["filtered"] = kf
        if fix and speed is not None:
            speed = float(speed)
            if self.prev_gps is not None:
                dt = (now_ns - self.prev_gps["t_ns"]) / 1e9
                if 0.2 <= dt <= 3.0:
                    raw_accel = (speed - self.prev_gps["speed"]) / dt
                    self.gps_speed_accel = raw_accel if self.gps_speed_accel is None else 0.75*self.gps_speed_accel + 0.25*raw_accel


            self.prev_gps = {
                "t_ns": now_ns,
                "speed": speed,
                "lat": None if p.get("lat") is None else float(p["lat"]),
                "lon": None if p.get("lon") is None else float(p["lon"]),
            }

        out["speed_accel_mps2"] = self.gps_speed_accel
        out["distance_total_m"] = self.distance_total_m
        self.state["gps"] = out
        self.state["navigation"] = {
            "speed_mps": float(speed) if fix and speed is not None else None,
            "speed_kmh": float(speed) * 3.6 if fix and speed is not None else None,
            "speed_source": "gps" if fix and speed is not None else None,
            "course_deg": float(course) if fix and course is not None else None,
            "distance_total_m": self.distance_total_m,
            "gps_speed_accel_mps2": self.gps_speed_accel,
            "note": "absolute speed/course from GPS; IMU delta-v is short-term only",
        }

    def _motion_quality(self, start_ns, end_ns):
        rows = [r for r in self.imu_history if start_ns <= r["host_mono_ns"] <= end_ns]
        if len(rows) < 50:
            return {"valid": False, "samples": len(rows)}
        da = np.asarray([r["dynamic_accel_mag"] for r in rows], dtype=np.float64)
        gg = np.asarray([r["gyro_mag"] for r in rows], dtype=np.float64)
        return {
            "valid": True,
            "samples": len(rows),
            "dynamic_accel_rms_mps2": float(np.sqrt(np.mean(da*da))),
            "dynamic_accel_p90_mps2": float(np.percentile(da, 90)),
            "gyro_rms_rads": float(np.sqrt(np.mean(gg*gg))),
        }

    def calculate_vitals(self, now_ns=None):
        """Heart rate from the chest ECG. ``now_ns`` is on the host monotonic clock."""
        now = (time.perf_counter_ns() if now_ns is None else now_ns) / 1e9
        still = bool((self.state.get("imu") or {}).get("stationary"))
        return self.ecg.estimate(now=now, still=still)

    async def update_vitals(self):
        async with self.lock:
            now_ns = time.perf_counter_ns()
            result = self.calculate_vitals(now_ns)
            if self.ecg.t:
                result["window_end_host_mono_ns"] = int(self.ecg.last_t * 1e9)
            self.state["vitals"]["rolling"] = result
            if result.get("valid"):
                self.stable_vitals.append({"t_ns": now_ns, "bpm": float(result["bpm"]),
                                           "quality": float(result.get("quality") or 0.0)})
            cutoff = now_ns - int(HR_STABLE_SECONDS * 1e9)
            while self.stable_vitals and self.stable_vitals[0]["t_ns"] < cutoff:
                self.stable_vitals.popleft()
            self.state["vitals"]["live"] = {
                "valid": bool(result.get("valid")),
                "bpm": result.get("bpm") if result.get("valid") else None,
                "rmssd_ms": result.get("rmssd_ms"),
                "quality": result.get("quality"),
                "reason": "ok" if result.get("valid") else result.get("reason"),
                "stale": not bool(result.get("valid")),
                "source": "ecg",
            }
            if len(self.stable_vitals) >= 4:
                bpm = np.asarray([x["bpm"] for x in self.stable_vitals])
                self.state["vitals"]["stable_15s"] = {
                    "valid": True,
                    "bpm": round(float(np.median(bpm)), 1),
                    "quality": round(float(np.median([x["quality"] for x in self.stable_vitals])), 2),
                    "n": len(self.stable_vitals),
                    "window_seconds": HR_STABLE_SECONDS,
                    "computed_host_mono_ns": now_ns,
                    "computed_unix_ns": mono_to_unix_ns(now_ns),
                    "window_end_unix_ns": mono_to_unix_ns(result.get("window_end_host_mono_ns") or now_ns),
                    "medical_calibrated": False,
                }
            else:
                self.state["vitals"]["stable_15s"] = {"valid": False, "reason": "need_more_valid_windows",
                                                     "n": len(self.stable_vitals), "medical_calibrated": False}

    async def ecg_trace(self, seconds: float = 4.0):
        async with self.lock:
            return self.ecg.trace(seconds)

    async def snapshot(self):
        async with self.lock:
            state = json.loads(json.dumps(self.state))
        last = state["device"].get("last_packet_mono_ns")
        state["device"]["connected"] = bool(last and (time.perf_counter_ns() - last) < 15_000_000_000)
        state["server"] = {
            "host_mono_ns": time.perf_counter_ns(),
            "host_unix_ns": time.time_ns(),
            "build": BUILD_ID,
        }
        imu = state.get("imu") or {}
        if imu:
            imu["earth_acceleration_mps2"] = imu.get("linear_accel_world_mps2")
            imu["earth_acceleration_magnitude_mps2"] = imu.get("linear_accel_magnitude_mps2")
            imu["gyro_corrected_rads"] = imu.get("gyro_body_rads")
            imu["ahrs"] = {
                "startup": not bool(imu.get("calibrated_gyro_bias")),
                "accelerometer_ignored": False,
            }
            imu["accel_scale_calibrated"] = self.accel_scale_calibrated
            state["imu"] = imu
        vit = state.get("vitals") or {}
        stable = vit.get("stable_15s") or {}
        if stable.get("valid"):
            vit["seconds_to_next_stable"] = 0.0
        else:
            n = int(stable.get("n") or (vit.get("stable_15s") or {}).get("n") or 0)
            vit["seconds_to_next_stable"] = round(max(0.0, HR_STABLE_SECONDS - n), 1)
        state["vitals"] = vit
        nav = state.get("navigation") or {}
        imu = state.get("imu") or {}
        gps = state.get("gps") or {}
        lin = imu.get("linear_accel_world_mps2") or [None, None, None]
        course = nav.get("course_deg")
        gps_speed = nav.get("speed_mps") if bool(gps.get("fix")) else None
        # Speed only from GNSS Doppler (or the camera elsewhere); an accelerometer
        # alone cannot measure speed, so none is shown without a fix.
        active_speed = gps_speed
        speed_src = "wearable GPS" if gps_speed is not None else None
        state["kinematics"] = {
            "speed_mps": active_speed,
            "speed_kmh": round(active_speed * 3.6, 2) if active_speed is not None else None,
            "speed_source": speed_src,
            "course_deg": course,
            "heading_cardinal": _cardinal(course),
            "gps_accel_mps2": nav.get("gps_speed_accel_mps2"),
            "linear_accel_mps2": lin,
            "linear_accel_mag_mps2": imu.get("linear_accel_magnitude_mps2"),
            "body_accel_mps2": imu.get("accel_body_mps2"),
            "jerk_mag_mps3": None if not imu.get("jerk_world_mps3") else float(np.linalg.norm(imu["jerk_world_mps3"])),
            "roll_deg": (imu.get("orientation_deg") or {}).get("roll"),
            "pitch_deg": (imu.get("orientation_deg") or {}).get("pitch"),
            "yaw_relative_deg": (imu.get("orientation_deg") or {}).get("yaw_relative"),
            "stationary": imu.get("stationary"),
            "gps_fix": bool(gps.get("fix")),
        }
        return state


async def vitals_loop(engine):
    while True:
        await asyncio.sleep(1.0)
        try:
            await engine.update_vitals()
        except Exception:
            pass


def relay_envelope_to_packet(envelope):
    """Convert one validated relay envelope back to the raw sensor packet."""
    if not isinstance(envelope, dict):
        return None
    sample_type = envelope.get("sample_type")
    payload = envelope.get("payload")
    if sample_type not in {"imu", "ecg", "gps", "status"} or not isinstance(payload, dict):
        return None
    packet = dict(payload)
    packet.update({
        "t": sample_type,
        "boot_id": envelope.get("device_boot_id"),
        "seq": envelope.get("source_seq"),
        "source_unix_ns": envelope.get("source_timestamp_ns"),
        "relay_received_ns": envelope.get("relay_received_ns"),
        "relay_seq": envelope.get("relay_seq"),
        "relay_clock": envelope.get("clock") or {"valid": False},
        "player_id": envelope.get("player_id"),
        "match_id": envelope.get("match_id"),
    })
    return packet


def _relay_websocket_url(relay_url, after_seq, match_id, player_id):
    parsed = urlparse(relay_url)
    query = urlencode({
        "after_seq": int(after_seq),
        "match_id": match_id,
        "player_id": int(player_id),
    })
    scheme = "ws" if parsed.scheme == "http" else "wss"          # plain only for a loopback test relay
    return urlunparse((scheme, parsed.netloc, "/ws/sensors", "", query, ""))


async def relay_reader(engine, relay_url, token, player_id, match_id):
    """Consume replay-safe raw telemetry from the public relay."""
    after_seq = 0
    relay_instance_id = None
    backoff = 0.5
    cache_gaps = 0
    rejected = 0

    async def process(envelope):
        nonlocal after_seq, rejected
        try:
            sequence = int(envelope.get("relay_seq"))
        except (AttributeError, TypeError, ValueError):
            rejected += 1
            return
        if sequence <= after_seq:
            return
        packet = relay_envelope_to_packet(envelope)
        after_seq = sequence
        if packet is None:
            rejected += 1
            return
        rx_mono_ns = time.perf_counter_ns()
        rx_unix_ns = time.time_ns()
        await engine.ingest(packet, rx_mono_ns, rx_unix_ns)

    while True:
        uri = _relay_websocket_url(relay_url, after_seq, match_id, player_id)
        try:
            async with connect(
                uri,
                additional_headers={"X-Auth": token},
                open_timeout=8,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
                # Legacy relays send their entire 4 MiB cache in one frame.
                # One queued frame bounds memory while allowing hub-first rollout.
                max_size=8 * 1024 * 1024,
                max_queue=1,
                compression=None,
            ) as websocket:
                async with engine.lock:
                    engine.state["relay"] = {
                        "configured": True,
                        "connected": True,
                        "endpoint": relay_url,
                        "last_relay_seq": after_seq,
                        "cache_gaps": cache_gaps,
                        "rejected": rejected,
                    }
                backoff = 0.5

                async for raw_message in websocket:
                    message = json.loads(raw_message)
                    message_instance = message.get("relay_instance_id")
                    if message.get("type") == "replay":
                        replay = message.get("data") or {}
                        message_instance = replay.get("relay_instance_id") or message_instance
                        if relay_instance_id is not None and message_instance != relay_instance_id:
                            after_seq = 0
                        relay_instance_id = message_instance
                        cache_gaps += int(bool(replay.get("cache_gap")))
                        items = replay.get("items") or []
                        for index, envelope in enumerate(items):
                            await process(envelope)
                            if index % 64 == 63:
                                await asyncio.sleep(0)
                        if replay.get("has_more") or (len(items) >= 5000 and replay.get("next_seq", after_seq) < replay.get("latest_seq", after_seq)):
                            # Reconnect at the last consumed item, never at an
                            # unseen server watermark. The cache bounds recovery.
                            break
                    elif message.get("type") == "wearable_sample":
                        if relay_instance_id is not None and message_instance != relay_instance_id:
                            after_seq = 0
                        relay_instance_id = message_instance
                        await process(message.get("data") or {})
                    elif message.get("type") == "wearable_batch":
                        if relay_instance_id is not None and message_instance != relay_instance_id:
                            after_seq = 0
                        relay_instance_id = message_instance
                        for envelope in message.get("data") or []:
                            await process(envelope)
                    elif message.get("type") == "stream_gap":
                        # Keep the last consumed cursor and recover from replay.
                        break

                    async with engine.lock:
                        engine.state["relay"] = {
                            "configured": True,
                            "connected": True,
                            "endpoint": relay_url,
                            "relay_instance_id": relay_instance_id,
                            "last_relay_seq": after_seq,
                            "cache_gaps": cache_gaps,
                            "rejected": rejected,
                            "last_received_unix_ns": time.time_ns(),
                        }
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            async with engine.lock:
                engine.state["relay"] = {
                    "configured": True,
                    "connected": False,
                    "endpoint": relay_url,
                    "last_relay_seq": after_seq,
                    "cache_gaps": cache_gaps,
                    "rejected": rejected,
                    "reason": str(exc),
                }
                engine.state["device"]["connected"] = False
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 15.0)


# ============================================================================
# FastAPI
# ============================================================================

ARGS = None
ENGINE = None


@asynccontextmanager
async def lifespan(app):
    tasks = [
        asyncio.create_task(vitals_loop(ENGINE)),
        asyncio.create_task(relay_reader(
            ENGINE, ARGS.relay_url, ARGS.relay_token,
            ARGS.player_id, ARGS.match_id,
        )),
        asyncio.create_task(relay_clock_loop(ENGINE, ARGS.relay_url)),
    ]
    yield
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(title="Synchronized Wearable Sensor Hub", lifespan=lifespan)

BUILD_ID = "sensor-hub-ecg-2026-11-19"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get(
            "CPG44_HUB_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)



@app.get("/", include_in_schema=False)
async def index():
    """The sensor processor has no screen of its own; the app's Vest readings page shows it."""
    return {"service": "cpg44 sensor hub", "build": BUILD_ID,
            "endpoints": ["/health", "/api/latest", "/api/vitals", "/api/ecg", "/api/stream", "/api/raw-stream"]}


@app.get("/debug/routes", include_in_schema=False)
async def debug_routes():
    return {
        "build": BUILD_ID,
        "routes": sorted(
            getattr(route, "path", "")
            for route in app.routes
            if getattr(route, "path", None)
        ),
    }



@app.get("/health")
async def health():
    s = await ENGINE.snapshot()
    return {
        "ok": True,
        "device_connected": s["device"]["connected"],
        "clock_synced": s["clock"]["valid"],
        "relay_connected": bool((s.get("relay") or {}).get("connected")),
        "build": BUILD_ID,
    }


@app.get("/api/v1/observations/wearable")
async def wearable_observation():
    """Capstone join contract: match_id + global_player_id + timestamp."""
    from .hub_bridge import sample_to_observation, snapshot_to_sample

    s = await ENGINE.snapshot()
    player_id = int((s.get("player_id") or 7))
    sample = snapshot_to_sample(s, player_id)
    if sample is None:
        return JSONResponse({"ok": False, "reason": "no_sample_yet"}, status_code=404)
    return JSONResponse(sample_to_observation(sample, match_id=s.get("match_id") or "live"))


@app.get("/api/latest")
async def latest():
    return JSONResponse(await ENGINE.snapshot())


@app.get("/api/ecg")
async def ecg_trace(seconds: float = 4.0):
    """The last few seconds of the chest ECG with detected beats, for display."""
    return JSONResponse(await ENGINE.ecg_trace(min(max(seconds, 1.0), 10.0)))


@app.get("/api/vitals")
async def vitals():
    s = await ENGINE.snapshot()
    return JSONResponse(s["vitals"])


@app.get("/api/stream")
async def processed_stream():
    """Processed state as NDJSON at 10 Hz."""
    async def generate():
        while True:
            state = await ENGINE.snapshot()
            yield json.dumps(state, separators=(",", ":")) + "\n"
            await asyncio.sleep(0.1)
    return StreamingResponse(generate(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"})


@app.get("/api/raw-stream")
async def raw_stream():
    """Every synchronized raw sensor packet as NDJSON."""
    q = ENGINE.raw_broadcast.subscribe()

    async def generate():
        try:
            while True:
                item = await q.get()
                yield json.dumps(item, separators=(",", ":")) + "\n"
        finally:
            ENGINE.raw_broadcast.unsubscribe(q)

    return StreamingResponse(generate(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"})


# ============================================================================
# Main
# ============================================================================

def main():
    global ARGS, ENGINE

    parser = argparse.ArgumentParser()
    parser.add_argument("--http-port", type=int, default=8081)
    parser.add_argument("--record", default=None, help="Optional synchronized raw NDJSON output file")
    parser.add_argument("--calibration", default=None, help="Optional IMU calibration JSON")
    parser.add_argument("--player-id", type=int, default=7, help="Roster id fused with vision tracks")
    parser.add_argument("--match-id", default="live")
    parser.add_argument("--relay-url", default=os.environ.get("CPG44_RELAY_URL", ""),
                        help="Required relay origin: https://cpg44.nivaspms.com")
    parser.add_argument("--relay-token", default=os.environ.get("CPG44_RELAY_TOKEN", ""),
                        help="X-Auth token for the VPS relay (prefer environment variable)")
    parser.add_argument("--allow-local-relay", action="store_true",
                        help="testing only: accept a relay on http://127.0.0.1")
    ARGS = parser.parse_args()

    local = ARGS.allow_local_relay and ARGS.relay_url.startswith("http://127.0.0.1:")
    if local and len(ARGS.relay_token) < 32:
        raise SystemExit("CPG44_RELAY_TOKEN must contain at least 32 characters")
    if not ARGS.relay_url:
        raise SystemExit("CPG44_RELAY_URL is required and must be https://cpg44.nivaspms.com")
    if ARGS.relay_url and not local:
        from urllib.parse import urlparse

        parsed = urlparse(ARGS.relay_url)
        host = parsed.hostname or ""
        if parsed.scheme != "https":
            raise SystemExit("The remote relay URL must use HTTPS")
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            raise SystemExit("Use the relay DNS hostname, not a direct IP address")
        if (
            host.lower() != "cpg44.nivaspms.com"
            or parsed.port not in (None, 443)
            or parsed.path not in ("", "/")
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            raise SystemExit("The relay origin must be exactly https://cpg44.nivaspms.com")
        if len(ARGS.relay_token) < 32:
            raise SystemExit("CPG44_RELAY_TOKEN must contain at least 32 characters")

    ENGINE = TelemetryEngine(
        record_path=ARGS.record,
        calibration_path=ARGS.calibration,
    )
    ENGINE.state["player_id"] = ARGS.player_id
    ENGINE.state["match_id"] = ARGS.match_id

    print(f"\n=== {BUILD_ID} ===")
    print(f"Sensor hub API: http://127.0.0.1:{ARGS.http_port}/api/latest (shown in the app under Vest readings)")
    print(f"Debug:     http://127.0.0.1:{ARGS.http_port}/debug/routes")
    print("Registered routes:")
    for route in sorted(
        getattr(r, "path", "")
        for r in app.routes
        if getattr(r, "path", None)
    ):
        print("  ", route)
    print()

    uvicorn.run(app, host="0.0.0.0", port=ARGS.http_port, log_level="info")


if __name__ == "__main__":
    main()
