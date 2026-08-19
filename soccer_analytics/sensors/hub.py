#!/usr/bin/env python3
"""
ESP32-S3 wearable sensor hub.

- Subscribes to the ESP32 raw TCP NDJSON stream.
- NTP-style bidirectional clock synchronization maps ESP32 device_us into the
  host time.perf_counter_ns() timeline for later CV fusion.
- Preserves raw packets with acquisition timestamps.
- Processes MAX30102 PPG on the host using a Python port of the Maxim/SparkFun
  reference algorithm's peak/AC/DC ratio method, with float evaluation of the
  reference SpO2 curve (avoiding MCU integer/table limitations).
- Adds signal-quality and motion-artifact gates.
- Processes MPU6050 acceleration/gyro, auto-calibrates gyro bias while still,
  estimates 6-DoF orientation, removes gravity, computes jerk and short-term
  delta-v. Absolute speed comes from GPS, not long-term IMU integration.
- Exposes latest JSON plus processed and raw streaming HTTP endpoints.
- Optionally records synchronized NDJSON for later CV/sensor fusion.
"""

import argparse
import asyncio
import json
import math
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt, welch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
import uvicorn

G = 9.80665
PPG_FS = 25.0
PPG_REF_N = 150                # 6 s at 25 Hz — enough peaks for robust IBI
PPG_STABLE_SECONDS = 12.0
MAXIM_RATIO_MIN = 2
MAXIM_RATIO_MAX = 184


def c_div(a: int, b: int) -> int:
    """C/C++ integer division semantics: truncate toward zero."""
    if b == 0:
        raise ZeroDivisionError
    sign = -1 if (a < 0) ^ (b < 0) else 1
    return sign * (abs(a) // abs(b))


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
# Maxim reference HR / SpO2 method (host-side port)
# ============================================================================

def _peaks_above_min_height(x: np.ndarray, min_height: int):
    locs = []
    i = 1
    n = len(x)
    while i < n - 1:
        if x[i] > min_height and x[i] > x[i - 1]:
            width = 1
            while i + width < n and x[i] == x[i + width]:
                width += 1
            if i + width < n and x[i] > x[i + width]:
                if len(locs) < 15:
                    locs.append(i)
                i += width + 1
            else:
                i += width
        else:
            i += 1
    return locs


def _remove_close_peaks(x: np.ndarray, locs, min_distance: int):
    # Equivalent intent to Maxim: keep stronger peaks first, enforce spacing,
    # then restore chronological order.
    ranked = sorted(locs, key=lambda i: int(x[i]), reverse=True)
    kept = []
    for loc in ranked:
        if all(abs(loc - k) > min_distance for k in kept):
            kept.append(loc)
    return sorted(kept)[:15]


def maxim_reference_hr_spo2(ir_values, red_values, fs=PPG_FS):
    """
    Python implementation of the Maxim/SparkFun reference structure.

    The original source uses an integer lookup table for SpO2 because it targets
    small MCUs. Here we evaluate the reference polynomial in float on the host:
      SpO2 = -45.060*R^2 + 30.354*R + 94.845
    where R is the median ratio-of-ratios.
    """
    if len(ir_values) < PPG_REF_N or len(red_values) < PPG_REF_N:
        return {"hr_valid": False, "spo2_valid": False, "reason": "collecting"}

    ir = np.asarray(ir_values[-PPG_REF_N:], dtype=np.int64)
    red = np.asarray(red_values[-PPG_REF_N:], dtype=np.int64)

    # Reference code: DC removal, inversion, 4-point moving average.
    ir_mean = int(np.sum(ir) // PPG_REF_N)
    x = -(ir - ir_mean)
    x = x.astype(np.int64, copy=True)
    for k in range(PPG_REF_N - 4):
        x[k] = int((x[k] + x[k + 1] + x[k + 2] + x[k + 3]) // 4)

    threshold = int(np.sum(x) // PPG_REF_N)
    threshold = clamp(threshold, 30, 60)

    valleys = _peaks_above_min_height(x, threshold)
    valleys = _remove_close_peaks(x, valleys, 4)

    hr = None
    hr_valid = False
    if len(valleys) >= 2:
        interval_sum = sum(valleys[k] - valleys[k - 1] for k in range(1, len(valleys)))
        mean_interval = c_div(interval_sum, len(valleys) - 1)
        if mean_interval > 0:
            hr = (float(fs) * 60.0) / float(mean_interval)
            hr_valid = 30.0 <= hr <= 240.0

    # Reload raw values for the AC/DC ratio calculation.
    xraw = ir.astype(np.int64)
    yraw = red.astype(np.int64)
    ratios = []

    for k in range(max(0, len(valleys) - 1)):
        left = valleys[k]
        right = valleys[k + 1]
        if right - left <= 3:
            continue

        xseg = xraw[left:right]
        yseg = yraw[left:right]
        if len(xseg) == 0:
            continue

        x_rel = int(np.argmax(xseg))
        y_rel = int(np.argmax(yseg))
        x_dc_idx = left + x_rel
        y_dc_idx = left + y_rel
        x_dc_max = int(xraw[x_dc_idx])
        y_dc_max = int(yraw[y_dc_idx])

        # Red AC above a linear baseline between successive IR valleys.
        y_line = int(yraw[left]) + c_div(
            (int(yraw[right]) - int(yraw[left])) * (y_dc_idx - left),
            right - left,
        )
        y_ac = int(yraw[y_dc_idx]) - y_line

        # IR AC. Preserve the reference implementation's indexing behavior.
        x_line = int(xraw[left]) + c_div(
            (int(xraw[right]) - int(xraw[left])) * (x_dc_idx - left),
            right - left,
        )
        x_ac = int(xraw[y_dc_idx]) - x_line

        nume = (y_ac * x_dc_max) >> 7
        denom = (x_ac * y_dc_max) >> 7

        if denom > 0 and nume != 0 and len(ratios) < 5:
            ratios.append(c_div(nume * 100, denom))

    spo2 = None
    ratio_average = None
    spo2_valid = False

    if ratios:
        ratios.sort()
        middle = len(ratios) // 2
        if middle > 1:
            ratio_average = c_div(ratios[middle - 1] + ratios[middle], 2)
        else:
            ratio_average = ratios[middle]

        if MAXIM_RATIO_MIN < ratio_average < MAXIM_RATIO_MAX:
            r = ratio_average / 100.0
            spo2 = -45.060 * r * r + 30.354 * r + 94.845
            spo2_valid = math.isfinite(spo2) and 50.0 <= spo2 <= 100.5

    return {
        "hr_valid": bool(hr_valid),
        "bpm_reference": None if hr is None else round(float(hr), 3),
        "spo2_valid": bool(spo2_valid),
        "spo2_reference_pct": None if spo2 is None else round(float(spo2), 3),
        "ratio_average_x100": ratio_average,
        "ratio_count": len(ratios),
        "valleys": valleys,
    }


def _ppg_bandpass(x, fs=PPG_FS, lo=0.7, hi=4.0, order=2):
    """Zero-phase Butterworth bandpass (SciPy sosfiltfilt — prefer SOS over ba).

    Cardiac PPG lives ~0.7–4 Hz (42–240 BPM). DC / slow wander and 50/60 Hz
    junk stay out. filtfilt/sosfiltfilt apply the filter forward and backward
    so peaks are not time-shifted.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 24:
        return x - float(np.mean(x))
    sos = butter(order, [lo, hi], btype="bandpass", output="sos", fs=fs)
    return sosfiltfilt(sos, x)


def estimate_ppg_vitals(ir_values, red_values, fs=PPG_FS):
    """Host-side HR / SpO2 (standard AC/DC ratio-of-ratios + IBI).

    Replaces the jumpy MCU integer Maxim port as the *displayed* estimate.
    The Maxim routine can still be logged; this is what the GUI uses.

    HR: bandpass IR → find_peaks → median inter-beat interval.
    SpO2: R = (ACred/DCred) / (ACir/DCir) with AC = RMS of bandpass,
    then the Maxim/SparkFun quadratic (float, not the MCU lookup table).
    """
    if len(ir_values) < 75 or len(red_values) < 75:
        return {"valid": False, "reason": "collecting", "bpm": None, "spo2_estimate_pct": None}

    ir = np.asarray(ir_values, dtype=np.float64)
    red = np.asarray(red_values, dtype=np.float64)
    ir_dc = float(np.mean(ir))
    red_dc = float(np.mean(red))
    if ir_dc <= 0 or red_dc <= 0:
        return {"valid": False, "reason": "no_or_weak_contact", "bpm": None, "spo2_estimate_pct": None}

    ir_ac = _ppg_bandpass(ir, fs)
    red_ac = _ppg_bandpass(red, fs)

    # Peaks = systolic pulses on the AC IR waveform.
    min_dist = max(2, int(fs * 60.0 / 180.0))   # at most 180 BPM
    prominence = max(0.25 * float(np.std(ir_ac)), 1.0)
    peaks, _ = find_peaks(ir_ac, distance=min_dist, prominence=prominence)

    hr = None
    ibi_cv = None
    n_ibi = 0
    if len(peaks) >= 4:
        ibi = np.diff(peaks) / float(fs)
        med = float(np.median(ibi))
        if med > 0:
            ibi = ibi[np.abs(ibi - med) <= 0.35 * med]
            n_ibi = int(len(ibi))
            if n_ibi >= 2:
                mean_ibi = float(np.median(ibi))
                hr = 60.0 / mean_ibi
                ibi_cv = float(np.std(ibi) / mean_ibi) if mean_ibi > 0 else None

    ir_rms = float(np.sqrt(np.mean(ir_ac * ir_ac)))
    red_rms = float(np.sqrt(np.mean(red_ac * red_ac)))
    corr = float(np.corrcoef(ir_ac, red_ac)[0, 1]) if ir_rms > 0 and red_rms > 0 else float("nan")
    pi_ir_pct = 100.0 * ir_rms / max(ir_dc, 1.0)

    R = None
    spo2 = None
    denom = ir_rms / ir_dc
    if denom > 1e-9:
        R = (red_rms / red_dc) / denom
        spo2 = -45.060 * R * R + 30.354 * R + 94.845

    hr_ok = hr is not None and 40.0 <= hr <= 180.0 and (ibi_cv is None or ibi_cv < 0.28)
    spo2_ok = spo2 is not None and math.isfinite(spo2) and 70.0 <= spo2 <= 100.5
    optical_ok = math.isfinite(corr) and corr > 0.45 and pi_ir_pct > 0.04
    valid = bool(hr_ok and spo2_ok and optical_ok)

    reason = "ok"
    if not optical_ok:
        reason = "poor_optical_quality"
    elif not hr_ok:
        reason = "hr_peaks_unreliable"
    elif not spo2_ok:
        reason = "spo2_out_of_range"

    return {
        "valid": valid,
        "reason": reason,
        "bpm": None if hr is None else round(float(hr), 2),
        "spo2_estimate_pct": None if spo2 is None else round(float(spo2), 2),
        "r_ratio": None if R is None else round(float(R), 4),
        "peak_count": int(len(peaks)),
        "ibi_count": n_ibi,
        "ibi_cv": None if ibi_cv is None else round(ibi_cv, 4),
        "perfusion_index_ir_pct": round(pi_ir_pct, 4),
        "red_ir_correlation": None if not math.isfinite(corr) else round(corr, 4),
        "method": "bandpass+find_peaks+rms_ratio",
    }


# ============================================================================
# Clock synchronization
# ============================================================================

@dataclass
class SyncPoint:
    device_us: float
    host_ns: float
    rtt_ns: float


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
    def __init__(self, finger_ir_min=20000.0, record_path=None, calibration_path=None):
        self.lock = asyncio.Lock()
        self.clock = ClockMapper()
        self.raw_broadcast = Broadcast()
        self.ppg = deque(maxlen=2500)
        self.imu_history = deque(maxlen=6000)
        self.stable_vitals = deque(maxlen=40)
        self.filter = Mahony6D()
        self.finger_ir_min = float(finger_ir_min)
        self.disp_hr = None
        self.disp_spo2 = None

        self.current_boot_id = None
        self.last_imu_device_us = None
        self.gyro_bias = np.zeros(3, dtype=np.float64)
        self.accel_bias = np.zeros(3, dtype=np.float64)
        self.accel_scale = np.ones(3, dtype=np.float64)
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

        self.record_fp = open(record_path, "a", buffering=1) if record_path else None

        self.state = {
            "device": {"connected": False, "ip": None, "boot_id": None, "last_packet_mono_ns": None},
            "clock": self.clock.status(),
            "imu": {"valid": False},
            "gps": {"rx": False, "fix": False},
            "navigation": {"speed_mps": None, "speed_source": None, "course_deg": None},
            "vitals": {
                "rolling": {"valid": False, "reason": "collecting"},
                "live": {"valid": False, "bpm": None, "spo2_estimate_pct": None, "reason": "collecting"},
                "stable_15s": {"valid": False, "reason": "collecting"},
                "medical_calibrated": False,
            },
            "ppg": {"contact": False, "ir": None, "red": None},
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
            self.ppg.clear()
            self.imu_history.clear()
            self.stable_vitals.clear()
            self.last_imu_device_us = None
            self.filter.reset()

    def map_packet_timestamp(self, packet, rx_mono_ns, rx_unix_ns):
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

            t = packet.get("t")
            if t == "hello":
                self.state["device"].update({k: v for k, v in packet.items() if k not in {"t"}})
            elif t == "imu":
                self._process_imu(packet)
            elif t == "ppg":
                self._process_ppg(packet)
            elif t == "gps":
                self._process_gps(packet)
            elif t == "status":
                self.state["device"].update({"ip": packet.get("ip"), "rssi_dbm": packet.get("rssi_dbm"), "heap": packet.get("heap")})

            self.state["clock"] = self.clock.status()

        self._record(packet)
        self.raw_broadcast.publish(packet)

    def _process_imu(self, p):
        a_raw = np.asarray(p["a"], dtype=np.float64)
        g_raw = np.asarray(p["g"], dtype=np.float64)
        device_us = int(p["device_us"])

        if self.last_imu_device_us is None:
            dt = 0.01
        else:
            dt = (device_us - self.last_imu_device_us) / 1e6
            if not (0.002 <= dt <= 0.05):
                dt = 0.01
        self.last_imu_device_us = device_us

        # Auto gyro-bias calibration only while plausibly stationary.
        amag_raw = float(np.linalg.norm(a_raw))
        gmag_raw = float(np.linalg.norm(g_raw))
        still_for_cal = abs(amag_raw - G) < 0.30 and gmag_raw < 0.12
        if not self.auto_gyro_calibrated and still_for_cal:
            self.auto_gyro_samples.append(g_raw.copy())
            if len(self.auto_gyro_samples) >= 300:
                self.gyro_bias = np.mean(np.asarray(self.auto_gyro_samples), axis=0)
                self.auto_gyro_calibrated = True

        accel = (a_raw - self.accel_bias) * self.accel_scale
        gyro = g_raw - self.gyro_bias

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

        stationary = linear_mag < 0.18 and gyro_mag < 0.08
        # Do not integrate accel into speed — that ramps and drifts. GPS is speed.

        row = {
            "host_mono_ns": int(p["host_mono_ns"]),
            "dynamic_accel_mag": dyn_mag,
            "gyro_mag": gyro_mag,
            "linear_world": self.linear_lp.copy(),
        }
        self.imu_history.append(row)

        self.state["imu"] = {
            "valid": True,
            "timestamp": {
                "device_us": device_us,
                "host_mono_ns": int(p["host_mono_ns"]),
                "host_unix_ns": int(p["host_unix_ns"]),
                "clock_synced": bool(p["clock_synced"]),
            },
            "calibrated_gyro_bias": self.auto_gyro_calibrated,
            "accel_body_mps2": [float(x) for x in accel],
            "gyro_body_rads": [float(x) for x in gyro],
            "accel_magnitude_mps2": float(np.linalg.norm(accel)),
            "dynamic_accel_magnitude_mps2": dyn_mag,
            "linear_accel_world_mps2": [float(x) for x in self.linear_lp],
            "linear_accel_magnitude_mps2": linear_mag,
            "jerk_world_mps3": [float(x) for x in jerk],
            "orientation_deg": q_euler_deg(q),
            "stationary": bool(stationary),
            "temperature_c": p.get("temp_c"),
            "gyro_bias_rads": [float(x) for x in self.gyro_bias],
            "yaw_note": "relative only; MPU6050 has no magnetometer",
        }

    def _process_ppg(self, p):
        ir = float(p["ir"])
        red = float(p["red"])
        self.ppg.append({
            "host_mono_ns": int(p["host_mono_ns"]),
            "host_unix_ns": int(p["host_unix_ns"]),
            "device_us": int(p["device_us"]),
            "ir": ir,
            "red": red,
        })
        self.state["ppg"] = {
            "contact": ir >= self.finger_ir_min,
            "ir": ir,
            "red": red,
        }

    def _process_gps(self, p):
        out = dict(p)
        fix = bool(p.get("fix"))
        speed = p.get("speed_mps")
        course = p.get("course_deg")
        now_ns = int(p["host_mono_ns"])

        if fix and speed is not None:
            speed = float(speed)
            if self.prev_gps is not None:
                dt = (now_ns - self.prev_gps["t_ns"]) / 1e9
                if 0.2 <= dt <= 3.0:
                    raw_accel = (speed - self.prev_gps["speed"]) / dt
                    self.gps_speed_accel = raw_accel if self.gps_speed_accel is None else 0.75*self.gps_speed_accel + 0.25*raw_accel

                lat, lon = p.get("lat"), p.get("lon")
                if lat is not None and lon is not None and self.prev_gps.get("lat") is not None:
                    d = haversine_m(self.prev_gps["lat"], self.prev_gps["lon"], float(lat), float(lon))
                    hdop = p.get("hdop")
                    if d < 100.0 and (hdop is None or float(hdop) <= 5.0):
                        self.distance_total_m += d

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

    def _spectral_hr(self, ir):
        if len(ir) < 100:
            return None
        x = detrend(np.asarray(ir, dtype=np.float64), type="linear")
        f, p = welch(x, fs=PPG_FS, nperseg=min(len(x), 200))
        band = (f >= 0.58) & (f <= 3.67)  # ~35..220 BPM
        if not np.any(band):
            return None
        return float(f[band][np.argmax(p[band])] * 60.0)

    def calculate_vitals(self):
        if len(self.ppg) < 75:
            return {"valid": False, "reason": "collecting", "samples": len(self.ppg),
                    "bpm": None, "spo2_estimate_pct": None}

        rows = list(self.ppg)[-PPG_REF_N:]
        ir = np.asarray([r["ir"] for r in rows], dtype=np.float64)
        red = np.asarray([r["red"] for r in rows], dtype=np.float64)
        start_ns = rows[0]["host_mono_ns"]
        end_ns = rows[-1]["host_mono_ns"]

        mean_ir = float(np.mean(ir))
        mean_red = float(np.mean(red))
        if mean_ir < self.finger_ir_min or mean_red <= 0:
            return {
                "valid": False,
                "reason": "no_or_weak_contact",
                "bpm": None,
                "spo2_estimate_pct": None,
                "ir_mean": mean_ir,
                "red_mean": mean_red,
                "window_start_host_mono_ns": start_ns,
                "window_end_host_mono_ns": end_ns,
            }

        est = estimate_ppg_vitals(ir, red, PPG_FS)
        spectral_rows = list(self.ppg)[-min(len(self.ppg), 200):]
        spectral_hr = self._spectral_hr([r["ir"] for r in spectral_rows])
        hr = est.get("bpm")
        hr_diff = None if hr is None or spectral_hr is None else abs(float(hr) - spectral_hr)

        motion = self._motion_quality(start_ns, end_ns)
        if motion.get("valid"):
            motion_ok = motion["dynamic_accel_rms_mps2"] < 1.6 and motion["gyro_rms_rads"] < 1.2
        else:
            motion_ok = True

        agreement_ok = hr_diff is None or hr_diff <= 25.0
        valid = bool(est.get("valid") and motion_ok and agreement_ok)

        reason = est.get("reason") or "ok"
        if est.get("valid") and not motion_ok:
            reason = "motion_artifact"
        elif est.get("valid") and not agreement_ok:
            reason = "hr_estimators_disagree"

        quality_parts = []
        corr = est.get("red_ir_correlation")
        quality_parts.append(clamp((corr - 0.4) / 0.5, 0.0, 1.0) if corr is not None and math.isfinite(corr) else 0.0)
        pi = est.get("perfusion_index_ir_pct") or 0.0
        quality_parts.append(clamp(pi / 0.5, 0.0, 1.0))
        if motion.get("valid"):
            quality_parts.append(clamp(1.0 - motion["dynamic_accel_rms_mps2"] / 1.6, 0.0, 1.0))
        quality = float(np.mean(quality_parts)) if quality_parts else 0.0

        # Live display: EMA of *valid* windows only — tracks the signal, does
        # not ramp, and does not flash invalid jumps.
        if valid and hr is not None and est.get("spo2_estimate_pct") is not None:
            a = 0.40
            self.disp_hr = float(hr) if self.disp_hr is None else a * float(hr) + (1 - a) * self.disp_hr
            self.disp_spo2 = (
                float(est["spo2_estimate_pct"]) if self.disp_spo2 is None
                else a * float(est["spo2_estimate_pct"]) + (1 - a) * self.disp_spo2
            )

        return {
            "valid": valid,
            "reason": reason,
            "bpm": None if not valid else round(self.disp_hr, 1) if self.disp_hr is not None else None,
            "spo2_estimate_pct": None if not valid else (
                round(self.disp_spo2, 1) if self.disp_spo2 is not None else None),
            "bpm_raw": hr,
            "spo2_raw": est.get("spo2_estimate_pct"),
            "spectral_bpm": None if spectral_hr is None else round(spectral_hr, 2),
            "bpm_difference": None if hr_diff is None else round(hr_diff, 2),
            "r_ratio": est.get("r_ratio"),
            "peak_count": est.get("peak_count"),
            "ibi_cv": est.get("ibi_cv"),
            "method": est.get("method"),
            "ir_mean": mean_ir,
            "red_mean": mean_red,
            "perfusion_index_ir_pct": est.get("perfusion_index_ir_pct"),
            "red_ir_correlation": est.get("red_ir_correlation"),
            "quality": round(quality, 4),
            "motion": motion,
            "window_start_host_mono_ns": start_ns,
            "window_end_host_mono_ns": end_ns,
            "medical_calibrated": False,
        }

    async def update_vitals(self):
        async with self.lock:
            result = self.calculate_vitals()
            self.state["vitals"]["rolling"] = result

            now_ns = time.perf_counter_ns()
            if result.get("valid") and result.get("bpm") is not None:
                self.stable_vitals.append({
                    "t_ns": now_ns,
                    "bpm": float(result["bpm"]),
                    "spo2": float(result["spo2_estimate_pct"]),
                    "quality": float(result["quality"]),
                })

            cutoff = now_ns - int(PPG_STABLE_SECONDS * 1e9)
            while self.stable_vitals and self.stable_vitals[0]["t_ns"] < cutoff:
                self.stable_vitals.popleft()

            self.state["vitals"]["live"] = {
                "valid": self.disp_hr is not None and self.disp_spo2 is not None,
                "bpm": None if self.disp_hr is None else round(self.disp_hr, 1),
                "spo2_estimate_pct": None if self.disp_spo2 is None else round(self.disp_spo2, 1),
                "reason": result.get("reason"),
                "stale": not bool(result.get("valid")),
            }

            if len(self.stable_vitals) >= 4:
                bpm = np.asarray([x["bpm"] for x in self.stable_vitals])
                spo2 = np.asarray([x["spo2"] for x in self.stable_vitals])
                q = np.asarray([x["quality"] for x in self.stable_vitals])
                self.state["vitals"]["stable_15s"] = {
                    "valid": True,
                    "bpm": round(float(np.median(bpm)), 2),
                    "spo2_estimate_pct": round(float(np.median(spo2)), 2),
                    "quality": round(float(np.median(q)), 4),
                    "n": len(self.stable_vitals),
                    "window_seconds": PPG_STABLE_SECONDS,
                    "computed_host_mono_ns": now_ns,
                    "computed_unix_ns": mono_to_unix_ns(now_ns),
                    "medical_calibrated": False,
                }
            else:
                self.state["vitals"]["stable_15s"] = {
                    "valid": False,
                    "reason": "need_more_valid_windows",
                    "n": len(self.stable_vitals),
                    "medical_calibrated": False,
                }

    async def snapshot(self):
        async with self.lock:
            state = json.loads(json.dumps(self.state))
        last = state["device"].get("last_packet_mono_ns")
        state["device"]["connected"] = bool(last and (time.perf_counter_ns() - last) < 3_000_000_000)
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
            imu["accel_scale_calibrated"] = True
            state["imu"] = imu
        vit = state.get("vitals") or {}
        stable = vit.get("stable_15s") or {}
        if stable.get("valid"):
            vit["seconds_to_next_stable"] = 0.0
        else:
            n = int(stable.get("n") or (vit.get("stable_15s") or {}).get("n") or 0)
            vit["seconds_to_next_stable"] = round(max(0.0, PPG_STABLE_SECONDS - n), 1)
        state["vitals"] = vit
        nav = state.get("navigation") or {}
        imu = state.get("imu") or {}
        gps = state.get("gps") or {}
        lin = imu.get("linear_accel_world_mps2") or [None, None, None]
        course = nav.get("course_deg")
        state["kinematics"] = {
            "speed_mps": nav.get("speed_mps"),
            "speed_kmh": nav.get("speed_kmh"),
            "speed_source": nav.get("speed_source"),
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


# ============================================================================
# ESP32 connection + clock sync
# ============================================================================

async def sync_sender(writer, stop_event):
    sync_id = 0
    while not stop_event.is_set():
        try:
            t1 = time.perf_counter_ns()
            writer.write(f"SYNC,{sync_id},{t1}\n".encode())
            await writer.drain()
            sync_id += 1
        except Exception:
            return
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass


async def device_reader(engine, host, port):
    while True:
        writer = None
        sync_task = None
        stop_event = asyncio.Event()
        try:
            print(f"Connecting to ESP32 {host}:{port} ...")
            reader, writer = await asyncio.open_connection(host, port)
            print("ESP32 stream connected")
            sync_task = asyncio.create_task(sync_sender(writer, stop_event))

            while True:
                raw = await reader.readline()
                t4_mono_ns = time.perf_counter_ns()
                t4_unix_ns = time.time_ns()
                if not raw:
                    raise ConnectionError("ESP32 closed the stream")

                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                if line.startswith("SYNC_REPLY,"):
                    parts = line.split(",")
                    if len(parts) == 5:
                        _, _sid, t1, t2, t3 = parts
                        engine.clock.add_exchange(int(t1), int(t2), int(t3), t4_mono_ns)
                        async with engine.lock:
                            engine.state["clock"] = engine.clock.status()
                    continue

                try:
                    packet = json.loads(line)
                except json.JSONDecodeError:
                    continue

                await engine.ingest(packet, t4_mono_ns, t4_unix_ns)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print("ESP32 connection:", exc)
            async with engine.lock:
                engine.state["device"]["connected"] = False
            await asyncio.sleep(1.0)
        finally:
            stop_event.set()
            if sync_task:
                sync_task.cancel()
                await asyncio.gather(sync_task, return_exceptions=True)
            if writer:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass


async def vitals_loop(engine):
    while True:
        await asyncio.sleep(1.0)
        try:
            await engine.update_vitals()
        except Exception as exc:
            print("Vitals processor:", exc)


# ============================================================================
# FastAPI
# ============================================================================

ARGS = None
ENGINE = None


@asynccontextmanager
async def lifespan(app):
    tasks = [asyncio.create_task(vitals_loop(ENGINE))]
    if ARGS and ARGS.esp32:
        tasks.insert(0, asyncio.create_task(device_reader(ENGINE, ARGS.esp32, ARGS.stream_port)))
    else:
        print("Hub started without --esp32 (telemetry dashboard only; no device reader)")
    yield
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(title="Synchronized Wearable Sensor Hub", lifespan=lifespan)

BUILD_ID = "sensor-hub-capstone-2026-08-19-v6"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

DASHBOARD_HTML = '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width,initial-scale=1">\n<title>Wearable Telemetry</title>\n<style>\n  :root {\n    color-scheme: dark;\n    --bg:#0a0c10; --panel:#11151b; --panel2:#161b22; --line:#252c35;\n    --text:#edf1f5; --muted:#87919d; --good:#5ac47a; --warn:#d8ad59; --bad:#df6b6b;\n  }\n  *{box-sizing:border-box}\n  body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 Inter,system-ui,-apple-system,"Segoe UI",sans-serif}\n  main{width:min(1400px,calc(100% - 28px));margin:auto;padding:24px 0 48px}\n  header{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;margin-bottom:16px}\n  h1{margin:0;font-size:25px;letter-spacing:-.035em;font-weight:680}.sub{color:var(--muted);margin-top:5px;font-size:13px}\n  .status{display:flex;gap:8px;align-items:center;border:1px solid var(--line);border-radius:999px;padding:8px 11px;color:var(--muted)}\n  .dot{width:8px;height:8px;border-radius:50%;background:var(--bad)}.dot.ok{background:var(--good)}\n  .grid{display:grid;grid-template-columns:repeat(12,1fr);gap:12px}.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px;min-width:0}\n  .s2{grid-column:span 2}.s3{grid-column:span 3}.s4{grid-column:span 4}.s6{grid-column:span 6}.s8{grid-column:span 8}.s12{grid-column:span 12}\n  .label{text-transform:uppercase;letter-spacing:.08em;font-weight:650;color:var(--muted);font-size:10px;margin-bottom:9px}\n  .big{font-size:38px;line-height:1;letter-spacing:-.055em;font-weight:680}.unit{font-size:13px;color:var(--muted);margin-left:3px}\n  .meta{color:var(--muted);font-size:12px;margin-top:8px;min-height:18px}.good{color:var(--good)}.warn{color:var(--warn)}.bad{color:var(--bad)}\n  .rows{display:grid;gap:7px}.row{display:flex;justify-content:space-between;gap:18px;border-bottom:1px solid #20262e;padding-bottom:7px}.row:last-child{border:0;padding-bottom:0}.row span:last-child{text-align:right;font-variant-numeric:tabular-nums}\n  .chips{display:flex;gap:7px;flex-wrap:wrap}.chip{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:6px 8px;color:var(--muted);font-size:11px}.chip strong{color:var(--text);font-weight:650}\n  canvas{display:block;width:100%;height:150px;margin-top:8px}.chart-title{font-size:11px;color:var(--muted)}\n  pre{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;max-height:340px;overflow:auto;color:#aab2bc;font:11px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}\n  @media(max-width:1000px){.s2,.s3,.s4{grid-column:span 6}.s6,.s8{grid-column:span 12}}\n  @media(max-width:620px){header{flex-direction:column}.s2,.s3,.s4,.s6,.s8,.s12{grid-column:span 12}.big{font-size:34px}}\n</style>\n</head>\n<body>\n<main>\n<header>\n  <div><h1>Wearable Telemetry</h1><div class="sub">ESP32-S3 · MAX30102 · MPU6050 · NEO-6M · synchronized sensor timeline</div></div>\n  <div class="status"><span id="onlineDot" class="dot"></span><span id="onlineText">Waiting</span></div>\n</header>\n\n<div class="grid">\n  <section class="card s3"><div class="label">Stable heart rate · 15 s</div><span id="stableBpm" class="big">--</span><span class="unit">BPM</span><div id="stableBpmMeta" class="meta">Collecting</div></section>\n  <section class="card s3"><div class="label">Stable blood oxygen · 15 s</div><span id="stableSpo2" class="big">--</span><span class="unit">%</span><div id="stableSpo2Meta" class="meta">Estimate, not medically calibrated</div></section>\n  <section class="card s3"><div class="label">Rolling heart rate</div><span id="rollingBpm" class="big">--</span><span class="unit">BPM</span><div id="rollingMeta" class="meta">--</div></section>\n  <section class="card s3"><div class="label">GPS speed</div><span id="speed" class="big">--</span><span class="unit">m/s</span><div id="speedMeta" class="meta">GPS required; no IMU speed integration</div></section>\n\n  <section class="card s4">\n    <div class="label">PPG / optical quality</div>\n    <div class="rows">\n      <div class="row"><span>Contact</span><span id="contact">--</span></div>\n      <div class="row"><span>IR</span><span id="ir">--</span></div>\n      <div class="row"><span>Red</span><span id="red">--</span></div>\n      <div class="row"><span>PI IR</span><span id="pi">--</span></div>\n      <div class="row"><span>Red/IR correlation</span><span id="corr">--</span></div>\n      <div class="row"><span>Quality</span><span id="quality">--</span></div>\n      <div class="row"><span>Stable reason</span><span id="stableReason">--</span></div>\n    </div>\n  </section>\n\n  <section class="card s4">\n    <div class="label">Motion</div>\n    <div class="rows">\n      <div class="row"><span>Earth accel X/Y/Z</span><span id="earthAccel">--</span></div>\n      <div class="row"><span>Accel magnitude</span><span id="earthMag">--</span></div>\n      <div class="row"><span>Gyro X/Y/Z</span><span id="gyro">--</span></div>\n      <div class="row"><span>Stationary</span><span id="stationary">--</span></div>\n      <div class="row"><span>Roll / Pitch / Yaw*</span><span id="angles">--</span></div>\n      <div class="row"><span>Temperature</span><span id="temperature">--</span></div>\n    </div>\n    <div class="meta">* Yaw is relative: MPU6050 has no magnetometer.</div>\n  </section>\n\n  <section class="card s4">\n    <div class="label">GPS / network</div>\n    <div class="rows">\n      <div class="row"><span>GPS UART</span><span id="gpsRx">--</span></div>\n      <div class="row"><span>Fix</span><span id="gpsFix">--</span></div>\n      <div class="row"><span>Position</span><span id="position">--</span></div>\n      <div class="row"><span>Course</span><span id="course">--</span></div>\n      <div class="row"><span>Satellites</span><span id="sat">--</span></div>\n      <div class="row"><span>HDOP</span><span id="hdop">--</span></div>\n      <div class="row"><span>ESP RSSI</span><span id="rssi">--</span></div>\n      <div class="row"><span>Clock drift</span><span id="drift">--</span></div>\n      <div class="row"><span>Best sync RTT</span><span id="rtt">--</span></div>\n    </div>\n  </section>\n\n  <section class="card s12">\n    <div class="chips">\n      <div class="chip">AHRS startup <strong id="ahrsStartup">--</strong></div>\n      <div class="chip">Accel ignored <strong id="accelIgnored">--</strong></div>\n      <div class="chip">Accel scale calibrated <strong id="accelCal">--</strong></div>\n      <div class="chip">Next stable window <strong id="nextStable">--</strong></div>\n      <div class="chip">Device IP <strong id="deviceIp">--</strong></div>\n    </div>\n  </section>\n\n  <section class="card s6"><div class="chart-title">Raw MAX30102 · IR / Red</div><canvas id="ppgChart"></canvas></section>\n  <section class="card s6"><div class="chart-title">Earth-frame acceleration magnitude</div><canvas id="accelChart"></canvas></section>\n  <section class="card s6"><div class="chart-title">Rolling BPM</div><canvas id="bpmChart"></canvas></section>\n  <section class="card s6"><div class="chart-title">Rolling SpO₂ estimate</div><canvas id="spo2Chart"></canvas></section>\n\n  <section class="card s12"><div class="label">Latest processed JSON</div><pre id="rawJson">Waiting...</pre></section>\n</div>\n</main>\n<script>\nconst $ = id => document.getElementById(id);\nconst histories = {ir:[], red:[], accel:[], bpm:[], spo2:[]};\nconst MAX = {ppg:250, regular:180};\nfunction f(v,d=2){return Number.isFinite(Number(v))?Number(v).toFixed(d):"--"}\nfunction boolText(v){return v===true?"yes":v===false?"no":"--"}\nfunction push(arr,v,max){if(Number.isFinite(Number(v))){arr.push(Number(v));if(arr.length>max)arr.splice(0,arr.length-max)}}\nfunction draw(canvas, series){\n  const dpr=devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=w*dpr;canvas.height=h*dpr;\n  const c=canvas.getContext(\'2d\');c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,w,h);c.strokeStyle=\'#252c35\';c.lineWidth=1;\n  for(let i=1;i<4;i++){const y=h*i/4;c.beginPath();c.moveTo(0,y);c.lineTo(w,y);c.stroke()}\n  const vals=series.flatMap(s=>s.values).filter(Number.isFinite);if(vals.length<2)return;\n  let lo=Math.min(...vals),hi=Math.max(...vals);if(hi===lo){hi+=1;lo-=1}const p=(hi-lo)*.08;lo-=p;hi+=p;\n  series.forEach((s,si)=>{if(s.values.length<2)return;c.strokeStyle=si===0?\'#8da9d8\':\'#d18b77\';c.lineWidth=1.6;c.beginPath();s.values.forEach((v,i)=>{const x=i/(Math.max(2,s.values.length)-1)*w;const y=h-(v-lo)/(hi-lo)*h;i?c.lineTo(x,y):c.moveTo(x,y)});c.stroke()})\n}\nfunction update(s){\n  const dev=s.device||{}, imu=s.imu||{}, gps=s.gps||{}, nav=s.navigation||{}, vit=s.vitals||{}, roll=vit.rolling||{}, stable=vit.stable_15s||{}, ppg=s.ppg||{}, clock=s.clock||{};\n  $(\'onlineDot\').classList.toggle(\'ok\',!!dev.connected);$(\'onlineText\').textContent=dev.connected?\'Live\':\'Offline\';\n  $(\'stableBpm\').textContent=stable.bpm??\'--\'; $(\'stableSpo2\').textContent=stable.spo2_estimate_pct??\'--\';\n  $(\'stableBpmMeta\').textContent=`${stable.valid?\'valid\':\'not valid\'} · quality ${f(stable.quality,2)}`;\n  $(\'stableBpmMeta\').className=\'meta \'+(stable.valid?\'good\':\'warn\');\n  $(\'stableSpo2Meta\').textContent=stable.medical_calibrated?\'calibrated\':\'estimate · not medically calibrated\';\n  $(\'rollingBpm\').textContent=roll.bpm??\'--\'; $(\'rollingMeta\').textContent=`${roll.reason||\'--\'} · Q ${f(roll.quality,2)}`;\n  $(\'speed\').textContent=nav.speed_mps==null?\'--\':f(nav.speed_mps,2); $(\'speedMeta\').textContent=nav.speed_source===\'gps\'?`${f(nav.speed_kmh,1)} km/h · GPS`:\'GPS fix required; IMU speed disabled\';\n  $(\'contact\').textContent=boolText(ppg.contact); $(\'ir\').textContent=ppg.ir??\'--\'; $(\'red\').textContent=ppg.red??\'--\';\n  $(\'pi\').textContent=roll.perfusion_index_ir_pct==null?\'--\':f(roll.perfusion_index_ir_pct,3)+\' %\'; $(\'corr\').textContent=f(roll.red_ir_correlation,3); $(\'quality\').textContent=f(roll.quality,3); $(\'stableReason\').textContent=stable.reason||\'--\';\n  const ea=imu.earth_acceleration_mps2||[]; const gr=imu.gyro_corrected_rads||[]; const ang=imu.orientation_deg||{};\n  $(\'earthAccel\').textContent=ea.length?ea.map(x=>f(x,3)).join(\' / \'):\'--\'; $(\'earthMag\').textContent=imu.earth_acceleration_magnitude_mps2==null?\'--\':f(imu.earth_acceleration_magnitude_mps2,3)+\' m/s²\';\n  $(\'gyro\').textContent=gr.length?gr.map(x=>f(x,3)).join(\' / \'):\'--\'; $(\'stationary\').textContent=boolText(imu.stationary); $(\'angles\').textContent=`${f(ang.roll,1)} / ${f(ang.pitch,1)} / ${f(ang.yaw_relative,1)}°`; $(\'temperature\').textContent=imu.temperature_c==null?\'--\':f(imu.temperature_c,1)+\' °C\';\n  $(\'gpsRx\').textContent=boolText(gps.rx); $(\'gpsFix\').textContent=boolText(gps.fix); $(\'position\').textContent=gps.fix?`${f(gps.lat,6)}, ${f(gps.lon,6)}`:\'--\'; $(\'course\').textContent=gps.course_deg==null?\'--\':f(gps.course_deg,1)+\'°\'; $(\'sat\').textContent=gps.sat??\'--\'; $(\'hdop\').textContent=f(gps.hdop,2);\n  $(\'rssi\').textContent=dev.rssi_dbm==null?\'--\':`${dev.rssi_dbm} dBm`; $(\'drift\').textContent=clock.drift_ppm==null?\'--\':f(clock.drift_ppm,1)+\' ppm\'; $(\'rtt\').textContent=clock.best_rtt_ms==null?\'--\':f(clock.best_rtt_ms,2)+\' ms\';\n  $(\'ahrsStartup\').textContent=boolText(imu.ahrs?.startup); $(\'accelIgnored\').textContent=boolText(imu.ahrs?.accelerometer_ignored); $(\'accelCal\').textContent=boolText(imu.accel_scale_calibrated); $(\'nextStable\').textContent=f(vit.seconds_to_next_stable,1)+\' s\'; $(\'deviceIp\').textContent=dev.ip||\'--\';\n  push(histories.accel,imu.earth_acceleration_magnitude_mps2,MAX.regular); push(histories.bpm,roll.bpm,MAX.regular); push(histories.spo2,roll.spo2_estimate_pct,MAX.regular);\n  draw($(\'accelChart\'),[{values:histories.accel}]); draw($(\'bpmChart\'),[{values:histories.bpm}]); draw($(\'spo2Chart\'),[{values:histories.spo2}]);\n  $(\'rawJson\').textContent=JSON.stringify(s,null,2);\n}\nconst API_BASE = location.protocol === \'file:\' ? \'http://127.0.0.1:8081\' : \'\';\nfunction apiUrl(path){ return API_BASE + path; }\n\nasync function consume(url,onItem){\n  url = apiUrl(url);\n  while(true){\n    try{const r=await fetch(url,{cache:\'no-store\'});if(!r.ok)throw new Error(r.status);const rd=r.body.getReader();const dec=new TextDecoder();let buf=\'\';\n      while(true){const {value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});const lines=buf.split(\'\\n\');buf=lines.pop();for(const line of lines){if(!line.trim())continue;try{onItem(JSON.parse(line))}catch{}}}\n    }catch(e){await new Promise(r=>setTimeout(r,700))}\n  }\n}\nconsume(\'/api/stream\',update);\nconsume(\'/api/raw-stream\',p=>{if(p.t!==\'ppg\')return;push(histories.ir,p.ir,MAX.ppg);push(histories.red,p.red,MAX.ppg);draw($(\'ppgChart\'),[{values:histories.ir},{values:histories.red}])});\n</script>\n</body>\n</html>\n'


def _dashboard_html() -> str:
    return Path(__file__).with_name("hub_dashboard.html").read_text(encoding="utf-8")


@app.get("/", include_in_schema=False)
async def dashboard():
    return HTMLResponse(
        content=_dashboard_html(),
        status_code=200,
        headers={
            "Cache-Control": "no-store",
            "X-Sensor-Hub-Build": BUILD_ID,
        },
    )


@app.get("/dashboard", include_in_schema=False)
async def dashboard_alias():
    return await dashboard()


@app.get("/how-it-works", include_in_schema=False)
async def how_it_works():
    return HTMLResponse(
        """<!doctype html><html><head><meta charset="utf-8"><title>How CPG44 works</title>
<style>body{font:18px/1.5 system-ui;max-width:720px;margin:40px auto;padding:0 20px;background:#0a0c10;color:#edf1f5}
a{color:#8da9d8} code{background:#161b22;padding:2px 6px;border-radius:6px}</style></head><body>
<h1>How this product works</h1>
<p>Three machines talk to each other:</p>
<ol>
<li><b>The vest</b> (ESP32) reads motion, heart, oxygen, GPS and waits for the laptop.</li>
<li><b>The sensor hub</b> in WSL connects to the vest, lines up clocks, and computes heart rate.</li>
<li><b>The camera app</b> finds players in video and glues wearable numbers onto the matching player.</li>
</ol>
<p>Open the live wearable page at <a href="/">/</a>. Vision fusion is a second command (see the beginner report).</p>
</body></html>""",
        headers={"Cache-Control": "no-store"},
    )


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
    parser.add_argument("--esp32", default=None, help="ESP32 LAN IP printed by the firmware")
    parser.add_argument("--stream-port", type=int, default=9000)
    parser.add_argument("--http-port", type=int, default=8081)
    parser.add_argument("--finger-ir-min", type=float, default=20000.0)
    parser.add_argument("--record", default=None, help="Optional synchronized raw NDJSON output file")
    parser.add_argument("--calibration", default=None, help="Optional IMU calibration JSON")
    parser.add_argument("--player-id", type=int, default=7, help="Roster id fused with vision tracks")
    parser.add_argument("--match-id", default="live")
    ARGS = parser.parse_args()

    ENGINE = TelemetryEngine(
        finger_ir_min=ARGS.finger_ir_min,
        record_path=ARGS.record,
        calibration_path=ARGS.calibration,
    )
    ENGINE.state["player_id"] = ARGS.player_id
    ENGINE.state["match_id"] = ARGS.match_id

    print(f"\n=== {BUILD_ID} ===")
    print(f"Dashboard: http://127.0.0.1:{ARGS.http_port}/")
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
