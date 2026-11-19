#!/usr/bin/env python3
"""
ESP32-S3 wearable sensor processor.

- Subscribes to the authenticated VPS relay. It never opens a LAN connection
  to an ESP32.
- Maps the firmware's SNTP-anchored sample time into the host monotonic clock
  for later CV fusion.
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
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt, welch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
import uvicorn
from websockets.asyncio.client import connect

from soccer_analytics.sensors.motion_artifact import clean_heart_rate
from soccer_analytics.sensors.ecg import ECGProcessor
from soccer_analytics.sensors.calibration import (AccelAutoCal, GpsKalman, GyroBias, SpO2Calibration,
                                                  battery_state)

G = 9.80665
PPG_FS = 25.0
PPG_REF_N = 150                # 6 s at 25 Hz, enough peaks for robust IBI
PPG_STABLE_SECONDS = 12.0
HR_MOTION_SQI_MIN = 0.5        # spectral concentration gate for exercise HR (tune vs ECG)
SPO2_STILL_ACCEL_RMS = 1.5     # m/s^2 (~0.15 g): SpO2 only in still recovery windows
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
    """Zero-phase Butterworth bandpass (SciPy sosfiltfilt, prefer SOS over ba).

    Cardiac PPG lives ~0.7-4 Hz (42-240 BPM). DC / slow wander and 50/60 Hz
    junk stay out. filtfilt/sosfiltfilt apply the filter forward and backward
    so peaks are not time-shifted.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 24:
        return x - float(np.mean(x))
    sos = butter(order, [lo, hi], btype="bandpass", output="sos", fs=fs)
    return sosfiltfilt(sos, x)


def estimate_ppg_vitals(ir_values, red_values, fs=PPG_FS):
    """High-accuracy HR / SpO2 via Autocorrelation & Peak Inter-Beat fusion."""
    if len(ir_values) < 20 or len(red_values) < 20:
        return {"valid": False, "reason": "collecting", "bpm": None, "spo2_estimate_pct": None}

    ir = np.asarray(ir_values, dtype=np.float64)
    red = np.asarray(red_values, dtype=np.float64)
    ir_dc = float(np.mean(ir))
    red_dc = float(np.mean(red))
    if ir_dc <= 0 or red_dc <= 0:
        return {"valid": False, "reason": "no_or_weak_contact", "bpm": None, "spo2_estimate_pct": None}

    ir_ac = _ppg_bandpass(ir, fs)
    red_ac = _ppg_bandpass(red, fs)

    # 1. Fundamental pulse rate via Autocorrelation
    autocorr = np.correlate(ir_ac, ir_ac, mode="full")[len(ir_ac) - 1 :]
    min_lag = max(6, int(fs * 60.0 / 200.0))   # ~200 BPM
    max_lag = min(len(autocorr) - 1, int(fs * 60.0 / 48.0))  # ~48 BPM
    autocorr_hr = None
    if max_lag > min_lag + 2:
        lag_search = autocorr[min_lag:max_lag]
        best_idx = int(np.argmax(lag_search))
        best_lag = float(min_lag + best_idx)
        if 0 < best_idx < len(lag_search) - 1:
            y0, y1, y2 = lag_search[best_idx - 1], lag_search[best_idx], lag_search[best_idx + 1]
            denom_p = (y0 - 2 * y1 + y2)
            if abs(denom_p) > 1e-9:
                delta = 0.5 * (y0 - y2) / denom_p
                best_lag += delta
        if best_lag > 0:
            autocorr_hr = 60.0 * fs / best_lag

    # 2. Peaks = systolic arterial pulses
    min_dist = max(6, int(fs * 0.30))
    prominence = max(0.20 * float(np.std(ir_ac)), 0.4)
    peaks, _ = find_peaks(ir_ac, distance=min_dist, prominence=prominence)

    peak_hr = None
    ibi_cv = None
    n_ibi = 0
    if len(peaks) >= 2:
        ibi = np.diff(peaks) / float(fs)
        valid_ibi = ibi[(ibi >= 0.30) & (ibi <= 1.4)]
        if len(valid_ibi) >= 1:
            med = float(np.median(valid_ibi))
            peak_hr = 60.0 / med
            ibi_cv = float(np.std(valid_ibi) / med) if len(valid_ibi) > 1 else 0.05
            n_ibi = len(valid_ibi)

    # Fuse autocorrelation and peak intervals
    if autocorr_hr is not None and peak_hr is not None:
        if abs(autocorr_hr - peak_hr) < 22.0:
            hr = 0.65 * autocorr_hr + 0.35 * peak_hr
        elif 55.0 <= autocorr_hr <= 180.0:
            hr = autocorr_hr
        else:
            hr = peak_hr
    elif autocorr_hr is not None and 50.0 <= autocorr_hr <= 190.0:
        hr = autocorr_hr
    elif peak_hr is not None and 50.0 <= peak_hr <= 190.0:
        hr = peak_hr
    else:
        hr = None

    ir_rms = float(np.sqrt(np.mean(ir_ac * ir_ac)))
    red_rms = float(np.sqrt(np.mean(red_ac * red_ac)))
    corr = float(np.corrcoef(ir_ac, red_ac)[0, 1]) if ir_rms > 0 and red_rms > 0 else float("nan")
    pi_ir_pct = 100.0 * ir_rms / max(ir_dc, 1.0)

    R = None
    spo2 = None
    denom = ir_rms / ir_dc
    if denom > 1e-9 and red_dc > 0:
        R = (red_rms / red_dc) / denom
        # Maxim reference curve (MAX30102 application algorithm). Values outside
        # the curve's physiological range are reported as unavailable; they are
        # never replaced by a healthy-looking constant.
        spo2_calc = -45.060 * R * R + 30.354 * R + 94.845
        if 70.0 <= spo2_calc <= 100.5:
            spo2 = round(min(spo2_calc, 100.0), 1)

    hr_ok = hr is not None and 45.0 <= hr <= 210.0 and (ibi_cv is None or ibi_cv < 0.35)
    spo2_ok = spo2 is not None and math.isfinite(spo2)
    optical_ok = math.isfinite(corr) and corr > 0.40 and pi_ir_pct > 0.03
    # Heart rate does not depend on the red channel being calibrated; SpO2 does.
    valid = bool(hr_ok and optical_ok)

    reason = "ok"
    if not optical_ok:
        reason = "poor_optical_quality"
    elif not hr_ok:
        reason = "hr_peaks_unreliable"
    elif not spo2_ok:
        reason = "spo2_out_of_range"

    return {
        "valid": valid,
        "spo2_valid": bool(valid and spo2_ok),
        "reason": reason,
        "bpm": None if hr is None else round(float(hr), 2),
        "spo2_estimate_pct": None if spo2 is None or not optical_ok else round(float(spo2), 2),
        "r_ratio": None if R is None else round(float(R), 4),
        "peak_count": int(len(peaks)),
        "ibi_count": n_ibi,
        "ibi_cv": None if ibi_cv is None else round(ibi_cv, 4),
        "perfusion_index_ir_pct": round(pi_ir_pct, 4),
        "red_ir_correlation": None if not math.isfinite(corr) else round(corr, 4),
        "method": "autocorr+find_peaks+rms_ratio",
    }


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

    def probe():
        t0 = time.time_ns()
        with urllib.request.urlopen(url, timeout=3) as r:
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
    def __init__(self, finger_ir_min=4000.0, record_path=None, calibration_path=None):
        self.lock = asyncio.Lock()
        self.clock = ClockMapper()
        self.relay_clock = RelayClockOffset()
        self.raw_broadcast = Broadcast()
        self.ppg = deque(maxlen=2500)
        self.imu_history = deque(maxlen=6000)
        self.stable_vitals = deque(maxlen=40)
        self.filter = Mahony6D()
        self.finger_ir_min = float(finger_ir_min)
        self.disp_hr = None
        self.disp_spo2 = None
        self.ecg = ECGProcessor()
        self.gyro_cal = GyroBias(fs=50.0)
        self.accel_cal = AccelAutoCal()
        self.spo2_cal = SpO2Calibration()
        self.gps_kf = GpsKalman()
        self.battery_hist = deque(maxlen=400)
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
            self.spo2_cal = SpO2Calibration([tuple(x) for x in cfg.get("spo2_pairs", [])])

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
            if t in ("imu", "ppg", "gps", "ecg"):
                self._count_rate(t, int(packet["host_mono_ns"]))
            if t == "hello":
                self.state["device"].update({k: v for k, v in packet.items() if k not in {"t"}})
            elif t == "imu":
                self._process_imu(packet)
            elif t == "ppg":
                self._process_ppg(packet)
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
                    "max30102": packet.get("max30102"),
                    "gps_rx": packet.get("gps_rx"),
                    "dropped_samples": packet.get("dropped_samples"),
                    "queued_samples": packet.get("queued_samples"),
                    "imu_rate_hz": packet.get("imu_rate_hz"),
                    "imu_accel_range_g": packet.get("imu_accel_range_g"),
                    "imu_gyro_range_dps": packet.get("imu_gyro_range_dps"),
                    "ppg_output_rate_hz": packet.get("ppg_output_rate_hz"),
                    "ppg_raw_rate_hz": packet.get("ppg_raw_rate_hz"),
                    "ppg_sample_average": packet.get("ppg_sample_average"),
                    "clock_anchor_age_ms": packet.get("clock_anchor_age_ms"),
                    "gps_rate_hz": packet.get("gps_rate_hz"),
                    "battery": battery_state(self.battery_hist, rx_mono_ns / 1e9, packet.get("battery_pct"),
                                             packet.get("battery_v"), packet.get("battery_rate_pct_h")),
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
            cfg.update({"gyro_bias_rads": [float(x) for x in self.gyro_cal.bias],
                        "spo2_pairs": [list(x) for x in self.spo2_cal.pairs]})
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

    def add_spo2_reference(self, reference_pct):
        """Pair the current still-window ratio R with a finger oximeter reading."""
        R = ((self.state.get("vitals") or {}).get("rolling") or {}).get("ratio_r")
        if R is None:
            raise ValueError("no still, good-contact pulse window right now; sit still for 20 s and retry")
        status = self.spo2_cal.add(float(R), float(reference_pct))
        self._save_calibration()
        return status

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
        if self.gyro_cal.update(g_raw, accel):
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

    def _spectral_hr(self, ir):
        if len(ir) < 100:
            return None
        x = detrend(np.asarray(ir, dtype=np.float64), type="linear")
        f, p = welch(x, fs=PPG_FS, nperseg=min(len(x), 200))
        band = (f >= 0.58) & (f <= 3.67)  # ~35..220 BPM
        if not np.any(band):
            return None
        return float(f[band][np.argmax(p[band])] * 60.0)

    def calculate_vitals(self, now_ns=None):
        if len(self.ppg) < 125:
            return {"valid": False, "reason": "collecting", "samples": len(self.ppg),
                    "bpm": None, "spo2_estimate_pct": None}

        rows = list(self.ppg)[-min(len(self.ppg), PPG_REF_N):]
        ir = np.asarray([r["ir"] for r in rows], dtype=np.float64)
        red = np.asarray([r["red"] for r in rows], dtype=np.float64)
        start_ns = rows[0]["host_mono_ns"]
        end_ns = rows[-1]["host_mono_ns"]
        age_s = ((time.perf_counter_ns() if now_ns is None else now_ns) - end_ns) / 1e9
        ts = np.asarray([r["host_mono_ns"] for r in rows], dtype=np.float64) / 1e9
        gaps = np.diff(ts)
        # Short transport gaps (a few lost samples) are bridged on a uniform
        # grid; a gap of a third of a second or more, or a window that is more
        # than 15% bridged, is too much of a heartbeat to invent.
        if age_s > 2.0 or age_s < 0 or np.any(gaps <= 0) or np.any(gaps > 0.32):
            return {"valid": False, "reason": "stale_or_discontinuous_pulse",
                    "bpm": None, "spo2_estimate_pct": None,
                    "window_end_host_mono_ns": end_ns}
        grid = np.arange(ts[0], ts[-1] + 1e-9, 1.0 / PPG_FS)
        bridged = float(np.sum(np.clip(np.round(gaps * PPG_FS) - 1, 0, None))) / max(len(grid), 1)
        if bridged > 0.15:
            return {"valid": False, "reason": "stale_or_discontinuous_pulse",
                    "bpm": None, "spo2_estimate_pct": None, "bridged_fraction": round(bridged, 3),
                    "window_end_host_mono_ns": end_ns}
        if bridged > 0:
            ir, red = np.interp(grid, ts, ir), np.interp(grid, ts, red)

        mean_ir = float(np.mean(ir))
        mean_red = float(np.mean(red))
        if mean_ir < self.finger_ir_min or mean_red <= 0:
            self.disp_hr = None
            self.disp_spo2 = None
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
        if est.get("r_ratio") is not None and est.get("spo2_estimate_pct") is not None:
            # Per-device curve from paired finger-oximeter readings, when available.
            cal = self.spo2_cal.spo2(float(est["r_ratio"]))
            est = dict(est, spo2_estimate_pct=round(min(cal, 100.0), 2) if 70.0 <= cal <= 100.5 else None)
        spectral_rows = list(self.ppg)[-min(len(self.ppg), 200):]
        spectral_hr = self._spectral_hr([r["ir"] for r in spectral_rows])
        hr = est.get("bpm")
        hr_diff = None if hr is None or spectral_hr is None else abs(float(hr) - spectral_hr)

        motion = self._motion_quality(start_ns, end_ns)
        still = bool(motion.get("valid") and motion["dynamic_accel_rms_mps2"] < SPO2_STILL_ACCEL_RMS
                     and motion["gyro_rms_rads"] < 1.0)
        motion_ok = (
            not motion.get("valid")
            or (motion["dynamic_accel_rms_mps2"] < 3.0 and motion["gyro_rms_rads"] < 2.5)
        )

        # During movement, step cadence contaminates the optical pulse. Instead
        # of always abstaining, cancel the accelerometer-correlated component
        # (multi-axis NLMS) and accept the spectral estimate only when the
        # cleaned spectrum is concentrated (SQI gate).
        cleaned = None
        if motion.get("valid") and not still:
            imu_rows = [r for r in self.imu_history
                        if start_ns - 5e8 <= r["host_mono_ns"] <= end_ns + 5e8 and "accel_body" in r]
            if len(imu_rows) >= 50:
                cleaned = clean_heart_rate(
                    ir, [r["host_mono_ns"] / 1e9 for r in rows],
                    np.asarray([r["accel_body"] for r in imu_rows]),
                    [r["host_mono_ns"] / 1e9 for r in imu_rows], PPG_FS,
                    previous_bpm=self.disp_hr)
        if cleaned is not None and cleaned.get("bpm") is not None and cleaned["sqi"] >= HR_MOTION_SQI_MIN \
                and est.get("reason") != "poor_optical_quality":
            est = dict(est, valid=True, bpm=round(cleaned["bpm"], 2), reason="ok",
                       method=cleaned["method"])
            hr = est["bpm"]
            hr_diff = None if spectral_hr is None else abs(float(hr) - spectral_hr)
            motion_ok = True

        agreement_ok = hr_diff is None or hr_diff <= 35.0 or cleaned is not None
        valid = bool(est.get("valid") and motion_ok and agreement_ok)
        spo2_ok = bool(valid and still and est.get("spo2_estimate_pct") is not None)

        reason = est.get("reason") or "ok"
        if est.get("valid") and not motion.get("valid"):
            reason = "imu_quality_unavailable"
        elif est.get("valid") and not motion_ok:
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
        if cleaned is not None:
            quality_parts.append(float(cleaned["sqi"]))
        quality = float(np.mean(quality_parts)) if quality_parts else 0.0

        # Live display: EMA of *valid* windows only, tracks the signal, does
        # not ramp, and does not flash invalid jumps.
        if valid and hr is not None:
            a = 0.40
            self.disp_hr = float(hr) if self.disp_hr is None else a * float(hr) + (1 - a) * self.disp_hr
        if spo2_ok:
            a = 0.40
            self.disp_spo2 = (
                float(est["spo2_estimate_pct"]) if self.disp_spo2 is None
                else a * float(est["spo2_estimate_pct"]) + (1 - a) * self.disp_spo2
            )

        return {
            "valid": valid,
            "reason": reason,
            "bpm": None if not valid else round(self.disp_hr, 1) if self.disp_hr is not None else None,
            "spo2_estimate_pct": None if not spo2_ok else (
                round(self.disp_spo2, 1) if self.disp_spo2 is not None else None),
            "spo2_valid": spo2_ok,
            "spo2_note": None if spo2_ok else "SpO2 is only estimated in still recovery windows",
            "spo2_calibration": self.spo2_cal.status(),
            "ratio_r": est.get("r_ratio") if spo2_ok else None,
            "motion_cleaning": cleaned,
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
                    "spo2": (float(result["spo2_estimate_pct"])
                             if result.get("spo2_estimate_pct") is not None else None),
                    "quality": float(result["quality"]),
                })

            cutoff = now_ns - int(PPG_STABLE_SECONDS * 1e9)
            while self.stable_vitals and self.stable_vitals[0]["t_ns"] < cutoff:
                self.stable_vitals.popleft()

            self.state["vitals"]["live"] = {
                "valid": self.disp_hr is not None,
                "bpm": None if self.disp_hr is None else round(self.disp_hr, 1),
                "spo2_estimate_pct": None if self.disp_spo2 is None else round(self.disp_spo2, 1),
                "reason": result.get("reason"),
                "stale": not bool(result.get("valid")),
                "source": "ppg",
            }
            # The chest ECG, when worn, is the heart-rate source: it keeps beat
            # timing while running, where the optical pulse fails.
            if self.ecg.t:
                still = bool((self.state.get("imu") or {}).get("stationary"))
                ecg = self.ecg.estimate(now=now_ns / 1e9, still=still)
                self.state["vitals"]["ecg"] = ecg
                if ecg.get("valid"):
                    self.state["vitals"]["live"].update({"valid": True, "bpm": ecg["bpm"], "source": "ecg",
                                                         "stale": False, "reason": "ok",
                                                         "rmssd_ms": ecg.get("rmssd_ms")})
                elif self.disp_hr is None:
                    self.state["vitals"]["live"]["reason"] = ecg.get("reason")

            if result.get("valid") and len(self.stable_vitals) >= 4:
                bpm = np.asarray([x["bpm"] for x in self.stable_vitals])
                spo2 = np.asarray([x["spo2"] for x in self.stable_vitals if x["spo2"] is not None])
                q = np.asarray([x["quality"] for x in self.stable_vitals])
                self.state["vitals"]["stable_15s"] = {
                    "valid": True,
                    "bpm": round(float(np.median(bpm)), 2),
                    "spo2_estimate_pct": round(float(np.median(spo2)), 2) if len(spo2) else None,
                    "quality": round(float(np.median(q)), 4),
                    "n": len(self.stable_vitals),
                    "window_seconds": PPG_STABLE_SECONDS,
                    "computed_host_mono_ns": now_ns,
                    "computed_unix_ns": mono_to_unix_ns(now_ns),
                    "window_end_host_mono_ns": result.get("window_end_host_mono_ns"),
                    "window_end_unix_ns": mono_to_unix_ns(result["window_end_host_mono_ns"]),
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
            vit["seconds_to_next_stable"] = round(max(0.0, PPG_STABLE_SECONDS - n), 1)
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
    if sample_type not in {"imu", "ppg", "ecg", "gps", "status"} or not isinstance(payload, dict):
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

BUILD_ID = "sensor-hub-relay-only-2026-08-20-v8"

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
<style>body{font:18px/1.5 system-ui;max-width:720px;margin:40px auto;padding:0 20px;background:#f5f6f2;color:#18231f}
a{color:#245a46} code{background:#e6e9e4;padding:2px 6px;border-radius:4px}</style></head><body>
<h1>How this product works</h1>
<p>Three machines talk to each other:</p>
<ol>
<li><b>The vest</b> reads the sensors and sends timestamped batches to the relay.</li>
<li><b>The local processor</b> receives the relay stream and checks motion and pulse quality.</li>
<li><b>The match pipeline</b> links the processed wearable values to the tagged player.</li>
</ol>
<p>Open the live wearable page at <a href="/">/</a>.</p>
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


@app.post("/api/calibrate/spo2")
async def calibrate_spo2(body: dict):
    """Pair the vest's current still reading with a finger pulse oximeter value."""
    try:
        ref = float(body.get("reference_pct"))
        async with ENGINE.lock:
            status = ENGINE.add_spo2_reference(ref)
    except (TypeError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=422)
    return {"ok": True, "calibration": status}


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
    parser.add_argument("--finger-ir-min", type=float, default=4000.0)
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
