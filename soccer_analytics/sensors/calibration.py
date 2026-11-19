"""Sensor calibration and filtering for the vest, all running on the host.

Each piece fixes a measured or well-documented error source:

* ``GyroBias``: MEMS gyro offset (MPU6050: up to +-20 deg/s at power-up,
  drifting about 0.02 deg/s per degC). Estimated only in windows that are
  genuinely still (low gyro variance and gravity-only acceleration), then
  tracked slowly so warm-up drift is followed without absorbing motion.
* ``AccelAutoCal``: per-axis offset and scale of the accelerometer without a
  calibration rig (Frosio, Pedersini & Borghese 2009, "Autocalibration of
  MEMS accelerometers", IEEE TIM). At rest the true reading has magnitude g in
  any orientation, so still poses in several orientations constrain the six
  unknowns; a least-squares fit runs once the poses span enough directions.
* ``player_load``: PlayerLoad as defined by Barrett, Midgley & Lovell (2014):
  the accumulated change of acceleration in g, scaled by 1/100.
* ``GpsKalman``: constant-velocity Kalman filter in local metres. Speed comes
  from the receiver's Doppler measurement (RMC), which is several times more
  accurate than differencing noisy positions (Townshend et al. 2008), and
  distance is integrated from the filtered velocity so standing still does
  not accumulate position jitter as distance.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

G = 9.80665


# --------------------------------------------------------------------- gyro --
class GyroBias:
    def __init__(self, fs: float = 50.0, still_s: float = 1.0, track_alpha: float = 0.05):
        self.n = max(10, int(fs * still_s))
        self.win: deque = deque(maxlen=self.n)
        self.bias = np.zeros(3)
        self.calibrated = False
        self.alpha = track_alpha
        self.windows = 0

    def update(self, gyro_raw: np.ndarray, accel: np.ndarray) -> bool:
        """Feed one raw sample; returns True while the device is still."""
        still_now = abs(float(np.linalg.norm(accel)) - G) < 0.35 and \
            float(np.linalg.norm(gyro_raw - self.bias)) < (0.6 if not self.calibrated else 0.08)
        if not still_now:
            self.win.clear()
            return False
        self.win.append(np.asarray(gyro_raw, float))
        if len(self.win) < self.n:
            return False
        w = np.asarray(self.win)
        if float(np.max(np.std(w, axis=0))) > 0.01:          # rad/s: someone is touching it
            self.win.popleft()
            return False
        m = w.mean(axis=0)
        self.bias = m if not self.calibrated else (1 - self.alpha) * self.bias + self.alpha * m
        self.calibrated = True
        self.windows += 1
        self.win.clear()
        return True


# -------------------------------------------------------------------- accel --
class AccelAutoCal:
    def __init__(self, min_poses: int = 6, min_sep_deg: float = 30.0):
        self.poses: List[np.ndarray] = []
        self.min_poses = min_poses
        self.min_cos = math.cos(math.radians(min_sep_deg))
        self.bias = np.zeros(3)
        self.scale = np.ones(3)
        self.calibrated = False
        self.residual_g: Optional[float] = None

    def add_still_mean(self, accel_mean: np.ndarray) -> bool:
        """Add the mean raw acceleration of a still window; refit if it is a new pose."""
        a = np.asarray(accel_mean, float)
        u = a / max(np.linalg.norm(a), 1e-9)
        if any(float(u @ (p / np.linalg.norm(p))) > self.min_cos for p in self.poses):
            return False
        self.poses.append(a)
        if len(self.poses) > 24:
            self.poses.pop(0)
        return self.fit()

    def _spans(self) -> bool:
        U = np.asarray([p / np.linalg.norm(p) for p in self.poses])
        # every axis must have been seen pointing both up and down at least partly
        return bool(np.all(U.max(axis=0) > 0.5) and np.all(U.min(axis=0) < -0.5))

    def fit(self) -> bool:
        if len(self.poses) < self.min_poses or not self._spans():
            return False
        from scipy.optimize import least_squares
        A = np.asarray(self.poses)

        def res(p):
            return np.linalg.norm((A - p[:3]) * p[3:], axis=1) - G
        sol = least_squares(res, np.r_[np.zeros(3), np.ones(3)])
        b, s = sol.x[:3], sol.x[3:]
        rms = float(np.sqrt(np.mean(sol.fun ** 2))) / G
        if rms < 0.02 and np.all(np.abs(s - 1) < 0.1) and np.all(np.abs(b) < 1.5):
            self.bias, self.scale, self.calibrated, self.residual_g = b, s, True, rms
            return True
        return False

    def apply(self, a_raw: np.ndarray) -> np.ndarray:
        return (np.asarray(a_raw, float) - self.bias) * self.scale


# ---------------------------------------------------------------- PlayerLoad --
def player_load(accel_g: np.ndarray) -> float:
    """PlayerLoad (Barrett et al. 2014) of an (n,3) accelerometer series in g.

    The definition assumes 100 Hz sampling. At 50 Hz each step spans twice
    the time and sees a larger change, so values are comparable within this
    system but not with commercial 100 Hz units.
    """
    a = np.asarray(accel_g, float)
    if len(a) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(a, axis=0), axis=1)) / 100.0)


# --------------------------------------------------------------------- GNSS --
class GpsKalman:
    """Position + velocity in local east/north metres, 1 kHz-free and stateless
    about wall time: call ``update`` with each fix's own timestamp."""

    UERE_M = 2.5            # user equivalent range error for a patch-antenna receiver
    DOPPLER_SIGMA = 0.25    # m/s, single-frequency Doppler speed under motion
    ACCEL_SIGMA = 3.0       # m/s^2 process noise: football accelerations
    MAX_SPEED = 11.5        # m/s, above any player's top speed

    def __init__(self):
        self.origin: Optional[Tuple[float, float]] = None
        self.x: Optional[np.ndarray] = None            # [e, n, ve, vn]
        self.P = np.eye(4) * 100.0
        self.t: Optional[float] = None
        self.distance_m = 0.0
        self.rejected = 0

    def _enu(self, lat: float, lon: float) -> np.ndarray:
        lat0, lon0 = self.origin
        k = 111_320.0
        return np.array([(lon - lon0) * k * math.cos(math.radians(lat0)), (lat - lat0) * k])

    def update(self, t: float, lat: float, lon: float, speed_mps: Optional[float] = None,
               course_deg: Optional[float] = None, hdop: Optional[float] = None,
               sats: Optional[int] = None) -> Optional[dict]:
        if sats is not None and sats < 5 or hdop is not None and hdop > 4.0:
            self.rejected += 1
            return None
        if self.origin is None:
            self.origin = (lat, lon)
        z = self._enu(lat, lon)
        if self.x is None:
            self.x, self.t = np.r_[z, 0.0, 0.0], t
            return self.state()
        dt = t - self.t
        if not (0.0 < dt <= 5.0):
            return self.state()
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        q = self.ACCEL_SIGMA ** 2
        Q = q * np.array([[dt**4 / 4, 0, dt**3 / 2, 0], [0, dt**4 / 4, 0, dt**3 / 2],
                          [dt**3 / 2, 0, dt**2, 0], [0, dt**3 / 2, 0, dt**2]])
        x = F @ self.x
        P = F @ self.P @ F.T + Q
        H, Rm, zz = [np.eye(2, 4)], [np.eye(2) * (self.UERE_M * (hdop or 1.5)) ** 2], [z]
        if speed_mps is not None and course_deg is not None and np.isfinite(speed_mps) and np.isfinite(course_deg):
            c = math.radians(course_deg)
            v = np.array([speed_mps * math.sin(c), speed_mps * math.cos(c)])
            H.append(np.hstack([np.zeros((2, 2)), np.eye(2)]))
            Rm.append(np.eye(2) * self.DOPPLER_SIGMA ** 2 + (np.eye(2) * 0.5 ** 2 if speed_mps < 1.0 else 0))
            zz.append(v)
        Hm = np.vstack(H)
        Rb = np.zeros((2 * len(Rm), 2 * len(Rm)))
        for k, r in enumerate(Rm):
            Rb[2 * k:2 * k + 2, 2 * k:2 * k + 2] = r
        innov = np.concatenate(zz) - Hm @ x
        S = Hm @ P @ Hm.T + Rb
        if float(innov @ np.linalg.solve(S, innov)) > 30.0:      # chi-square gate: a multipath jump
            self.rejected += 1
            self.x, self.P, self.t = x, P, t
            return self.state()
        K = P @ Hm.T @ np.linalg.inv(S)
        x = x + K @ innov
        P = (np.eye(4) - K @ Hm) @ P
        sp = float(np.hypot(x[2], x[3]))
        if sp > self.MAX_SPEED:
            x[2:] *= self.MAX_SPEED / sp
        mean_speed = 0.5 * (float(np.hypot(self.x[2], self.x[3])) + float(np.hypot(x[2], x[3])))
        if mean_speed > 0.3:                                       # below walking pace: no distance
            self.distance_m += mean_speed * dt
        self.x, self.P, self.t = x, P, t
        return self.state()

    def state(self) -> dict:
        e, n, ve, vn = self.x
        return {"east_m": float(e), "north_m": float(n), "speed_mps": float(np.hypot(ve, vn)),
                "course_deg": float(math.degrees(math.atan2(ve, vn)) % 360), "distance_m": round(self.distance_m, 1),
                "pos_sigma_m": float(math.sqrt(max(self.P[0, 0], self.P[1, 1]))), "rejected": self.rejected}


def battery_state(history: deque, now_s: float, pct: Optional[float], volts: Optional[float],
                  rate_pct_h: Optional[float] = None) -> Optional[Dict[str, object]]:
    """Battery percentage, time left and a warning level.

    The vest reads the cell through a 100k/100k divider and converts voltage to
    charge with a LiPo curve, so single readings wobble with load; time left is
    the slope of the last 10 minutes of charge once enough history exists."""
    if pct is None or not np.isfinite(pct):
        return None
    history.append((now_s, float(pct)))
    while history and now_s - history[0][0] > 600:
        history.popleft()
    rate = None
    if len(history) >= 10 and history[-1][0] - history[0][0] >= 120:
        t, p = np.asarray(history).T
        slope = float(np.polyfit(t, p, 1)[0]) * 3600          # %/h
        rate = slope if slope < -0.1 else None
    if rate is None and rate_pct_h is not None and np.isfinite(rate_pct_h) and rate_pct_h < -0.1:
        rate = float(rate_pct_h)
    minutes_left = None if rate is None else max(0.0, pct / -rate * 60)
    level = "low" if pct < 15 or (minutes_left is not None and minutes_left < 20) else "ok"
    return {"pct": round(float(pct), 1), "volts": None if volts is None else round(float(volts), 3),
            "rate_pct_h": None if rate is None else round(rate, 1),
            "minutes_left": None if minutes_left is None else round(minutes_left), "level": level}
