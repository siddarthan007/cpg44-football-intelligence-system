"""Kalman filtering for player/ball trajectories.

A constant-acceleration Kalman filter turns noisy per-frame pitch positions into
smooth **position, velocity and acceleration** estimates. This is the single
biggest accuracy lever downstream: raw finite-difference speed is dominated by
detection jitter (which is why unfiltered speeds needed hard clamping), whereas
the filter yields physically plausible velocity and, crucially, acceleration,
which the metabolic-power / Catapult load model in :mod:`soccer_analytics.catapult`
depends on.

State per track: [x, vx, ax, y, vy, ay] (metres, m/s, m/s²). Measurement: (x, y).
During occlusions (no detection) the filter runs predict-only, so a briefly lost
player keeps a sensible trajectory instead of a teleport.

Pure NumPy, unit-testable without the CV stack.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class TrackKalman:
    """Constant-acceleration Kalman filter for one 2-D track (metres)."""

    def __init__(self, meas_noise: float = 0.35, accel_noise: float = 3.0,
                 max_speed: float = 12.0, innovation_gate: float = 13.82):
        # measurement noise (m): homography + foot-point error, ~0.3-0.5 m
        self.R = np.eye(2) * (meas_noise ** 2)
        self.q = accel_noise ** 2           # process (jerk) noise density
        self.max_speed = max_speed
        self.x: Optional[np.ndarray] = None  # 6-state
        self.P = np.eye(6) * 10.0
        self.H = np.zeros((2, 6))
        self.H[0, 0] = 1.0
        self.H[1, 3] = 1.0
        self._initialized = False
        self.innovation_gate = innovation_gate  # chi-square(2), 99.9% gate
        self.measurement_accepted = False
        self.rejected_measurements = 0

    @staticmethod
    def _F(dt: float) -> np.ndarray:
        f = np.eye(6)
        for base in (0, 3):                 # x-block, y-block
            f[base, base + 1] = dt
            f[base, base + 2] = 0.5 * dt * dt
            f[base + 1, base + 2] = dt
        return f

    def _Q(self, dt: float) -> np.ndarray:
        # discrete white-noise-jerk process noise for a [p,v,a] block
        t2, t3, t4, t5 = dt**2, dt**3, dt**4, dt**5
        block = np.array([[t5 / 20, t4 / 8, t3 / 6],
                          [t4 / 8,  t3 / 3, t2 / 2],
                          [t3 / 6,  t2 / 2, dt]]) * self.q
        Q = np.zeros((6, 6))
        Q[:3, :3] = block
        Q[3:, 3:] = block
        return Q

    def init(self, z: Tuple[float, float]):
        self.x = np.array([z[0], 0, 0, z[1], 0, 0], dtype=float)
        self.P = np.eye(6) * 5.0
        self._initialized = True

    def step(self, z: Optional[Tuple[float, float]], dt: float):
        """Advance one frame with measurement ``z`` (or None during occlusion)."""
        self.measurement_accepted = False
        valid = z is not None and np.isfinite(z).all()
        if not self._initialized:
            if valid:
                self.init(z)
                self.measurement_accepted = True
            return
        dt = max(dt, 1e-3)
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

        if valid:
            y = np.array([z[0], z[1]]) - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            innovation = float(y @ np.linalg.solve(S, y))
            if innovation <= self.innovation_gate:
                K = np.linalg.solve(S, self.H @ self.P).T
                self.x = self.x + K @ y
                correction = np.eye(6) - K @ self.H
                # Joseph form keeps covariance positive under long float runs.
                self.P = correction @ self.P @ correction.T + K @ self.R @ K.T
                self.P = 0.5 * (self.P + self.P.T)
                self.measurement_accepted = True
            else:
                self.rejected_measurements += 1

        # clamp velocity to physically plausible range (robust to bad detections)
        sp = self.speed
        if sp > self.max_speed:
            scale = self.max_speed / sp
            self.x[1] *= scale
            self.x[4] *= scale

    # ---- readouts ---- #
    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def position(self) -> Tuple[float, float]:
        return (float(self.x[0]), float(self.x[3])) if self._initialized else (np.nan, np.nan)

    @property
    def velocity(self) -> Tuple[float, float]:
        return (float(self.x[1]), float(self.x[4])) if self._initialized else (0.0, 0.0)

    @property
    def speed(self) -> float:
        if not self._initialized:
            return 0.0
        return float(np.hypot(self.x[1], self.x[4]))

    @property
    def acceleration(self) -> Tuple[float, float]:
        return (float(self.x[2]), float(self.x[5])) if self._initialized else (0.0, 0.0)

    @property
    def accel_mag(self) -> float:
        if not self._initialized:
            return 0.0
        return float(np.hypot(self.x[2], self.x[5]))


class BallKalman(TrackKalman):
    """Ball filter, higher process noise (fast, erratic) and higher speed cap."""

    def __init__(self):
        super().__init__(meas_noise=0.5, accel_noise=12.0, max_speed=40.0)


class KalmanBank:
    """One :class:`TrackKalman` per track id, created on demand.

    Call :meth:`update` every frame with the (possibly missing) measurement; it
    steps that track's filter and returns it so callers can read smoothed
    position / velocity / acceleration. Tracks unseen for ``max_missing`` frames
    are dropped so ids that never return don't leak memory.
    """

    def __init__(self, meas_noise: float = 0.35, accel_noise: float = 3.0,
                 max_speed: float = 12.0, max_missing: int = 50):
        self._kw = dict(meas_noise=meas_noise, accel_noise=accel_noise, max_speed=max_speed)
        self._filters: dict = {}
        self._missing: dict = {}
        self._last_time: dict = {}
        self.max_missing = max_missing

    def update(self, track_id: int, z, dt: float, time_s: Optional[float] = None) -> TrackKalman:
        kf = self._filters.get(track_id)
        if kf is None:
            kf = TrackKalman(**self._kw)
            self._filters[track_id] = kf
        if time_s is not None:
            dt = max(1e-3, time_s - self._last_time.get(track_id, time_s - dt))
            self._last_time[track_id] = time_s
        kf.step(z, dt)
        self._missing[track_id] = 0 if kf.measurement_accepted \
            else self._missing.get(track_id, 0) + 1
        return kf

    def prune(self, active_ids=None):
        if active_ids is not None:
            for tid in self._filters.keys() - set(active_ids):
                self._missing[tid] = self._missing.get(tid, 0) + 1
        dead = [t for t, m in self._missing.items() if m > self.max_missing]
        for t in dead:
            self._filters.pop(t, None)
            self._missing.pop(t, None)
            self._last_time.pop(t, None)
