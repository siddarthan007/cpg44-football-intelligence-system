"""Transparent football physical-load features from calibrated vision and IMU.

The centrepiece is **metabolic power** (di Prampero 2005 / Osgnach 2010): the
energy cost of accelerated running on grass is estimated from instantaneous
speed and acceleration, giving an external-load estimate. It supports
within-system comparisons after camera validation; it is not claimed to
reproduce a commercial device.

Inputs come from the Kalman filter (:mod:`soccer_analytics.filters`), which
supplies smooth velocity and acceleration — essential, because metabolic power is
extremely sensitive to acceleration noise.

Metrics produced per player:
  * total distance + distance in speed zones (walk/jog/run/HSR/sprint)
  * high-speed-running distance, sprint efforts
  * high acceleration / deceleration efforts
  * metabolic power (avg / peak), high-metabolic-power distance, energy (kcal)
  * vision acceleration-load proxy and, if worn, tri-axial IMU PlayerLoad
  * HR average / drift, min SpO2, and cross-session ACWR

Pure NumPy — unit-testable.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .loadtypes import WorkloadFeatures

G = 9.81

# speed zones (m/s): lower-inclusive bounds. km/h in comments.
SPEED_ZONES = [
    ("walk", 0.0, 2.0),     # <7.2
    ("jog", 2.0, 4.0),      # 7.2-14.4
    ("run", 4.0, 5.5),      # 14.4-19.8
    ("hsr", 5.5, 7.0),      # 19.8-25.2
    ("sprint", 7.0, 99.0),  # >25.2
]
HSR_SPEED = 5.5
SPRINT_SPEED = 7.0
ACC_THRESH = 2.0            # m/s² — high acceleration effort
DEC_THRESH = 2.0           # m/s² — high deceleration effort
HIGH_METABOLIC = 20.0      # W/kg — high metabolic power threshold (Osgnach)


def energy_cost(a_forward: float) -> float:
    """Energy cost of running (J/kg/m) at forward acceleration ``a_forward`` (m/s²),
    di Prampero equivalent-slope model with the Minetti gradient-cost polynomial.
    On flat constant speed (a=0) this returns 3.6 J/kg/m."""
    es = a_forward / G                          # equivalent slope
    es = max(-0.45, min(0.45, es))              # Minetti model validity range
    em = math.sqrt(es * es + 1.0)               # equivalent mass ratio
    c = (155.4 * es**5 - 30.4 * es**4 - 43.3 * es**3
         + 46.3 * es**2 + 19.5 * es + 3.6)
    return c * em


def _zone(speed: float) -> str:
    for name, lo, hi in SPEED_ZONES:
        if lo <= speed < hi:
            return name
    return "sprint"


class _Acc:
    def __init__(self):
        self.dist = 0.0
        self.zones = defaultdict(float)
        self.hsr = 0.0
        self.sprints = 0
        self.in_sprint = False
        self.accel_efforts = 0
        self.in_acc = False
        self.decel_efforts = 0
        self.in_dec = False
        self.pl_vision = 0.0
        self.pl_imu = 0.0
        self.top = 0.0
        self.power_sum = 0.0
        self.power_peak = 0.0
        self.power_n = 0
        self.energy_jkg = 0.0
        self.high_met = 0.0
        self.hr: List[float] = []
        self.spo2: List[float] = []
        self.t = 0.0
        self._last_a = None      # last positional accel vector
        self._last_imu = None    # last IMU accel vector


class LoadEngine:
    """Per-player Catapult-style load accumulator."""

    def __init__(self, mass_kg: float = 75.0):
        self.mass = mass_kg
        self._acc: Dict[int, _Acc] = defaultdict(_Acc)
        self._history: Dict[int, List[float]] = {}
        self._mass_of: Dict[int, float] = {}

    def set_mass(self, player_id: int, kg: float):
        self._mass_of[player_id] = kg

    def set_history(self, player_id: int, past_session_loads: List[float]):
        self._history[int(player_id)] = list(past_session_loads)

    def update(self, player_id: int, dt: float,
               vel: Tuple[float, float], acc: Tuple[float, float],
               hr: Optional[float] = None, spo2: Optional[float] = None,
               imu_accel: Optional[Tuple[float, float, float]] = None):
        """Fold one frame of smoothed (Kalman) motion + optional wearable data."""
        a = self._acc[player_id]
        if dt <= 0:
            return
        a.t += dt
        vx, vy = vel
        ax, ay = acc
        speed = math.hypot(vx, vy)
        a.top = max(a.top, speed)

        # distance + speed zones
        step = speed * dt
        a.dist += step
        a.zones[_zone(speed)] += step
        if speed >= HSR_SPEED:
            a.hsr += step
        # sprint efforts (rising edge with hysteresis so speed hovering around the
        # 7 m/s threshold doesn't count many spurious sprints)
        if speed >= SPRINT_SPEED and not a.in_sprint:
            a.sprints += 1
            a.in_sprint = True
        elif speed < SPRINT_SPEED - 0.5:
            a.in_sprint = False

        # forward (along-track) acceleration
        if speed > 0.3:
            a_fwd = (vx * ax + vy * ay) / speed
        else:
            a_fwd = math.hypot(ax, ay)
        # accel / decel efforts (hysteresis)
        if a_fwd >= ACC_THRESH and not a.in_acc:
            a.accel_efforts += 1; a.in_acc = True
        elif a_fwd < ACC_THRESH * 0.5:
            a.in_acc = False
        if a_fwd <= -DEC_THRESH and not a.in_dec:
            a.decel_efforts += 1; a.in_dec = True
        elif a_fwd > -DEC_THRESH * 0.5:
            a.in_dec = False

        # metabolic power (di Prampero) — needs smooth accel from Kalman.
        # Clamp to a physiological ceiling so a bad-calibration / tracking spike
        # cannot blow up energy totals (elite peaks are ~50-100 W/kg).
        p = min(energy_cost(a_fwd) * speed, 120.0)    # W/kg
        a.power_sum += p; a.power_n += 1
        a.power_peak = max(a.power_peak, p)
        a.energy_jkg += p * dt
        if p >= HIGH_METABOLIC:
            a.high_met += step

        # vision PlayerLoad proxy: rate of change of positional acceleration
        if a._last_a is not None:
            a.pl_vision += math.hypot(ax - a._last_a[0], ay - a._last_a[1]) / 100.0
        a._last_a = (ax, ay)

        # Tri-axial IMU PlayerLoad-style accumulation (if worn).
        if imu_accel is not None:
            if a._last_imu is not None:
                a.pl_imu += math.sqrt(sum((imu_accel[i] - a._last_imu[i]) ** 2
                                          for i in range(3))) / 100.0
            a._last_imu = imu_accel

        if hr is not None:
            a.hr.append(hr)
        if spo2 is not None:
            a.spo2.append(spo2)

    def features(self, player_id: int) -> WorkloadFeatures:
        a = self._acc[player_id]
        mass = self._mass_of.get(player_id, self.mass)
        f = WorkloadFeatures(player_id=player_id)
        f.total_distance = round(a.dist, 1)
        f.distance_by_zone = {k: round(v, 1) for k, v in a.zones.items()}
        f.hsr_distance = round(a.hsr, 1)
        f.sprint_count = a.sprints
        f.accel_efforts = a.accel_efforts
        f.decel_efforts = a.decel_efforts
        f.player_load = round(a.pl_vision, 2)
        f.player_load_imu = round(a.pl_imu, 2) if a._last_imu is not None else float("nan")
        f.metabolic_power_avg = round(a.power_sum / a.power_n, 2) if a.power_n else 0.0
        f.metabolic_power_peak = round(a.power_peak, 2)
        f.high_metabolic_distance = round(a.high_met, 1)
        f.energy_kcal = round(a.energy_jkg * mass / 4184.0, 1)   # J/kg → kcal
        f.top_speed = round(a.top, 2)
        f.duration_s = round(a.t, 1)
        if a.hr:
            f.avg_hr = round(float(np.mean(a.hr)), 1)
            f.max_hr = round(float(np.max(a.hr)), 1)
            n = len(a.hr)
            if n >= 6:
                f.hr_drift = round(float(np.mean(a.hr[-n // 3:]) - np.mean(a.hr[: n // 3])), 1)
        if a.spo2:
            f.min_spo2 = round(float(np.min(a.spo2)), 1)
        f.acwr = self._acwr(player_id, a.dist)
        return f

    def _acwr(self, player_id: int, current_load: float) -> float:
        hist = self._history.get(player_id, [])
        series = hist + [current_load]
        if len(series) < 2:
            return float("nan")
        acute = float(np.mean(series[-7:]))
        chronic = float(np.mean(series[-28:]))
        return round(acute / chronic, 2) if chronic > 1e-6 else float("nan")

    def all_features(self) -> Dict[int, WorkloadFeatures]:
        return {pid: self.features(pid) for pid in self._acc}
