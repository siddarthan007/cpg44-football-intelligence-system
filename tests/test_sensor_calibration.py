"""Vest calibration and filtering against synthetic ground truth."""

import math
from collections import deque

import numpy as np

from soccer_analytics.sensors.calibration import (G, AccelAutoCal, GpsKalman, GyroBias, battery_state, player_load)


def test_gyro_bias_learnt_only_when_still():
    rng = np.random.default_rng(0)
    true = np.array([0.03, -0.05, 0.02])
    gb = GyroBias(fs=50)
    for _ in range(200):                                       # moving: rotation + accelerations
        gb.update(true + rng.normal(0, 1.5, 3), np.array([3.0, 1.0, G + 4]))
    assert not gb.calibrated
    for _ in range(60):                                        # 1.2 s on the bench
        gb.update(true + rng.normal(0, 0.003, 3), np.array([0, 0, G]) + rng.normal(0, 0.02, 3))
    assert gb.calibrated and np.allclose(gb.bias, true, atol=0.002)


def test_accel_autocal_recovers_bias_and_scale_from_still_poses():
    rng = np.random.default_rng(1)
    b, s = np.array([0.25, -0.18, 0.4]), np.array([1.03, 0.97, 1.015])
    cal = AccelAutoCal()
    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (1, 1, 1), (-1, 1, -1), (1, -1, 1)]
    for d in dirs:
        u = np.asarray(d, float) / np.linalg.norm(d)
        cal.add_still_mean(G * u / s + b + rng.normal(0, 0.005, 3))
    assert cal.calibrated
    assert np.allclose(cal.bias, b, atol=0.02) and np.allclose(cal.scale, s, atol=0.004)
    assert abs(np.linalg.norm(cal.apply(G * np.array([0, 0, 1]) / s + b)) - G) < 0.02


def test_player_load_is_zero_at_rest_and_grows_with_impacts():
    rest = np.tile([0, 0, 1.0], (100, 1))
    assert player_load(rest) == 0.0
    steps = rest.copy(); steps[::10, 2] += 1.5
    assert player_load(steps) > 0.25


def test_gps_kalman_speed_and_distance_with_doppler():
    rng = np.random.default_rng(2)
    k = GpsKalman()
    lat0, lon0 = 30.35, 76.37
    m_per_deg_lat, m_per_deg_lon = 111_320.0, 111_320.0 * math.cos(math.radians(lat0))
    # 10 s standing, then 20 s running north-east at 6 m/s, at 5 Hz
    for i in range(150):
        t = i * 0.2
        v = 0.0 if t < 10 else 6.0
        d = 0.0 if t < 10 else 6.0 * (t - 10)
        e, n = d * math.sin(math.radians(45)), d * math.cos(math.radians(45))
        lat = lat0 + (n + rng.normal(0, 2.0)) / m_per_deg_lat
        lon = lon0 + (e + rng.normal(0, 2.0)) / m_per_deg_lon
        st = k.update(t, lat, lon, v + rng.normal(0, 0.2), 45 + rng.normal(0, 3) if v else 0.0, hdop=1.0, sats=8)
    assert abs(st["speed_mps"] - 6.0) < 0.4
    assert abs(st["distance_m"] - 6.0 * 19.8) < 8.0               # no jitter distance while standing


def test_battery_time_left_starts_from_the_vests_draw():
    """A flat stretch of the LiPo curve must not promise 25 hours."""
    h = deque()
    out = None
    for i in range(40):                                          # 13 min reading only 1.7 %/h
        out = battery_state(h, i * 20.0, 42.7 - 1.7 * i * 20 / 3600, 3.78)
    assert out["estimate"] == "typical draw" and 350 < out["minutes_left"] < 420   # about 6.5 h, not 25


def test_battery_uses_the_measured_slope_once_it_is_long_and_plausible():
    h = deque()
    out = None
    for i in range(80):                                          # 26 min at 8 %/h
        out = battery_state(h, i * 20.0, 80 - 8 * i * 20 / 3600, 3.9)
    assert out["estimate"] == "measured" and abs(out["rate_pct_h"] + 8) < 0.5
    flat = deque()
    for i in range(80):                                          # 26 min that looks like 1 %/h: not trusted
        out = battery_state(flat, i * 20.0, 60 - 1 * i * 20 / 3600, 3.8)
    assert out["estimate"] == "typical draw"
    assert battery_state(deque(), 0.0, 10.0, 3.5)["level"] == "low"


def _running_profiles(n=20, secs=150, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, secs, 0.2)
    out = {}
    for k in range(n):
        v = np.zeros(len(t)); x = 2.0
        for i in range(len(t)):
            x += 0.2 * (2.5 - x) * 0.2 + rng.normal(0, 0.45)
            if rng.random() < 0.004:
                x += rng.uniform(3, 5)                       # a sprint burst
            x = float(np.clip(x, 0, 8.5)); v[i] = x
        out[k] = (t, v)
    return out


def _vest_from_speed(t, v, offset_s, seed=3):
    rng = np.random.default_rng(seed)
    ti = np.arange(t[0], t[-1], 0.02)                         # 50 Hz IMU
    vi = np.interp(ti, t, v)
    cadence = 1.3 + 0.22 * vi                                # Hz
    phase = 2 * np.pi * np.cumsum(cadence) * 0.02
    impact = (1.0 + 1.7 * vi) * np.clip(np.sin(phase), 0, None) ** 3
    a = np.column_stack([rng.normal(0, 0.5, len(ti)) + 0.4 * impact,
                         rng.normal(0, 0.5, len(ti)) + 1.2,
                         G + impact + rng.normal(0, 0.6, len(ti))])
    return ti + offset_s, a                                   # vest clock runs ahead by offset_s


def test_vest_is_matched_to_its_wearer_and_clock_offset_recovered():
    from soccer_analytics.sensors.association import associate, imu_envelope, track_envelope
    prof = _running_profiles()
    tracks = {}
    for k, (t, v) in prof.items():
        v = v.copy()
        if k % 3 == 0:
            v[(t > 40) & (t < 90)] = np.nan                  # out of the camera view for a while
        tracks[k] = track_envelope(t, v)
    tv, a = _vest_from_speed(*prof[7], offset_s=0.8)
    res = associate({1: imu_envelope(tv, a)}, tracks)[1]
    assert res.track == 7 and res.confident
    assert abs(res.lag_s - 0.8) <= 0.4


def test_two_vests_never_claim_the_same_player():
    from soccer_analytics.sensors.association import associate, imu_envelope, track_envelope
    prof = _running_profiles(seed=5)
    tracks = {k: track_envelope(t, v) for k, (t, v) in prof.items()}
    vests = {1: imu_envelope(*_vest_from_speed(*prof[4], 0.0, seed=1)),
             2: imu_envelope(*_vest_from_speed(*prof[11], -0.6, seed=2))}
    res = associate(vests, tracks)
    assert res[1].track == 4 and res[2].track == 11


def test_live_associator_binds_after_two_confident_runs():
    from soccer_analytics.sensors.association import LiveAssociator
    prof = _running_profiles(n=12, secs=150, seed=7)
    tv, a = _vest_from_speed(*prof[5], offset_s=0.3, seed=4)
    la = LiveAssociator()
    t0 = 1.8e9
    for t, ax in zip(tv, a):
        la.add_vest(9, t0 + t, ax / 9.80665)
    binds = {}
    for k, (t, v) in prof.items():
        for ti, vi in zip(t, v):
            la.add_track(k, t0 + ti, vi)
    for now in np.arange(t0 + 60, t0 + 150, 10):
        binds.update(la.step(now))
    assert binds == {9: 5}


def test_gyro_calibrates_even_with_a_4_percent_accelerometer_scale_error():
    """A real MPU6050 read 9.4 m/s2 at rest; the strict gravity test never saw it as still."""
    rng = np.random.default_rng(2)
    bias = np.array([0.03, -0.02, 0.01])
    cal = GyroBias(fs=50.0)
    for _ in range(120):
        accel = np.array([0.0, 0.0, 9.4]) + rng.normal(0, 0.01, 3)
        cal.update(bias + rng.normal(0, 0.002, 3), accel, accel_tolerance=1.5)
    assert cal.calibrated and np.allclose(cal.bias, bias, atol=0.003)
    moving = GyroBias(fs=50.0)
    for k in range(120):                                   # handled pod: gravity reading wobbles
        accel = np.array([0.0, 0.3 * np.sin(k / 3), 9.4 + 0.4 * np.sin(k / 2)])
        moving.update(bias + rng.normal(0, 0.002, 3), accel, accel_tolerance=1.5)
    assert not moving.calibrated


def test_vest_lying_still_counts_as_still_before_accel_calibration():
    """An uncalibrated accelerometer offset (10.5 m/s2 upside down) must not
    read as constant motion: stillness also follows how steady the reading is."""
    import numpy as np
    from soccer_analytics.sensors.hub import TelemetryEngine
    engine = TelemetryEngine()
    rng = np.random.default_rng(3)
    t0 = 10_000_000_000
    for k in range(150):
        a = np.array([0.7, -0.3, -10.45]) + rng.normal(0, 0.01, 3)
        g = np.array([-0.0445, 0.0019, 0.0050]) + rng.normal(0, 0.001, 3)
        engine._process_imu({"a": a.tolist(), "g": g.tolist(), "m": [27.9, 10.1, -20.0], "device_us": 1_000_000 + k * 20_000,
                             "host_mono_ns": t0 + k * 20_000_000, "host_unix_ns": 1_800_000_000_000_000_000 + k * 20_000_000,
                             "clock_synced": True, "transport_delay_ms": 120.0, "temp_c": 31.0})
    assert engine.state["imu"]["stationary"] is True
