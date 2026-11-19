"""Controlled synthetic evaluation of the 3D ball and vest heart-rate methods.

These are *method-correctness* checks with known ground truth, not field
accuracy: real footage adds calibration error and detector noise, and a real
chest ECG adds electrode movement and sweat. Results: data/benchmarks/physics_synthetic.json

    PYTHONPATH=. python scripts/evaluate_physics.py
"""

import json
from pathlib import Path

import numpy as np

from soccer_analytics.ball3d import _ballistic, reconstruct
from scipy.signal import find_peaks

from soccer_analytics.sensors.ecg import ECGProcessor, detect_r_peaks

ROOT = Path(__file__).resolve().parents[1]


def camera(C=(52.5, 100., 25.), target=(60., 34., 0.), f=1800):
    C, target = np.asarray(C), np.asarray(target)
    z = (target - C) / np.linalg.norm(target - C)
    x = np.cross(z, [0, 0, 1.]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    K = np.array([[f, 0, 960], [0, f, 540], [0, 0, 1.]])
    return K @ np.column_stack((R, -R @ C))


def ball3d_sweep(trials=40):
    P = camera()
    Hi2p = np.linalg.inv(P[:, [0, 1, 3]])
    out = {}
    for noise in (0.5, 1.0, 2.0, 3.0):
        err, peak_err, naive, ok = [], [], [], 0
        for seed in range(trials):
            rng = np.random.default_rng(seed)
            v0 = np.array([rng.uniform(8, 22), rng.uniform(-6, 6), rng.uniform(3, 12)])
            tl = (v0[2] + np.sqrt(v0[2] ** 2 + 2 * 9.81 * 0.1)) / 9.81
            t = np.arange(0, tl, 1 / 25)
            X = _ballistic(np.array([rng.uniform(25, 80), rng.uniform(10, 58), 0.1]), v0, t)
            roll = np.column_stack((X[-1, 0] + 0.6 * v0[0] * np.arange(1, 15) / 25,
                                    X[-1, 1] + 0.6 * v0[1] * np.arange(1, 15) / 25, np.zeros(14)))
            Xa = np.vstack([X, roll])
            h = np.column_stack((Xa, np.ones(len(Xa)))) @ P.T
            uv = h[:, :2] / h[:, 2:3] + rng.normal(0, noise, (len(Xa), 2))
            r = reconstruct(np.arange(len(Xa)), uv, Hi2p, (1920, 1080), 25)
            err.append(float(np.nanmean(np.linalg.norm(r.xyz - Xa, axis=1))))
            air = [s for s in r.segments if s.kind == "air"]
            if air:
                ok += 1
                peak_err.append(abs(max(s.peak_height_m for s in air) - X[:, 2].max()))
            g = np.column_stack((uv, np.ones(len(uv)))) @ Hi2p.T
            naive.append(float(np.mean(np.linalg.norm(g[:, :2] / g[:, 2:3] - Xa[:, :2], axis=1))))
        out[f"noise_{noise}px"] = {
            "mean_3d_error_m": round(float(np.mean(err)), 3), "median_3d_error_m": round(float(np.median(err)), 3),
            "aerial_arc_detected": f"{ok}/{trials}", "median_peak_height_error_m": round(float(np.median(peak_err)), 3) if peak_err else None,
            "ground_homography_error_m": round(float(np.mean(naive)), 3)}
    return out


def false_flight_sweep(trials=60):
    """Rolling balls, many heading towards or away from the camera: how often
    is a flight invented? (The failure seen on real clips.)"""
    P = camera()
    Hi2p = np.linalg.inv(P[:, [0, 1, 3]])
    out = {}
    for noise in (1.0, 2.0, 3.0):
        false_air, peak = 0, []
        for seed in range(trials):
            rng = np.random.default_rng(100 + seed)
            n = rng.integers(6, 30)
            ang = rng.uniform(0, 2 * np.pi)
            sp = rng.uniform(4, 20)
            t = np.arange(n) / 25
            p0 = np.array([rng.uniform(25, 80), rng.uniform(10, 58)])
            xy = p0 + np.outer(sp * t - 1.5 * t * t, [np.cos(ang), np.sin(ang)])
            X = np.column_stack((xy, np.zeros(n)))
            h = np.column_stack((X, np.ones(n))) @ P.T
            uv = h[:, :2] / h[:, 2:3] + rng.normal(0, noise, (n, 2))
            r = reconstruct(np.arange(n), uv, Hi2p, (1920, 1080), 25)
            air = [s for s in r.segments if s.kind == "air"]
            false_air += bool(air)
            if air:
                peak.append(max(s.peak_height_m for s in air))
        out[f"noise_{noise}px"] = {"rolling_balls_given_a_flight": f"{false_air}/{trials}",
                                   "median_invented_peak_m": round(float(np.median(peak)), 2) if peak else None}
    return out


def synth_ecg(bpm, secs, rng, fs=250.0, motion=0.0, noise=0.04):
    """PQRST on a jittered beat train in AD8232 counts; ``motion`` scales step
    impacts, baseline wander and muscle noise (0 at rest, 1 hard running)."""
    t = np.arange(0, secs, 1 / fs)
    beats = [0.4]
    while beats[-1] < secs:
        beats.append(beats[-1] + 60 / bpm * (1 + rng.normal(0, 0.03)))
    beats = np.asarray(beats[:-1])
    x = np.zeros_like(t)
    for b in beats:
        for dt, amp, wid in ((-0.2, 0.12, 0.025), (-0.03, -0.15, 0.010), (0.0, 1.2, 0.012),
                             (0.03, -0.25, 0.010), (0.25, 0.3, 0.045)):
            x += amp * np.exp(-0.5 * ((t - b - dt) / wid) ** 2)
    cad = 2.4 + 0.8 * motion                                   # steps per second
    x += (0.3 + 0.6 * motion) * np.sin(2 * np.pi * 0.3 * t) + rng.normal(0, noise, len(t))
    x += motion * (0.5 * np.sin(np.pi * cad * t) ** 10 + rng.normal(0, 0.12, len(t)) * (np.sin(2 * np.pi * cad / 2 * t) > 0))
    return t, np.clip(2048 + 124 * x, 0, 4095), beats


def ecg_sweep():
    """Beat detection and heart rate on synthetic ECG from rest to hard running.

    The baseline is what a naive monitor does: count peaks above half the
    signal range. The method is the Pan-Tompkins detector and interval
    cleaning used by the hub."""
    rng = np.random.default_rng(0)
    fs, rows = 250.0, []
    for bpm in (60, 90, 120, 150, 175, 190):
        for motion in (0.0, 0.5, 1.0):
            t, x, beats = synth_ecg(bpm, 30.0, rng, fs, motion)
            peaks = detect_r_peaks(x, fs)
            found = t[peaks]
            inside = beats[(beats > 0.5) & (beats < t[-1] - 0.5)]
            tp = sum(np.min(np.abs(found - b)) < 0.05 for b in inside) if len(found) else 0
            sens, ppv = tp / len(inside), tp / max(len(found), 1)
            p = ECGProcessor()
            for k in range(0, len(x), 50):
                p.add_block(t[k], fs, x[k:k + 50], lead_off=False)
            est = p.estimate(window_s=10.0)
            true = 60 / np.median(np.diff(beats[beats >= t[-1] - 10.0]))
            naive_peaks, _ = find_peaks(x, height=np.percentile(x, 50) + 0.5 * (x.max() - np.percentile(x, 50)), distance=int(0.25 * fs))
            naive = 60 / np.median(np.diff(t[naive_peaks][t[naive_peaks] >= t[-1] - 10.0])) if len(naive_peaks) > 3 else np.nan
            err = abs(est["bpm"] - true) if est.get("valid") else np.nan
            rows.append((bpm, motion, sens, ppv, err, abs(naive - true)))
    a = np.asarray(rows, float)
    ok = ~np.isnan(a[:, 4])
    return {"cases": len(a), "beat_sensitivity": round(float(a[:, 2].mean()), 3), "beat_ppv": round(float(a[:, 3].mean()), 3),
            "hr_mae_bpm": round(float(np.nanmean(a[:, 4])), 2), "hr_within_3bpm": f"{int((a[ok, 4] <= 3).sum())}/{len(a)}",
            "hr_reported": f"{int(ok.sum())}/{len(a)}",
            "naive_peak_mae_bpm": round(float(np.nanmean(a[:, 5])), 2),
            "hard_running_mae_bpm": round(float(np.nanmean(a[a[:, 1] == 1.0, 4])), 2)}


if __name__ == "__main__":
    res = {"ball3d_monocular": ball3d_sweep(), "ball3d_false_flights": false_flight_sweep(), "ecg_heart_rate": ecg_sweep(),
           "note": "Synthetic ground truth: verifies the methods; field accuracy needs the protocol in docs/CAPSTONE_ACCURACY_PROTOCOL.md"}
    out = ROOT / "data" / "benchmarks" / "physics_synthetic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
