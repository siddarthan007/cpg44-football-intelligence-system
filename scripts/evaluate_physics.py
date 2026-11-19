"""Controlled synthetic evaluation of the 3D ball and vest heart-rate methods.

These are *method-correctness* checks with known ground truth, not field
accuracy: real footage adds calibration error and detector noise, and real PPG
adds skin contact and tissue effects. Results: data/benchmarks/physics_synthetic.json

    PYTHONPATH=. python scripts/evaluate_physics.py
"""

import json
from pathlib import Path

import numpy as np

from soccer_analytics.ball3d import _ballistic, reconstruct
from soccer_analytics.sensors.motion_artifact import _bandpass, clean_heart_rate, spectral_hr

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


def nlms_sweep():
    rng = np.random.default_rng(0)
    fs, t, ta = 25, np.arange(0, 8, 1 / 25), np.arange(0, 8, 1 / 50)
    rows = []
    for hr_bpm in (70, 95, 120, 150, 175):
        for cad in (1.6, 2.2, 2.8, 3.2):
            acc = np.column_stack([3 * np.sin(2 * np.pi * cad * ta), 1.5 * np.sin(2 * np.pi * cad * ta + .7),
                                   .8 * np.sin(2 * np.pi * 2 * cad * ta)]) + .2 * rng.standard_normal((len(ta), 3))
            ap = np.column_stack([np.interp(t, ta, acc[:, k]) for k in range(3)])
            ir = 80000 + 120 * np.sin(2 * np.pi * hr_bpm / 60 * t) + 500 * ap[:, 0] + 300 * ap[:, 1] + 200 * ap[:, 2] \
                + 5 * rng.standard_normal(len(t))
            base = spectral_hr(_bandpass(ir, fs), fs)["bpm"]
            clean = clean_heart_rate(ir, t, acc, ta, fs)
            rows.append((hr_bpm, cad * 60, abs(base - hr_bpm), abs(clean["bpm"] - hr_bpm), clean["sqi"]))
    a = np.asarray(rows)
    gated = a[a[:, 4] >= 0.5]
    return {"cases": len(a), "bandpass_only_mae_bpm": round(float(a[:, 2].mean()), 2),
            "nlms_mae_bpm": round(float(a[:, 3].mean()), 2),
            "nlms_within_5bpm": f"{int((a[:, 3] <= 5).sum())}/{len(a)}",
            "bandpass_within_5bpm": f"{int((a[:, 2] <= 5).sum())}/{len(a)}",
            "sqi_gate_0.5_yield": f"{len(gated)}/{len(a)}",
            "nlms_mae_after_gate_bpm": round(float(gated[:, 3].mean()), 2) if len(gated) else None}


if __name__ == "__main__":
    res = {"ball3d_monocular": ball3d_sweep(), "ball3d_false_flights": false_flight_sweep(), "ppg_motion_artifact": nlms_sweep(),
           "note": "Synthetic ground truth: verifies the methods; field accuracy needs the protocol in docs/CAPSTONE_ACCURACY_PROTOCOL.md"}
    out = ROOT / "data" / "benchmarks" / "physics_synthetic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
