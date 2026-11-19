"""Offline report of a recorded vest session (wearable_raw.ndjson).

Replays the raw relay log through the same processing used live:

* sampling health: achieved IMU, ECG and GNSS rates, gaps, transport delay;
* external load: PlayerLoad (sum of tri-axial jerk / 100, in g), dynamic
  acceleration and impacts above 4 g;
* internal load: chest-ECG heart rate every 4 s over 10 s windows (beat
  detection in ``ecg.py``). Windows with electrodes off or too few clean beats
  are reported as withheld, never filled.

Output is a compact JSON for the vest page and the evidence ledger. The vest
is a separate recording: nothing here is paired with video unless a capture
epoch anchor links them.

    PYTHONPATH=. python -m soccer_analytics.sensors.session_report data/live_sessions/<id>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .ecg import ECGProcessor

G = 9.80665
ECG_BLOCK = 50            # samples per ECG packet from the firmware


def load(path: Path):
    imu, ecg, gps, delays = [], [], [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            t = r.get("source_unix_ns") or r.get("host_unix_ns")
            if t is None:
                continue
            t = t / 1e9
            if r.get("transport_delay_ms") is not None:
                delays.append(float(r["transport_delay_ms"]))
            kind = r.get("t")
            if kind == "imu" and "a" in r:
                imu.append((t, *r["a"], *r.get("g", [np.nan] * 3)))
            elif kind == "ecg" and r.get("v"):
                ecg.append((t, float(r.get("fs") or 250), [float(x) for x in r["v"]], bool(r.get("lead_off"))))
            elif kind == "gps":
                gps.append((t, bool(r.get("fix")), r.get("lat"), r.get("lon"), r.get("speed_mps")))
    ecg.sort(key=lambda b: b[0])
    return np.asarray(imu, float).reshape(-1, 7), ecg, gps, np.asarray(delays, float)


def _rate(t):
    if len(t) < 3:
        return None, None
    d = np.diff(np.sort(t))
    return float(1.0 / np.median(d)), float((d > 5 * np.median(d)).sum())


def report(session_dir: str | Path) -> dict:
    session_dir = Path(session_dir)
    imu, ecg, gps, delays = load(session_dir / "wearable_raw.ndjson")
    if len(imu) == 0 and not ecg:
        return {"session": session_dir.name, "valid": False, "reason": "no samples"}
    et = np.asarray([b[0] for b in ecg], float)
    t0 = float(np.nanmin([imu[:, 0].min() if len(imu) else np.inf, et.min() if len(et) else np.inf]))
    imu_rate, imu_gaps = _rate(imu[:, 0]) if len(imu) else (None, None)
    block_rate, ecg_gaps = _rate(et) if len(et) else (None, None)
    out = {
        "session": session_dir.name, "valid": True, "start_unix": t0,
        "duration_s": round(float(max(imu[:, 0].max() if len(imu) else 0, et.max() if len(et) else 0) - t0), 1),
        "sampling": {"imu_hz": imu_rate and round(imu_rate, 1), "imu_gaps": imu_gaps,
                     "ecg_hz": block_rate and round(block_rate * ECG_BLOCK, 1), "ecg_gaps": ecg_gaps,
                     "imu_completeness": None,
                     "ecg_completeness": None,
                     "gps_messages": len(gps), "gps_fix_share": round(sum(g[1] for g in gps) / max(1, len(gps)), 3),
                     "transport_delay_ms_p50": None if not len(delays) else round(float(np.percentile(delays, 50)), 1),
                     "transport_delay_ms_p95": None if not len(delays) else round(float(np.percentile(delays, 95)), 1)},
    }
    def active(t):
        """Seconds actually streaming (inter-sample gaps < 5 s) and longest outage."""
        if len(t) < 2:
            return 0.0, 0.0
        d = np.diff(np.sort(t))
        return float(d[d < 5.0].sum()), float(d.max())
    imu_act, imu_out = active(imu[:, 0]) if len(imu) else (0.0, 0.0)
    ecg_act, _ = active(et) if len(et) else (0.0, 0.0)
    out["sampling"]["active_s"] = round(imu_act, 1)
    out["sampling"]["longest_outage_s"] = round(imu_out, 1)
    out["sampling"]["imu_completeness"] = round(min(1.0, len(imu) / max(50.0 * imu_act, 1)), 3)
    if len(et):
        out["sampling"]["ecg_completeness"] = round(min(1.0, len(et) * ECG_BLOCK / max(ecg[0][1] * ecg_act, 1)), 3)
    # ---- external load
    series = []
    if len(imu) >= 10:
        order = np.argsort(imu[:, 0])
        imu = imu[order]
        a_g = imu[:, 1:4] / G
        jerk = np.sqrt((np.diff(a_g, axis=0) ** 2).sum(1))
        pl = np.r_[0.0, np.cumsum(jerk) / 100.0]
        dyn = np.abs(np.linalg.norm(imu[:, 1:4], axis=1) - G)
        impacts = int((np.linalg.norm(a_g, axis=1) > 4.0).sum())
        out["external"] = {"player_load": round(float(pl[-1]), 2),
                           "player_load_per_min": round(float(pl[-1]) / max(out["duration_s"] / 60, 1e-6), 2),
                           "impacts_over_4g": impacts,
                           "note": "PlayerLoad at 50 Hz under-reads 100 Hz commercial units; compare like with like"}
    # ---- internal load: ECG heart rate every 4 s over 10 s windows
    hr_rows = []
    if ecg:
        proc = ECGProcessor(seconds=75.0)
        k = 0
        for now in np.arange(et[0] + 10.0, et[-1] + 1e-9, 4.0):
            while k < len(ecg) and ecg[k][0] <= now:
                t_b, fs, v, off = ecg[k]
                proc.add_block(t_b, fs, v, off)
                k += 1
            if now - proc.last_t > 3.0:
                hr_rows.append((now - t0, None, 0.0, "gap"))
                continue
            est = proc.estimate(now=now, window_s=10.0)
            ok = bool(est.get("valid"))
            reason = "ok" if ok else ("electrode_off" if est.get("reason") == "electrode_off" else "withheld")
            hr_rows.append((now - t0, est["bpm"] if ok else None, float(est.get("quality") or 0.0), reason))
        valid = [r for r in hr_rows if r[1] is not None]
        with_data = [r for r in hr_rows if r[3] != "gap"]
        out["internal"] = {
            "windows": len(hr_rows), "valid_windows": len(valid), "windows_with_data": len(with_data),
            "electrode_contact_share": round(sum(r[3] != "electrode_off" for r in with_data) / max(1, len(with_data)), 3),
            "hr_median": round(float(np.median([r[1] for r in valid])), 1) if valid else None,
            "hr_max": round(float(np.max([r[1] for r in valid])), 1) if valid else None,
            "method": "Pan-Tompkins R peaks, interval cleaning, 10 s windows",
        }
    # ---- compact per-10 s series for charts
    if len(imu):
        for s in np.arange(t0, t0 + out["duration_s"], 10.0):
            mi = (imu[:, 0] >= s) & (imu[:, 0] < s + 10)
            dyn_rms = float(np.sqrt(np.mean(dyn[mi] ** 2))) if mi.any() else None
            hrs = [r[1] for r in hr_rows if s - t0 <= r[0] < s - t0 + 10 and r[1] is not None]
            series.append({"t": round(s - t0, 1), "dyn_rms": None if dyn_rms is None else round(dyn_rms, 3),
                           "hr": round(float(np.median(hrs)), 1) if hrs else None})
    out["series"] = series
    return out


def main(argv=None):
    argv = argv or sys.argv[1:]
    out_dir = Path(__file__).resolve().parents[2] / "data" / "vest_sessions"
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in argv:
        r = report(d)
        (out_dir / f"{Path(d).name}.json").write_text(json.dumps(r, indent=1))
        print(json.dumps({k: v for k, v in r.items() if k != "series"}, indent=1))


if __name__ == "__main__":
    main()
