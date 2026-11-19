"""ECG heart rate against synthetic recordings with known beat times."""

import numpy as np

from soccer_analytics.sensors.ecg import ECGProcessor, clean_intervals, detect_r_peaks

FS = 250.0


def synth_ecg(bpm, secs=20.0, seed=0, noise=0.04, wander=0.6, motion=False):
    """PQRST as Gaussians on a jittered beat train, in AD8232 ADC counts."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, secs, 1 / FS)
    beats = [0.4]
    while beats[-1] < secs:
        beats.append(beats[-1] + 60 / bpm * (1 + rng.normal(0, 0.03)))
    beats = np.asarray(beats[:-1])
    x = np.zeros_like(t)
    for b in beats:
        for dt, amp, wid in ((-0.2, 0.12, 0.025), (-0.03, -0.15, 0.010), (0.0, 1.2, 0.012),
                             (0.03, -0.25, 0.010), (0.25, 0.3, 0.045)):
            x += amp * np.exp(-0.5 * ((t - b - dt) / wid) ** 2)
    x += wander * np.sin(2 * np.pi * 0.3 * t) + rng.normal(0, noise, len(t))
    if motion:                                           # running: step impacts and EMG bursts
        x += 0.35 * np.sin(2 * np.pi * 2.8 * t) ** 8 + rng.normal(0, 0.08, len(t)) * (np.sin(2 * np.pi * 1.4 * t) > 0)
    counts = 2048 + 124 * x                             # 1 mV ~ 124 counts at gain 100, 12 bit, 3.3 V
    return t, np.clip(counts, 0, 4095), beats


def test_r_peaks_found_at_true_beats():
    t, x, beats = synth_ecg(150, seed=1)
    peaks = detect_r_peaks(x, FS)
    found = t[peaks]
    inside = beats[(beats > 0.5) & (beats < t[-1] - 0.5)]
    hits = sum(np.min(np.abs(found - b)) < 0.03 for b in inside)
    assert hits >= 0.97 * len(inside) and len(found) <= len(beats) + 1


def test_heart_rate_within_two_bpm_at_rest_and_running():
    for bpm, motion in ((60, False), (150, True), (185, True)):
        t, x, _ = synth_ecg(bpm, seed=bpm, motion=motion)
        p = ECGProcessor()
        for k in range(0, len(x), 50):
            p.add_block(t[k], FS, x[k:k + 50], lead_off=False)
        r = p.estimate()
        assert r["valid"] and abs(r["bpm"] - bpm) < 2.5, (bpm, r)


def test_electrode_off_and_flat_signal_are_not_reported():
    t, x, _ = synth_ecg(70)
    p = ECGProcessor()
    for k in range(0, len(x), 50):
        p.add_block(t[k], FS, x[k:k + 50], lead_off=k > len(x) - 400)
    assert p.estimate()["reason"] == "electrode_off"
    q = ECGProcessor()
    for k in range(0, 5000, 50):
        q.add_block(k / FS, FS, np.full(50, 2048.0), lead_off=False)
    assert q.estimate()["reason"] == "flat_signal"


def test_artefact_intervals_are_dropped():
    rr = np.array([0.8, 0.81, 0.79, 0.4, 0.82, 0.8, 1.6, 0.81])     # an extra and a missed beat
    ok = clean_intervals(rr)
    assert list(ok) == [True, True, True, False, True, True, False, True]


def test_rmssd_only_when_still():
    t, x, _ = synth_ecg(60, secs=40, seed=3)
    p = ECGProcessor()
    for k in range(0, len(x), 50):
        p.add_block(t[k], FS, x[k:k + 50], lead_off=False)
    assert "rmssd_ms" not in p.estimate(still=False)
    r = p.estimate(still=True)
    assert 10 < r["rmssd_ms"] < 80                       # 3 % beat jitter at 60 bpm is about 40 ms


def test_display_trace_marks_the_true_beats():
    t, x, beats = synth_ecg(120, secs=12.0, seed=3)
    p = ECGProcessor()
    for k in range(0, len(x), 50):
        p.add_block(t[k], FS, x[k:k + 50], lead_off=False)
    tr = p.trace(4.0)
    assert tr["ok"] and not tr["lead_off"] and abs(len(tr["values"]) - 4.0 * tr["fs"]) <= 2
    shown = beats[beats >= tr["t0"]] - tr["t0"]
    assert len(tr["beats"]) == len(shown)
    assert np.max(np.abs(np.asarray(tr["beats"]) - shown)) < 0.03
