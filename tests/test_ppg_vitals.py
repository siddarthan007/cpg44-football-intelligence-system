"""Synthetic PPG: bandpass + peaks should recover ~72 BPM."""

import numpy as np
from soccer_analytics.sensors.hub import estimate_ppg_vitals


def test_estimate_ppg_vitals_sine_72bpm():
    fs = 25.0
    t = np.arange(0, 8.0, 1.0 / fs)
    hr_hz = 72.0 / 60.0
    ir = 120000 + 4000 * np.sin(2 * np.pi * hr_hz * t)
    red = 90000 + 2200 * np.sin(2 * np.pi * hr_hz * t)
    out = estimate_ppg_vitals(ir, red, fs)
    assert out["valid"], out
    assert abs(out["bpm"] - 72.0) < 6.0
    assert 90.0 <= out["spo2_estimate_pct"] <= 100.5


if __name__ == "__main__":
    test_estimate_ppg_vitals_sine_72bpm()
    print("ppg vitals ok")
