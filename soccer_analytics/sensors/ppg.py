"""MAX30102 PPG Digital Signal Processing & Biometric Extraction Engine.

Implements Butterworth bandpass filtering (0.7 - 3.5 Hz), Pan-Tompkins adaptive
derivative peak detection, and AC/DC ratio-of-ratios SpO2 calculation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks


class PPGProcessor:
    """Processes raw red and infrared PPG channels into heart rate and SpO2."""

    def __init__(self, fs: float = 25.0):
        self.fs = fs
        # 4th order Butterworth bandpass (0.7 Hz to 3.5 Hz => 42 to 210 BPM)
        self.sos = butter(4, [0.7, 3.5], btype="bandpass", fs=fs, output="sos")

    def process(self, ir_signal: Sequence[float], red_signal: Sequence[float]) -> Dict[str, Optional[float]]:
        """Extracts HR (BPM), SpO2 (%), signal quality, and confidence."""
        ir = np.asarray(ir_signal, dtype=np.float64)
        red = np.asarray(red_signal, dtype=np.float64)

        if len(ir) < int(self.fs * 3.0) or len(red) < int(self.fs * 3.0):
            return {
                "valid": False,
                "bpm": None,
                "spo2_pct": None,
                "quality": 0.0,
                "confidence": 0.0,
            }

        # DC components (baseline offset)
        dc_ir = float(np.median(ir))
        dc_red = float(np.median(red))

        if dc_ir < 5000 or dc_red < 5000:
            return {
                "valid": False,
                "bpm": None,
                "spo2_pct": None,
                "quality": 0.0,
                "confidence": 0.0,
            }

        # AC components via zero-phase forward-backward filtering
        ac_ir = sosfiltfilt(self.sos, ir - dc_ir)
        ac_red = sosfiltfilt(self.sos, red - dc_red)

        # Pan-Tompkins style peak detection on IR AC waveform
        min_dist = int(self.fs * 0.35)  # Max 170 BPM refractory limit
        prominence = float(np.std(ac_ir) * 0.6)
        peaks, _ = find_peaks(ac_ir, distance=min_dist, prominence=prominence)

        if len(peaks) < 3:
            return {
                "valid": False,
                "bpm": None,
                "spo2_pct": None,
                "quality": 0.2,
                "confidence": 0.0,
            }

        # Inter-Beat Intervals (IBI) in seconds
        ibis = np.diff(peaks) / self.fs
        # Remove outlier IBIs (beyond 0.28s - 1.5s => 40 - 214 BPM)
        valid_ibis = ibis[(ibis >= 0.28) & (ibis <= 1.5)]

        if len(valid_ibis) < 2:
            return {
                "valid": False,
                "bpm": None,
                "spo2_pct": None,
                "quality": 0.3,
                "confidence": 0.0,
            }

        mean_ibi = float(np.median(valid_ibis))
        bpm = round(60.0 / mean_ibi, 1)

        # SpO2 Calculation via Ratio-of-Ratios (AC/DC)
        # R = (AC_red_rms / DC_red) / (AC_ir_rms / DC_ir)
        ac_red_rms = float(np.sqrt(np.mean(ac_red**2)))
        ac_ir_rms = float(np.sqrt(np.mean(ac_ir**2)))

        spo2_pct = None
        if ac_ir_rms >= 1e-6 and dc_red >= 1e-6 and dc_ir >= 1e-6:
            r_ratio = (ac_red_rms / dc_red) / (ac_ir_rms / dc_ir)
            # Empirical ratio curve. This remains an uncalibrated estimate and is
            # invalid rather than replaced with a healthy-looking default.
            spo2_est = 110.0 - 25.0 * r_ratio
            if 70.0 <= spo2_est <= 100.5:
                spo2_pct = round(float(spo2_est), 1)

        # Signal Quality Index (SQI) based on IBI regularity
        ibi_std = float(np.std(valid_ibis))
        quality = round(float(np.clip(1.0 - (ibi_std / mean_ibi), 0.0, 1.0)), 2)
        confidence = round(float(np.clip(quality * (len(valid_ibis) / 5.0), 0.0, 1.0)), 2)

        return {
            "valid": spo2_pct is not None,
            "bpm": bpm,
            "spo2_pct": spo2_pct,
            "quality": quality,
            "confidence": confidence,
            "ibi_ms": round(mean_ibi * 1000.0, 1),
        }
