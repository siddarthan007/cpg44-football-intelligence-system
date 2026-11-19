"""Accelerometer-referenced PPG motion-artifact suppression and quality index.

While a player runs, chest/back tissue moves at the step cadence (about
1.5-3.5 Hz), which sits inside the cardiac band (1-3.5 Hz = 60-210 bpm). A
plain band-pass filter cannot separate them, which is why the previous code
simply withheld heart rate whenever the IMU showed movement.

This module follows the classical two-stage approach:

1. **Multi-axis NLMS adaptive noise cancellation** (Widrow et al. 1975; for PPG
   see Ram et al. 2012, and the per-axis NLMS of the Essex EMBC study). The
   three band-passed accelerometer axes, resampled onto the PPG clock, form
   the reference; the filter output is subtracted from the PPG.
2. **Spectral tracking with cadence suppression** (a light version of TROIKA,
   Zhang et al. IEEE TBME 2015): the cleaned PPG periodogram is attenuated
   around the dominant accelerometer frequency and its harmonic, and the peak
   is chosen with continuity to the previous estimate.

The returned ``sqi`` (0..1) is the fraction of in-band power concentrated at
the chosen fundamental and its harmonic. The enhancement guide's gate is
``sqi >= 0.65`` for exercise heart rate; below that the caller must abstain.
SpO2 is *not* computed here: it is only meaningful in still recovery windows.

Nothing here is a medical device algorithm; validate against a chest-strap
ECG (e.g. Polar H10) using docs/CAPSTONE_ACCURACY_PROTOCOL.md.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.signal import butter, sosfiltfilt, periodogram


def _bandpass(x: np.ndarray, fs: float, lo: float = 0.5, hi: float = 4.0, order: int = 2) -> np.ndarray:
    x = np.asarray(x, float)
    if len(x) < 4 * order * 3:
        return x - np.mean(x)
    sos = butter(order, [lo, min(hi, 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x - np.mean(x))


def resample_to(t_src: Sequence[float], values: np.ndarray, t_dst: Sequence[float]) -> np.ndarray:
    """Linear resampling of (n, k) samples onto new timestamps (same units)."""
    t_src = np.asarray(t_src, float)
    v = np.asarray(values, float).reshape(len(t_src), -1)
    order = np.argsort(t_src)
    t_src, v = t_src[order], v[order]
    return np.column_stack([np.interp(t_dst, t_src, v[:, k]) for k in range(v.shape[1])])


def nlms_cancel(ppg: np.ndarray, ref: np.ndarray, taps: int = 16, mu: float = 0.05,
                eps: float = 1e-6, passes: int = 2) -> np.ndarray:
    """Return ``ppg`` minus the part linearly predictable from ``ref`` (n, k).

    Two passes let the weights converge on short windows; weights are
    normalised by the instantaneous reference power (NLMS) so the step size
    is scale-free. Divergence (non-finite weights) returns the input unchanged.
    """
    d = np.asarray(ppg, float)
    X = np.asarray(ref, float).reshape(len(d), -1)
    n, k = X.shape
    w = np.zeros(taps * k)
    e = d.copy()
    for _ in range(passes):
        for i in range(taps, n):
            u = X[i - taps:i][::-1].reshape(-1)
            y = w @ u
            e[i] = d[i] - y
            w += (mu / (eps + u @ u)) * e[i] * u
        if not np.isfinite(w).all() or np.linalg.norm(w) > 1e3:
            return d
    e[:taps] = d[:taps]
    return e


def spectral_hr(ppg: np.ndarray, fs: float, accel: Optional[np.ndarray] = None,
                previous_bpm: Optional[float] = None, lo_bpm: float = 45.0,
                hi_bpm: float = 210.0, nfft: int = 2048) -> dict:
    """Heart-rate peak with cadence suppression and a spectral quality index."""
    x = np.asarray(ppg, float)
    f, p = periodogram(x * np.hanning(len(x)), fs=fs, nfft=nfft)
    band = (f >= lo_bpm / 60) & (f <= hi_bpm / 60)
    if not band.any() or p[band].sum() <= 0:
        return {"bpm": None, "sqi": 0.0, "cadence_hz": None}
    weight = np.ones_like(p)
    cadence = None
    if accel is not None and len(accel) == len(x):
        axes = np.asarray(accel, float).reshape(len(x), -1)
        pa = 0
        for j in range(axes.shape[1]):
            col = axes[:, j] - axes[:, j].mean()
            fa, pj = periodogram(col * np.hanning(len(col)), fs=fs, nfft=nfft)
            pa = pa + pj
        aband = (fa >= 1.0) & (fa <= 4.0)
        if aband.any() and pa[aband].max() > 0:
            # Only treat the accel peak as cadence when it clearly dominates.
            peak = fa[aband][np.argmax(pa[aband])]
            if pa[aband].max() > 4 * np.median(pa[aband] + 1e-12):
                cadence = float(peak)
                for h in (1.0, 2.0, 0.5):
                    weight *= 1 - 0.85 * np.exp(-0.5 * ((f - h * cadence) / 0.08) ** 2)
    score = p * weight
    if previous_bpm is not None and np.isfinite(previous_bpm):
        # Heart rate changes by at most ~10-15 bpm in a few seconds.
        score = score * (0.35 + 0.65 * np.exp(-0.5 * ((f * 60 - previous_bpm) / 18.0) ** 2))
    idx = np.nonzero(band)[0]
    k = idx[np.argmax(score[band])]
    f0 = f[k]
    # parabolic interpolation for sub-bin precision
    if 0 < k < len(p) - 1 and p[k - 1] > 0 and p[k + 1] > 0:
        a, b, c = np.log(p[k - 1]), np.log(max(p[k], 1e-30)), np.log(p[k + 1])
        denom = a - 2 * b + c
        if abs(denom) > 1e-12:
            delta = 0.5 * (a - c) / denom
            if abs(delta) <= 1.0:
                f0 = f0 + delta * (f[1] - f[0])
    def power_near(freq, width=0.12):
        return p[band & (np.abs(f - freq) <= width)].sum()
    concentrated = power_near(f0) + power_near(2 * f0)
    sqi = float(np.clip(concentrated / p[band].sum(), 0.0, 1.0))
    return {"bpm": float(f0 * 60), "sqi": sqi, "cadence_hz": cadence}


def clean_heart_rate(ir: Sequence[float], ppg_t: Sequence[float], accel: np.ndarray,
                     accel_t: Sequence[float], fs: float = 25.0,
                     previous_bpm: Optional[float] = None) -> dict:
    """End-to-end exercise heart-rate estimate from raw IR and 3-axis accel.

    ``ppg_t`` and ``accel_t`` are in the same clock (seconds). Accel windows
    that do not cover the PPG window fall back to band-pass only.
    """
    ir = np.asarray(ir, float)
    ppg_t = np.asarray(ppg_t, float)
    s = _bandpass(ir, fs)
    ref = None
    if accel is not None and len(accel) >= 8:
        at = np.asarray(accel_t, float)
        if at.min() <= ppg_t.min() + 0.5 and at.max() >= ppg_t.max() - 0.5:
            ref = resample_to(at, accel, ppg_t)
            ref = np.column_stack([_bandpass(ref[:, j], fs) for j in range(ref.shape[1])])
    # MAX30102 IR falls when blood volume rises; invert so systolic peaks are up.
    s = -s
    cleaned = nlms_cancel(s, ref) if ref is not None else s
    est = spectral_hr(cleaned, fs, ref, previous_bpm)
    est["method"] = "nlms+spectral" if ref is not None else "bandpass+spectral"
    est["accel_reference"] = ref is not None
    return est
