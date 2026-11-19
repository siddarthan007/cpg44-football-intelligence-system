"""Heart rate and heart-rate variability from the AD8232 chest ECG.

A chest ECG measures the heart's electrical activity directly, so its beat
times stay usable while running, unlike an optical pulse, whose signal is
swamped by step impacts. Chest-strap ECG is the usual reference that optical
sensors are validated against.

Detection follows Pan & Tompkins (1985), "A real-time QRS detection
algorithm", IEEE TBME 32(3): band-pass 5 to 15 Hz to keep the QRS energy,
derivative, squaring, a 150 ms moving-window integral, then peak picking with
a 250 ms refractory period and an adaptive threshold. Each detection is moved
to the largest deflection of the 0.5 to 40 Hz signal nearby, so beat times are
not delayed by the integrator.

Intervals are cleaned the way HRV software does it: an interval outside 0.27
to 2.0 s (220 to 30 bpm) or more than 30 % away from the median of its
neighbours is an artefact (a missed or extra beat) and is dropped, not
interpolated. RMSSD, the short-term vagal HRV index, is reported only for
still windows of at least 30 s with few artefacts (Task Force of the ESC and
NASPE 1996; Shaffer & Ginsberg 2017).
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


def detect_r_peaks(x: np.ndarray, fs: float) -> np.ndarray:
    """Sample indices of R peaks in a raw ECG segment."""
    x = np.asarray(x, float)
    if len(x) < fs * 2:
        return np.zeros(0, int)
    qrs = sosfiltfilt(butter(2, [5, 15], "bandpass", fs=fs, output="sos"), x)
    wide = sosfiltfilt(butter(2, [0.5, 40], "bandpass", fs=fs, output="sos"), x)
    d = np.gradient(qrs)
    w = max(1, int(0.15 * fs))
    mwi = np.convolve(d * d, np.ones(w) / w, mode="same")
    # adaptive threshold: a fraction of the robust peak level in 2 s blocks
    level = np.zeros_like(mwi)
    blk = int(2 * fs)
    for k in range(0, len(mwi), blk):
        seg = mwi[max(0, k - blk):k + 2 * blk]
        level[k:k + blk] = np.percentile(seg, 98)
    cand, _ = find_peaks(mwi, height=0.3 * level, distance=int(0.25 * fs))
    # the integrator lags the R wave by about half its window: search back
    out = []
    half = int(0.12 * fs)
    sign = 1.0 if np.percentile(wide, 99.5) >= -np.percentile(wide, 0.5) else -1.0
    for c in cand:
        lo, hi = max(0, c - 2 * half), min(len(wide), c + half // 2)
        if hi > lo:
            out.append(lo + int(np.argmax(sign * wide[lo:hi])))
    out = np.unique(np.asarray(out, int))
    keep = [out[0]] if len(out) else []
    for p in out[1:]:
        if p - keep[-1] >= int(0.25 * fs):
            keep.append(p)
    return np.asarray(keep, int)


def clean_intervals(rr: np.ndarray, window: int = 5) -> np.ndarray:
    """Boolean mask of plausible beat-to-beat intervals (seconds)."""
    rr = np.asarray(rr, float)
    ok = (rr >= 0.27) & (rr <= 2.0)
    for i in range(len(rr)):
        lo, hi = max(0, i - window), min(len(rr), i + window + 1)
        ref = np.median(rr[lo:hi][ok[lo:hi]]) if ok[lo:hi].any() else np.nan
        if np.isfinite(ref) and abs(rr[i] - ref) > 0.3 * ref:
            ok[i] = False
    return ok


class ECGProcessor:
    """Rolling ECG buffer on the host clock with heart rate and RMSSD."""

    def __init__(self, seconds: float = 75.0):
        self.t: deque = deque(maxlen=int(seconds * 500))
        self.v: deque = deque(maxlen=int(seconds * 500))
        self.fs = 250.0
        self.lead_off_until = 0.0
        self.last_t = 0.0

    def add_block(self, t0: float, fs: float, values, lead_off: bool) -> None:
        self.fs = float(fs)
        n = len(values)
        ts = t0 + np.arange(n) / self.fs
        if ts[0] < self.last_t:                      # replayed or out-of-order block
            return
        self.t.extend(ts.tolist())
        self.v.extend(float(x) for x in values)
        self.last_t = float(ts[-1])
        if lead_off:
            self.lead_off_until = self.last_t + 2.0   # electrodes off: distrust the next 2 s too

    def estimate(self, now: Optional[float] = None, window_s: float = 10.0, still: bool = False) -> dict:
        if not self.t:
            return {"valid": False, "reason": "no_ecg"}
        now = self.last_t if now is None else now
        if now - self.last_t > 3.0:
            return {"valid": False, "reason": "ecg_stale"}
        if self.lead_off_until > now - window_s:
            return {"valid": False, "reason": "electrode_off"}
        t = np.asarray(self.t)
        v = np.asarray(self.v)
        m = t >= self.last_t - max(window_s, 60.0)
        t, v = t[m], v[m]
        if len(v) < self.fs * 4 or np.std(v) < 2.0:  # flat line: board unpowered or cable out
            return {"valid": False, "reason": "flat_signal"}
        peaks = detect_r_peaks(v, self.fs)
        beat_t = t[peaks]
        rr = np.diff(beat_t)
        if len(rr) < 4:
            return {"valid": False, "reason": "too_few_beats"}
        ok = clean_intervals(rr)
        recent = (beat_t[1:] >= self.last_t - window_s)
        good = rr[ok & recent]
        quality = float(ok[recent].mean()) if recent.any() else 0.0
        if len(good) < 3 or quality < 0.6:
            return {"valid": False, "reason": "noisy_ecg", "quality": round(quality, 2)}
        bpm = 60.0 / float(np.median(good))
        out = {"valid": True, "bpm": round(bpm, 1), "quality": round(quality, 2), "beats": int(len(good)),
               "source": "ecg"}
        # RMSSD on successive clean pairs over the last 30 s, still periods only
        span = beat_t[-1] - beat_t[0]
        if still and span >= 30 and ok.mean() >= 0.9:
            pair = ok[1:] & ok[:-1]
            d = np.diff(rr)[pair]
            if len(d) >= 20:
                out["rmssd_ms"] = round(float(np.sqrt(np.mean(d * d)) * 1000), 1)
        return out
