"""Quality-gated coaching cues from paired external work and internal response.

This is an interpretable within-session comparison, not a trained fatigue or
injury probability. No missing modality is filled with a population average.
"""

from collections import defaultdict, deque
import math
import statistics


class ReadinessEngine:
    def __init__(self):
        self._history = defaultdict(lambda: deque(maxlen=600))
        self._identity = {}

    def update(self, track_id, t, speed, sample, *, player_id=None, position_ok=True, offset_ms=None):
        base = {"status": "signal_limited", "action": "Check the player link",
                "reason": "Paired video and vest readings are needed.",
                "sources": ["video"] if position_ok else [], "model": "paired_response_v1",
                "medical_prediction": False}
        if player_id is None or not position_ok or speed is None or not math.isfinite(speed):
            return base
        if self._identity.get(track_id) != player_id:
            self._history[track_id].clear()
            self._identity[track_id] = player_id
        if (sample is None or sample.clock_synced is not True or sample.hr is None
            or not math.isfinite(sample.hr) or not 35 <= sample.hr <= 230
            or sample.signal_quality is None or sample.signal_quality < 0.6
            or offset_ms is None or abs(offset_ms) > 250):
            base.update(action="Check the vest signal", reason="A fresh, synchronized pulse reading is needed.")
            return base
        history = self._history[track_id]
        if history and t < history[-1][0]:
            history.clear()
        if not history or t - history[-1][0] >= 1.0:
            history.append((float(t), float(speed), float(sample.hr)))
        base.update(status="building_baseline", action="Building a player baseline",
                    reason="Keep video and vest connected for three minutes.", sources=["video", "vest"],
                    observed_seconds=len(history))
        if len(history) < 150 or history[-1][0] - history[0][0] < 180:
            return base
        # Contiguous observations matter: a short burst after an outage is not
        # a minute of supported measurements.
        early = [p for p in history if p[0] <= history[0][0] + 120]
        recent = [p for p in history if p[0] >= t - 60]
        if len(early) < 90 or len(recent) < 45:
            base.update(status="signal_limited", action="Restore paired coverage",
                        reason="There are gaps in this player's comparison window.")
            return base
        drift, matches = [], 0
        for band in range(12):
            old = [p[2] for p in early if band <= p[1] < band + 1]
            new = [p[2] for p in recent if band <= p[1] < band + 1]
            if len(old) >= 10 and len(new) >= 10:
                weight = min(len(old), len(new))
                drift.extend([statistics.median(new) - statistics.median(old)] * weight)
                matches += weight
        if matches < 20:
            base.update(status="signal_limited", action="Wait for comparable activity",
                        reason="The early and recent movement levels differ too much.")
            return base
        early_speed = statistics.mean(p[1] for p in early)
        recent_speed = statistics.mean(p[1] for p in recent)
        change = (recent_speed / early_speed - 1) if early_speed > 0.5 else None
        hr_drift = statistics.median(drift)
        review = change is not None and change <= -0.25 and hr_drift >= 10
        base.update(status="check_in" if review else "monitoring",
                    action="Check in with this player" if review else "Continue monitoring",
                    reason=("Work rate has fallen while pulse is higher at comparable speeds."
                            if review else "No paired decline cue in the current comparison."),
                    hr_drift_bpm=round(hr_drift, 1),
                    work_rate_change_pct=None if change is None else round(change * 100, 1),
                    comparable_samples=matches,
                    comparison="first 120 supported seconds vs recent 60 seconds")
        return base
