"""Bounded wall-time diagnostics, kept separate from source-time measurements."""

from collections import defaultdict, deque
import time
import numpy as np


class StageProfiler:
    def __init__(self, capacity=300):
        self.samples = defaultdict(lambda: deque(maxlen=capacity))
        self._last = time.perf_counter()

    def start(self):
        self._last = time.perf_counter()

    def mark(self, stage):
        now = time.perf_counter()
        self.samples[stage].append((now - self._last) * 1000)
        self._last = now

    def summary(self):
        return {key: {"p50_ms": round(float(np.percentile(values, 50)), 2),
                      "p95_ms": round(float(np.percentile(values, 95)), 2), "samples": len(values)}
                for key, values in self.samples.items() if values}
