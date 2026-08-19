"""Wearable data sources — the single hardware-abstraction seam.

All sources push :class:`SensorSample` objects into a thread-safe queue on a
background thread, so the main (vision) loop never blocks waiting on the wearable
— essential for near-real-time operation. Implement a new transport by
subclassing :class:`SensorSource` and yielding samples from :meth:`_produce`.
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from typing import Iterable, List, Optional

from .schema import SensorSample


class SensorSource:
    """Base class. Runs :meth:`_produce` on a daemon thread feeding a queue."""

    def __init__(self, maxsize: int = 10000):
        self._q: "queue.Queue[SensorSample]" = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---- override this ---- #
    def _produce(self) -> Iterable[SensorSample]:
        raise NotImplementedError

    # ---- lifecycle ---- #
    def start(self) -> "SensorSource":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def push(self, s: SensorSample):
        """Enqueue a sample (drop-oldest if full). Safe to call from any thread —
        used by push-based sources like the HTTP/WebSocket endpoint."""
        try:
            self._q.put_nowait(s)
        except queue.Full:
            try:
                self._q.get_nowait()              # drop oldest, keep fresh
                self._q.put_nowait(s)
            except queue.Empty:
                pass

    def _run(self):
        try:
            for s in self._produce():
                if self._stop.is_set():
                    break
                self.push(s)
        except Exception as e:  # never let a sensor thread take down the pipeline
            print(f"[sensor] source stopped: {e}")

    def drain(self) -> List[SensorSample]:
        """Non-blocking: return all samples queued since the last call."""
        out = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return out

    def stop(self):
        self._stop.set()


class SimulatedSensorSource(SensorSource):
    """Generates plausible HR / SpO2 / IMU for a set of players — for development
    and end-to-end testing before the real wearable exists."""

    def __init__(self, player_ids: List[int], hz: float = 10.0, seed: int = 0,
                 duration_s: Optional[float] = None):
        super().__init__()
        self.player_ids = player_ids
        self.hz = hz
        self.duration_s = duration_s
        import numpy as np
        self._rng = np.random.default_rng(seed)

    def _produce(self) -> Iterable[SensorSample]:
        import numpy as np
        dt = 1.0 / self.hz
        base_hr = {p: self._rng.uniform(70, 85) for p in self.player_ids}
        t0 = time.time()
        while not self._stop.is_set():
            now = time.time()
            if self.duration_s and now - t0 > self.duration_s:
                break
            elapsed = now - t0
            for p in self.player_ids:
                # HR rises with match time + noise (fatigue drift)
                hr = base_hr[p] + 55 * (1 - math.exp(-elapsed / 900)) + self._rng.normal(0, 3)
                spo2 = float(np.clip(98 - self._rng.gamma(1.2, 0.8), 90, 100))
                a = self._rng.normal(0, 1.2, 3)
                a[2] += 1.0                       # gravity on z
                yield SensorSample(player_id=p, t=now, hr=round(float(hr), 1),
                                   spo2=round(spo2, 1), accel=tuple(float(v) for v in a),
                                   source="sim")
            time.sleep(dt)


class SerialSensorSource(SensorSource):
    """Reads newline-delimited JSON from a serial port (e.g. ESP32 over USB).

    Each line: ``{"player_id":7,"t":1699...,"hr":142,"spo2":97,"accel":[..],...}``
    """

    def __init__(self, port: str, baud: int = 115200):
        super().__init__()
        self.port, self.baud = port, baud

    def _produce(self) -> Iterable[SensorSample]:
        import serial  # pyserial
        with serial.Serial(self.port, self.baud, timeout=1) as ser:
            while not self._stop.is_set():
                line = ser.readline().decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    yield SensorSample.from_json(json.loads(line))
                except (json.JSONDecodeError, KeyError):
                    continue


class UdpSensorSource(SensorSource):
    """Reads JSON datagrams over UDP (Wi-Fi microcontrollers) — lowest-latency
    wireless option for live matches."""

    def __init__(self, host: str = "0.0.0.0", port: int = 9999):
        super().__init__()
        self.host, self.port = host, port

    def _produce(self) -> Iterable[SensorSample]:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.host, self.port))
        sock.settimeout(1.0)
        while not self._stop.is_set():
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                continue
            try:
                yield SensorSample.from_json(json.loads(data.decode("utf-8", "replace")))
            except (json.JSONDecodeError, KeyError):
                continue
        sock.close()
