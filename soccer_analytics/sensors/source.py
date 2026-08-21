"""Wearable data sources — the single hardware-abstraction seam.

All sources push :class:`SensorSample` objects into a thread-safe queue on a
background thread, so the main (vision) loop never blocks waiting on the wearable
— essential for near-real-time operation. Implement a new transport by
subclassing :class:`SensorSource` and yielding samples from :meth:`_produce`.
"""

from __future__ import annotations

import json
import queue
import threading
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
