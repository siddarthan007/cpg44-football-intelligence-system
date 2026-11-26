"""Two persistent, latest-value workers; neither network nor JPEG blocks inference."""

from __future__ import annotations

import json
import threading
import time

import cv2
import httpx


class LiveSnapshotPublisher:
    def __init__(self, api_base: str, preview_hz: float = 10.0, preview_width: int = 960,
                 camera_id: str = ""):
        self.url = f"{api_base.rstrip('/')}/api/v1/live" if api_base else ""
        # lets the API keep the tracker it started in charge of the Live page
        self.camera_headers = {"X-Camera-Id": camera_id} if camera_id else {}
        self.preview_width = preview_width
        self.preview_interval = 1.0 / max(preview_hz, 1.0)
        self.last_error = None
        self.replaced = {"json": 0, "frame": 0}
        self.sent = {"json": 0, "frame": 0}
        self._pending = {"json": None, "frame": None}
        self._condition = threading.Condition()
        self._closed = False
        self._last_frame = 0.0
        self._workers = []
        if self.url:
            for kind in self._pending:
                thread = threading.Thread(target=self._work, args=(kind,), daemon=True)
                thread.start()
                self._workers.append(thread)

    def _put(self, kind, value):
        with self._condition:
            if self._closed:
                return
            self.replaced[kind] += int(self._pending[kind] is not None)
            self._pending[kind] = value
            self._condition.notify_all()

    def publish(self, payload: dict):
        if self.url:
            self._put("json", payload)

    def publish_frame(self, frame):
        now = time.perf_counter()
        if not self.url or now - self._last_frame < self.preview_interval:
            return
        self._last_frame = now
        height, width = frame.shape[:2]
        if width > self.preview_width:
            frame = cv2.resize(frame, (self.preview_width, round(height * self.preview_width / width)))
        else:
            frame = frame.copy()
        self._put("frame", frame)

    def _work(self, kind):
        with httpx.Client(timeout=0.6, trust_env=False) as client:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._closed or self._pending[kind] is not None)
                    item = self._pending[kind]
                    self._pending[kind] = None
                    if self._closed:
                        return
                try:
                    if kind == "frame":
                        ok, encoded = cv2.imencode(".jpg", item, [cv2.IMWRITE_JPEG_QUALITY, 78])
                        if not ok:
                            raise ValueError("could not encode preview")
                        response = client.post(self.url + "/frame", content=encoded.tobytes(),
                                               headers={"Content-Type": "image/jpeg", **self.camera_headers})
                    else:
                        response = client.post(self.url, content=json.dumps(item, allow_nan=False),
                                               headers={"Content-Type": "application/json", **self.camera_headers})
                    response.raise_for_status()
                    self.sent[kind] += 1
                    self.last_error = None
                except (httpx.HTTPError, ValueError) as exc:
                    self.last_error = str(exc)

    def close(self):
        with self._condition:
            self._closed = True
            self._pending = {"json": None, "frame": None}
            self._condition.notify_all()
        for thread in self._workers:
            thread.join(timeout=1.0)
