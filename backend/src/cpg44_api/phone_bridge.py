"""The API's own link to each paired phone.

The API, not the tracking process, holds the phone's WebRTC link. Video then
flows as soon as the phone is paired, so the Cameras page can show a preview
for aiming, Pitch setup can take a still for calibration, and starting or
stopping tracking never renegotiates the link. Tracking reads the frames back
as MJPEG from ``/api/v1/cameras/<id>/mjpeg``, the same way it reads a browser
camera.

Frames are encoded to JPEG only when something asks for them, at most once per
frame and size.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional, Tuple

import cv2

from soccer_analytics.phone_camera import PhoneCamera


class PhoneBridge:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._links: Dict[str, PhoneCamera] = {}
        self._jpeg: Dict[Tuple[str, int], Tuple[int, bytes]] = {}

    def ensure(self, camera_id: str, source: str) -> None:
        with self._lock:
            link = self._links.get(camera_id)
            if link is not None and link.isOpened():
                return
            self._links[camera_id] = PhoneCamera(source, first_frame_timeout_s=0)

    def drop(self, camera_id: str) -> None:
        with self._lock:
            link = self._links.pop(camera_id, None)
            for k in [k for k in self._jpeg if k[0] == camera_id]:
                del self._jpeg[k]
        if link is not None:
            link.release()

    def close(self) -> None:
        for cid in list(self._links):
            self.drop(cid)

    def status(self, camera_id: str) -> Optional[dict]:
        link = self._links.get(camera_id)
        return link.status() if link is not None else None

    def frame_seq(self, camera_id: str) -> int:
        link = self._links.get(camera_id)
        return link.latest()[0] if link is not None else 0

    def jpeg(self, camera_id: str, max_width: int = 0, quality: int = 85) -> Optional[Tuple[int, bytes]]:
        """Newest frame as JPEG, with its sequence number. Blocking (encodes)."""
        link = self._links.get(camera_id)
        if link is None:
            return None
        seq, frame, at = link.latest()
        if frame is None or time.time() - at > 5.0:
            return None
        cache_key = (camera_id, max_width)
        hit = self._jpeg.get(cache_key)
        if hit is not None and hit[0] == seq:
            return hit
        if max_width and frame.shape[1] > max_width:
            h = round(frame.shape[0] * max_width / frame.shape[1])
            frame = cv2.resize(frame, (max_width, h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None
        out = (seq, buf.tobytes())
        self._jpeg[cache_key] = out
        return out
