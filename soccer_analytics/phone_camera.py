"""Phone camera as a live video source (WebRTC, relay-signalled).

``PhoneCamera("phone://<room>?key=<key>&relay=https://cpg44.example")`` behaves
like the other capture classes in :mod:`soccer_analytics.realtime`: ``read()``
returns the newest frame and never queues old ones, so slow inference lowers
the processed frame rate instead of adding delay. See
``backend/src/cpg44_api/relay_camera.py`` for the relay half and the reasons
for the design.

Two transports, chosen automatically:

* ``direct``: WebRTC media from the phone to this machine. The relay only
  forwards the offer and answer. Frames are decoded by aiortc (libavcodec).
* ``relay``: JPEG frames forwarded by the relay when no direct path exists.

Each frame is stamped on arrival with this machine's clock, which is the clock
the vest hub maps sensor time into, so video and vest share one timeline.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Optional
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

ICE_SERVERS = ("stun:stun.cloudflare.com:3478", "stun:stun.l.google.com:19302")


def parse_source(source: str):
    u = urlparse(source)
    if u.scheme != "phone" or not u.netloc:
        raise ValueError("phone source must look like phone://<room>?key=...&relay=https://...")
    q = parse_qs(u.query)
    relay = (q.get("relay") or [""])[0].rstrip("/")
    key = (q.get("key") or [""])[0]
    if not relay.startswith(("https://", "http://")) or not key:
        raise ValueError("phone source needs key= and relay= (https URL)")
    ws = relay.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    return u.netloc, key, f"{ws}/ws/cam/{u.netloc}?role=base&key={key}"


class PhoneCamera:
    def __init__(self, source: str, first_frame_timeout_s: float = 60.0):
        self.room, self._key, self._url = parse_source(source)
        self._cond = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._delivered = 0
        self._running = True
        self.mode = ""
        self.phone_stats: dict = {}
        self.frames_received = 0
        self.overwritten_frames = 0
        self.last_frame_epoch = 0.0
        self.last_frame_mono = 0.0
        self.error: Optional[str] = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        with self._cond:
            self._cond.wait_for(lambda: self._frame is not None or not self._running, timeout=first_frame_timeout_s)
        if self._frame is None:
            self.release()
            raise RuntimeError(self.error or "the phone did not start sending within the timeout; open the QR link and press Start")

    # ------------------------------------------------------------- capture API
    def isOpened(self) -> bool:
        return self._running

    def get(self, prop: int) -> float:
        f = self._frame
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(f.shape[1]) if f is not None else 0.0
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(f.shape[0]) if f is not None else 0.0
        if prop == cv2.CAP_PROP_FPS:
            return float(self.phone_stats.get("fps") or 25.0)
        return 0.0

    def read(self):
        with self._cond:
            self._cond.wait_for(lambda: self._seq > self._delivered or not self._running, timeout=2.0)
            if self._seq <= self._delivered:
                return False, None
            self._delivered = self._seq
            return True, self._frame

    def release(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify_all()
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(lambda: None)
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=3.0)

    def status(self) -> dict:
        return {"room": self.room, "mode": self.mode, "frames": self.frames_received,
                "overwritten": self.overwritten_frames, "phone": self.phone_stats,
                "age_s": round(time.time() - self.last_frame_epoch, 2) if self.last_frame_epoch else None}

    # ---------------------------------------------------------------- internals
    def _push(self, img: np.ndarray, mode: str) -> None:
        with self._cond:
            self.overwritten_frames += int(self._seq > self._delivered)
            self._frame = img
            self.mode = mode
            self.last_frame_epoch = time.time()
            self.last_frame_mono = time.monotonic()
            self._seq += 1
            self.frames_received += 1
            self._cond.notify_all()

    def _thread_main(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as e:                       # surfaced to the caller
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self._running = False
            with self._cond:
                self._cond.notify_all()

    async def _main(self) -> None:
        import websockets
        backoff = 1.0
        while self._running:
            try:
                async with websockets.connect(self._url, max_size=2 ** 21, ping_interval=15) as ws:
                    backoff = 1.0
                    await self._session(ws)
            except (OSError, websockets.exceptions.WebSocketException) as e:
                self.error = f"relay connection: {e}"
            if not self._running:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10.0)

    async def _session(self, ws) -> None:
        from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
        pc: Optional[RTCPeerConnection] = None
        consumer: Optional[asyncio.Task] = None

        async def consume(track):
            try:
                while self._running:
                    frame = await track.recv()
                    self._push(frame.to_ndarray(format="bgr24"), "direct")
            except Exception:                        # track ended: accept relayed frames again
                self.mode = ""

        try:
            while self._running:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if isinstance(msg, (bytes, bytearray)):
                    img = cv2.imdecode(np.frombuffer(msg, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None and self.mode != "direct":
                        self._push(img, "relay")
                    continue
                body = json.loads(msg)
                kind = body.get("type")
                if kind == "stats":
                    self.phone_stats = {k: body.get(k) for k in ("mode", "width", "height", "fps", "kbps", "rtt_ms")}
                    if body.get("mode") == "relay" and self.mode == "direct":
                        self.mode = "relay"
                elif kind == "offer":
                    if pc is not None:
                        await pc.close()
                    pc = RTCPeerConnection(RTCConfiguration([RTCIceServer(u) for u in ICE_SERVERS]))

                    @pc.on("track")
                    def on_track(track):
                        nonlocal consumer
                        if track.kind == "video":
                            consumer = asyncio.ensure_future(consume(track))

                    await pc.setRemoteDescription(RTCSessionDescription(body["sdp"], "offer"))
                    await pc.setLocalDescription(await pc.createAnswer())
                    await ws.send(json.dumps({"type": "answer", "sdp": pc.localDescription.sdp}))
        finally:
            if consumer is not None:
                consumer.cancel()
            if pc is not None:
                await pc.close()
