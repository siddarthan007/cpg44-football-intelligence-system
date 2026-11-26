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
PING_EVERY_S = 15.0            # the relay drops a socket that says nothing for 75 s
CLOSE_ENDED, CLOSE_REPLACED = 4404, 4409


class PairingEnded(RuntimeError):
    """The relay refused the code for good: ended, expired or taken over."""


def parse_source(source: str):
    u = urlparse(source)
    if u.scheme != "phone" or not u.netloc:
        raise ValueError("phone source must look like phone://<room>?key=...&relay=https://...")
    q = parse_qs(u.query)
    relay = (q.get("relay") or [""])[0].rstrip("/")
    key = (q.get("key") or [""])[0]
    exp = (q.get("exp") or [""])[0]
    if not relay.startswith(("https://", "http://")) or not key:
        raise ValueError("phone source needs key= and relay= (https URL)")
    ws = relay.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    url = f"{ws}/ws/cam/{u.netloc}?role=base&key={key}" + (f"&exp={exp}" if exp.isdigit() else "")
    return u.netloc, key, url


def source_expiry(source: str) -> Optional[float]:
    exp = (parse_qs(urlparse(source).query).get("exp") or [""])[0]
    return float(exp) if exp.isdigit() else None


class PhoneCamera:
    """Holds one phone link. With ``first_frame_timeout_s`` > 0 the constructor
    waits for video and raises if none comes (the capture-class contract);
    with 0 it returns at once and keeps trying in the background."""

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
        self.relay_connected = False
        self.phone_present = False
        self.ended = False
        self.error: Optional[str] = None
        self._fps = 0.0
        self._pc = None
        self._consumer: Optional[asyncio.Task] = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._thread_main, daemon=True, name=f"phone-{self.room}")
        self._thread.start()
        if first_frame_timeout_s > 0:
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
            return float(self.phone_stats.get("fps") or self._fps or 25.0)
        return 0.0

    def read(self):
        with self._cond:
            self._cond.wait_for(lambda: self._seq > self._delivered or not self._running, timeout=2.0)
            if self._seq <= self._delivered:
                return False, None
            self._delivered = self._seq
            return True, self._frame

    def latest(self):
        """(sequence number, frame, arrival epoch) without consuming it."""
        with self._cond:
            return self._seq, self._frame, self.last_frame_epoch

    def release(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify_all()
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(lambda: None)
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=3.0)

    def status(self) -> dict:
        age = round(time.time() - self.last_frame_epoch, 2) if self.last_frame_epoch else None
        f = self._frame
        return {"room": self.room, "relay": self.relay_connected, "phone": self.phone_present,
                "mode": self.mode if age is not None and age < 3 else "",
                "fps": round(self._fps, 1), "frames": self.frames_received,
                "width": int(f.shape[1]) if f is not None else None, "height": int(f.shape[0]) if f is not None else None,
                "age_s": age, "ended": self.ended, "error": self.error, "phone_stats": self.phone_stats}

    # ---------------------------------------------------------------- internals
    def _push(self, img: np.ndarray, mode: str) -> None:
        now = time.time()
        with self._cond:
            if self.last_frame_epoch:
                gap = now - self.last_frame_epoch
                if 0 < gap < 2:
                    self._fps = 0.9 * self._fps + 0.1 / gap if self._fps else 1.0 / gap
                else:
                    self._fps = 0.0
            self.overwritten_frames += int(self._seq > self._delivered)
            self._frame = img
            self.mode = mode
            self.last_frame_epoch = now
            self.last_frame_mono = time.monotonic()
            self._seq += 1
            self.frames_received += 1
            self._cond.notify_all()

    def _needs_offer(self, since_offer_s: float) -> bool:
        """Ask the phone for a new offer only when the current link is not going
        to deliver: none yet, failed, or connected with no video. A link that is
        still connecting gets 15 s; restarting it sooner can loop forever."""
        state = getattr(self._pc, "connectionState", None)
        if state in (None, "failed", "closed"):
            return since_offer_s > 6
        if state == "connected":
            return since_offer_s > 8
        return since_offer_s > 15

    def _fresh(self) -> bool:
        return self.last_frame_epoch > 0 and time.time() - self.last_frame_epoch < 3.0

    def _video_live(self) -> bool:
        return self.mode == "direct" and self._fresh()

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
        try:
            while self._running:
                try:
                    async with websockets.connect(self._url, max_size=2 ** 21, ping_interval=20,
                                                  open_timeout=15) as ws:
                        self.relay_connected, backoff = True, 1.0
                        self.error = None
                        await self._session(ws)
                except websockets.exceptions.ConnectionClosed as e:
                    code = e.rcvd.code if e.rcvd is not None else None
                    if code in (CLOSE_ENDED, CLOSE_REPLACED):
                        self.ended = True
                        self.error = ("the pairing code has ended; pair the phone again" if code == CLOSE_ENDED
                                      else "another program on this laptop took over the phone")
                        return
                    self.error = f"relay connection closed ({code})"
                except (OSError, asyncio.TimeoutError, websockets.exceptions.WebSocketException) as e:
                    self.error = f"relay connection: {e}"
                finally:
                    self.relay_connected = self.phone_present = False
                if not self._running:
                    break
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10.0)
        finally:
            await self._close_peer()

    async def _close_peer(self) -> None:
        if self._consumer is not None:
            self._consumer.cancel()
            self._consumer = None
        if self._pc is not None:
            pc, self._pc = self._pc, None
            await pc.close()

    async def _session(self, ws) -> None:
        """Signalling on one socket. The WebRTC link itself outlives the socket,
        so a relay restart does not interrupt video that already flows."""
        from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription

        async def consume(track):
            try:
                while self._running:
                    frame = await track.recv()
                    self._push(frame.to_ndarray(format="bgr24"), "direct")
            except Exception:                        # track ended: accept relayed frames again
                if self.mode == "direct":
                    self.mode = ""

        last_ping = time.monotonic()
        asked_at, offered_at = 0.0, 0.0
        while self._running:
            now = time.monotonic()
            if now - last_ping >= PING_EVERY_S:
                await ws.send(json.dumps({"type": "ping", "t": time.time()}))
                last_ping = now
            # The phone offers when it sees us. If no video arrives a few seconds
            # after it is present, ask for a fresh offer (once every 10 s).
            if self.phone_present and not self._fresh() and self._needs_offer(now - offered_at) and now - asked_at > 10:
                await ws.send(json.dumps({"type": "control", "offer": True}))
                asked_at = now
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if isinstance(msg, (bytes, bytearray)):
                img = cv2.imdecode(np.frombuffer(msg, np.uint8), cv2.IMREAD_COLOR)
                if img is not None and not self._video_live():
                    self._push(img, "relay")
                continue
            try:
                body = json.loads(msg)
            except ValueError:
                continue
            kind = body.get("type")
            if kind == "peer":
                self.phone_present = bool(body.get("present"))
                if self.phone_present:
                    asked_at = time.monotonic() - 7        # give the phone's own offer ~3 s first
            elif kind == "stats":
                self.phone_stats = {k: body.get(k) for k in ("mode", "width", "height", "fps", "kbps", "rtt_ms")}
            elif kind == "offer":
                offered_at = time.monotonic()
                await self._close_peer()
                pc = RTCPeerConnection(RTCConfiguration([RTCIceServer(u) for u in ICE_SERVERS]))
                self._pc = pc

                @pc.on("track")
                def on_track(track):
                    if track.kind == "video" and self._pc is pc:
                        self._consumer = asyncio.ensure_future(consume(track))

                await pc.setRemoteDescription(RTCSessionDescription(body["sdp"], "offer"))
                await pc.setLocalDescription(await pc.createAnswer())
                await ws.send(json.dumps({"type": "answer", "id": body.get("id"), "sdp": pc.localDescription.sdp}))
