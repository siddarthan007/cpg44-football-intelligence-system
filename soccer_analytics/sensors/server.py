"""HTTP / WebSocket ingestion endpoint for wearable data.

The ESP32 streams JSON samples to this endpoint; the analytics pipeline consumes
them through :class:`EndpointSensorSource` (a normal :class:`SensorSource`), so
nothing else in the pipeline changes.

Two deployment shapes, same code:

* **Laptop endpoint** (recommended outdoors) — run the endpoint in the SAME
  process as the pipeline. ESP32 + laptop join one Wi-Fi (e.g. a phone hotspot,
  which sidesteps the university NAT / AP-isolation); the ESP32 POSTs to the
  laptop's LAN IP.
* **Cloud relay** — run this endpoint standalone on a public server. The ESP32
  POSTs there; the laptop pipeline subscribes to the same server. Works through
  NAT because both sides make *outbound* connections.

Wire format (one JSON object per sample; POST a single object or a list to
``/ingest``, or stream objects over ``/ws``):

    {"player_id":7,"t":1699999999.12,"hr":150,"spo2":97,
     "accel":[0.10,-0.20,9.91],"gyro":[...],"mag":[...],"gps":[30.1,76.2]}

``t`` should be epoch seconds; if omitted the server stamps arrival time.
"""
# NOTE: no `from __future__ import annotations` here — FastAPI needs real (not
# stringized) type annotations to inject Request / WebSocket parameters.

import threading
import time
from typing import Callable, List, Optional, Union

from .schema import SensorSample
from .source import SensorSource


def create_app(on_sample: Callable[[SensorSample], None], token: Optional[str] = None):
    """Build the FastAPI app. ``on_sample`` is called for every received sample.
    If ``token`` is set, requests must send header ``X-Auth: <token>``."""
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException

    app = FastAPI(title="soccer-wearable-ingest")

    def _auth(req_token: Optional[str]):
        if token and req_token != token:
            raise HTTPException(status_code=401, detail="bad token")

    def _to_sample(d: dict) -> SensorSample:
        d = dict(d)
        d.setdefault("t", time.time())
        return SensorSample.from_json(d)

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/ingest")
    async def ingest(request: Request):
        _auth(request.headers.get("X-Auth"))
        body = await request.json()
        items = body if isinstance(body, list) else [body]
        ok = 0
        for it in items:
            try:
                on_sample(_to_sample(it))
                ok += 1
            except (KeyError, TypeError, ValueError):
                continue
        return {"received": ok}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                try:
                    d = await websocket.receive_json()   # inside try: a malformed
                    on_sample(_to_sample(d))             # frame is dropped, not fatal
                except (KeyError, TypeError, ValueError):
                    continue
        except WebSocketDisconnect:
            return

    return app


class EndpointSensorSource(SensorSource):
    """A :class:`SensorSource` fed by an embedded HTTP/WebSocket server.

    Runs uvicorn on a daemon thread; incoming samples are pushed onto the queue
    the pipeline drains each frame.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8000,
                 token: Optional[str] = None):
        super().__init__()
        self.host, self.port, self.token = host, port, token
        self._server = None

    def start(self) -> "EndpointSensorSource":
        import uvicorn
        app = create_app(self.push, token=self.token)
        config = uvicorn.Config(app, host=self.host, port=self.port,
                                log_level="warning", loop="asyncio")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        print(f"[sensors] ingest endpoint on http://{self.host}:{self.port} "
              f"(POST /ingest, WS /ws)")
        return self

    def _produce(self):   # not used — samples arrive via HTTP handlers
        return iter(())

    def stop(self):
        super().stop()
        if self._server is not None:
            self._server.should_exit = True


def run_relay(host: str = "0.0.0.0", port: int = 8000, token: Optional[str] = None):
    """Run a standalone cloud-relay endpoint that also re-broadcasts every sample
    to connected WebSocket subscribers (so a remote laptop can subscribe)."""
    import asyncio
    import uvicorn
    from collections import deque
    from fastapi import WebSocket, WebSocketDisconnect

    subscribers: List = []
    lock = threading.Lock()

    def fanout(sample: SensorSample):
        # bounded per-subscriber queues (deque maxlen) → a stalled subscriber drops
        # old samples instead of growing memory without bound.
        with lock:
            subs = list(subscribers)
        for wsq in subs:
            wsq.append(sample)

    app = create_app(fanout, token=token)

    @app.websocket("/subscribe")
    async def subscribe(websocket: WebSocket):
        await websocket.accept()
        q = deque(maxlen=2000)
        with lock:
            subscribers.append(q)
        try:
            while True:
                if q:
                    s = q.popleft()
                    await websocket.send_json({
                        "player_id": s.player_id, "t": s.t, "hr": s.hr,
                        "spo2": s.spo2, "accel": s.accel, "gps": s.gps})
                else:
                    await asyncio.sleep(0.02)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            with lock:
                if q in subscribers:
                    subscribers.remove(q)

    print(f"[relay] listening on :{port} — POST /ingest, subscribe WS /subscribe")
    uvicorn.run(app, host=host, port=port, log_level="warning")
