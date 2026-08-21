#!/usr/bin/env python3
"""Stateless, authenticated CPG44 telemetry relay.

The field PC remains the authority for sensor processing and clock mapping. This
service preserves the source timestamp, adds a relay receive timestamp and a
monotonic relay sequence, and holds only a bounded in-memory replay window. It
has no database, filesystem volume, Redis, MinIO or Postgres dependency.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

from fastapi import Body, FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn


RELAY_TOKEN = os.environ.get("CPG44_RELAY_TOKEN", "").strip()
MAX_CACHE_BYTES = max(
    1_048_576,
    min(int(os.environ.get("CPG44_RELAY_CACHE_BYTES", 4_194_304)), 16_777_216),
)
MAX_BODY_BYTES = max(
    65_536,
    min(int(os.environ.get("CPG44_RELAY_MAX_BODY_BYTES", 262_144)), 1_048_576),
)
MATCH_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
SOURCE_VALUES = {"synchronized_local_hub", "wearable"}
RAW_SAMPLE_TYPES = {"imu", "ppg", "gps", "status"}
RELAY_INSTANCE_ID = secrets.token_hex(8)


def _float(value: Any, minimum: float, maximum: float) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not minimum <= number <= maximum:
        return None
    return number


def _vector(value: Any, length: int, limit: float) -> Optional[List[float]]:
    if not isinstance(value, list) or len(value) != length:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) and abs(item) <= limit for item in result):
        return None
    return result


def _integer(value: Any, minimum: int, maximum: int) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if isinstance(value, bool) or not minimum <= number <= maximum:
        return None
    return number


def _raw_payload(sample_type: Optional[str], value: Any) -> Optional[Dict[str, Any]]:
    """Return the small, validated raw sample needed by the field processor."""
    if sample_type is None:
        return None
    if sample_type not in RAW_SAMPLE_TYPES or not isinstance(value, dict):
        return None
    device_us = _integer(value.get("device_us"), 0, 9_000_000_000_000_000)
    if device_us is None:
        return None
    clean: Dict[str, Any] = {"device_us": device_us}

    if sample_type == "imu":
        accel = _vector(value.get("a"), 3, 320.0)
        gyro = _vector(value.get("g"), 3, 80.0)
        if accel is None or gyro is None:
            return None
        clean.update({"a": accel, "g": gyro})
        temperature = _float(value.get("temp_c"), -50.0, 125.0)
        if temperature is not None:
            clean["temp_c"] = temperature
        return clean

    if sample_type == "ppg":
        red = _integer(value.get("red"), 0, 262_143)
        infrared = _integer(value.get("ir"), 0, 262_143)
        if red is None or infrared is None:
            return None
        clean.update({"red": red, "ir": infrared})
        return clean

    if sample_type == "gps":
        clean.update({
            "rx": bool(value.get("rx")),
            "fix": bool(value.get("fix")),
            "sat": _integer(value.get("sat"), 0, 128) or 0,
            "chars": _integer(value.get("chars"), 0, 4_294_967_295) or 0,
        })
        if clean["fix"]:
            latitude = _float(value.get("lat"), -90.0, 90.0)
            longitude = _float(value.get("lon"), -180.0, 180.0)
            if latitude is None or longitude is None:
                return None
            clean.update({"lat": latitude, "lon": longitude})
            for key, low, high in (
                ("speed_mps", 0.0, 150.0),
                ("course_deg", 0.0, 360.0),
                ("alt_m", -500.0, 20_000.0),
                ("hdop", 0.0, 99.99),
            ):
                number = _float(value.get(key), low, high)
                if number is not None:
                    clean[key] = number
        return clean

    rssi = _integer(value.get("rssi_dbm"), -127, 20)
    heap = _integer(value.get("heap"), 0, 64_000_000)
    if rssi is None or heap is None:
        return None
    clean.update({
        "rssi_dbm": rssi,
        "heap": heap,
        "mpu6050": bool(value.get("mpu6050")),
        "max30102": bool(value.get("max30102")),
        "gps_rx": bool(value.get("gps_rx")),
        "dropped_samples": _integer(value.get("dropped_samples"), 0, 4_294_967_295) or 0,
        "queued_samples": _integer(value.get("queued_samples"), 0, 100_000) or 0,
    })
    return clean


def _source_timestamp_ns(raw: dict) -> Optional[int]:
    direct = raw.get("source_timestamp_ns")
    if direct is not None:
        try:
            value = int(direct)
        except (TypeError, ValueError):
            return None
    else:
        seconds = _float(raw.get("timestamp", raw.get("t")), 1_577_836_800.0, 4_102_444_800.0)
        if seconds is None:
            return None
        value = int(seconds * 1_000_000_000)
    # Allow delayed reconnection while rejecting accidental monotonic/device clocks.
    if abs(value - time.time_ns()) > 7 * 24 * 60 * 60 * 1_000_000_000:
        return None
    return value


@dataclass
class RelayEnvelope:
    relay_seq: int
    event_id: str
    match_id: str
    player_id: int
    source: str
    source_seq: Optional[int]
    source_timestamp_ns: int
    relay_received_ns: int
    relay_delay_ms: float
    device_boot_id: Optional[str]
    sample_type: Optional[str]
    payload: Optional[Dict[str, Any]]
    hr: Optional[float]
    spo2: Optional[float]
    accel: Optional[List[float]]
    accel_unit: Optional[str]
    gyro: Optional[List[float]]
    gps: Optional[List[float]]
    player_load: Optional[float]
    signal_quality: float
    clock: Dict[str, Any]
    tags: Dict[str, Any]


class MemoryRelay:
    """Byte-bounded replay cache with idempotent event IDs."""

    def __init__(self, max_bytes: int = MAX_CACHE_BYTES):
        self.max_bytes = int(max_bytes)
        self.history: Deque[Tuple[RelayEnvelope, int]] = deque()
        self.history_bytes = 0
        self.latest: Dict[Tuple[str, int], RelayEnvelope] = {}
        self.event_sequences: Dict[str, int] = {}
        self.sequence = 0
        self.accepted_count = 0
        self.duplicate_count = 0
        self.rejected_count = 0
        self.started_at = time.time()
        self.ws_clients: Dict[WebSocket, Tuple[asyncio.Queue, Optional[str], Optional[int]]] = {}
        self._lock = threading.RLock()

    def _event_id(self, raw: dict, match_id: str, player_id: int,
                  source_timestamp_ns: int, source_seq: Optional[int]) -> str:
        supplied = str(raw.get("event_id") or "").strip()
        if supplied and len(supplied) <= 160 and all(31 < ord(char) < 127 for char in supplied):
            return supplied
        material = (
            f"{match_id}|{player_id}|{raw.get('sample_type')}|"
            f"{source_timestamp_ns}|{source_seq}"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _clock(raw: dict) -> Dict[str, Any]:
        value = raw.get("clock")
        if not isinstance(value, dict):
            return {"valid": False}
        return {
            "valid": bool(value.get("valid")),
            "method": str(value.get("method") or "unspecified")[:64],
            "drift_ppm": _float(value.get("drift_ppm"), -5000.0, 5000.0),
            "best_rtt_ms": _float(value.get("best_rtt_ms"), 0.0, 60_000.0),
            "samples": int(value.get("samples") or 0),
        }

    @staticmethod
    def _tags(raw: dict, match_id: str, player_id: int,
              device_boot_id: Optional[str]) -> Dict[str, Any]:
        supplied = raw.get("tags") if isinstance(raw.get("tags"), dict) else {}
        tags: Dict[str, Any] = {}
        for key in ("session_id", "device_id", "jersey", "squad", "role"):
            value = supplied.get(key)
            if isinstance(value, (str, int, float, bool)) and len(str(value)) <= 64:
                tags[key] = value
        tags.update({"match_id": match_id, "player_id": player_id})
        if device_boot_id is not None:
            tags["device_boot_id"] = device_boot_id
        return tags

    def register(self, raw: Any) -> Tuple[Optional[RelayEnvelope], bool]:
        if not isinstance(raw, dict):
            self.rejected_count += 1
            return None, False
        try:
            player_id = int(raw.get("player_id"))
        except (TypeError, ValueError):
            self.rejected_count += 1
            return None, False
        match_id = str(raw.get("match_id") or "").strip()
        source = str(raw.get("source") or "").strip()
        source_timestamp_ns = _source_timestamp_ns(raw)
        if (
            not 1 <= player_id <= 9999
            or not MATCH_PATTERN.fullmatch(match_id)
            or source not in SOURCE_VALUES
            or source_timestamp_ns is None
        ):
            self.rejected_count += 1
            return None, False

        source_seq = raw.get("source_seq")
        try:
            source_seq = int(source_seq) if source_seq is not None else None
        except (TypeError, ValueError):
            self.rejected_count += 1
            return None, False
        if source_seq is not None and source_seq < 0:
            self.rejected_count += 1
            return None, False

        device_boot_id = raw.get("device_boot_id")
        if device_boot_id is not None:
            device_boot_id = str(device_boot_id)[:64]
        event_id = self._event_id(raw, match_id, player_id, source_timestamp_ns, source_seq)
        sample_type_value = raw.get("sample_type")
        sample_type = str(sample_type_value).strip() if sample_type_value is not None else None
        payload = _raw_payload(sample_type, raw.get("payload"))
        if sample_type is not None and payload is None:
            self.rejected_count += 1
            return None, False

        with self._lock:
            existing_seq = self.event_sequences.get(event_id)
            if existing_seq is not None:
                self.duplicate_count += 1
                for envelope, _ in reversed(self.history):
                    if envelope.relay_seq == existing_seq:
                        return envelope, True

            accel_unit = raw.get("accel_unit") if raw.get("accel_unit") in {"g", "m/s2"} else None
            accel_limit = 32.0 if accel_unit == "g" else 320.0
            gps = raw.get("gps")
            gps = _vector(gps, len(gps), 100_000.0) if isinstance(gps, list) and len(gps) in (2, 3) else None
            if gps and (not -90 <= gps[0] <= 90 or not -180 <= gps[1] <= 180):
                gps = None

            received_ns = time.time_ns()
            self.sequence += 1
            envelope = RelayEnvelope(
                relay_seq=self.sequence,
                event_id=event_id,
                match_id=match_id,
                player_id=player_id,
                source=source,
                source_seq=source_seq,
                source_timestamp_ns=source_timestamp_ns,
                relay_received_ns=received_ns,
                relay_delay_ms=round((received_ns - source_timestamp_ns) / 1_000_000.0, 3),
                device_boot_id=device_boot_id,
                sample_type=sample_type,
                payload=payload,
                hr=_float(raw.get("hr"), 25.0, 240.0),
                spo2=_float(raw.get("spo2"), 50.0, 100.0),
                accel=_vector(raw.get("accel"), 3, accel_limit),
                accel_unit=accel_unit,
                gyro=_vector(raw.get("gyro"), 3, 4000.0),
                gps=gps,
                player_load=_float(raw.get("player_load"), 0.0, 1_000_000.0),
                signal_quality=_float(raw.get("signal_quality"), 0.0, 1.0) or 0.0,
                clock=self._clock(raw),
                tags=self._tags(raw, match_id, player_id, device_boot_id),
            )
            encoded_size = len(json.dumps(asdict(envelope), separators=(",", ":")))
            self.history.append((envelope, encoded_size))
            self.history_bytes += encoded_size
            self.latest[(match_id, player_id)] = envelope
            self.event_sequences[event_id] = envelope.relay_seq
            self.accepted_count += 1

            while self.history and self.history_bytes > self.max_bytes:
                old, old_size = self.history.popleft()
                self.history_bytes -= old_size
                self.event_sequences.pop(old.event_id, None)
                if self.latest.get((old.match_id, old.player_id)) is old:
                    self.latest.pop((old.match_id, old.player_id), None)
            return envelope, False

    def replay(self, after_seq: int = 0, limit: int = 1000,
               match_id: Optional[str] = None, player_id: Optional[int] = None) -> dict:
        with self._lock:
            items = [
                asdict(envelope)
                for envelope, _ in self.history
                if envelope.relay_seq > after_seq
                and (match_id is None or envelope.match_id == match_id)
                and (player_id is None or envelope.player_id == player_id)
            ][:limit]
            oldest = self.history[0][0].relay_seq if self.history else self.sequence + 1
            latest = self.sequence
        return {
            "relay_instance_id": RELAY_INSTANCE_ID,
            "items": items,
            "oldest_seq": oldest,
            "latest_seq": latest,
            "next_seq": items[-1]["relay_seq"] if items else after_seq,
            "cache_gap": bool(after_seq and after_seq < oldest - 1),
        }

    def latest_payload(self, match_id: Optional[str] = None) -> dict:
        with self._lock:
            streams = [
                asdict(value) for (match, _), value in self.latest.items()
                if match_id is None or match == match_id
            ]
        streams.sort(key=lambda item: (item["match_id"], item["player_id"]))
        players: Dict[str, dict] = {}
        for item in streams:
            key = str(item["player_id"])
            if key not in players or item["relay_seq"] > players[key]["relay_seq"]:
                players[key] = item
        return {
            "relay_instance_id": RELAY_INSTANCE_ID,
            "relay_timestamp_ns": time.time_ns(),
            "players": players,
            "streams": streams,
            "cache": self.cache_status(),
        }

    def cache_status(self) -> dict:
        with self._lock:
            return {
                "storage": "memory_only",
                "items": len(self.history),
                "bytes": self.history_bytes,
                "max_bytes": self.max_bytes,
                "oldest_seq": self.history[0][0].relay_seq if self.history else None,
                "latest_seq": self.sequence,
            }

    def subscribe(
        self,
        ws: WebSocket,
        after_seq: int,
        match_id: Optional[str],
        player_id: Optional[int],
    ) -> Tuple[asyncio.Queue, dict]:
        queue: asyncio.Queue = asyncio.Queue(maxsize=2048)
        with self._lock:
            replay = self.replay(after_seq, 5000, match_id, player_id)
            self.ws_clients[ws] = (queue, match_id, player_id)
        return queue, replay

    def unsubscribe(self, ws: WebSocket) -> None:
        with self._lock:
            self.ws_clients.pop(ws, None)

    async def broadcast(self, envelope: RelayEnvelope) -> None:
        payload = {
            "type": "wearable_sample",
            "relay_instance_id": RELAY_INSTANCE_ID,
            "data": asdict(envelope),
        }
        with self._lock:
            subscribers = list(self.ws_clients.values())
        for queue, match_id, player_id in subscribers:
            if match_id is not None and envelope.match_id != match_id:
                continue
            if player_id is not None and envelope.player_id != player_id:
                continue
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass


RELAY = MemoryRelay()


def _authorized(token: Optional[str]) -> bool:
    return bool(RELAY_TOKEN and token and hmac.compare_digest(token, RELAY_TOKEN))


app = FastAPI(
    title="CPG44 stateless telemetry relay",
    version="3.0.0",
    description="Authenticated in-memory replay relay for raw and processed wearable observations.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get("CPG44_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Auth"],
)


@app.middleware("http")
async def body_limit(request: Request, call_next):
    length = request.headers.get("content-length")
    if length:
        try:
            if int(length) > MAX_BODY_BYTES:
                return JSONResponse(
                    {"detail": "request body too large"},
                    status_code=413,
                    headers={"Cache-Control": "no-store"},
                )
        except ValueError:
            return JSONResponse(
                {"detail": "invalid content-length"},
                status_code=400,
                headers={"Cache-Control": "no-store"},
            )
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "cpg44-stateless-relay",
        "relay_instance_id": RELAY_INSTANCE_ID,
        "uptime_s": round(time.time() - RELAY.started_at, 1),
        "accepted": RELAY.accepted_count,
        "duplicates": RELAY.duplicate_count,
        "rejected": RELAY.rejected_count,
        "websocket_clients": len(RELAY.ws_clients),
        "auth_required": True,
        "cache": RELAY.cache_status(),
    }


@app.post("/api/v1/sensors/ingest")
async def ingest(
    body: Any = Body(...),
    x_auth: Optional[str] = Header(None, alias="X-Auth"),
):
    if not _authorized(x_auth):
        raise HTTPException(status_code=401, detail="invalid relay token")
    rows = body if isinstance(body, list) else [body]
    if not 1 <= len(rows) <= 100:
        raise HTTPException(status_code=422, detail="send 1 to 100 observations")
    accepted: List[RelayEnvelope] = []
    duplicates = 0
    for row in rows:
        envelope, duplicate = RELAY.register(row)
        if envelope is None:
            continue
        accepted.append(envelope)
        duplicates += int(duplicate)
        if not duplicate:
            await RELAY.broadcast(envelope)
    if not accepted:
        raise HTTPException(status_code=422, detail="no valid timestamped observations")
    return {
        "status": "ok",
        "accepted": len(accepted),
        "duplicates": duplicates,
        "last_relay_seq": accepted[-1].relay_seq,
        "event_ids": [item.event_id for item in accepted],
    }


@app.get("/api/v1/sensors/latest")
async def latest(
    match_id: Optional[str] = Query(None),
    x_auth: Optional[str] = Header(None, alias="X-Auth"),
):
    if not _authorized(x_auth):
        raise HTTPException(status_code=401, detail="invalid relay token")
    return RELAY.latest_payload(match_id)


@app.get("/api/v1/sensors/history")
async def history(
    after_seq: int = Query(0, ge=0),
    limit: int = Query(1000, ge=1, le=5000),
    match_id: Optional[str] = Query(None),
    player_id: Optional[int] = Query(None, ge=1, le=9999),
    x_auth: Optional[str] = Header(None, alias="X-Auth"),
):
    if not _authorized(x_auth):
        raise HTTPException(status_code=401, detail="invalid relay token")
    return RELAY.replay(after_seq, limit, match_id, player_id)


@app.websocket("/ws/sensors")
async def websocket_sensors(
    ws: WebSocket,
    after_seq: int = Query(0, ge=0),
    match_id: Optional[str] = Query(None),
    player_id: Optional[int] = Query(None, ge=1, le=9999),
):
    if not _authorized(ws.headers.get("x-auth")):
        await ws.close(code=1008, reason="invalid relay token")
        return
    await ws.accept()
    queue, replay = RELAY.subscribe(ws, after_seq, match_id, player_id)
    try:
        await ws.send_json({"type": "replay", "data": replay})
        while True:
            await ws.send_json(await queue.get())
    except WebSocketDisconnect:
        pass
    finally:
        RELAY.unsubscribe(ws)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def status_page():
    cache = RELAY.cache_status()
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>CPG44 telemetry relay</title><style>
    body{{margin:0;background:#f4f5f1;color:#17211d;font:16px/1.5 system-ui;padding:32px}}
    main{{max-width:720px;margin:auto;background:white;border:1px solid #d8ded9;border-radius:8px;padding:28px}}
    h1{{font:600 32px Georgia,serif;margin:0 0 8px}}dl{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
    dt,dd{{margin:0;padding:8px 0;border-bottom:1px solid #e7ebe8}}dd{{text-align:right;font-variant-numeric:tabular-nums}}
    code{{background:#edf0ed;padding:2px 5px;border-radius:3px}}</style></head><body><main>
    <h1>CPG44 telemetry relay</h1>
    <p>Authenticated, memory-only replay service. Sensor processing remains on the field PC.</p>
    <dl><dt>Cached observations</dt><dd>{cache['items']}</dd>
    <dt>Cache used</dt><dd>{cache['bytes']} / {cache['max_bytes']} bytes</dd>
    <dt>Latest relay sequence</dt><dd>{cache['latest_seq']}</dd>
    <dt>Connected subscribers</dt><dd>{len(RELAY.ws_clients)}</dd></dl>
    <p>Public status: <code>/health</code>. Measurement routes require <code>X-Auth</code>.</p>
    </main></body></html>"""


if __name__ == "__main__":
    if len(RELAY_TOKEN) < 32:
        raise SystemExit("CPG44_RELAY_TOKEN must contain at least 32 characters")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8081)))
