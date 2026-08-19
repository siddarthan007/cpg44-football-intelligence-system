"""
FastAPI application factory for the CPG44 Football Intelligence & Sensor Fusion Platform.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from cpg44_api import __version__
from cpg44_api.store import ProductStore
from cpg44_api.training_manager import TrainingManager
from cpg44_api.flasher import ESP32Flasher
from cpg44_api.tagger import MatchTagger
from soccer_analytics.modes import EngineManager, MatchMode

ROOT = Path(__file__).resolve().parents[3]
STATIC_DIR = ROOT / "backend" / "src" / "cpg44_api" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

STORE = ProductStore(
    hub_url=os.environ.get("CPG44_HUB_URL", "http://127.0.0.1:8081"),
    hostinger_url=os.environ.get("HOSTINGER_RELAY_URL", ""),
)
ENGINE = EngineManager(ROOT)
TRAINER = TrainingManager(ROOT)
FLASHER = ESP32Flasher()
TAGGER = MatchTagger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = STORE
    app.state.engine = ENGINE
    app.state.trainer = TRAINER
    yield


def create_app() -> FastAPI:
    origins = os.environ.get(
        "CPG44_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    
    app = FastAPI(
        title="CPG44 Football Intelligence Platform",
        version=__version__,
        description=(
            "Full-stack Computer Vision, Multi-Camera Tracking, ESP32 Wearable Sync, "
            "and Tactical AI Engine."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins if o.strip()] + ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- System & Health ---
    @app.get("/api/v1/health")
    def health():
        return {"status": "ok", "version": __version__, "mode": STORE.active_mode}

    @app.get("/api/v1/system/info")
    def system_info():
        return STORE.system_info()

    # --- Match Management ---
    @app.get("/api/v1/matches")
    def list_matches():
        return [STORE.match]

    @app.get("/api/v1/matches/{match_id}")
    def get_match(match_id: str):
        m = dict(STORE.match)
        m["match_id"] = match_id
        return m

    @app.post("/api/v1/matches/{match_id}/mode")
    def set_match_mode(match_id: str, body: dict):
        mode = body.get("mode", "demo")
        STORE.set_mode(mode)
        return {"ok": True, "active_mode": STORE.active_mode}

    @app.post("/api/v1/matches/{match_id}/start")
    def start_match(match_id: str):
        STORE.match["engine_running"] = True
        STORE.match["status"] = "playing"
        return {"ok": True, "message": "match marked playing"}

    @app.post("/api/v1/matches/{match_id}/stop")
    def stop_match(match_id: str):
        STORE.match["engine_running"] = False
        STORE.match["status"] = "paused"
        return {"ok": True, "message": "match paused"}

    @app.get("/api/v1/matches/{match_id}/analytics")
    def get_match_analytics(match_id: str):
        return STORE.live_frame_snapshot()

    @app.get("/api/v1/matches/{match_id}/players")
    def get_match_players(match_id: str):
        return STORE.roster

    @app.get("/api/v1/matches/{match_id}/passing-network")
    def get_passing_network(match_id: str):
        return STORE.passing_network

    # --- Video Streaming Endpoint with HTTP 206 Partial Content ---
    @app.get("/api/v1/video/stream/{match_id}")
    async def stream_match_video(match_id: str, request: Request):
        video_path = STORE.demo_video_path
        if not video_path.is_file():
            # Fallback test video check
            for cand in [ROOT / "demo" / "sample_match.mp4", ROOT / "data" / "sample_match.mp4"]:
                if cand.is_file():
                    video_path = cand
                    break

        if not video_path.is_file():
            raise HTTPException(status_code=404, detail="Video file not found")

        file_size = video_path.stat().st_size
        range_header = request.headers.get("range")

        if not range_header:
            return FileResponse(video_path, media_type="video/mp4")

        # Parse range header: e.g. "bytes=0-1048575"
        try:
            h = range_header.replace("bytes=", "").split("-")
            start = int(h[0]) if h[0] else 0
            end = int(h[1]) if len(h) > 1 and h[1] else file_size - 1
        except Exception:
            start, end = 0, file_size - 1

        end = min(end, file_size - 1)
        chunk_size = end - start + 1

        def iterfile():
            with open(video_path, "rb") as f:
                f.seek(start)
                bytes_left = chunk_size
                while bytes_left > 0:
                    read_size = min(64 * 1024, bytes_left)
                    data = f.read(read_size)
                    if not data:
                        break
                    bytes_left -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_size),
            "Content-Type": "video/mp4",
        }
        return StreamingResponse(iterfile(), status_code=206, headers=headers)

    # --- Video Upload & Processing Mode ---
    @app.post("/api/v1/matches/upload")
    async def upload_match_video(
        file: UploadFile = File(...),
        match_name: str = Form("College Match"),
        mode: str = Form("inference"),
    ):
        file_path = UPLOAD_DIR / f"{int(time.time())}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        match_id = f"match_{int(time.time())}"
        ENGINE.start_upload_processing(
            match_id=match_id,
            video_path=str(file_path),
        )

        return {
            "ok": True,
            "match_id": match_id,
            "filename": file.filename,
            "saved_path": str(file_path),
            "status": "processing",
        }

    @app.get("/api/v1/matches/{match_id}/progress")
    def get_processing_progress(match_id: str):
        prog = ENGINE.get_progress(match_id)
        if not prog:
            return {"match_id": match_id, "status": "completed", "progress_pct": 100.0}
        return prog

    # --- Training Manager API (YOLO on SoccerNet & Strain Model) ---
    @app.get("/api/v1/training/status")
    def get_training_status():
        return TRAINER.get_status()

    @app.post("/api/v1/training/start-yolo")
    def start_yolo_train(body: dict):
        return TRAINER.start_yolo_training(
            data_path=body.get("data_path", "/home/siddartha/SoccerNet_YOLO/data.yaml"),
            model_name=body.get("model_name", "yolov8m.pt"),
            epochs=int(body.get("epochs", 50)),
            imgsz=int(body.get("imgsz", 1280)),
            batch=int(body.get("batch", 4)),
        )

    @app.post("/api/v1/training/stop-yolo")
    def stop_yolo_train():
        return TRAINER.stop_yolo_training()

    @app.post("/api/v1/training/train-strain-model")
    def train_strain():
        return TRAINER.train_strain_model()

    # --- Camera Network (Single & Multi-Camera Gateway) ---
    @app.get("/api/v1/cameras")
    def list_cameras():
        return list(STORE.cameras.values())

    @app.post("/api/v1/cameras/register")
    def register_camera(cam: dict):
        return STORE.register_camera(cam)

    # --- Observations Endpoint for API Compatibility ---
    @app.get("/api/v1/observations")
    def list_observations(source: str = ""):
        rows = STORE.wearable_log
        if source:
            rows = [r for r in rows if r.get("source") == source]
        return rows[-200:]

    @app.get("/api/v1/observations/wearable")
    def list_wearables():
        hub_data = STORE.fetch_hub()
        if hub_data and "players" in hub_data:
            return hub_data["players"]
        return STORE.wearable_log[-100:]

    @app.post("/api/v1/observations/wearable")
    def post_wearable(body: dict):
        return STORE.record_wearable(body)

    # --- Smartphone Ingest HTML ---
    @app.get("/camera", response_class=HTMLResponse)
    def smartphone_camera_page():
        html_file = STATIC_DIR / "camera.html"
        if html_file.is_file():
            return html_file.read_text(encoding="utf-8")
        return """
        <!DOCTYPE html>
        <html>
        <head><title>Mobile Camera Streaming</title></head>
        <body style="background:#111;color:#fff;font-family:sans-serif;text-align:center;padding:2rem;">
            <h2>CPG44 Smartphone Camera Streamer</h2>
            <p>Point this camera at the pitch. Video frames will stream directly to the CPG44 Gateway.</p>
            <video id="v" autoplay playsinline style="width:100%;max-width:480px;border-radius:8px;"></video>
            <script>
                navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'}}).then(s=>{
                    document.getElementById('v').srcObject=s;
                });
            </script>
        </body>
        </html>
        """

    # --- Real-Time WebSocket Feeds ---
    @app.websocket("/ws/live")
    async def websocket_live_feed(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                snap = STORE.live_frame_snapshot()
                await ws.send_json(snap)
                await asyncio.sleep(0.04)  # ~25 FPS live match broadcast
        except WebSocketDisconnect:
            pass

    @app.websocket("/ws/wearables")
    async def websocket_wearable_feed(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                hub = STORE.fetch_hub() or {"players": {}}
                await ws.send_json({
                    "timestamp": time.time(),
                    "telemetry": hub,
                })
                await asyncio.sleep(0.1)  # 10 Hz telemetry push
        except WebSocketDisconnect:
            pass


    # --- ESP32 Hardware & Flashing ---
    @app.get("/api/v1/hardware/ports")
    def list_serial_ports():
        return FLASHER.list_ports()

    @app.get("/api/v1/hardware/chip-info")
    def get_esp32_chip_info(port: str = "/dev/ttyUSB0"):
        return FLASHER.get_chip_info(port=port)

    @app.post("/api/v1/hardware/flash")
    def flash_esp32_firmware(body: dict):
        return FLASHER.flash_device(
            port=body.get("port", "/dev/ttyUSB0"),
            player_id=int(body.get("player_id", 27)),
            wifi_ssid=body.get("wifi_ssid", "Field_WiFi"),
            wifi_pass=body.get("wifi_pass", "FieldPass123"),
            endpoint=body.get("endpoint", "http://192.168.1.100:8000/ingest"),
        )

    @app.get("/api/v1/hardware/flash/status")
    def get_flash_job_status():
        return FLASHER.get_flash_status()

    # --- Player Tagging & Event Logging ---
    @app.get("/api/v1/tagging/teams")
    def get_team_profiles():
        return TAGGER.team_profiles

    @app.post("/api/v1/tagging/teams")
    def update_team_profiles(body: dict):
        return TAGGER.save_team_profiles(body)

    @app.get("/api/v1/tagging/events")
    def list_match_events():
        return TAGGER.events

    @app.post("/api/v1/tagging/events")
    def create_match_event(body: dict):
        ev = TAGGER.log_event(body)
        STORE.timeline_tags.append(ev)
        return ev

    return app


app = create_app()
