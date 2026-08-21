"""
FastAPI application factory for the CPG44 Football Intelligence & Sensor Fusion Platform.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
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
CAMERA_FRAME_DIR = STATIC_DIR / "camera_frames"
CAMERA_FRAME_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

STORE = ProductStore(
    hub_url=os.environ.get("CPG44_HUB_URL", "http://127.0.0.1:8081"),
    hostinger_url=os.environ.get("CPG44_RELAY_URL", os.environ.get("HOSTINGER_RELAY_URL", "")),
    relay_token=os.environ.get("CPG44_RELAY_TOKEN", os.environ.get("HOSTINGER_RELAY_TOKEN", "")),
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
        description="Local football vision, synchronized wearable evidence and tactical analytics.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins if o.strip()],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- System & Health ---
    @app.get("/api/v1/health")
    async def health():
        return {"status": "ok", "version": __version__, "mode": STORE.active_mode}

    @app.get("/api/v1/system/info")
    async def system_info():
        return await asyncio.to_thread(STORE.system_info)

    # --- Match Management ---
    @app.get("/api/v1/matches")
    async def list_matches():
        return [STORE.match]

    @app.get("/api/v1/matches/{match_id}")
    async def get_match(match_id: str):
        if match_id != STORE.match["match_id"]:
            raise HTTPException(status_code=404, detail="match not found")
        m = dict(STORE.match)
        return m

    @app.post("/api/v1/matches/{match_id}/mode")
    async def set_match_mode(match_id: str, body: dict):
        if match_id != STORE.match["match_id"]:
            raise HTTPException(status_code=404, detail="match not found")
        try:
            STORE.set_mode(str(body.get("mode", "recorded")))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"ok": True, "active_mode": STORE.active_mode}

    @app.post("/api/v1/matches/{match_id}/start")
    async def start_match(match_id: str):
        if match_id != STORE.match["match_id"]:
            raise HTTPException(status_code=404, detail="match not found")
        STORE.match["engine_running"] = True
        STORE.match["status"] = "waiting_for_pipeline"
        return {"ok": True, "message": "waiting for live pipeline frames"}

    @app.post("/api/v1/matches/{match_id}/stop")
    async def stop_match(match_id: str):
        if match_id != STORE.match["match_id"]:
            raise HTTPException(status_code=404, detail="match not found")
        STORE.match["engine_running"] = False
        STORE.match["status"] = "paused"
        return {"ok": True, "message": "match paused"}

    @app.get("/api/v1/matches/{match_id}/status")
    async def get_match_status(match_id: str):
        if match_id != STORE.match["match_id"]:
            raise HTTPException(status_code=404, detail="match not found")
        snapshot = STORE.live_frame_snapshot()
        return {
            "match": STORE.match,
            "provenance": snapshot.get("provenance"),
            "data_quality": snapshot.get("data_quality"),
        }

    @app.get("/api/v1/matches/{match_id}/analytics")
    async def get_match_analytics(match_id: str):
        if match_id != STORE.match["match_id"]:
            raise HTTPException(status_code=404, detail="match not found")
        return STORE.live_frame_snapshot()

    @app.get("/api/v1/matches/{match_id}/players")
    async def get_match_players(match_id: str):
        if match_id != STORE.match["match_id"]:
            raise HTTPException(status_code=404, detail="match not found")
        return STORE.live_frame_snapshot().get("players", [])

    @app.get("/api/v1/matches/{match_id}/passing-network")
    async def get_passing_network(match_id: str):
        if match_id != STORE.match["match_id"]:
            raise HTTPException(status_code=404, detail="match not found")
        return STORE.live_frame_snapshot().get("passing_network", {})

    @app.get("/api/v1/live")
    async def get_live_snapshot():
        return STORE.live_frame_snapshot()

    @app.post("/api/v1/live")
    async def post_live_snapshot(body: dict):
        if not isinstance(body.get("players", []), list):
            raise HTTPException(status_code=422, detail="players must be a list")
        normalized = dict(body)
        normalized.setdefault("match_id", STORE.match["match_id"])
        result = STORE.ingest_live_frame(normalized)
        result["training_samples_recorded"] = await asyncio.to_thread(
            TRAINER.record_live_snapshot, normalized
        )
        return result

    @app.get("/api/v1/live/frame")
    async def get_live_annotated_frame():
        image = STORE.live_jpeg()
        if image is None:
            raise HTTPException(status_code=404, detail="no fresh annotated frame")
        return Response(
            content=image,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.post("/api/v1/live/frame")
    async def post_live_annotated_frame(request: Request):
        content_type = request.headers.get("content-type", "").split(";", 1)[0]
        if content_type != "image/jpeg":
            raise HTTPException(status_code=415, detail="expected image/jpeg")
        body = await request.body()
        if not body or len(body) > 3_000_000:
            raise HTTPException(status_code=413, detail="JPEG must be between 1 byte and 3 MB")
        try:
            return STORE.ingest_live_jpeg(body)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    # --- Video Streaming Endpoint with HTTP 206 Partial Content ---
    @app.get("/api/v1/video/stream/{match_id}")
    async def stream_match_video(match_id: str, request: Request):
        if match_id != STORE.match["match_id"]:
            raise HTTPException(status_code=404, detail="match not found")
        video_path = STORE.recorded_video_path
        if video_path is None or not video_path.is_file():
            raise HTTPException(status_code=404, detail="no recorded video is selected")

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
        calibration: Optional[UploadFile] = File(None),
    ):
        safe_name = Path(file.filename or "match.mp4").name
        if Path(safe_name).suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
            raise HTTPException(status_code=415, detail="unsupported video type")
        file_path = UPLOAD_DIR / f"{int(time.time())}_{safe_name}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        match_id = f"match_{int(time.time())}"
        calibration_path = None
        if calibration and calibration.filename:
            calibration_name = Path(calibration.filename).name
            if Path(calibration_name).suffix.lower() not in {".yaml", ".yml"}:
                raise HTTPException(status_code=415, detail="calibration must be a YAML file")
            calibration_file = UPLOAD_DIR / f"{match_id}_{calibration_name}"
            with open(calibration_file, "wb") as buffer:
                shutil.copyfileobj(calibration.file, buffer)
            calibration_path = str(calibration_file)

        def activate_result(_stats: dict) -> None:
            output_dir = ROOT / "data" / "matches" / match_id
            STORE.activate_recorded_result(
                output_dir / "stats.json",
                output_dir / "annotated.mp4",
                match_id,
                match_name,
            )

        ENGINE.start_upload_processing(
            match_id=match_id,
            video_path=str(file_path),
            calibration_path=calibration_path,
            on_complete=activate_result,
        )

        job = ENGINE.get_progress(match_id)
        if job and job.status != "failed":
            STORE.set_mode("upload")
            STORE.match.update({
                "match_id": match_id,
                "name": match_name,
                "status": job.status,
                "engine_running": True,
            })
        return {
            "ok": bool(job and job.status != "failed"),
            "match_id": match_id,
            "filename": safe_name,
            "status": job.status if job else "failed",
            "error": job.error_message if job else "processing job was not created",
            "metric_calibration": calibration_path is not None,
        }

    @app.get("/api/v1/matches/{match_id}/progress")
    async def get_processing_progress(match_id: str):
        prog = ENGINE.get_progress(match_id)
        if not prog:
            raise HTTPException(status_code=404, detail="processing job not found")
        return prog

    # --- Training Manager API (YOLO on SoccerNet & Strain Model) ---
    @app.get("/api/v1/training/status")
    async def get_training_status():
        return TRAINER.get_status()

    @app.post("/api/v1/training/start-yolo")
    async def start_yolo_train(body: dict):
        return TRAINER.start_yolo_training(
            data_path=body.get("data_path", "/home/siddartha/SoccerNet/yolo/data.yaml"),
            model_name=body.get("model_name", "yolov8m.pt"),
            epochs=int(body.get("epochs", 50)),
            imgsz=int(body.get("imgsz", 1280)),
            batch=int(body.get("batch", 4)),
        )

    @app.post("/api/v1/training/stop-yolo")
    async def stop_yolo_train():
        return TRAINER.stop_yolo_training()

    @app.post("/api/v1/training/train-strain-model")
    async def train_strain():
        return await asyncio.to_thread(TRAINER.train_strain_model)

    @app.post("/api/v1/training/outcome-label")
    async def label_player_session(body: dict):
        try:
            player_id = int(body.get("player_id"))
            injury_label = int(body.get("injury_label"))
            outcome_window_days = int(
                body.get("outcome_window_days", TRAINER.outcome_window_days)
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="player_id, injury_label and outcome window must be integers") from exc
        result = await asyncio.to_thread(
            TRAINER.label_player_session,
            str(body.get("match_id") or ""),
            player_id,
            injury_label,
            str(body.get("label_source") or ""),
            outcome_window_days,
            str(body.get("notes") or ""),
        )
        if not result.get("ok"):
            raise HTTPException(status_code=422, detail=result.get("error"))
        return result

    # --- Camera Network (Single & Multi-Camera Gateway) ---
    @app.get("/api/v1/cameras")
    async def list_cameras():
        return STORE.camera_list()

    @app.post("/api/v1/cameras/register")
    async def register_camera(cam: dict):
        return STORE.register_camera(cam)

    @app.post("/api/v1/cameras/{camera_id}/frame")
    async def ingest_camera_frame(camera_id: str, request: Request):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", camera_id):
            raise HTTPException(status_code=422, detail="invalid camera id")
        if request.headers.get("content-type", "").split(";", 1)[0] != "image/jpeg":
            raise HTTPException(status_code=415, detail="expected image/jpeg")
        body = await request.body()
        if not body or len(body) > 3_000_000 or not body.startswith(b"\xff\xd8") or not body.endswith(b"\xff\xd9"):
            raise HTTPException(status_code=413, detail="invalid or oversized JPEG")
        path = CAMERA_FRAME_DIR / f"{camera_id}.jpg"
        await asyncio.to_thread(path.write_bytes, body)
        STORE.update_camera_health(camera_id, status="online", last_frame_at=time.time())
        return {"ok": True, "camera_id": camera_id, "bytes": len(body)}

    @app.get("/api/v1/cameras/{camera_id}/frame")
    async def get_camera_frame(camera_id: str):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", camera_id):
            raise HTTPException(status_code=422, detail="invalid camera id")
        path = CAMERA_FRAME_DIR / f"{camera_id}.jpg"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="camera has not sent a frame")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    # --- Observations Endpoint for API Compatibility ---
    @app.get("/api/v1/observations")
    async def list_observations(source: str = ""):
        rows = STORE.wearable_log
        if source:
            rows = [r for r in rows if r.get("source") == source]
        return rows[-200:]

    @app.get("/api/v1/observations/wearable")
    async def list_wearables():
        hub_data = await asyncio.to_thread(STORE.fetch_hub)
        if hub_data and "players" in hub_data:
            return hub_data["players"]
        if hub_data:
            from soccer_analytics.sensors.hub_bridge import sample_to_observation, snapshot_to_sample
            player_id = int(hub_data.get("player_id") or 1)
            sample = snapshot_to_sample(hub_data, player_id)
            if sample is not None:
                return [sample_to_observation(sample, match_id=hub_data.get("match_id") or "live")]
        return STORE.wearable_log[-100:]

    @app.post("/api/v1/observations/wearable")
    async def post_wearable(body: dict):
        if not body.get("global_player_id") and body.get("player_id") is None:
            raise HTTPException(status_code=422, detail="global_player_id or player_id is required")
        return STORE.record_wearable(body)

    # --- Smartphone Ingest HTML ---
    @app.get("/camera", response_class=HTMLResponse)
    async def smartphone_camera_page():
        html_file = STATIC_DIR / "camera.html"
        if html_file.is_file():
            return html_file.read_text(encoding="utf-8")
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head><meta name="viewport" content="width=device-width,initial-scale=1"><title>CPG44 phone camera</title></head>
        <body style="margin:0;background:#f3f5f6;color:#17212b;font:16px/1.5 system-ui;padding:24px;">
          <main style="max-width:760px;margin:auto"><h1 style="font:600 34px Georgia,serif">Phone camera preview</h1>
            <p>This sends measured JPEG frames to the local gateway. Keep this page open and the screen awake.</p>
            <video id="v" autoplay muted playsinline style="width:100%;background:#202a31;border:1px solid #d9e0e4;border-radius:8px;"></video>
            <p id="s" style="padding:10px;background:#fff;border:1px solid #d9e0e4;border-radius:6px">Requesting camera permission…</p>
            <canvas id="c" hidden></canvas>
          </main>
            <script>
              const v=document.getElementById('v'),c=document.getElementById('c'),s=document.getElementById('s');
              const cameraId=new URLSearchParams(location.search).get('id')||'phone_1'; let sending=false;
              async function send(){
                if(sending||!v.videoWidth)return; sending=true;
                const width=Math.min(1280,v.videoWidth),height=Math.round(width*v.videoHeight/v.videoWidth);
                c.width=width;c.height=height;c.getContext('2d').drawImage(v,0,0,width,height);
                c.toBlob(async blob=>{try{const r=await fetch('/api/v1/cameras/'+encodeURIComponent(cameraId)+'/frame',{method:'POST',headers:{'Content-Type':'image/jpeg'},body:blob});s.textContent=r.ok?'Sending '+width+'×'+height+' frames as '+cameraId:'Gateway rejected frame: '+r.status}catch(e){s.textContent='Gateway unavailable: '+e.message}finally{sending=false}},'image/jpeg',.84);
              }
              navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:1920}},audio:false}).then(async stream=>{
                v.srcObject=stream;
                await fetch('/api/v1/cameras/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:cameraId,name:'Phone camera',type:'browser_jpeg',source:location.href})});
                setInterval(send,200);
              }).catch(error=>{s.textContent='Camera permission failed: '+error.message});
            </script>
        </body></html>
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
                rows = await list_wearables()
                await ws.send_json({
                    "timestamp": time.time(),
                    "wearables": rows,
                })
                await asyncio.sleep(0.1)  # 10 Hz telemetry push
        except WebSocketDisconnect:
            pass


    # --- ESP32 Hardware & Flashing ---
    @app.get("/api/v1/hardware/ports")
    async def list_serial_ports():
        return FLASHER.list_ports()

    @app.get("/api/v1/hardware/status")
    async def get_hardware_status():
        return await asyncio.to_thread(FLASHER.toolchain_status)

    @app.get("/api/v1/hardware/chip-info")
    async def get_esp32_chip_info(port: str):
        return await asyncio.to_thread(FLASHER.get_chip_info, port)

    @app.post("/api/v1/hardware/flash")
    async def flash_esp32_firmware(body: dict):
        if not body.get("port") or not body.get("wifi_ssid"):
            raise HTTPException(status_code=422, detail="port and wifi_ssid are required")
        return FLASHER.flash_device(
            port=body["port"],
            player_id=int(body.get("player_id", 1)),
            wifi_ssid=body["wifi_ssid"],
            wifi_pass=body.get("wifi_pass", ""),
            endpoint="https://cpg44.nivaspms.com/api/v1/sensors/ingest",
            match_id=str(body.get("match_id") or "live"),
        )

    @app.get("/api/v1/hardware/flash/status")
    async def get_flash_job_status():
        return FLASHER.get_flash_status()

    # --- Player Tagging & Event Logging ---
    @app.get("/api/v1/tagging/teams")
    async def get_team_profiles():
        return TAGGER.team_profiles

    @app.post("/api/v1/tagging/teams")
    async def update_team_profiles(body: dict):
        try:
            return TAGGER.save_team_profiles(body)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/v1/tagging/events")
    async def list_match_events():
        return TAGGER.events

    @app.post("/api/v1/tagging/events")
    async def create_match_event(body: dict):
        try:
            return TAGGER.log_event(body)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
