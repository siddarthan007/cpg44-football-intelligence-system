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

import httpx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from cpg44_api import __version__
from cpg44_api.store import ProductStore
from cpg44_api.training_manager import TrainingManager
from cpg44_api.flasher import ESP32Flasher
from cpg44_api.tagger import MatchTagger
from cpg44_api.analyses import AnalysisLibrary
from cpg44_api.calibration import CalibrationStudio
from cpg44_api.usb_bridge import ensure_esp32_usb_attached
from soccer_analytics.modes import EngineManager

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
CALIB = CalibrationStudio(ROOT / "data" / "calibration_sessions", ROOT / "configs" / "calibrations")
ANALYSES = AnalysisLibrary([ROOT / "data" / "matches", ROOT / "demo" / "analysis"])


def _range_response(path: Path, request: Request):
    """Serve a video with HTTP Range support so the browser can seek."""
    size = path.stat().st_size
    header = request.headers.get("range")
    if not header:
        return FileResponse(path, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})
    try:
        a, b = header.replace("bytes=", "").split("-")[:2]
        start = int(a) if a else 0
        end = int(b) if b else size - 1
    except ValueError:
        start, end = 0, size - 1
    end = min(end, size - 1)
    if start > end:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    def body():
        with open(path, "rb") as fh:
            fh.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = fh.read(min(256 * 1024, left))
                if not chunk:
                    break
                left -= len(chunk)
                yield chunk
    return StreamingResponse(body(), status_code=206, media_type="video/mp4", headers={
        "Content-Range": f"bytes {start}-{end}/{size}", "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1)})


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = STORE
    app.state.engine = ENGINE
    app.state.trainer = TRAINER
    app.state.hub_client = httpx.AsyncClient(timeout=0.5, trust_env=False)
    try:
        yield
    finally:
        if ENGINE.live_status().get("running"):
            await run_in_threadpool(ENGINE.stop_live_processing)
        await app.state.hub_client.aclose()


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

    async def fetch_hub_async() -> Optional[dict]:
        """Read the local sensor processor using persistent client pool (< 0.5ms)."""
        client = getattr(app.state, "hub_client", None)
        if client is None or client.is_closed:
            client = app.state.hub_client = httpx.AsyncClient(timeout=0.3, trust_env=False)
        try:
            response = await client.get(
                f"{STORE.hub_url}/api/latest",
                headers={"Accept": "application/json", "Cache-Control": "no-store"},
            )
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None

    # --- System & Health ---
    @app.get("/api/v1/health")
    async def health():
        return {"status": "ok", "version": __version__, "mode": STORE.active_mode}

    @app.get("/api/v1/system/info")
    async def system_info():
        return await run_in_threadpool(STORE.system_info)

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
        camera_id = normalized.get("camera_id")
        if isinstance(camera_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", camera_id):
            ENGINE.note_live_frame(camera_id)
            STORE.update_camera_health(
                camera_id,
                status="online",
                fps=normalized.get("processing_fps"),
                calibrated=bool(normalized.get("metric")),
                last_frame_at=time.time(),
            )
        result["training_samples_recorded"] = await run_in_threadpool(
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

    @app.get("/api/v1/live/mjpeg", include_in_schema=False)
    async def stream_live_annotated_frames():
        async def frames():
            last_version: Optional[float] = None
            while True:
                image, version = STORE.live_jpeg_snapshot()
                if image is not None and version is not None and version != last_version:
                    last_version = version
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(image)}\r\n\r\n".encode("ascii")
                        + image
                        + b"\r\n"
                    )
                await asyncio.sleep(0.01)

        return StreamingResponse(
            frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "X-Accel-Buffering": "no",
            },
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

    # --- Measured evidence for the report ---
    @app.get("/api/v1/evidence")
    async def evidence():
        out = {}
        bench = ROOT / "data" / "benchmarks"
        for f in sorted(bench.glob("*.json")) if bench.is_dir() else []:
            try:
                out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
        return out

    # --- Pitch setup (interactive calibration) ---
    @app.post("/api/v1/calibration/sessions")
    async def calibration_session(file: UploadFile = File(...), at_s: float = Form(2.0)):
        data = await file.read()
        if len(data) > 1024 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="file too large")
        try:
            return await run_in_threadpool(CALIB.create, data, file.filename or "clip.mp4", at_s)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.get("/api/v1/calibration/sessions/{sid}/frame")
    async def calibration_frame(sid: str):
        try:
            return FileResponse(CALIB.frame_path(sid), media_type="image/jpeg")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.post("/api/v1/calibration/sessions/{sid}/frame")
    async def calibration_regrab(sid: str, body: dict):
        try:
            return await run_in_threadpool(CALIB.regrab, sid, float(body.get("at_s", 0)))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.get("/api/v1/calibration/landmarks")
    async def calibration_landmarks(length: float = 105.0, width: float = 68.0, kind: str = "football"):
        if not (20 <= length <= 130 and 15 <= width <= 100 and width < length) or kind not in ("football", "futsal"):
            raise HTTPException(status_code=422, detail="pitch size or kind out of range")
        return CALIB.landmarks(length, width, kind)

    def _calib_call(fn, sid: str, body: dict, *extra):
        try:
            kind = str(body.get("kind") or "football")
            if kind not in ("football", "futsal"):
                raise ValueError("pitch kind must be football or futsal")
            return fn(sid, body.get("points") or {}, float(body.get("length", 105)), float(body.get("width", 68)), *extra,
                      kind=kind)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post("/api/v1/calibration/sessions/{sid}/solve")
    async def calibration_solve(sid: str, body: dict):
        return await run_in_threadpool(_calib_call, CALIB.solve, sid, body, body.get("k1"), body.get("homography"))

    @app.post("/api/v1/calibration/sessions/{sid}/snap")
    async def calibration_snap(sid: str, body: dict):
        if body.get("homography") is None:
            raise HTTPException(status_code=422, detail="solve first")
        return await run_in_threadpool(_calib_call, CALIB.snap, sid, body, body.get("k1", 0.0), body["homography"])

    @app.post("/api/v1/calibration/sessions/{sid}/save")
    async def calibration_save(sid: str, body: dict):
        try:
            return CALIB.save(sid, str(body.get("name") or "campus"), body.get("points") or {},
                              float(body.get("length", 105)), float(body.get("width", 68)),
                              float(body.get("k1") or 0.0), body["homography"], str(body.get("camera", "fixed")),
                              str(body.get("kind") or "football"))
        except KeyError as exc:
            raise HTTPException(status_code=404 if "session" in str(exc) else 422, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.get("/api/v1/calibrations")
    async def calibrations():
        return CALIB.list()

    @app.get("/api/v1/calibrations/{cid}/thumb")
    async def calibration_thumb(cid: str):
        p = CALIB.saved_path(cid)
        if p is None or not p.with_suffix(".jpg").is_file():
            raise HTTPException(status_code=404, detail="no preview")
        return FileResponse(p.with_suffix(".jpg"), media_type="image/jpeg")

    # --- Adapt the detector to a ground (campus fine-tuning) ---
    from soccer_analytics import adapt as ADAPT

    def _adapt(fn, *args):
        try:
            return fn(*args)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post("/api/v1/adapt/{project}/videos")
    async def adapt_video(project: str, file: UploadFile = File(...), frames: int = Form(60)):
        d = _adapt(ADAPT._proj, project) / "videos"
        d.mkdir(exist_ok=True)
        path = d / Path(file.filename or "clip.mp4").name
        with open(path, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
        kept = await run_in_threadpool(_adapt, ADAPT.add_video, project, str(path), max(10, min(frames, 300)))
        return {"added": kept, "frames": ADAPT.frames(project)}

    @app.post("/api/v1/adapt/{project}/prelabel")
    async def adapt_prelabel(project: str):
        weights = ENGINE._default_weights()
        if weights is None:
            raise HTTPException(status_code=409, detail="no detector checkpoint available")
        n = await run_in_threadpool(_adapt, ADAPT.prelabel, project, str(weights))
        return {"labelled": n, "frames": ADAPT.frames(project)}

    @app.get("/api/v1/adapt/{project}/frames")
    async def adapt_frames(project: str):
        return _adapt(ADAPT.frames, project)

    @app.get("/api/v1/adapt/{project}/frames/{frame_id}.jpg")
    async def adapt_frame(project: str, frame_id: str):
        if not re.fullmatch(r"[a-z0-9_-]{1,64}", frame_id):
            raise HTTPException(status_code=404, detail="frame not found")
        p = _adapt(ADAPT._proj, project) / "frames" / f"{frame_id}.jpg"
        if not p.is_file():
            raise HTTPException(status_code=404, detail="frame not found")
        return FileResponse(p, media_type="image/jpeg")

    @app.get("/api/v1/adapt/{project}/labels/{frame_id}")
    async def adapt_get_labels(project: str, frame_id: str):
        return _adapt(ADAPT.get_labels, project, frame_id)

    @app.put("/api/v1/adapt/{project}/labels/{frame_id}")
    async def adapt_put_labels(project: str, frame_id: str, body: dict):
        return _adapt(ADAPT.save_labels, project, frame_id, body.get("boxes") or [])

    @app.post("/api/v1/adapt/{project}/train")
    async def adapt_train(project: str, body: dict):
        if ADAPT.status(project).get("state") == "training":
            raise HTTPException(status_code=409, detail="already training")
        weights = ENGINE._default_weights()
        if weights is None:
            raise HTTPException(status_code=409, detail="no detector checkpoint available")
        return await run_in_threadpool(_adapt, ADAPT.start_training, project, str(weights), int(body.get("epochs", 20)))

    @app.post("/api/v1/adapt/{project}/promote")
    async def adapt_promote(project: str):
        st = ADAPT.status(project)
        res = st.get("results") or {}
        if st.get("state") != "done" or res.get("recommend") != "adapted":
            raise HTTPException(status_code=409, detail="the adapted model did not beat the base model")
        (ROOT / "configs").mkdir(exist_ok=True)
        (ROOT / "configs" / "active_detector.json").write_text(json.dumps(
            {"weights": res["adapted"]["weights"], "project": project, "at": time.time(), "results": res}))
        return {"active": res["adapted"]["weights"]}

    @app.get("/api/v1/adapt/{project}/status")
    async def adapt_status(project: str):
        return _adapt(ADAPT.status, project)

    # --- Pre-kick-off camera check ---
    _check_detector = {}

    def _camera_check(data: bytes) -> dict:
        import cv2
        import numpy as np
        from soccer_analytics.camera_check import check_frame
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("not an image")
        det = _check_detector.get("d")
        if det is None:
            weights = ENGINE._default_weights()
            if weights is not None:
                try:
                    from soccer_analytics.tracker import Detector
                    det = _check_detector["d"] = Detector(str(weights), imgsz=1280, person_conf=0.3)
                except Exception:
                    det = None
        return check_frame(frame, det)

    @app.post("/api/v1/camera-check")
    async def camera_check(image: UploadFile = File(...)):
        data = await image.read()
        if len(data) > 25 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="image too large")
        try:
            return await run_in_threadpool(_camera_check, data)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    # --- Recorded match analyses (review workspace) ---
    @app.get("/api/v1/analyses")
    async def list_analyses():
        return ANALYSES.list()

    @app.get("/api/v1/analyses/{analysis_id}")
    async def get_analysis(analysis_id: str):
        a = await run_in_threadpool(ANALYSES.get, analysis_id)
        if a is None:
            raise HTTPException(status_code=404, detail="analysis not found")
        return a

    @app.get("/api/v1/analyses/{analysis_id}/video")
    async def analysis_video(analysis_id: str, request: Request):
        path = ANALYSES.video_path(analysis_id)
        if path is None:
            raise HTTPException(status_code=404, detail="no annotated video for this analysis")
        return _range_response(path, request)

    @app.get("/api/v1/analyses/{analysis_id}/thumb")
    async def analysis_thumb(analysis_id: str):
        path = await run_in_threadpool(ANALYSES.thumbnail, analysis_id)
        if path is None:
            raise HTTPException(status_code=404, detail="no preview")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/v1/analyses/{analysis_id}/clip")
    async def analysis_clip(analysis_id: str, t: float, before: float = 3.0, after: float = 5.0):
        if not (0 <= before <= 15 and 0 < after <= 20):
            raise HTTPException(status_code=422, detail="clip window out of range")
        path = await run_in_threadpool(ANALYSES.clip, analysis_id, t, before, after)
        if path is None:
            raise HTTPException(status_code=404, detail="no video to cut")
        return FileResponse(path, media_type="video/mp4", filename=f"{analysis_id}_{int(t)}s.mp4")

    @app.post("/api/v1/analyses/{analysis_id}/review")
    async def review_event(analysis_id: str, payload: dict):
        try:
            return ANALYSES.decide(analysis_id, str(payload.get("event_id", "")),
                                   str(payload.get("decision", "")), str(payload.get("note", "")))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post("/api/v1/analyses/{analysis_id}/vest")
    async def tag_vest(analysis_id: str, payload: dict):
        try:
            vest = payload.get("vest")
            return ANALYSES.tag_vest(analysis_id, int(payload["player"]), None if vest in (None, "") else int(vest))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    # --- Video Upload & Processing Mode ---
    @app.post("/api/v1/matches/upload")
    async def upload_match_video(
        file: UploadFile = File(...),
        match_name: str = Form("College Match"),
        mode: str = Form("inference"),
        calibration: Optional[UploadFile] = File(None),
        calibration_id: str = Form(""),
    ):
        safe_name = Path(file.filename or "match.mp4").name
        if Path(safe_name).suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
            raise HTTPException(status_code=415, detail="unsupported video type")
        file_path = UPLOAD_DIR / f"{int(time.time())}_{safe_name}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        match_id = f"match_{int(time.time())}"
        calibration_path = None
        if calibration_id:
            saved = CALIB.saved_path(calibration_id)
            if saved is None:
                raise HTTPException(status_code=404, detail="saved calibration not found")
            calibration_path = str(saved)
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
            batch=int(body.get("batch", 2)),
        )

    @app.post("/api/v1/training/stop-yolo")
    async def stop_yolo_train():
        return TRAINER.stop_yolo_training()

    @app.post("/api/v1/training/train-strain-model")
    async def train_strain():
        return await run_in_threadpool(TRAINER.train_strain_model)

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
        result = await run_in_threadpool(
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
        cameras = STORE.camera_list()
        live = ENGINE.live_status()
        if live.get("running"):
            for camera in cameras:
                if camera["id"] == live.get("camera_id") and camera["status"] != "online":
                    camera["status"] = live.get("status") or "starting"
        return cameras

    @app.post("/api/v1/cameras/register")
    async def register_camera(cam: dict):
        camera = dict(cam)
        camera_id = str(camera.get("id") or f"cam_{len(STORE.camera_list()) + 1}")
        camera_type = str(camera.get("type") or "file")
        source = str(camera.get("source") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", camera_id):
            raise HTTPException(status_code=422, detail="camera id may contain letters, numbers, _ and -")
        if camera_type not in {"rtsp", "usb", "file", "browser_jpeg", "phone"}:
            raise HTTPException(status_code=422, detail="unsupported camera transport")
        if not source:
            raise HTTPException(status_code=422, detail="camera source is required")
        if camera_type == "usb" and not source.isdigit():
            raise HTTPException(status_code=422, detail="USB source must be a camera number such as 0")
        if camera_type == "rtsp" and not source.lower().startswith(("rtsp://", "rtsps://")):
            raise HTTPException(status_code=422, detail="RTSP source must begin with rtsp:// or rtsps://")
        camera.update({"id": camera_id, "type": camera_type, "source": source})
        return STORE.register_camera(camera)

    def _relay_client():
        import httpx
        relay, token = STORE.hostinger_url, STORE.relay_token
        if not relay or len(token or "") < 32:
            raise HTTPException(status_code=422, detail="Set CPG44_RELAY_URL and CPG44_RELAY_TOKEN to pair a phone")
        ca = os.environ.get("CPG44_RELAY_CA_FILE", "").strip()
        return relay, httpx.AsyncClient(timeout=8.0, verify=ca if ca and Path(ca).is_file() else True,
                                        headers={"X-Auth": token})

    @app.post("/api/v1/phone-camera/pair")
    async def pair_phone_camera(body: Optional[dict] = None):
        """Open a relay room for a phone and return the QR the phone scans."""
        import io
        import qrcode
        import qrcode.image.svg
        relay, client = _relay_client()
        async with client:
            try:
                r = await client.post(f"{relay}/api/v1/cam/rooms")
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"relay unreachable: {exc}") from exc
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"relay refused the room ({r.status_code})")
        room = r.json()
        url = f"{relay}{room['phone_path']}"
        buf = io.BytesIO()
        qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, box_size=10, border=2).save(buf)
        calibration = str((body or {}).get("calibration_path") or "")
        if (body or {}).get("calibration_id"):
            saved = CALIB.saved_path(str(body["calibration_id"]))
            calibration = str(saved) if saved else calibration
        camera = STORE.register_camera({
            "id": f"phone_{room['room']}", "name": "Phone camera", "type": "phone",
            "source": f"phone://{room['room']}?key={room['key']}&relay={relay}",
            "calibration_path": calibration})
        return {"camera_id": camera["id"], "room": room["room"], "url": url,
                "qr_svg": buf.getvalue().decode(), "expires_at": room["expires_at"]}

    @app.get("/api/v1/phone-camera/{room}")
    async def phone_camera_status(room: str):
        if not re.fullmatch(r"[0-9a-f]{8}", room):
            raise HTTPException(status_code=422, detail="invalid room")
        relay, client = _relay_client()
        async with client:
            try:
                r = await client.get(f"{relay}/api/v1/cam/rooms/{room}")
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"relay unreachable: {exc}") from exc
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail="room closed")
        return r.json()

    SQUAD_FILE = Path(os.environ.get("CPG44_SQUAD_FILE") or ROOT / "data" / "squad.json")

    @app.get("/api/v1/squad")
    async def get_squad():
        if SQUAD_FILE.is_file():
            return json.loads(SQUAD_FILE.read_text())
        return {"team_name": "", "players": []}

    @app.put("/api/v1/squad")
    async def put_squad(body: dict):
        """Our squad: shirt number, name, position and vest, the link between
        what the camera reads, what the vest sends and who the player is."""
        rows = body.get("players")
        if not isinstance(rows, list) or len(rows) > 40:
            raise HTTPException(status_code=422, detail="players must be a list of at most 40")
        clean, numbers, vests = [], set(), set()
        for r in rows:
            name = str(r.get("name") or "").strip()[:40]
            num = r.get("number")
            vest = r.get("vest")
            num = None if num in (None, "") else int(num)
            vest = None if vest in (None, "") else int(vest)
            if not name:
                raise HTTPException(status_code=422, detail="every player needs a name")
            if num is not None and not 0 <= num <= 99:
                raise HTTPException(status_code=422, detail=f"{name}: shirt numbers run from 0 to 99")
            if num is not None and num in numbers:
                raise HTTPException(status_code=422, detail=f"shirt {num} is used twice")
            if vest is not None and (not 1 <= vest <= 9999 or vest in vests):
                raise HTTPException(status_code=422, detail=f"vest {vest} is invalid or used twice")
            numbers.add(num) if num is not None else None
            vests.add(vest) if vest is not None else None
            clean.append({"name": name, "number": num, "vest": vest,
                          "position": (str(r.get("position") or "").strip()[:20] or None)})
        data = {"team_name": str(body.get("team_name") or "").strip()[:40], "players": clean}
        SQUAD_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = SQUAD_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(SQUAD_FILE)
        return data

    @app.get("/api/v1/vision/status")
    async def live_vision_status():
        return ENGINE.live_status()

    @app.post("/api/v1/cameras/{camera_id}/start")
    async def start_camera_processing(camera_id: str, body: Optional[dict] = None):
        camera = STORE.get_camera(camera_id)
        if camera is None:
            raise HTTPException(status_code=404, detail="camera not found")
        if (body or {}).get("calibration_id"):
            saved = CALIB.saved_path(str(body["calibration_id"]))
            if saved is None:
                raise HTTPException(status_code=404, detail="calibration not found")
            camera = STORE.register_camera({**camera, "calibration_path": str(saved)})
        product_api = os.environ.get("CPG44_PRODUCT_API", "http://127.0.0.1:8000").rstrip("/")
        source = camera["source"]
        if camera["type"] == "browser_jpeg":
            source = f"{product_api}/api/v1/cameras/{camera_id}/mjpeg"
        result = await run_in_threadpool(
            ENGINE.start_live_processing,
            camera_id,
            source,
            camera["type"],
            camera.get("calibration_path") or None,
            product_api,
        )
        if not result.get("ok"):
            raise HTTPException(status_code=409, detail=result.get("error") or "vision did not start")
        STORE.update_camera_health(
            camera_id,
            status="starting",
            calibrated=bool(camera.get("calibration_path")),
        )
        STORE.set_mode("live")
        STORE.match.update({"status": "starting", "engine_running": True})
        return result

    @app.post("/api/v1/vision/stop")
    async def stop_camera_processing():
        camera_id = ENGINE.live_status().get("camera_id")
        result = await run_in_threadpool(ENGINE.stop_live_processing)
        if not result.get("ok"):
            raise HTTPException(status_code=409, detail=result.get("error") or "vision did not stop")
        if camera_id:
            STORE.update_camera_health(camera_id, status="registered")
        STORE.match.update({"status": "paused", "engine_running": False})
        return result

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
        await run_in_threadpool(path.write_bytes, body)
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

    @app.get("/api/v1/cameras/{camera_id}/mjpeg", include_in_schema=False)
    async def stream_camera_frames(camera_id: str):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", camera_id):
            raise HTTPException(status_code=422, detail="invalid camera id")
        path = CAMERA_FRAME_DIR / f"{camera_id}.jpg"

        async def frames():
            last_version = None
            while True:
                try:
                    version = path.stat().st_mtime_ns
                except FileNotFoundError:
                    version = None
                if version is not None and version != last_version:
                    image = await run_in_threadpool(path.read_bytes)
                    last_version = version
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(image)}\r\n\r\n".encode("ascii")
                        + image
                        + b"\r\n"
                    )
                await asyncio.sleep(0.01)

        return StreamingResponse(
            frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    # --- Observations Endpoint for API Compatibility ---
    @app.get("/api/v1/observations")
    async def list_observations(source: str = ""):
        rows = STORE.wearable_log
        if source:
            rows = [r for r in rows if r.get("source") == source]
        return rows[-200:]

    @app.get("/api/v1/observations/wearable")
    async def list_wearables():
        hub_data = await fetch_hub_async()
        if hub_data and "players" in hub_data:
            return hub_data["players"]
        if hub_data:
            from soccer_analytics.sensors.hub_bridge import sample_to_observation, snapshot_to_sample
            player_id = int(hub_data.get("player_id") or 1)
            sample = snapshot_to_sample(hub_data, player_id)
            if sample is not None:
                return [sample_to_observation(sample, match_id=hub_data.get("match_id") or "live")]
        return STORE.wearable_log[-100:]

    @app.get("/api/v1/wearables/status")
    async def wearable_status():
        state = await fetch_hub_async()
        if not state:
            return {"connected": False, "clock_synced": False}
        device = state.get("device") or {}
        clock = state.get("clock") or {}
        imu = state.get("imu") or {}
        vitals = state.get("vitals") or {}
        stable = vitals.get("stable_15s") or {}
        rolling = vitals.get("rolling") or {}
        live_v = vitals.get("live") or {}
        ppg_contact = bool((state.get("ppg") or {}).get("contact"))
        return {
            "connected": bool(device.get("connected")),
            "clock_synced": bool(clock.get("valid") or (imu.get("timestamp") or {}).get("clock_synced")),
            "transport_delay_ms": (imu.get("timestamp") or {}).get("transport_delay_ms"),
            "gyro_calibrated": bool(imu.get("calibrated_gyro_bias")),
            "accel_scale_calibrated": bool(imu.get("accel_scale_calibrated")),
            "ppg_valid": bool(stable.get("valid") or rolling.get("valid") or live_v.get("valid") or ppg_contact),
            "ppg_reason": live_v.get("reason") or rolling.get("reason") or stable.get("reason"),
            "signal_quality": stable.get("quality") if stable.get("valid") else rolling.get("quality"),
            "dropped_samples": device.get("dropped_samples"),
            "queued_samples": device.get("queued_samples"),
            "gps_fix": bool((state.get("gps") or {}).get("fix")),
            "medical_calibrated": bool(vitals.get("medical_calibrated")),
            # delivery and timing: what arrived against what the vest was set to send
            "rates_hz": device.get("rates_hz") or {},
            "expected_hz": {"imu": device.get("imu_rate_hz"), "ppg": device.get("ppg_output_rate_hz"),
                            "gps": device.get("gps_rate_hz")},
            "clock_offset": state.get("relay_clock_offset"),
            "battery": device.get("battery"),
            "rssi_dbm": device.get("rssi_dbm"),
            "gps": {k: (state.get("gps") or {}).get(k) for k in ("fix", "sat", "hdop", "filtered")},
            "imu": {k: imu.get(k) for k in ("accel_cal_poses", "accel_calibrated", "player_load", "player_load_per_min",
                                            "stationary", "temperature_c")},
            "pulse": {"bpm": live_v.get("bpm") if live_v.get("valid") else None,
                      "source": live_v.get("source"), "rmssd_ms": live_v.get("rmssd_ms"),
                      "ecg": vitals.get("ecg"),
                      "spo2": rolling.get("spo2_estimate_pct") if rolling.get("spo2_valid") else None,
                      "quality": rolling.get("quality"), "reason": rolling.get("reason"),
                      "spo2_calibration": rolling.get("spo2_calibration")},
        }

    @app.post("/api/v1/wearables/calibrate/spo2")
    async def calibrate_spo2(body: dict):
        """Forward a finger-oximeter reference to the hub (it pairs it with the
        vest's current still reading)."""
        try:
            async with httpx.AsyncClient(timeout=3.0, trust_env=False) as client:
                r = await client.post(f"{STORE.hub_url}/api/calibrate/spo2", json={"reference_pct": body.get("reference_pct")})
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"sensor hub unavailable: {exc}") from exc
        data = r.json()
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=data.get("error") or "calibration rejected")
        return data

    @app.post("/api/v1/observations/wearable")
    async def post_wearable(body: dict):
        if not body.get("global_player_id") and body.get("player_id") is None:
            raise HTTPException(status_code=422, detail="global_player_id or player_id is required")
        return STORE.record_wearable(body)

    @app.websocket("/ws/live")
    async def websocket_live_feed(ws: WebSocket):
        await ws.accept()
        last_version = None
        try:
            while True:
                snap = STORE.live_frame_snapshot()
                provenance = snap.get("provenance") or {}
                version = (
                    snap.get("frame_index"),
                    snap.get("timestamp"),
                    provenance.get("received_at"),
                )
                if version != last_version:
                    await ws.send_json(snap)
                    last_version = version
                await asyncio.sleep(0.02)
        except (WebSocketDisconnect, Exception):
            pass

    @app.websocket("/ws/wearables")
    async def websocket_wearable_feed(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                rows = await list_wearables()
                status = await wearable_status()
                await ws.send_json({
                    "timestamp": time.time(),
                    "wearables": rows,
                    "status": status,
                })
                await asyncio.sleep(0.033)  # 30 Hz ultra-low latency push
        except (WebSocketDisconnect, Exception):
            pass


    # --- ESP32 Hardware & Flashing ---
    @app.get("/api/v1/hardware/ports")
    async def list_serial_ports():
        return FLASHER.list_ports()

    @app.get("/api/v1/hardware/status")
    async def get_hardware_status():
        return await run_in_threadpool(FLASHER.toolchain_status)

    @app.get("/api/v1/hardware/chip-info")
    async def get_esp32_chip_info(port: str):
        return await run_in_threadpool(FLASHER.get_chip_info, port)

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

    # --- Serial Monitor & USB pass-through ---
    @app.post("/api/v1/hardware/monitor/start")
    async def start_serial_monitor(body: dict):
        port = str(body.get("port") or "").strip()
        if not port:
            raise HTTPException(status_code=422, detail="port is required")
        result = await run_in_threadpool(FLASHER.start_monitor, port)
        if not result.get("ok"):
            raise HTTPException(status_code=409, detail=result.get("error"))
        return result

    @app.post("/api/v1/hardware/monitor/stop")
    async def stop_serial_monitor():
        return await run_in_threadpool(FLASHER.stop_monitor)

    @app.get("/api/v1/hardware/monitor")
    async def serial_monitor_status():
        return await run_in_threadpool(FLASHER.monitor_status)

    @app.post("/api/v1/hardware/usb/attach")
    async def attach_esp32_usb():
        """Bind and attach the ESP32 from Windows into WSL via usbipd.

        Runs the Windows usbipd tool with elevation when required. Safe to call
        repeatedly: an already-attached device returns ok with attached=False."""
        result = await run_in_threadpool(ensure_esp32_usb_attached)
        if not result.get("ok"):
            raise HTTPException(status_code=409, detail=result.get("error"))
        return result

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
