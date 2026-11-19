"""Operational job manager for real uploaded-video analysis."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Optional


class MatchMode(str, Enum):
    RECORDED = "recorded"
    UPLOAD = "upload"
    LIVE = "live"
    TRAIN = "train"


@dataclass
class ProcessingProgress:
    match_id: str
    mode: MatchMode
    status: str = "idle"
    progress_pct: Optional[float] = None
    current_frame: int = 0
    total_frames: int = 0
    fps: Optional[float] = None
    error_message: Optional[str] = None
    output_video: Optional[str] = None
    output_stats: Optional[str] = None
    log_path: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class EngineManager:
    """Run uploaded-video jobs and one local live-vision process."""

    def __init__(self, workspace_root: Path):
        self.root = workspace_root
        self.jobs: Dict[str, ProcessingProgress] = {}
        self.lock = threading.RLock()
        self._processes: Dict[str, subprocess.Popen] = {}
        self._live_process: Optional[subprocess.Popen] = None
        self._live_log = None
        self._live_state = {
            "status": "idle",
            "camera_id": None,
            "source_kind": None,
            "started_at": None,
            "stopped_at": None,
            "return_code": None,
            "error": None,
            "log_path": None,
        }

    def get_progress(self, match_id: str) -> Optional[ProcessingProgress]:
        with self.lock:
            return self.jobs.get(match_id)

    def _default_weights(self) -> Optional[Path]:
        return active_detector(self.root)


    @staticmethod
    def _bounded_env_int(name: str, default: int, low: int, high: int) -> int:
        try:
            value = int(os.environ.get(name, str(default)))
        except ValueError:
            value = default
        return max(low, min(value, high))

    def live_status(self) -> dict:
        with self.lock:
            state = dict(self._live_state)
            process = self._live_process
            state["running"] = bool(process is not None and process.poll() is None)
            state["pid"] = process.pid if state["running"] else None
            return state

    def note_live_frame(self, camera_id: str) -> None:
        with self.lock:
            if (
                self._live_process is not None
                and self._live_process.poll() is None
                and self._live_state.get("camera_id") == camera_id
            ):
                self._live_state["status"] = "running"

    def start_live_processing(
        self,
        camera_id: str,
        source: str,
        source_kind: str,
        calibration_path: Optional[str] = None,
        product_api: str = "http://127.0.0.1:8000",
    ) -> dict:
        """Start the same measured real-time pipeline used by the demo script."""
        source = str(source).strip()
        weights = self._default_weights()
        calibration = None
        if calibration_path:
            calibration = Path(calibration_path).expanduser()
            if not calibration.is_absolute():
                calibration = self.root / calibration
            calibration = calibration.resolve()

        if not source:
            return {"ok": False, "error": "camera source is empty"}
        if source_kind == "file":
            source_path = Path(source).expanduser()
            if not source_path.is_absolute():
                source_path = self.root / source_path
            if not source_path.is_file():
                return {"ok": False, "error": f"video file not found: {source_path}"}
            source = str(source_path.resolve())
        if source_kind == "usb" and not source.isdigit():
            return {"ok": False, "error": "USB source must be a camera number"}
        if source_kind == "rtsp" and not source.lower().startswith(("rtsp://", "rtsps://")):
            return {"ok": False, "error": "invalid RTSP source"}
        if source_kind == "phone":
            try:
                from .phone_camera import parse_source
                parse_source(source)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
        if weights is None:
            return {"ok": False, "error": "no trained detector checkpoint is configured"}
        if calibration is not None and not calibration.is_file():
            return {"ok": False, "error": f"calibration not found: {calibration}"}
        if not product_api.startswith(("http://127.0.0.1:", "http://localhost:")):
            return {"ok": False, "error": "live vision must publish to the local product API"}

        log_dir = self.root / "data" / "live"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "vision.log"
        command = [
            sys.executable,
            "-u",
            "-m",
            "soccer_analytics.realtime",
            "--video",
            source,
            "--weights",
            str(weights),
            "--imgsz",
            str(self._bounded_env_int("CPG44_IMGSZ", 960, 640, 1920)),
            "--analytics-every",
            str(self._bounded_env_int("CPG44_ANALYTICS_EVERY", 10, 1, 120)),
            "--snapshot-every",
            str(self._bounded_env_int("CPG44_SNAPSHOT_EVERY", 2, 1, 30)),
            "--latency-budget-ms",
            str(self._bounded_env_int("CPG44_LATENCY_BUDGET_MS", 240, 80, 1000)),
            "--product-api",
            product_api,
            "--camera-id",
            camera_id,
            "--no-window",
        ]
        if calibration is not None:
            command.extend(["--calibration", str(calibration)])
        if source_kind == "file":
            command.append("--loop")
        sensor_hub = os.environ.get("CPG44_HUB_URL", "").strip()
        if sensor_hub:
            command.extend(["--sensor-hub", sensor_hub])
        roster = os.environ.get("CPG44_ROSTER", "").strip()
        if roster:
            command.extend(["--roster", roster])
        roster_numbers = os.environ.get("CPG44_ROSTER_NUMBERS", "").strip()
        if roster_numbers:
            command.extend(["--roster-numbers", roster_numbers])

        with self.lock:
            if self._live_process is not None and self._live_process.poll() is None:
                return {
                    "ok": False,
                    "error": f"camera {self._live_state.get('camera_id')} is already processing",
                }
            try:
                self._live_log = log_path.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    command,
                    cwd=self.root,
                    stdout=self._live_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except Exception as exc:
                if self._live_log is not None:
                    self._live_log.close()
                    self._live_log = None
                return {"ok": False, "error": str(exc)}
            self._live_process = process
            self._live_state = {
                "status": "starting",
                "camera_id": camera_id,
                "source_kind": source_kind,
                "started_at": time.time(),
                "stopped_at": None,
                "return_code": None,
                "error": None,
                "log_path": str(log_path),
            }
        threading.Thread(target=self._monitor_live_process, args=(process,), daemon=True).start()
        return {"ok": True, **self.live_status()}

    def _monitor_live_process(self, process: subprocess.Popen) -> None:
        return_code = process.wait()
        with self.lock:
            if process is not self._live_process:
                return
            requested_stop = self._live_state.get("status") == "stopping"
            self._live_state.update({
                "status": "stopped" if requested_stop else ("completed" if return_code == 0 else "failed"),
                "stopped_at": time.time(),
                "return_code": return_code,
                "error": None if requested_stop or return_code == 0 else "vision process exited; check its log",
            })
            self._live_process = None
            if self._live_log is not None:
                self._live_log.close()
                self._live_log = None

    def stop_live_processing(self) -> dict:
        with self.lock:
            process = self._live_process
            if process is None or process.poll() is not None:
                return {"ok": False, "error": "no live vision process is running", **self.live_status()}
            self._live_state["status"] = "stopping"
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        return {"ok": True, **self.live_status()}

    def start_upload_processing(
        self,
        match_id: str,
        video_path: str,
        calibration_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        on_complete: Optional[Callable[[dict], None]] = None,
    ):
        video = Path(video_path).resolve()
        weights = Path(weights_path).expanduser().resolve() if weights_path else self._default_weights()
        calibration = Path(calibration_path).expanduser().resolve() if calibration_path else None
        with self.lock:
            progress = ProcessingProgress(
                match_id=match_id,
                mode=MatchMode.UPLOAD,
                status="queued",
                started_at=time.time(),
            )
            if not video.is_file():
                progress.status = "failed"
                progress.error_message = f"video not found: {video}"
            elif weights is None or not weights.is_file():
                progress.status = "failed"
                progress.error_message = "no trained detector checkpoint is configured"
            elif calibration and not calibration.is_file():
                progress.status = "failed"
                progress.error_message = f"calibration not found: {calibration}"
            self.jobs[match_id] = progress
        if progress.status == "failed":
            return

        thread = threading.Thread(
            target=self._run_upload_pipeline,
            args=(match_id, video, weights, calibration, on_complete),
            daemon=True,
        )
        thread.start()

    def _run_upload_pipeline(
        self,
        match_id: str,
        video: Path,
        weights: Path,
        calibration: Optional[Path],
        on_complete: Optional[Callable[[dict], None]],
    ):
        import cv2
        capture = cv2.VideoCapture(str(video))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
        capture.release()

        out_dir = self.root / "data" / "matches" / match_id
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "pipeline.log"
        try:
            upload_imgsz = int(os.environ.get("CPG44_UPLOAD_IMGSZ", "1280"))
        except ValueError:
            upload_imgsz = 1280
        upload_imgsz = max(640, min(upload_imgsz, 1920))
        command = [
            sys.executable,
            "-m",
            "soccer_analytics.match_analysis",
            "--video",
            str(video),
            "--weights",
            str(weights),
            "--out",
            str(out_dir),
            "--imgsz",
            str(upload_imgsz),
            "--name",
            match_id,
        ]
        if calibration:
            command.extend(["--calibration", str(calibration)])

        started = time.time()
        try:
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=self.root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                with self.lock:
                    self._processes[match_id] = process
                    self.jobs[match_id].status = "processing"
                    self.jobs[match_id].total_frames = total_frames
                    self.jobs[match_id].log_path = str(log_path)
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    match = re.search(r"\[analyze\]\s+(\d+)/(\d+|\?)\s+frames", line)
                    if not match:
                        continue
                    current = int(match.group(1))
                    elapsed = max(time.time() - started, 1e-3)
                    with self.lock:
                        job = self.jobs[match_id]
                        job.current_frame = current
                        job.fps = round(current / elapsed, 2)
                        if total_frames:
                            # Phase 1 is most of the work; reserve the final 10% for
                            # analytics, heatmaps, and video rendering.
                            job.progress_pct = round(min(current / total_frames, 1.0) * 90, 1)
                return_code = process.wait()

            stats_path = out_dir / "stats.json"
            video_path = out_dir / "annotated.mp4"
            with self.lock:
                job = self.jobs[match_id]
                job.status = "completed" if return_code == 0 and stats_path.is_file() else "failed"
                job.progress_pct = 100.0 if job.status == "completed" else job.progress_pct
                job.output_stats = str(stats_path) if stats_path.is_file() else None
                job.output_video = str(video_path) if video_path.is_file() else None
                job.completed_at = time.time()
                if job.status == "failed":
                    job.error_message = f"pipeline exited with code {return_code}; see {log_path}"
                self._processes.pop(match_id, None)
            if on_complete and stats_path.is_file():
                on_complete(json.loads(stats_path.read_text(encoding="utf-8")))
        except Exception as exc:
            with self.lock:
                job = self.jobs[match_id]
                job.status = "failed"
                job.error_message = str(exc)
                job.completed_at = time.time()
                self._processes.pop(match_id, None)


def active_detector(root: Path) -> Optional[Path]:
    """The detector live and offline analysis use: a promoted model, then the
    environment override, then the best measured checkpoint. One function so
    the status page shows exactly what runs."""
    configured = Path(str(Path.cwd())) / "__missing__"
    active = root / "configs" / "active_detector.json"
    if active.is_file():                      # set by "Use this model" after a measured win
        try:
            configured = Path(json.loads(active.read_text())["weights"])
        except (ValueError, KeyError):
            pass
    if os.environ.get("CPG44_MODEL_PATH"):
        configured = Path(os.environ["CPG44_MODEL_PATH"]).expanduser()
    candidates = [
        configured,
        # Ordered by measured held-out SoccerNet test mAP (docs/TRACKING_BENCHMARK.md).
        # soccernet_v2 scored 0.27 mAP50-95 vs 0.42 for soccernet, so it is last.
        root / "runs" / "detect" / "soccernet" / "weights" / "best.pt",
        root / "runs" / "detect" / "soccernet_v3_s1" / "weights" / "best.pt",
        root / "runs" / "detect" / "soccernet_v2" / "weights" / "best.pt",
    ]
    return next((path.resolve() for path in candidates if path.is_file()), None)
