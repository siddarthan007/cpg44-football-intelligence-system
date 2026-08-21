"""Operational job manager for real uploaded-video analysis."""

from __future__ import annotations

import json
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
    """Run the existing CV pipeline in an isolated subprocess per upload."""

    def __init__(self, workspace_root: Path):
        self.root = workspace_root
        self.jobs: Dict[str, ProcessingProgress] = {}
        self.lock = threading.RLock()
        self._processes: Dict[str, subprocess.Popen] = {}

    def get_progress(self, match_id: str) -> Optional[ProcessingProgress]:
        with self.lock:
            return self.jobs.get(match_id)

    def _default_weights(self) -> Optional[Path]:
        configured = Path(str(Path.cwd())) / "__missing__"
        import os
        if os.environ.get("CPG44_MODEL_PATH"):
            configured = Path(os.environ["CPG44_MODEL_PATH"]).expanduser()
        candidates = [
            configured,
            self.root / "runs" / "detect" / "soccernet_v2" / "weights" / "best.pt",
            self.root / "runs" / "detect" / "soccernet" / "weights" / "best.pt",
        ]
        return next((path.resolve() for path in candidates if path.is_file()), None)

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
        command = [
            sys.executable,
            "-m",
            "soccer_analytics.pipeline",
            "--video",
            str(video),
            "--weights",
            str(weights),
            "--out",
            str(out_dir),
            "--render",
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
