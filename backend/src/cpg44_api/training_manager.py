"""Real training-process orchestration and evidence reporting for CPG44."""

from __future__ import annotations

import copy
import csv
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]


def _number(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class TrainingManager:
    """Start real Ultralytics jobs and expose metrics written by the trainer."""

    def __init__(self, workspace_root: Path):
        self.root = workspace_root
        self.active_job: Optional[dict] = None
        self.lock = threading.RLock()
        self._process: Optional[subprocess.Popen] = None
        self._last_multimodal_capture: Dict[tuple[str, int], float] = {}
        try:
            configured_window = int(os.environ.get("CPG44_OUTCOME_WINDOW_DAYS", "7"))
        except ValueError:
            configured_window = 7
        self.outcome_window_days = max(1, min(configured_window, 90))
        self.jobs_dir = self.root / "data" / "training_jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.root / "models" / "strain_xgboost.pkl"
        self.strain_model_status = {
            "trained": model_path.is_file(),
            "backend": "xgboost" if model_path.is_file() else None,
            "model_path": str(model_path) if model_path.is_file() else None,
            "validation": None,
            "training_source": "independently reviewed player-session outcomes required",
        }

    def _default_dataset(self) -> Path:
        configured = os.environ.get("CPG44_DATASET_YAML")
        candidates = [
            Path(configured) if configured else None,
            Path("/home/siddartha/SoccerNet/yolo/data.yaml"),
            Path("/home/siddartha/SoccerNet_YOLO/data.yaml"),
        ]
        return next((path for path in candidates if path and path.is_file()), candidates[1])

    @staticmethod
    def _read_last_metrics(path: Path) -> Optional[dict]:
        if not path.is_file():
            return None
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                return None
            row = rows[-1]
            return {
                "epoch": int(float(row.get("epoch", len(rows) - 1))) + 1,
                "precision": _number(row.get("metrics/precision(B)")),
                "recall": _number(row.get("metrics/recall(B)")),
                "map50": _number(row.get("metrics/mAP50(B)")),
                "map50_95": _number(row.get("metrics/mAP50-95(B)")),
                "box_loss": _number(row.get("train/box_loss")),
                "class_loss": _number(row.get("train/cls_loss")),
            }
        except (OSError, ValueError):
            return None

    def _existing_runs(self) -> List[dict]:
        rows = []
        for results in sorted((self.root / "runs" / "detect").glob("*/results.csv")):
            metrics = self._read_last_metrics(results)
            if metrics is None:
                continue
            run_dir = results.parent
            best = run_dir / "weights" / "best.pt"
            rows.append({
                "name": run_dir.name,
                "metrics": metrics,
                "best_weights": str(best.relative_to(self.root)) if best.is_file() else None,
                "updated_at": results.stat().st_mtime,
            })
        rows.sort(key=lambda row: row["updated_at"], reverse=True)
        return rows

    def _refresh_active_metrics(self):
        if not self.active_job:
            return
        metrics = self._read_last_metrics(Path(self.active_job["results_csv"]))
        if metrics:
            self.active_job["metrics"] = metrics
            epochs = max(int(self.active_job["epochs"]), 1)
            self.active_job["progress_pct"] = round(min(metrics["epoch"] / epochs, 1.0) * 100, 1)

    def get_status(self) -> dict:
        with self.lock:
            self._refresh_active_metrics()
            dataset = self._default_dataset()
            active = copy.deepcopy(self.active_job)
            strain = copy.deepcopy(self.strain_model_status)
        label_status = {
            "collected_samples": 0,
            "collected_player_sessions": 0,
            "labelled_samples": 0,
            "positive_outcomes": 0,
            "ready": False,
        }
        db_path = self.root / "data" / "multimodal_dataset.db"
        if db_path.is_file():
            try:
                with sqlite3.connect(str(db_path)) as connection:
                    self._ensure_multimodal_schema(connection)
                    collected, sessions = connection.execute(
                        "SELECT COUNT(*), COUNT(DISTINCT match_id || ':' || player_id) "
                        "FROM multimodal_samples"
                    ).fetchone()
                    count, positives = connection.execute(
                        "SELECT COUNT(*), COALESCE(SUM(injury_label), 0) "
                        "FROM outcome_labels WHERE outcome_window_days = ?",
                        (self.outcome_window_days,),
                    ).fetchone()
                label_status = {
                    "collected_samples": int(collected),
                    "collected_player_sessions": int(sessions),
                    "labelled_samples": int(count),
                    "positive_outcomes": int(positives),
                    "ready": int(count) >= 100 and int(positives) >= 10
                    and int(count) - int(positives) >= 10,
                }
            except sqlite3.Error:
                pass
        return {
            "active_job": active,
            "strain_model": strain,
            "dataset_path": str(dataset),
            "dataset_ready": dataset.is_file(),
            "runs": self._existing_runs()[:8],
            "gpu_required_for_practical_training": True,
            "outcome_labels": label_status,
            "outcome_window_days": self.outcome_window_days,
        }

    def start_yolo_training(
        self,
        data_path: str,
        model_name: str = "yolov8m.pt",
        epochs: int = 50,
        imgsz: int = 1280,
        batch: int = 4,
        lr: float = 0.01,
    ) -> dict:
        del lr  # the CLI owns its tuned learning-rate schedule
        dataset = Path(data_path).expanduser().resolve()
        if not dataset.is_file():
            return {"ok": False, "error": f"dataset YAML not found: {dataset}"}
        if not 1 <= epochs <= 500:
            return {"ok": False, "error": "epochs must be between 1 and 500"}
        if imgsz not in {640, 768, 960, 1024, 1280, 1536, 1920}:
            return {"ok": False, "error": "unsupported image size"}
        if batch < 1 or batch > 128:
            return {"ok": False, "error": "batch must be between 1 and 128"}

        local_model = self.root / model_name
        model = str(local_model) if local_model.is_file() else model_name
        job_id = f"dashboard_{int(time.time())}"
        run_dir = self.root / "runs" / "detect" / job_id
        log_path = self.jobs_dir / f"{job_id}.log"
        command = [
            sys.executable,
            "-m",
            "soccer_analytics.train",
            "base",
            "--data",
            str(dataset),
            "--model",
            model,
            "--epochs",
            str(epochs),
            "--imgsz",
            str(imgsz),
            "--batch",
            str(batch),
            "--name",
            job_id,
        ]
        with self.lock:
            if self.active_job and self.active_job.get("status") in {"starting", "running"}:
                return {"ok": False, "error": "a training process is already running"}
            self.active_job = {
                "id": job_id,
                "model": model_name,
                "data": str(dataset),
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "status": "starting",
                "progress_pct": 0.0,
                "metrics": None,
                "pid": None,
                "started_at": time.time(),
                "results_csv": str(run_dir / "results.csv"),
                "log_path": str(log_path),
                "best_weights": str(run_dir / "weights" / "best.pt"),
            }
        threading.Thread(
            target=self._run_yolo_job,
            args=(job_id, command, log_path),
            daemon=True,
        ).start()
        return {"ok": True, "job": copy.deepcopy(self.active_job)}

    def _run_yolo_job(self, job_id: str, command: List[str], log_path: Path):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=self.root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True,
                )
                with self.lock:
                    if not self.active_job or self.active_job.get("id") != job_id:
                        process.terminate()
                        return
                    self._process = process
                    self.active_job.update({"status": "running", "pid": process.pid})
                return_code = process.wait()
            with self.lock:
                self._refresh_active_metrics()
                if self.active_job and self.active_job.get("id") == job_id:
                    stopped = self.active_job.get("status") == "stopping"
                    self.active_job["status"] = "stopped" if stopped else (
                        "completed" if return_code == 0 else "failed"
                    )
                    self.active_job["return_code"] = return_code
                    self.active_job["completed_at"] = time.time()
                    self._process = None
        except Exception as exc:
            with self.lock:
                if self.active_job and self.active_job.get("id") == job_id:
                    self.active_job.update({
                        "status": "failed",
                        "error": str(exc),
                        "completed_at": time.time(),
                    })
                    self._process = None

    def stop_yolo_training(self) -> dict:
        with self.lock:
            process = self._process
            if not process or process.poll() is not None:
                return {"ok": False, "error": "no active training process"}
            self.active_job["status"] = "stopping"
            process.terminate()
            return {"ok": True, "message": f"termination requested for PID {process.pid}"}

    def record_multimodal_sample(
        self,
        match_id: str,
        player_id: int,
        external_metrics: dict,
        internal_metrics: dict,
        source_timestamp: Optional[float] = None,
        frame_index: Optional[int] = None,
    ):
        """Persist synchronized features; injury outcomes remain NULL until labelled."""
        db_path = self.root / "data" / "multimodal_dataset.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(db_path)) as connection:
            self._ensure_multimodal_schema(connection)
            connection.execute(
                """
                INSERT INTO multimodal_samples (
                    match_id, player_id, timestamp, speed_mps, distance_m, hsr_m,
                    sprints, accel_efforts, decel_efforts, metabolic_power, hr,
                    hr_drift, spo2, player_load, acwr, signal_quality, frame_index,
                    recorded_at, injury_label, label_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    match_id,
                    player_id,
                    source_timestamp or time.time(),
                    external_metrics.get("speed_mps"),
                    external_metrics.get("distance_m"),
                    external_metrics.get("hsr_m"),
                    external_metrics.get("sprints"),
                    external_metrics.get("accel_efforts"),
                    external_metrics.get("decel_efforts"),
                    external_metrics.get("metabolic_power"),
                    internal_metrics.get("hr"),
                    internal_metrics.get("hr_drift"),
                    internal_metrics.get("spo2"),
                    internal_metrics.get("player_load"),
                    internal_metrics.get("acwr"),
                    internal_metrics.get("signal_quality"),
                    frame_index,
                    time.time(),
                    None,
                    None,
                ),
            )

    @staticmethod
    def _ensure_multimodal_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS multimodal_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                player_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                speed_mps REAL, distance_m REAL, hsr_m REAL, sprints INTEGER,
                accel_efforts INTEGER, decel_efforts INTEGER,
                metabolic_power REAL, hr REAL, hr_drift REAL, spo2 REAL,
                player_load REAL, acwr REAL, signal_quality REAL,
                frame_index INTEGER, recorded_at REAL,
                injury_label INTEGER NULL, label_source TEXT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_multimodal_session_time
                ON multimodal_samples(match_id, player_id, timestamp);
            CREATE TABLE IF NOT EXISTS outcome_labels (
                match_id TEXT NOT NULL,
                player_id INTEGER NOT NULL,
                injury_label INTEGER NOT NULL CHECK(injury_label IN (0, 1)),
                outcome_window_days INTEGER NOT NULL,
                label_source TEXT NOT NULL,
                notes TEXT,
                labelled_at REAL NOT NULL,
                PRIMARY KEY(match_id, player_id)
            );
            """
        )
        existing = {
            row[1] for row in connection.execute("PRAGMA table_info(multimodal_samples)")
        }
        for name, kind in (
            ("signal_quality", "REAL"),
            ("frame_index", "INTEGER"),
            ("recorded_at", "REAL"),
        ):
            if name not in existing:
                connection.execute(f"ALTER TABLE multimodal_samples ADD COLUMN {name} {kind}")

    def record_live_snapshot(self, snapshot: dict, interval_s: float = 5.0) -> int:
        """Collect real, calibrated vision + wearable rows at a bounded cadence."""
        if not snapshot.get("metric") or not isinstance(snapshot.get("players"), list):
            return 0
        match_id = str(snapshot.get("match_id") or "live")[:64]
        try:
            source_timestamp = float(snapshot.get("timestamp"))
        except (TypeError, ValueError):
            return 0
        if not source_timestamp > 0:
            return 0

        recorded = 0
        for player in snapshot["players"]:
            if not isinstance(player, dict):
                continue
            wearable = player.get("wearable_metrics")
            load = player.get("load")
            if not isinstance(wearable, dict) or not isinstance(load, dict):
                continue
            try:
                player_id = int(player.get("player_id"))
            except (TypeError, ValueError):
                continue
            capture_key = (match_id, player_id)
            with self.lock:
                previous = self._last_multimodal_capture.get(capture_key, 0.0)
                if source_timestamp <= previous or source_timestamp - previous < interval_s:
                    continue
                self._last_multimodal_capture[capture_key] = source_timestamp
            self.record_multimodal_sample(
                match_id=match_id,
                player_id=player_id,
                source_timestamp=source_timestamp,
                frame_index=snapshot.get("frame_index"),
                external_metrics={
                    "speed_mps": player.get("top_speed_mps")
                    if player.get("top_speed_mps") is not None else player.get("speed_mps"),
                    "distance_m": player.get("distance_m"),
                    "hsr_m": load.get("hsr_m"),
                    "sprints": load.get("sprints"),
                    "accel_efforts": load.get("accel_efforts"),
                    "decel_efforts": load.get("decel_efforts"),
                    "metabolic_power": load.get("metabolic_power_wkg"),
                },
                internal_metrics={
                    "hr": wearable.get("hr"),
                    "spo2": wearable.get("spo2"),
                    "signal_quality": wearable.get("signal_quality"),
                    "hr_drift": load.get("hr_drift"),
                    "player_load": load.get("player_load_imu")
                    if load.get("player_load_imu") is not None else load.get("player_load"),
                    "acwr": load.get("acwr"),
                },
            )
            recorded += 1
        return recorded

    def label_player_session(
        self,
        match_id: str,
        player_id: int,
        injury_label: int,
        label_source: str,
        outcome_window_days: int = 7,
        notes: str = "",
    ) -> dict:
        """Attach one independently reviewed outcome to one player-session."""
        match_id = str(match_id).strip()
        label_source = str(label_source).strip()
        if not match_id or len(match_id) > 64:
            return {"ok": False, "error": "match_id is required and must be at most 64 characters"}
        if not 1 <= int(player_id) <= 9999 or int(injury_label) not in (0, 1):
            return {"ok": False, "error": "player_id or binary outcome is invalid"}
        if len(label_source) < 4 or len(label_source) > 200:
            return {"ok": False, "error": "label_source must identify the reviewer or record"}
        if int(outcome_window_days) != self.outcome_window_days:
            return {
                "ok": False,
                "error": f"this model uses a fixed {self.outcome_window_days}-day outcome window",
            }
        db_path = self.root / "data" / "multimodal_dataset.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(db_path)) as connection:
            self._ensure_multimodal_schema(connection)
            exists = connection.execute(
                "SELECT 1 FROM multimodal_samples WHERE match_id = ? AND player_id = ? LIMIT 1",
                (match_id, int(player_id)),
            ).fetchone()
            if not exists:
                return {"ok": False, "error": "no measured player-session exists for this label"}
            connection.execute(
                """
                INSERT INTO outcome_labels (
                    match_id, player_id, injury_label, outcome_window_days,
                    label_source, notes, labelled_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(match_id, player_id) DO UPDATE SET
                    injury_label=excluded.injury_label,
                    outcome_window_days=excluded.outcome_window_days,
                    label_source=excluded.label_source,
                    notes=excluded.notes,
                    labelled_at=excluded.labelled_at
                """,
                (
                    match_id, int(player_id), int(injury_label), int(outcome_window_days),
                    label_source, str(notes)[:1000], time.time(),
                ),
            )
        return {"ok": True, "match_id": match_id, "player_id": int(player_id)}

    def train_strain_model(self) -> dict:
        """Train only from real labelled outcomes with a chronological holdout."""
        from soccer_analytics.loadtypes import WorkloadFeatures
        from soccer_analytics.sensors.injury import InjuryRiskModel, features_to_array

        db_path = self.root / "data" / "multimodal_dataset.db"
        if not db_path.is_file():
            return {"ok": False, "error": "no multimodal dataset has been collected"}
        with sqlite3.connect(str(db_path)) as connection:
            connection.row_factory = sqlite3.Row
            self._ensure_multimodal_schema(connection)
            rows = connection.execute(
                """
                SELECT sample.*, outcome.injury_label AS outcome_label,
                       outcome.label_source AS outcome_source
                FROM multimodal_samples AS sample
                JOIN outcome_labels AS outcome
                  ON outcome.match_id = sample.match_id
                 AND outcome.player_id = sample.player_id
                WHERE sample.id = (
                    SELECT latest.id FROM multimodal_samples AS latest
                    WHERE latest.match_id = sample.match_id
                      AND latest.player_id = sample.player_id
                    ORDER BY latest.timestamp DESC, latest.id DESC LIMIT 1
                )
                  AND outcome.outcome_window_days = ?
                ORDER BY sample.timestamp
                """,
                (self.outcome_window_days,),
            ).fetchall()
        positives = sum(int(row["outcome_label"]) for row in rows)
        if len(rows) < 100 or positives < 10 or len(rows) - positives < 10:
            return {
                "ok": False,
                "error": "need at least 100 independently labelled player-sessions with 10 outcomes in each class",
                "labelled_samples": len(rows),
                "positive_outcomes": positives,
            }

        features = [
            WorkloadFeatures(
                player_id=int(row["player_id"]),
                total_distance=float(row["distance_m"] or 0),
                hsr_distance=float(row["hsr_m"] or 0),
                sprint_count=int(row["sprints"] or 0),
                accel_efforts=int(row["accel_efforts"] or 0),
                decel_efforts=int(row["decel_efforts"] or 0),
                player_load=float(row["player_load"] or 0),
                metabolic_power_avg=float(row["metabolic_power"] or 0),
                top_speed=float(row["speed_mps"] or 0),
                avg_hr=float(row["hr"] or float("nan")),
                hr_drift=float(row["hr_drift"] or 0),
                min_spo2=float(row["spo2"] or float("nan")),
                acwr=float(row["acwr"] or float("nan")),
            )
            for row in rows
        ]
        labels = [int(row["outcome_label"]) for row in rows]
        split = int(len(rows) * 0.8)
        if len(set(labels[:split])) < 2 or len(set(labels[split:])) < 2:
            return {"ok": False, "error": "chronological train and holdout periods both need two classes"}

        validation_model = InjuryRiskModel().fit(features[:split], labels[:split])
        import numpy as np
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

        X_holdout = np.vstack([features_to_array(feature) for feature in features[split:]])
        probabilities = validation_model.model.predict_proba(X_holdout)[:, 1]
        y_holdout = np.asarray(labels[split:])
        validation = {
            "split": "chronological_last_20_percent",
            "samples": int(len(y_holdout)),
            "roc_auc": round(float(roc_auc_score(y_holdout, probabilities)), 4),
            "average_precision": round(float(average_precision_score(y_holdout, probabilities)), 4),
            "brier_score": round(float(brier_score_loss(y_holdout, probabilities)), 4),
        }

        final_model = InjuryRiskModel().fit(features, labels)
        out_dir = self.root / "models"
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "strain_xgboost.pkl"
        final_model.save(str(model_path))
        with self.lock:
            self.strain_model_status = {
                "trained": True,
                "backend": final_model._backend,
                "model_path": str(model_path),
                "last_trained": time.time(),
                "labelled_samples": len(rows),
                "positive_outcomes": positives,
                "validation": validation,
                "training_source": "labelled multimodal_dataset.db",
            }
        return {"ok": True, "strain_model": copy.deepcopy(self.strain_model_status)}
