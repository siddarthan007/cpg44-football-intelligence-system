"""Evidence-backed runtime state for the CPG44 product API.

A dashboard snapshot is a fresh frame posted by the vision pipeline, a recorded
analysis artifact, or an explicit empty state. It is never invented telemetry.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
import urllib.request
from urllib.parse import urlparse
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _finite_number(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        number = float(value)
        if number == number and abs(number) != float("inf"):
            return number
    except (TypeError, ValueError):
        pass
    return default


def _numeric_player_id(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value)
    for prefix in ("PLAYER_", "TRACK_", "P_"):
        if text.upper().startswith(prefix):
            text = text[len(prefix):]
            break
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _validated_relay_origin(value: str) -> str:
    origin = str(value or "").strip().rstrip("/")
    if not origin:
        return ""
    parsed = urlparse(origin)
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").lower() != "cpg44.nivaspms.com"
        or parsed.port not in (None, 443)
        or parsed.path
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("CPG44 relay origin must be exactly https://cpg44.nivaspms.com")
    return origin


class ProductStore:
    """Bounded live state plus file-backed artifacts for a single-node demo."""

    LIVE_MAX_AGE_S = 3.0
    WEARABLE_MAX_AGE_S = 10.0

    def __init__(
        self,
        stats_path: Optional[Path] = None,
        hub_url: str = "http://127.0.0.1:8081",
        hostinger_url: Optional[str] = None,
        relay_token: Optional[str] = None,
    ):
        configured_stats = os.environ.get("CPG44_STATS_PATH", "").strip()
        configured_video = os.environ.get("CPG44_RECORDED_VIDEO", "").strip()
        self.stats_path = stats_path or (
            Path(configured_stats).expanduser()
            if configured_stats
            else ROOT / "data" / "recorded" / "stats.json"
        )
        self.recorded_video_path: Optional[Path] = (
            Path(configured_video).expanduser() if configured_video else None
        )
        self.hub_url = hub_url.rstrip("/")
        self.hostinger_url = _validated_relay_origin(
            hostinger_url
            or os.environ.get("CPG44_RELAY_URL", "")
            or os.environ.get("HOSTINGER_RELAY_URL", "")
        )
        self.relay_token = (
            relay_token
            or os.environ.get("CPG44_RELAY_TOKEN", "")
            or os.environ.get("HOSTINGER_RELAY_TOKEN", "")
        )
        self.started_at = time.time()
        self.wearable_log: List[dict] = []
        self.active_mode = "recorded"
        self._lock = threading.RLock()
        self._latest_live_frame: Optional[dict] = None
        self._latest_live_received_at: Optional[float] = None
        self._latest_live_jpeg: Optional[bytes] = None
        self._latest_live_jpeg_received_at: Optional[float] = None

        self.match = {
            "match_id": "live",
            "name": os.environ.get("CPG44_MATCH_NAME", "Campus football session"),
            "period": None,
            "status": "ready",
            "mode": self.active_mode,
            "pitch_length_m": 105.0,
            "pitch_width_m": 68.0,
            "venue": os.environ.get("CPG44_MATCH_VENUE") or None,
            "notes": "Live or explicitly selected recorded evidence appears here.",
            "created_at": self.started_at,
            "engine_running": False,
            "home_team": os.environ.get("CPG44_HOME_TEAM", "Team 1"),
            "away_team": os.environ.get("CPG44_AWAY_TEAM", "Team 2"),
        }

        # Empty until a source is actually registered. Previous versions exposed
        # invented RTSP cameras with invented FPS and latency.
        self.cameras: Dict[str, dict] = {}

    @property
    def roster(self) -> List[dict]:
        return [
            {
                "global_player_id": player["global_player_id"],
                "jersey": player.get("jersey"),
                "name": player["name"],
                "team_id": player["team_id"],
                "team": player["team"],
                "position": None,
                "wearable": player["wearable"],
                "wearable_id": player["track_id"] if player["wearable"] else None,
            }
            for player in self._recorded_snapshot()["players"]
        ]

    @property
    def passing_network(self) -> dict:
        return self.live_frame_snapshot().get("passing_network") or {}

    @property
    def timeline_tags(self) -> list:
        return self.live_frame_snapshot().get("timeline_tags") or []

    def stats(self) -> dict:
        return _load_json(self.stats_path)

    def activate_recorded_result(
        self, stats_path: Path, video_path: Optional[Path], match_id: str, match_name: str
    ) -> None:
        """Select a completed, real pipeline artifact for dashboard playback."""
        with self._lock:
            self.stats_path = Path(stats_path)
            self.recorded_video_path = Path(video_path) if video_path else None
            self.active_mode = "recorded"
            self.match.update({
                "match_id": str(match_id),
                "name": str(match_name),
                "mode": "recorded",
                "status": "completed",
                "engine_running": False,
            })

    def set_mode(self, mode: str):
        if mode not in {"recorded", "upload", "live", "train"}:
            raise ValueError(f"unsupported mode: {mode}")
        with self._lock:
            self.active_mode = mode
            self.match["mode"] = mode

    def register_camera(self, cam_dict: dict) -> dict:
        with self._lock:
            cid = str(cam_dict.get("id") or f"cam_{len(self.cameras) + 1}")
            camera = {
                "id": cid,
                "name": str(cam_dict.get("name") or cid),
                "type": str(cam_dict.get("type") or "file"),
                "source": str(cam_dict.get("source") or ""),
                "status": "registered",
                "fps": None,
                "latency_ms": None,
                "calibrated": False,
                "registered_at": time.time(),
                "last_frame_at": None,
            }
            self.cameras[cid] = camera
            return copy.deepcopy(camera)

    def update_camera_health(self, camera_id: str, **values: Any) -> dict:
        with self._lock:
            if camera_id not in self.cameras:
                self.register_camera({"id": camera_id})
            allowed = {"status", "fps", "latency_ms", "calibrated", "last_frame_at"}
            self.cameras[camera_id].update({k: v for k, v in values.items() if k in allowed})
            return copy.deepcopy(self.cameras[camera_id])

    def camera_list(self) -> List[dict]:
        now = time.time()
        with self._lock:
            rows = copy.deepcopy(list(self.cameras.values()))
        for row in rows:
            last_frame = row.get("last_frame_at")
            if row.get("status") == "online" and (
                not isinstance(last_frame, (int, float)) or now - last_frame > self.LIVE_MAX_AGE_S
            ):
                row["status"] = "stale"
            row["frame_age_s"] = round(now - last_frame, 2) if isinstance(last_frame, (int, float)) else None
        return rows

    def fetch_hub(self) -> Optional[dict]:
        try:
            request = urllib.request.Request(
                f"{self.hub_url}/api/latest",
                headers={"Accept": "application/json", "Cache-Control": "no-store"},
            )
            with urllib.request.urlopen(request, timeout=0.5) as response:
                value = json.loads(response.read().decode("utf-8"))
                if isinstance(value, dict):
                    return value
        except Exception:
            pass
        return None

    def record_wearable(self, body: dict) -> dict:
        """Store one normalized observation using match + player + timestamp."""
        now = time.time()
        player_id = _numeric_player_id(body.get("global_player_id") or body.get("player_id"))
        global_id = body.get("global_player_id") or (
            f"PLAYER_{player_id}" if player_id is not None else None
        )
        metrics = dict(body.get("metrics") or {})
        for key in ("hr", "spo2", "player_load", "gps", "speed_mps", "acceleration"):
            if key in body and key not in metrics:
                metrics[key] = body[key]
        row = {
            "match_id": str(body.get("match_id") or "live"),
            "global_player_id": global_id,
            "player_id": player_id,
            "timestamp": _finite_number(body.get("timestamp") or body.get("t"), now),
            "source": str(body.get("source") or "wearable"),
            "metrics": metrics,
            "received_at": now,
        }
        with self._lock:
            self.wearable_log.append(row)
            if len(self.wearable_log) > 2000:
                del self.wearable_log[:-2000]
        return copy.deepcopy(row)

    def _fresh_wearables(self) -> Dict[int, dict]:
        now = time.time()
        fresh: Dict[int, dict] = {}
        with self._lock:
            rows = list(self.wearable_log)
        for row in rows:
            pid = _numeric_player_id(row.get("player_id") or row.get("global_player_id"))
            age = now - float(row.get("received_at") or 0)
            if pid is not None and age <= self.WEARABLE_MAX_AGE_S:
                fresh[pid] = row
        return fresh

    def ingest_live_frame(self, body: dict) -> dict:
        """Accept a real per-frame snapshot from ``soccer_analytics.realtime``."""
        now = time.time()
        frame = copy.deepcopy(body)
        frame.setdefault("timestamp", now)
        frame.setdefault("match_id", "live")
        frame["provenance"] = {
            "kind": "live_pipeline",
            "live": True,
            "input_kind": frame.get("source_kind") or "unknown",
            "received_at": now,
            "source_timestamp": frame.get("timestamp"),
        }
        with self._lock:
            self._latest_live_frame = frame
            self._latest_live_received_at = now
            self.active_mode = "live"
            self.match.update({"engine_running": True, "status": "playing", "mode": "live"})
        return {"ok": True, "received_at": now}

    def ingest_live_jpeg(self, body: bytes) -> dict:
        if not body.startswith(b"\xff\xd8") or not body.endswith(b"\xff\xd9"):
            raise ValueError("body is not a complete JPEG image")
        now = time.time()
        with self._lock:
            self._latest_live_jpeg = bytes(body)
            self._latest_live_jpeg_received_at = now
        return {"ok": True, "received_at": now, "bytes": len(body)}

    def live_jpeg(self) -> Optional[bytes]:
        with self._lock:
            image = self._latest_live_jpeg
            received_at = self._latest_live_jpeg_received_at
        if image is None or received_at is None or time.time() - received_at > self.LIVE_MAX_AGE_S:
            return None
        return image

    def _model_artifacts(self) -> List[dict]:
        rows = []
        for path in sorted((ROOT / "runs" / "detect").glob("*/weights/best.pt")):
            try:
                rows.append({
                    "name": path.parents[1].name,
                    "path": str(path.relative_to(ROOT)),
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
                    "modified_at": path.stat().st_mtime,
                })
            except OSError:
                continue
        return rows

    def system_info(self) -> dict:
        gpu_available = False
        gpu_name = None
        try:
            import torch
            gpu_available = bool(torch.cuda.is_available())
            if gpu_available:
                gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

        hub = self.fetch_hub()
        models = self._model_artifacts()
        return {
            "version": "2.1.0",
            "gpu": {"available": gpu_available, "name": gpu_name},
            "active_config": {
                "detector": models[-1]["name"] if models else "not configured",
                "tracker": "ByteTrack",
            },
            "hub_connected": hub is not None,
            "hub_url": self.hub_url,
            "hostinger_relay_url": self.hostinger_url or None,
            "active_mode": self.active_mode,
            "cameras_online": sum(c.get("status") == "online" for c in self.cameras.values()),
            "running_matches": ["live"] if self.match.get("engine_running") else [],
            "uptime_s": round(time.time() - self.started_at, 1),
            "storage": "process memory + JSON artifacts (single-node demo)",
            "models": models,
        }

    def _quality_from_stats(self, stats: dict, players: List[dict]) -> dict:
        if not stats:
            return {
                "metric_calibration": False,
                "unique_track_ids": 0,
                "speed_cap_fraction": 0.0,
                "status": "unavailable",
                "warnings": ["No live or recorded analysis is selected."],
            }
        warnings: List[str] = []
        metric = bool(stats.get("metric"))
        if not metric:
            warnings.append("No metric pitch calibration; geometry and load are not in metres.")

        unique_tracks = len(players)
        if unique_tracks > 30:
            warnings.append(
                f"{unique_tracks} unique track IDs indicate identity fragmentation; "
                "track IDs are not roster identities."
            )
        saturated = sum(
            abs(float(player.get("top_speed_mps") or 0) - 12.0) < 0.02
            for player in players
        )
        saturation_fraction = saturated / max(unique_tracks, 1)
        if saturation_fraction > 0.2:
            warnings.append(
                f"{saturation_fraction:.0%} of tracks reached the 12 m/s cap; "
                "speed and downstream load need calibration review."
            )
        return {
            "metric_calibration": metric,
            "unique_track_ids": unique_tracks,
            "speed_cap_fraction": round(saturation_fraction, 3),
            "status": "review" if warnings else "usable",
            "warnings": warnings,
        }

    def _recorded_snapshot(self) -> dict:
        now = time.time()
        stats = self.stats()
        fresh_wearables = self._fresh_wearables()
        indicators = stats.get("load_indicators") or {}
        raw_players = stats.get("players") or {}
        players: List[dict] = []

        if isinstance(raw_players, dict):
            for raw_id, values in raw_players.items():
                if not isinstance(values, dict):
                    continue
                pid = _numeric_player_id(raw_id)
                if pid is None:
                    continue
                wearable_row = fresh_wearables.get(pid)
                indicator = indicators.get(str(raw_id), {}) if isinstance(indicators, dict) else {}
                players.append({
                    "global_player_id": f"TRACK_{pid}",
                    "track_id": pid,
                    "jersey": None,
                    "name": f"Track {pid}",
                    "team_id": f"TEAM_{values.get('team', 0)}",
                    "team": int(values.get("team") or 0),
                    "x": None,
                    "y": None,
                    "speed_mps": None,
                    "top_speed_mps": _finite_number(values.get("top_speed_ms")),
                    "distance_m": _finite_number(values.get("distance_m"), 0.0),
                    "hsr_m": _finite_number(values.get("hsr_m"), 0.0),
                    "sprints": int(values.get("sprints") or 0),
                    "metabolic_power": _finite_number(values.get("metabolic_power_avg_wkg")),
                    "player_load": _finite_number(values.get("player_load")),
                    "wearable": wearable_row is not None,
                    "wearable_metrics": dict((wearable_row or {}).get("metrics") or {}) or None,
                    "load_indicator": {
                        "score": _finite_number(indicator.get("score")) if isinstance(indicator, dict) else None,
                        "severity": indicator.get("severity") if isinstance(indicator, dict) else None,
                        "factors": indicator.get("factors", {}) if isinstance(indicator, dict) else {},
                        "model": "heuristic_load_indicator",
                        "medical_prediction": False,
                    },
                })

        generated_at = None
        try:
            generated_at = self.stats_path.stat().st_mtime
        except OSError:
            pass
        quality = self._quality_from_stats(stats, players)
        source_file = None
        if self.stats_path.is_file():
            try:
                source_file = str(self.stats_path.relative_to(ROOT))
            except ValueError:
                source_file = str(self.stats_path)
        return {
            "timestamp": now,
            "match_id": self.match["match_id"],
            "match_name": self.match["name"],
            "match": copy.deepcopy(self.match),
            "mode": "recorded" if stats else "unavailable",
            "players": players,
            "ball": None,
            "possession_pct": stats.get("possession_pct") or {},
            "passes": stats.get("passes") or {},
            "passing_network": {},
            "timeline_tags": [],
            "shots_xg": stats.get("shots_xg") or {},
            "tactics": stats.get("tactics") or {},
            "substitution_watch": stats.get("substitution_watch") or [],
            "data_quality": quality,
            "wearables": {
                "connected_players": len(fresh_wearables),
                "players": {str(key): value for key, value in fresh_wearables.items()},
            },
            "provenance": {
                "kind": "recorded_analysis" if stats else "unavailable",
                "live": False,
                "source_file": source_file,
                "generated_at": generated_at,
                "age_s": round(now - generated_at, 1) if generated_at else None,
                "warnings": quality["warnings"],
            },
        }

    def live_frame_snapshot(self) -> dict:
        now = time.time()
        with self._lock:
            received_at = self._latest_live_received_at
            latest = copy.deepcopy(self._latest_live_frame)
        if latest is not None and received_at is not None and now - received_at <= self.LIVE_MAX_AGE_S:
            latest.setdefault("match", copy.deepcopy(self.match))
            latest.setdefault("data_quality", {"status": "unreported", "warnings": []})
            latest["provenance"]["age_s"] = round(now - received_at, 3)
            return latest
        if latest is not None:
            with self._lock:
                self.match.update({"engine_running": False, "status": "stale"})
        return self._recorded_snapshot()
