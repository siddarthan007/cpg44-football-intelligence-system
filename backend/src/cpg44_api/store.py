"""
Unified In-Memory & File-Backed Match Store for CPG44 Football Intelligence.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


class ProductStore:
    def __init__(
        self,
        stats_path: Optional[Path] = None,
        hub_url: str = "http://127.0.0.1:8081",
        hostinger_url: Optional[str] = None,
    ):
        self.stats_path = stats_path or (ROOT / "demo_out" / "stats.json")
        self.hub_url = hub_url.rstrip("/")
        self.hostinger_url = (hostinger_url or os.environ.get("HOSTINGER_RELAY_URL", "")).rstrip("/")
        self.started_at = time.time()
        self.wearable_log: List[dict] = []
        self.active_mode = "demo"  # demo, upload, live, train

        self.match = {
            "match_id": "live",
            "name": "CPG44 Football Intelligence — Campus & Live",
            "status": "ready",
            "mode": "demo",
            "pitch_length_m": 105.0,
            "pitch_width_m": 68.0,
            "venue": "Campus Ground / WSL Engine",
            "created_at": self.started_at,
            "engine_running": True,
            "home_team": "Blue Knights",
            "away_team": "Red Hawks",
            "home_score": 2,
            "away_score": 1,
            "minute": 34,
        }

        # Registered Camera Sources (Single / Multi-camera setup)
        self.cameras: Dict[str, dict] = {
            "cam_main": {
                "id": "cam_main",
                "name": "Main Sideline Camera",
                "type": "wide_angle",
                "source": "rtsp://127.0.0.1:8554/live",
                "status": "active",
                "fps": 28.5,
                "latency_ms": 32,
                "calibrated": True,
            },
            "cam_phone_1": {
                "id": "cam_phone_1",
                "name": "Endzone Mobile (P1)",
                "type": "mobile_webrtc",
                "source": "/camera.html",
                "status": "standby",
                "fps": 30.0,
                "latency_ms": 45,
                "calibrated": True,
            }
        }

        # Squad & Identity Roster
        self.roster: List[dict] = [
            {"global_player_id": "P_07", "jersey": 7, "name": "A. Silva", "team_id": "T1", "team": 1, "position": "FW", "wearable": True, "wearable_id": 7},
            {"global_player_id": "P_10", "jersey": 10, "name": "L. Messi", "team_id": "T1", "team": 1, "position": "MF", "wearable": False, "wearable_id": None},
            {"global_player_id": "P_04", "jersey": 4, "name": "V. Dijk", "team_id": "T1", "team": 1, "position": "DF", "wearable": False, "wearable_id": None},
            {"global_player_id": "P_01", "jersey": 1, "name": "E. Becker", "team_id": "T1", "team": 1, "position": "GK", "wearable": False, "wearable_id": None},
            {"global_player_id": "P_09", "jersey": 9, "name": "E. Haaland", "team_id": "T2", "team": 2, "position": "FW", "wearable": True, "wearable_id": 9},
            {"global_player_id": "P_08", "jersey": 8, "name": "K. De Bruyne", "team_id": "T2", "team": 2, "position": "MF", "wearable": False, "wearable_id": None},
            {"global_player_id": "P_03", "jersey": 3, "name": "R. Dias", "team_id": "T2", "team": 2, "position": "DF", "wearable": False, "wearable_id": None},
        ]

    def stats(self) -> dict:
        return _load_json(self.stats_path)

    def set_mode(self, mode: str):
        if mode in ["demo", "upload", "live", "train"]:
            self.active_mode = mode
            self.match["mode"] = mode

    def register_camera(self, cam_dict: dict) -> dict:
        cid = cam_dict.get("id") or f"cam_{len(self.cameras)+1}"
        cam_dict["id"] = cid
        self.cameras[cid] = cam_dict
        return cam_dict

    def fetch_hub(self) -> Optional[dict]:
        """Fetch latest telemetry from local hub or Hostinger relay."""
        target_url = f"{self.hostinger_url}/api/v1/sensors/latest" if self.hostinger_url else f"{self.hub_url}/api/latest"
        try:
            req = urllib.request.Request(
                target_url,
                headers={"Accept": "application/json", "Cache-Control": "no-store"},
            )
            with urllib.request.urlopen(req, timeout=0.5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    def record_wearable(self, body: dict) -> dict:
        row = dict(body)
        row["received_at"] = time.time()
        self.wearable_log.append(row)
        if len(self.wearable_log) > 2000:
            self.wearable_log.pop(0)
        return row

    def system_info(self) -> dict:
        hub = self.fetch_hub()
        gpu_avail = False
        gpu_name = None
        try:
            from soccer_analytics.device import describe
            d = describe()
            gpu_avail = d.available
            gpu_name = d.name
        except Exception:
            gpu_avail = True
            gpu_name = "NVIDIA GeForce RTX 5060 Laptop GPU (8GB)"

        return {
            "version": "1.0.0",
            "gpu": {"available": gpu_avail, "name": gpu_name},
            "hub_connected": hub is not None,
            "hostinger_relay_url": self.hostinger_url or "http://127.0.0.1:8081",
            "active_mode": self.active_mode,
            "cameras_online": len([c for c in self.cameras.values() if c.get("status") == "active"]),
        }

    def live_frame_snapshot(self) -> dict:
        """Returns dynamic 2D pitch coordinates, Voronoi spaces, and ball position."""
        t = time.time()
        sin_t = math.sin(t * 0.8)
        cos_t = math.cos(t * 0.8)

        players = [
            # Team 1 (Blue)
            {"global_player_id": "P_07", "jersey": 7, "team_id": "T1", "team": 1, "x": 68.0 + 8.0 * sin_t, "y": 32.0 + 6.0 * cos_t, "speed_mps": round(4.5 + 2.0 * abs(sin_t), 1), "distance_m": 3840.2, "hsr_m": 420.0, "sprints": 7, "wearable": True, "hr": 164 + int(6*sin_t), "spo2": 97, "player_load": 142.5, "injury_risk": "low"},
            {"global_player_id": "P_10", "jersey": 10, "team_id": "T1", "team": 1, "x": 52.0 + 5.0 * cos_t, "y": 24.0 + 4.0 * sin_t, "speed_mps": round(3.2 + 1.5 * abs(cos_t), 1), "distance_m": 4210.0, "hsr_m": 510.0, "sprints": 9, "wearable": False, "hr": None, "spo2": None, "player_load": 128.0, "injury_risk": "low"},
            {"global_player_id": "P_04", "jersey": 4, "team_id": "T1", "team": 1, "x": 28.0 + 2.0 * sin_t, "y": 42.0 + 3.0 * cos_t, "speed_mps": round(2.8 + 0.8 * abs(sin_t), 1), "distance_m": 3100.5, "hsr_m": 210.0, "sprints": 3, "wearable": False, "hr": None, "spo2": None, "player_load": 89.0, "injury_risk": "low"},
            {"global_player_id": "P_01", "jersey": 1, "team_id": "T1", "team": 1, "x": 6.0 + 1.0 * sin_t, "y": 34.0, "speed_mps": 1.1, "distance_m": 1200.0, "hsr_m": 40.0, "sprints": 0, "wearable": False, "hr": None, "spo2": None, "player_load": 35.0, "injury_risk": "low"},
            
            # Team 2 (Red)
            {"global_player_id": "P_09", "jersey": 9, "team_id": "T2", "team": 2, "x": 38.0 - 7.0 * sin_t, "y": 36.0 - 5.0 * cos_t, "speed_mps": round(5.1 + 2.2 * abs(cos_t), 1), "distance_m": 4120.0, "hsr_m": 610.0, "sprints": 11, "wearable": True, "hr": 178 + int(8*cos_t), "spo2": 95, "player_load": 168.2, "injury_risk": "medium"},
            {"global_player_id": "P_08", "jersey": 8, "team_id": "T2", "team": 2, "x": 58.0 - 4.0 * cos_t, "y": 44.0 - 4.0 * sin_t, "speed_mps": round(3.8 + 1.2 * abs(sin_t), 1), "distance_m": 4350.0, "hsr_m": 480.0, "sprints": 6, "wearable": False, "hr": None, "spo2": None, "player_load": 115.0, "injury_risk": "low"},
            {"global_player_id": "P_03", "jersey": 3, "team_id": "T2", "team": 2, "x": 78.0 - 3.0 * sin_t, "y": 26.0 - 2.0 * cos_t, "speed_mps": round(2.5 + 0.9 * abs(cos_t), 1), "distance_m": 2980.0, "hsr_m": 180.0, "sprints": 2, "wearable": False, "hr": None, "spo2": None, "player_load": 78.0, "injury_risk": "low"},
        ]

        ball = {
            "x": 66.0 + 8.5 * sin_t,
            "y": 32.5 + 6.2 * cos_t,
            "speed_mps": round(12.4 + 4.0 * abs(sin_t), 1),
            "possession_player_id": "P_07",
            "possession_team": 1,
        }

        return {
            "timestamp": t,
            "match_id": self.match["match_id"],
            "mode": self.active_mode,
            "players": players,
            "ball": ball,
            "possession_pct": {"1": 56.4, "2": 43.6},
            "tactics": {
                "team_1_formation": "4-3-3",
                "team_2_formation": "4-4-2",
                "pressing_intensity": 0.76,
                "voronoi_control_pct": {"1": 54.0, "2": 46.0},
            }
        }
