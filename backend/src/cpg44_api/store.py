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
        self.demo_video_path = ROOT / "demo" / "sample_match.mp4"
        self.hub_url = hub_url.rstrip("/")
        self.hostinger_url = (hostinger_url or os.environ.get("HOSTINGER_RELAY_URL", "")).rstrip("/")
        self.started_at = time.time()
        self.wearable_log: List[dict] = []
        self.active_mode = "demo"

        self.match = {
            "match_id": "live",
            "name": "Indy Eleven vs Louisville City",
            "period": "1st",
            "status": "playing",
            "mode": "demo",
            "pitch_length_m": 105.0,
            "pitch_width_m": 68.0,
            "venue": "Michael A. Carroll Stadium",
            "created_at": self.started_at,
            "engine_running": True,
            "home_team": "Indy Eleven",
            "away_team": "Louisville City",
            "home_score": 1,
            "away_score": 0,
            "timecode": "41:32",
            "total_duration": "90:00",
        }

        # Registered Camera Sources
        self.cameras: Dict[str, dict] = {
            "cam_main": {
                "id": "cam_main",
                "name": "Main Sideline Tactical Camera",
                "type": "wide_angle",
                "source": "rtsp://127.0.0.1:8554/live",
                "status": "active",
                "fps": 29.8,
                "latency_ms": 28,
                "calibrated": True,
            },
            "cam_phone_1": {
                "id": "cam_phone_1",
                "name": "Endzone Mobile (P1)",
                "type": "mobile_webrtc",
                "source": "/camera.html",
                "status": "standby",
                "fps": 30.0,
                "latency_ms": 42,
                "calibrated": True,
            }
        }

        # Squad & Identity Roster
        self.roster: List[dict] = [
            {"global_player_id": "P_27", "jersey": 27, "name": "R. Edwards", "team_id": "T1", "team": 1, "position": "CM", "wearable": True, "wearable_id": 27},
            {"global_player_id": "P_20", "jersey": 20, "name": "K. Koffie", "team_id": "T1", "team": 1, "position": "DM", "wearable": False, "wearable_id": None},
            {"global_player_id": "P_85", "jersey": 85, "name": "N. Hackshaw", "team_id": "T1", "team": 1, "position": "CB", "wearable": False, "wearable_id": None},
            {"global_player_id": "P_17", "jersey": 17, "name": "M. Arteaga", "team_id": "T1", "team": 1, "position": "RW", "wearable": True, "wearable_id": 17},
            {"global_player_id": "P_05", "jersey": 5, "name": "J. Cochran", "team_id": "T1", "team": 1, "position": "CB", "wearable": False, "wearable_id": None},
            {"global_player_id": "P_29", "jersey": 29, "name": "S. Guenzatti", "team_id": "T1", "team": 1, "position": "ST", "wearable": False, "wearable_id": None},
            {"global_player_id": "P_10", "jersey": 10, "name": "A. Quinn", "team_id": "T1", "team": 1, "position": "CAM", "wearable": False, "wearable_id": None},
            {"global_player_id": "P_09", "jersey": 9, "name": "C. Lancaster", "team_id": "T2", "team": 2, "position": "ST", "wearable": True, "wearable_id": 9},
            {"global_player_id": "P_08", "jersey": 8, "name": "N. Matsoso", "team_id": "T2", "team": 2, "position": "CM", "wearable": False, "wearable_id": None},
        ]

        # Passing Network Nodes and Connection Weights (Matching Reference Image)
        self.passing_network = {
            "team_1": {
                "nodes": [
                    {"id": 27, "name": "R. Edwards", "x": 52.5, "y": 34.0, "size": 36, "passes": 48},
                    {"id": 20, "name": "K. Koffie", "x": 42.0, "y": 24.0, "size": 28, "passes": 32},
                    {"id": 85, "name": "N. Hackshaw", "x": 32.0, "y": 36.0, "size": 26, "passes": 29},
                    {"id": 5, "name": "J. Cochran", "x": 62.0, "y": 22.0, "size": 27, "passes": 31},
                    {"id": 17, "name": "M. Arteaga", "x": 68.0, "y": 42.0, "size": 30, "passes": 38},
                    {"id": 29, "name": "S. Guenzatti", "x": 54.0, "y": 52.0, "size": 24, "passes": 22},
                    {"id": 19, "name": "T. Pasher", "x": 22.0, "y": 48.0, "size": 18, "passes": 16},
                    {"id": 13, "name": "E. Farrar", "x": 52.5, "y": 8.0, "size": 16, "passes": 14},
                    {"id": 10, "name": "A. Quinn", "x": 52.5, "y": 62.0, "size": 18, "passes": 19},
                    {"id": 2, "name": "K. Seaver", "x": 78.0, "y": 28.0, "size": 14, "passes": 11},
                ],
                "links": [
                    {"source": 27, "target": 20, "weight": 14},
                    {"source": 27, "target": 85, "weight": 18},
                    {"source": 27, "target": 5, "weight": 12},
                    {"source": 27, "target": 17, "weight": 22},
                    {"source": 27, "target": 29, "weight": 11},
                    {"source": 20, "target": 85, "weight": 9},
                    {"source": 5, "target": 17, "weight": 8},
                    {"source": 85, "target": 19, "weight": 7},
                    {"source": 5, "target": 2, "weight": 6},
                    {"source": 20, "target": 13, "weight": 5},
                    {"source": 29, "target": 10, "weight": 7},
                ]
            }
        }

        # Event Timeline Scrubber Tags (Shots, Passes, Tackles, Fatigue Spikes)
        self.timeline_tags = [
            {"id": "t1", "time": 240, "time_str": "04:00", "type": "pass", "label": "Build-up Pass", "team": 1, "player": 27},
            {"id": "t2", "time": 720, "time_str": "12:00", "type": "shot", "label": "Shot on Target", "team": 1, "player": 17},
            {"id": "t3", "time": 1180, "time_str": "19:40", "type": "tackle", "label": "Interception", "team": 2, "player": 8},
            {"id": "t4", "time": 1850, "time_str": "30:50", "type": "sprint", "label": "HSR Sprint (8.2 m/s)", "team": 1, "player": 27},
            {"id": "t5", "time": 2040, "time_str": "34:00", "type": "goal", "label": "Goal (1-0)", "team": 1, "player": 17},
            {"id": "t6", "time": 2492, "time_str": "41:32", "type": "possession", "label": "Current Play", "team": 1, "player": 27},
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
        return {
            "version": "2.0.0",
            "gpu": {"available": True, "name": "NVIDIA GeForce RTX 5060 Laptop GPU (8GB)"},
            "hub_connected": hub is not None,
            "hostinger_relay_url": self.hostinger_url or "http://127.0.0.1:8081",
            "active_mode": self.active_mode,
            "cameras_online": len([c for c in self.cameras.values() if c.get("status") == "active"]),
        }

    def live_frame_snapshot(self) -> dict:
        t = time.time()
        sin_t = math.sin(t * 0.8)
        cos_t = math.cos(t * 0.8)

        players = [
            # Team 1 (Indy Eleven / Blue)
            {"global_player_id": "P_27", "jersey": 27, "name": "R. Edwards", "team_id": "T1", "team": 1, "x": 52.5 + 4.0 * sin_t, "y": 34.0 + 3.0 * cos_t, "speed_mps": round(3.8 + 1.8 * abs(sin_t), 1), "distance_m": 4820.5, "hsr_m": 540.0, "sprints": 8, "wearable": True, "hr": 165 + int(5*sin_t), "spo2": 97, "player_load": 158.4, "metabolic_power": 12.8, "injury_risk": "low"},
            {"global_player_id": "P_20", "jersey": 20, "name": "K. Koffie", "team_id": "T1", "team": 1, "x": 42.0 + 3.0 * cos_t, "y": 24.0 + 2.0 * sin_t, "speed_mps": round(3.1 + 1.2 * abs(cos_t), 1), "distance_m": 4120.0, "hsr_m": 410.0, "sprints": 5, "wearable": False, "hr": None, "spo2": None, "player_load": 118.0, "metabolic_power": 10.4, "injury_risk": "low"},
            {"global_player_id": "P_85", "jersey": 85, "name": "N. Hackshaw", "team_id": "T1", "team": 1, "x": 32.0 + 2.0 * sin_t, "y": 36.0 + 2.0 * cos_t, "speed_mps": round(2.7 + 0.8 * abs(sin_t), 1), "distance_m": 3950.5, "hsr_m": 280.0, "sprints": 3, "wearable": False, "hr": None, "spo2": None, "player_load": 94.0, "metabolic_power": 8.9, "injury_risk": "low"},
            {"global_player_id": "P_17", "jersey": 17, "name": "M. Arteaga", "team_id": "T1", "team": 1, "x": 68.0 + 6.0 * sin_t, "y": 42.0 + 4.0 * cos_t, "speed_mps": round(4.9 + 2.4 * abs(sin_t), 1), "distance_m": 5340.0, "hsr_m": 720.0, "sprints": 12, "wearable": True, "hr": 174 + int(6*sin_t), "spo2": 96, "player_load": 176.2, "metabolic_power": 14.1, "injury_risk": "low"},
            {"global_player_id": "P_05", "jersey": 5, "name": "J. Cochran", "team_id": "T1", "team": 1, "x": 62.0 + 3.0 * cos_t, "y": 22.0 + 3.0 * sin_t, "speed_mps": round(3.4 + 1.1 * abs(cos_t), 1), "distance_m": 4410.0, "hsr_m": 390.0, "sprints": 6, "wearable": False, "hr": None, "spo2": None, "player_load": 124.0, "metabolic_power": 11.2, "injury_risk": "low"},
            
            # Team 2 (Louisville City / Red/White)
            {"global_player_id": "P_09", "jersey": 9, "name": "C. Lancaster", "team_id": "T2", "team": 2, "x": 48.0 - 5.0 * sin_t, "y": 38.0 - 3.0 * cos_t, "speed_mps": round(4.6 + 2.1 * abs(cos_t), 1), "distance_m": 4620.0, "hsr_m": 610.0, "sprints": 10, "wearable": True, "hr": 179 + int(7*cos_t), "spo2": 95, "player_load": 172.0, "metabolic_power": 13.6, "injury_risk": "moderate"},
            {"global_player_id": "P_08", "jersey": 8, "name": "N. Matsoso", "team_id": "T2", "team": 2, "x": 56.0 - 3.0 * cos_t, "y": 28.0 - 3.0 * sin_t, "speed_mps": round(3.5 + 1.0 * abs(sin_t), 1), "distance_m": 4280.0, "hsr_m": 440.0, "sprints": 6, "wearable": False, "hr": None, "spo2": None, "player_load": 119.0, "metabolic_power": 10.8, "injury_risk": "low"},
        ]

        ball = {
            "x": 54.0 + 4.2 * sin_t,
            "y": 34.8 + 3.1 * cos_t,
            "speed_mps": round(11.8 + 3.5 * abs(sin_t), 1),
            "possession_player_id": "P_27",
            "possession_player_jersey": 27,
            "possession_team": 1,
        }

        return {
            "timestamp": t,
            "match_id": self.match["match_id"],
            "match_name": self.match["name"],
            "timecode": "41:32",
            "mode": self.active_mode,
            "players": players,
            "ball": ball,
            "possession_pct": {"1": 58.2, "2": 41.8},
            "passing_network": self.passing_network,
            "timeline_tags": self.timeline_tags,
            "tactics": {
                "team_1_formation": "4-3-3",
                "team_2_formation": "4-4-2",
                "pressing_intensity": 0.74,
                "voronoi_control_pct": {"1": 56.5, "2": 43.5},
            }
        }
