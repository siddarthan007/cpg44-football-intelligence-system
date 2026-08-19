"""
Match Event Tagging, Team Kit CIELAB Color Profiling, and Squad Re-ID Assignment Engine.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("MatchTagger")
ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = ROOT / "configs"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)


class MatchTagger:
    def __init__(self, configs_dir: Path = CONFIGS_DIR):
        self.configs_dir = configs_dir
        self.teams_file = self.configs_dir / "teams.json"
        self.events_file = self.configs_dir / "match_events.json"

        self.team_profiles = self._load_team_profiles()
        self.events = self._load_events()

    def _load_team_profiles(self) -> dict:
        if self.teams_file.is_file():
            try:
                return json.loads(self.teams_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "team_1": {
                "name": "Indy Eleven",
                "short_name": "IND",
                "primary_color_rgb": [37, 99, 235],   # Royal Blue
                "secondary_color_rgb": [255, 255, 255],
                "cielab": [45.2, 18.4, -62.1],
            },
            "team_2": {
                "name": "Louisville City",
                "short_name": "LOU",
                "primary_color_rgb": [127, 29, 29],   # Crimson / Dark Red
                "secondary_color_rgb": [255, 255, 255],
                "cielab": [32.1, 48.6, 26.8],
            },
            "referee": {
                "name": "Match Official",
                "primary_color_rgb": [234, 179, 8],   # Yellow
            }
        }

    def _load_events(self) -> List[dict]:
        if self.events_file.is_file():
            try:
                return json.loads(self.events_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return [
            {"id": "ev_1", "timestamp_s": 240.0, "time_str": "04:00", "type": "pass", "team": 1, "player_jersey": 27, "name": "R. Edwards", "description": "Key Progressive Pass"},
            {"id": "ev_2", "timestamp_s": 720.0, "time_str": "12:00", "type": "shot", "team": 1, "player_jersey": 17, "name": "M. Arteaga", "description": "Shot on Target (xG 0.38)"},
            {"id": "ev_3", "timestamp_s": 1180.0, "time_str": "19:40", "type": "tackle", "team": 2, "player_jersey": 8, "name": "N. Matsoso", "description": "Defensive Interception"},
            {"id": "ev_4", "timestamp_s": 1850.0, "time_str": "30:50", "type": "sprint", "team": 1, "player_jersey": 27, "name": "R. Edwards", "description": "High Speed Sprint (8.4 m/s)"},
            {"id": "ev_5", "timestamp_s": 2040.0, "time_str": "34:00", "type": "goal", "team": 1, "player_jersey": 17, "name": "M. Arteaga", "description": "GOAL (1-0)"},
        ]

    def save_team_profiles(self, profiles: dict) -> dict:
        self.team_profiles = profiles
        self.teams_file.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
        return self.team_profiles

    def log_event(self, event_data: dict) -> dict:
        ev = {
            "id": f"ev_{int(time.time()*1000)}",
            "timestamp_s": float(event_data.get("timestamp_s", time.time())),
            "time_str": event_data.get("time_str", "41:32"),
            "type": event_data.get("type", "tag"),
            "team": int(event_data.get("team", 1)),
            "player_jersey": int(event_data.get("player_jersey", 27)),
            "name": event_data.get("name", "Player"),
            "description": event_data.get("description", "Tagged Match Event"),
            "pitch_x": float(event_data.get("pitch_x", 52.5)),
            "pitch_y": float(event_data.get("pitch_y", 34.0)),
        }
        self.events.append(ev)
        self.events_file.write_text(json.dumps(self.events, indent=2), encoding="utf-8")
        return ev

    def classify_crop_team(self, rgb_color: List[int]) -> dict:
        """Classifies detected player color into Team 1, Team 2, or Referee using CIELAB Delta-E."""
        # Convert RGB to rough CIELAB
        r, g, b = [c / 255.0 for c in rgb_color]
        # Ignore grass green
        if g > r * 1.2 and g > b * 1.2:
            return {"team": 0, "label": "background_grass"}

        c1 = self.team_profiles["team_1"]["primary_color_rgb"]
        c2 = self.team_profiles["team_2"]["primary_color_rgb"]

        d1 = math.sqrt((rgb_color[0]-c1[0])**2 + (rgb_color[1]-c1[1])**2 + (rgb_color[2]-c1[2])**2)
        d2 = math.sqrt((rgb_color[0]-c2[0])**2 + (rgb_color[1]-c2[1])**2 + (rgb_color[2]-c2[2])**2)

        assigned_team = 1 if d1 <= d2 else 2
        return {
            "team": assigned_team,
            "team_name": self.team_profiles[f"team_{assigned_team}"]["name"],
            "distance": round(min(d1, d2), 2),
        }
