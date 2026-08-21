"""
Match Event Tagging, Team Kit CIELAB Color Profiling, and Squad Re-ID Assignment Engine.
"""

from __future__ import annotations

import json
import logging
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
            "calibrated": False,
            "team_1": {"name": "Team 1", "primary_color_rgb": None, "cielab": None},
            "team_2": {"name": "Team 2", "primary_color_rgb": None, "cielab": None},
        }

    def _load_events(self) -> List[dict]:
        if self.events_file.is_file():
            try:
                rows = json.loads(self.events_file.read_text(encoding="utf-8"))
                # Older builds seeded fictional match events. Preserve the file,
                # but only expose observations carrying explicit provenance.
                return [row for row in rows if row.get("source") in {"manual", "pipeline", "imported"}]
            except Exception:
                pass
        return []

    def save_team_profiles(self, profiles: dict) -> dict:
        normalized = {"calibrated": True}
        for key in ("team_1", "team_2"):
            values = profiles.get(key)
            if not isinstance(values, dict):
                raise ValueError(f"{key} is required")
            rgb = values.get("primary_color_rgb")
            if not isinstance(rgb, list) or len(rgb) != 3 or any(
                not isinstance(value, (int, float)) or not 0 <= value <= 255 for value in rgb
            ):
                raise ValueError(f"{key}.primary_color_rgb must contain three values from 0 to 255")
            normalized[key] = {
                "name": str(values.get("name") or key.replace("_", " ").title()),
                "short_name": str(values.get("short_name") or "")[:8],
                "primary_color_rgb": [int(value) for value in rgb],
                "cielab": self._rgb_to_lab(rgb),
            }
        self.team_profiles = normalized
        self.teams_file.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
        return self.team_profiles

    def log_event(self, event_data: dict) -> dict:
        if event_data.get("timestamp_s") is None:
            raise ValueError("timestamp_s is required")
        event_type = str(event_data.get("type") or "").strip().lower()
        if event_type not in {"pass", "shot", "tackle", "interception", "goal", "sprint", "note"}:
            raise ValueError("unsupported event type")
        timestamp_s = float(event_data["timestamp_s"])
        if timestamp_s < 0:
            raise ValueError("timestamp_s must be non-negative")
        player_jersey = event_data.get("player_jersey")
        ev = {
            "id": f"ev_{int(time.time()*1000)}",
            "timestamp_s": timestamp_s,
            "time_str": f"{int(timestamp_s // 60):02d}:{int(timestamp_s % 60):02d}",
            "type": event_type,
            "team": int(event_data["team"]) if event_data.get("team") is not None else None,
            "player_jersey": int(player_jersey) if player_jersey not in (None, "") else None,
            "description": str(event_data.get("description") or event_type.title()),
            "pitch_x": float(event_data["pitch_x"]) if event_data.get("pitch_x") is not None else None,
            "pitch_y": float(event_data["pitch_y"]) if event_data.get("pitch_y") is not None else None,
            "source": "manual",
            "created_at": time.time(),
        }
        self.events.append(ev)
        self.events_file.write_text(json.dumps(self.events, indent=2), encoding="utf-8")
        return ev

    def classify_crop_team(self, rgb_color: List[int]) -> dict:
        """Classifies detected player color into Team 1, Team 2, or Referee using CIELAB Delta-E."""
        import numpy as np

        if not self.team_profiles.get("calibrated"):
            return {"team": 0, "label": "unassigned", "reason": "team colours are not calibrated"}
        lab = np.asarray(self._rgb_to_lab(rgb_color), dtype=float)
        reference = {
            team: np.asarray(self.team_profiles[f"team_{team}"]["cielab"], dtype=float)
            for team in (1, 2)
        }
        distances = {team: float(np.linalg.norm(lab - value)) for team, value in reference.items()}
        assigned_team = min(distances, key=distances.get)
        distance = distances[assigned_team]
        margin = abs(distances[1] - distances[2])
        if distance > 32.0 or margin < 5.0:
            return {
                "team": 0,
                "label": "unassigned",
                "distance": round(distance, 2),
                "margin": round(margin, 2),
                "reason": "colour evidence is out-of-profile or ambiguous",
            }
        return {
            "team": assigned_team,
            "team_name": self.team_profiles[f"team_{assigned_team}"]["name"],
            "distance": round(distance, 2),
            "margin": round(margin, 2),
            "method": "CIELAB_DeltaE76",
        }

    @staticmethod
    def _rgb_to_lab(rgb_color: List[int]) -> List[float]:
        import cv2
        import numpy as np

        rgb = np.asarray([[rgb_color]], dtype=np.uint8)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0, 0].astype(float)
        # OpenCV's 8-bit LAB encoding uses L* 0..255 and offsets a*/b* by 128.
        return [round(lab[0] * 100.0 / 255.0, 3), round(lab[1] - 128.0, 3), round(lab[2] - 128.0, 3)]
