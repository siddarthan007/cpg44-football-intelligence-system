"""Recorded-match analyses (analysis.json) and coach review decisions.

An analysis directory holds ``analysis.json`` written by
``soccer_analytics.match_analysis`` plus an optional ``annotated.mp4``. Review
decisions are stored next to it in ``review.json`` so the original evidence is
never rewritten: the API merges decisions at read time.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
DECISIONS = {"confirm", "dismiss", "reset"}


class AnalysisLibrary:
    def __init__(self, roots: List[Path]):
        self.roots = [Path(r) for r in roots]
        self._lock = threading.Lock()

    def _dir(self, analysis_id: str) -> Optional[Path]:
        if not SAFE_ID.match(analysis_id):
            return None
        for root in self.roots:
            d = root / analysis_id
            if (d / "analysis.json").is_file():
                return d
        return None

    def list(self) -> List[dict]:
        rows = []
        for root in self.roots:
            if not root.is_dir():
                continue
            for d in root.iterdir():
                f = d / "analysis.json"
                if not f.is_file():
                    continue
                try:
                    a = json.loads(f.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                q = a.get("quality", {})
                rows.append({
                    "id": d.name, "name": a.get("match", {}).get("name", d.name),
                    "duration_s": a.get("match", {}).get("duration_s"),
                    "generated_at": a.get("provenance", {}).get("generated_at"),
                    "metric": q.get("metric"), "ball_seen": q.get("ball_seen"),
                    "checks_pending": q.get("checks_pending"),
                    "has_video": (d / "annotated.mp4").is_file(),
                })
        rows.sort(key=lambda r: r.get("generated_at") or 0, reverse=True)
        return rows

    def get(self, analysis_id: str) -> Optional[dict]:
        d = self._dir(analysis_id)
        if d is None:
            return None
        a = json.loads((d / "analysis.json").read_text(encoding="utf-8"))
        decisions = self._decisions(d)
        pending = 0
        for e in a.get("events", []):
            dec = decisions.get(e.get("id"))
            if dec:
                e["decision"] = dec
                if dec["decision"] in ("confirm", "dismiss"):
                    e["status"] = "confirmed" if dec["decision"] == "confirm" else "dismissed"
            if e.get("status") == "check":
                pending += 1
        a.setdefault("quality", {})["checks_pending"] = pending
        tags = self._vest_tags(d)
        if tags:
            # a coach's tag wins over the automatic vest-to-track match
            for pl in a.get("players", []):
                if pl.get("vest") in tags.values() and str(pl["id"]) not in tags:
                    pl.pop("vest", None)
                if str(pl["id"]) in tags:
                    pl["vest"], pl["vest_source"] = tags[str(pl["id"])], "coach"
        a["id"] = analysis_id
        a["video_url"] = f"/api/v1/analyses/{analysis_id}/video" if (d / "annotated.mp4").is_file() else None
        return a

    def video_path(self, analysis_id: str) -> Optional[Path]:
        d = self._dir(analysis_id)
        p = d / "annotated.mp4" if d else None
        return p if p and p.is_file() else None

    def thumbnail(self, analysis_id: str) -> Optional[Path]:
        """A still from the annotated video (cached), for match cards."""
        d = self._dir(analysis_id)
        if d is None:
            return None
        thumb = d / "thumb.jpg"
        if thumb.is_file():
            return thumb
        video = d / "annotated.mp4"
        if not video.is_file():
            return None
        import cv2
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((cap.get(cv2.CAP_PROP_FPS) or 25) * 3))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        h, w = frame.shape[:2]
        cv2.imwrite(str(thumb), cv2.resize(frame, (480, int(480 * h / w))), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return thumb

    def clip(self, analysis_id: str, t: float, before: float = 3.0, after: float = 5.0) -> Optional[Path]:
        """A short H.264 clip around a moment, cut from the annotated video."""
        import shutil
        import subprocess
        d = self._dir(analysis_id)
        video = d / "annotated.mp4" if d else None
        if video is None or not video.is_file() or not shutil.which("ffmpeg"):
            return None
        start = max(0.0, float(t) - before)
        out = d / "clips" / f"clip_{int(start * 10)}_{int((before + after) * 10)}.mp4"
        out.parent.mkdir(exist_ok=True)
        if not out.is_file():
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", str(video),
                            "-t", f"{before + after:.2f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", "-an", str(out)], check=False, timeout=120)
        return out if out.is_file() else None

    def _decisions(self, d: Path) -> Dict[str, dict]:
        f = d / "review.json"
        try:
            return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
        except (OSError, ValueError):
            return {}

    def _vest_tags(self, d: Path) -> Dict[str, int]:
        f = d / "vest_tags.json"
        try:
            return {str(k): int(v) for k, v in json.loads(f.read_text(encoding="utf-8")).items()} if f.is_file() else {}
        except (OSError, ValueError, TypeError):
            return {}

    def tag_vest(self, analysis_id: str, player: int, vest: Optional[int]) -> Dict[str, int]:
        """Mark which tracked player wore which vest (or clear it)."""
        d = self._dir(analysis_id)
        if d is None:
            raise KeyError("analysis not found")
        a = json.loads((d / "analysis.json").read_text(encoding="utf-8"))
        if not any(p.get("id") == player for p in a.get("players", [])):
            raise KeyError("player not found")
        if vest is not None and not 1 <= int(vest) <= 9999:
            raise ValueError("vest number must be between 1 and 9999")
        with self._lock:
            tags = {k: v for k, v in self._vest_tags(d).items() if v != vest and k != str(player)}
            if vest is not None:
                tags[str(player)] = int(vest)
            tmp = d / "vest_tags.json.tmp"
            tmp.write_text(json.dumps(tags, indent=2), encoding="utf-8")
            tmp.replace(d / "vest_tags.json")
        return tags

    def decide(self, analysis_id: str, event_id: str, decision: str, note: str = "") -> dict:
        d = self._dir(analysis_id)
        if d is None:
            raise KeyError("analysis not found")
        if decision not in DECISIONS:
            raise ValueError("decision must be confirm, dismiss or reset")
        a = json.loads((d / "analysis.json").read_text(encoding="utf-8"))
        if not any(e.get("id") == event_id for e in a.get("events", [])):
            raise KeyError("event not found")
        with self._lock:
            decisions = self._decisions(d)
            if decision == "reset":
                decisions.pop(event_id, None)
            else:
                decisions[event_id] = {"decision": decision, "note": note[:280], "at": time.time()}
            tmp = d / "review.json.tmp"
            tmp.write_text(json.dumps(decisions, indent=2), encoding="utf-8")
            tmp.replace(d / "review.json")
        return decisions.get(event_id, {"decision": "reset"})
