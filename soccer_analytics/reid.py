"""Player re-identification — persistent IDs across occlusion / out-of-view.

ByteTrack recovers short gaps via its Kalman + lost buffer, but when a player
leaves the frame (or is occluded) long enough, ByteTrack issues a NEW track id on
return. This layer maps ByteTrack ids → **stable ids** that survive those gaps, so
a player who steps out and comes back is the SAME id — essential for per-player
stats and for keeping a wearable-tagged player locked.

Matching a re-appearing track to a recently-lost one uses cheap, robust cues:
  * **jersey number** (if OCR'd) — a hard lock, overrides everything;
  * **team** — must agree;
  * **appearance** — jersey colour (Lab a,b) distance;
  * **motion** — predicted position (last position + velocity × gap) within a radius.

Runs in image-pixel space, so it works with or without pitch calibration. Cost is
tiny: only NEW ByteTrack ids are matched, against a small gallery of lost tracks.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np


class ReIDManager:
    def __init__(self, reid_frames: int = 180, color_thr: float = 20.0,
                 base_radius: float = 90.0, radius_per_frame: float = 6.0,
                 max_radius: float = 500.0):
        self.reid_frames = reid_frames        # how long a lost track stays matchable
        self.color_thr = color_thr            # max Lab a,b distance to match appearance
        self.base_radius = base_radius
        self.radius_per_frame = radius_per_frame
        self.max_radius = max_radius
        self.gallery: Dict[int, dict] = {}    # stable_id -> entry
        self.t2s: Dict[int, int] = {}         # bytetrack_id -> stable_id
        self._next = 1
        self.reids = 0                        # count of successful re-identifications

    def update(self, frame_idx: int, tracks: List[dict]) -> Dict[int, int]:
        """``tracks`` = [{bt_id, center(x,y), team, color(3,)|None, jersey|None}, ...].
        Returns {bt_id: stable_id}. Call once per frame with the people tracks."""
        seen = {t["bt_id"] for t in tracks}
        # tracks whose bytetrack id vanished this frame → mark their stable id lost
        for bt in [b for b in self.t2s if b not in seen]:
            self.gallery[self.t2s[bt]]["active"] = False
            del self.t2s[bt]

        out: Dict[int, int] = {}
        for t in tracks:
            bt = t["bt_id"]
            if bt in self.t2s:
                sid = self.t2s[bt]
            else:
                sid = self._match(t, frame_idx)
                self.t2s[bt] = sid
            self._touch(sid, t, frame_idx)
            out[bt] = sid

        # drop gallery entries lost for too long
        for sid in [s for s, e in self.gallery.items()
                    if not e["active"] and frame_idx - e["last"] > self.reid_frames]:
            del self.gallery[sid]
        return out

    def _match(self, t: dict, fi: int) -> int:
        cx = t["center"]
        best, best_score = None, 1e18
        for sid, e in self.gallery.items():
            if e["active"]:
                continue                       # already tracked by another bt id
            gap = fi - e["last"]
            if gap > self.reid_frames:
                continue
            # jersey number is a hard lock
            if t.get("jersey") and e.get("jersey") and t["jersey"] == e["jersey"]:
                return sid
            if e.get("team") and t.get("team") and e["team"] != t["team"]:
                continue
            radius = min(self.base_radius + self.radius_per_frame * gap, self.max_radius)
            pred = (e["center"][0] + e["vel"][0] * gap, e["center"][1] + e["vel"][1] * gap)
            pd = math.hypot(pred[0] - cx[0], pred[1] - cx[1])
            if pd > radius:
                continue
            cd = 0.0
            if t.get("color") is not None and e.get("color") is not None:
                cd = float(np.hypot(t["color"][1] - e["color"][1],
                                    t["color"][2] - e["color"][2]))
                if cd > self.color_thr:
                    continue
            score = pd / radius + cd / max(self.color_thr, 1e-6)
            if score < best_score:
                best, best_score = sid, score
        if best is not None:
            self.reids += 1
            return best
        sid = self._next
        self._next += 1
        return sid

    def _touch(self, sid: int, t: dict, fi: int):
        e = self.gallery.get(sid)
        cx = t["center"]
        if e is None:
            self.gallery[sid] = {"center": cx, "vel": (0.0, 0.0), "team": t.get("team", 0),
                                 "color": t.get("color"), "jersey": t.get("jersey"),
                                 "last": fi, "active": True}
            return
        dt = max(1, fi - e["last"])
        e["vel"] = ((cx[0] - e["center"][0]) / dt, (cx[1] - e["center"][1]) / dt)
        e["center"] = cx
        if t.get("team"):
            e["team"] = t["team"]
        if t.get("color") is not None:
            e["color"] = t["color"]
        if t.get("jersey"):
            e["jersey"] = t["jersey"]
        e["last"] = fi
        e["active"] = True
