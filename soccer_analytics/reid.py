"""Player re-identification, persistent IDs across occlusion / out-of-view.

ByteTrack recovers short gaps via its Kalman + lost buffer, but when a player
leaves the frame (or is occluded) long enough, ByteTrack issues a NEW track id on
return. This layer maps ByteTrack ids to **stable ids** that survive those gaps, so
a player who steps out and comes back is the SAME id, essential for per-player
stats and for keeping a wearable-tagged player locked.

Matching a re-appearing track to a recently-lost one uses cheap, robust cues:
  * **jersey number** (if OCR'd), a hard lock, overrides everything;
  * **team**, must agree;
  * **appearance**, jersey colour (Lab a,b) distance;
  * **motion**, predicted position (last position + velocity × gap) within a radius.

Runs in image-pixel space, so it works with or without pitch calibration. Cost is
tiny: only NEW ByteTrack ids are matched, against a small gallery of lost tracks.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment


class ReIDManager:
    def __init__(self, reid_frames: int = 180, color_thr: float = 20.0,
                 base_radius: float = 90.0, radius_per_frame: float = 6.0,
                 max_radius: float = 500.0, ambiguity_margin: float = 0.15,
                 frame_rate: float = 25.0, max_speed_mps: float = 12.0):
        self.reid_frames = reid_frames        # how long a lost track stays matchable
        self.color_thr = color_thr            # max Lab a,b distance to match appearance
        self.base_radius = base_radius
        self.radius_per_frame = radius_per_frame
        self.max_radius = max_radius
        self.ambiguity_margin = ambiguity_margin
        self.frame_rate = frame_rate
        self.max_speed_mps = max_speed_mps
        self.gallery: Dict[int, dict] = {}    # stable_id -> entry
        self.t2s: Dict[int, int] = {}         # bytetrack_id -> stable_id
        self._next = 1
        self.reids = 0                        # count of successful re-identifications
        self.ambiguous = 0

    def update(self, frame_idx: int, tracks: List[dict]) -> Dict[int, int]:
        """``tracks`` = [{bt_id, center(x,y), team, color(3,)|None, jersey|None}, ...].
        Returns {bt_id: stable_id}. Call once per frame with the people tracks."""
        seen = {t["bt_id"] for t in tracks}
        # tracks whose bytetrack id vanished this frame to mark their stable id lost
        for bt in [b for b in self.t2s if b not in seen]:
            self.gallery[self.t2s[bt]]["active"] = False
            del self.t2s[bt]

        out: Dict[int, int] = {}
        newcomers = [t for t in tracks if t["bt_id"] not in self.t2s]
        lost = sorted(s for s, e in self.gallery.items()
                      if not e["active"] and 0 < frame_idx - e["last"] <= self.reid_frames)
        matches = {}
        if newcomers and lost:
            costs = np.asarray([[self._cost(t, self.gallery[s], frame_idx)
                                 for s in lost] for t in newcomers])
            # A dummy per row allows every track to abstain independently.
            unmatched = np.full((len(newcomers), len(newcomers)), 3.0)
            rows, cols = linear_sum_assignment(np.column_stack((costs, unmatched)))
            for row, col in zip(rows, cols):
                if col >= len(lost) or costs[row, col] >= 3.0:
                    continue
                score = costs[row, col]
                alternatives = np.r_[np.delete(costs[row], col), np.delete(costs[:, col], row)]
                if len(alternatives) and alternatives.min() - score < self.ambiguity_margin:
                    self.ambiguous += 1
                    continue
                matches[newcomers[row]["bt_id"]] = lost[col]
                self.reids += 1
        for t in tracks:
            bt = t["bt_id"]
            if bt in self.t2s:
                sid = self.t2s[bt]
            else:
                sid = matches.get(bt)
                if sid is None:
                    sid = self._next
                    self._next += 1
                self.t2s[bt] = sid
            self._touch(sid, t, frame_idx)
            out[bt] = sid

        # drop gallery entries lost for too long
        for sid in [s for s, e in self.gallery.items()
                    if not e["active"] and frame_idx - e["last"] > self.reid_frames]:
            del self.gallery[sid]
        return out

    def _cost(self, t: dict, e: dict, fi: int) -> float:
        # Shirt numbers repeat across teams. Neither OCR nor colour may bypass
        # a contradictory team/number or physically impossible displacement.
        reject = 1e6
        if e.get("team") and t.get("team") and e["team"] != t["team"]:
            return reject
        numbered = t.get("jersey") is not None and e.get("jersey") is not None
        if numbered and t["jersey"] != e["jersey"]:
            return reject
        gap = fi - e["last"]
        if t.get("pitch") is not None and e.get("pitch") is not None:
            radius = 0.75 + self.max_speed_mps * gap / self.frame_rate
            pd = float(np.linalg.norm(np.asarray(t["pitch"]) - e["pitch"]))
        else:
            radius = min(self.base_radius + self.radius_per_frame * gap, self.max_radius)
            pred = np.asarray(e["center"]) + np.asarray(e["vel"]) * min(gap, 15)
            pd = float(np.linalg.norm(pred - t["center"]))
        if not np.isfinite(pd) or pd > radius:
            return reject
        cd = None
        if t.get("color") is not None and e.get("color") is not None:
            cd = float(np.linalg.norm(np.asarray(t["color"])[1:] - np.asarray(e["color"])[1:]))
            if not np.isfinite(cd) or cd > self.color_thr:
                return reject
        if numbered and t.get("team") and t["team"] == e.get("team"):
            return 0.1 * pd / radius
        # Missing appearance is not evidence of identical appearance.
        if cd is None:
            return reject
        return pd / radius + cd / max(self.color_thr, 1e-6)

    def _touch(self, sid: int, t: dict, fi: int):
        e = self.gallery.get(sid)
        cx = t["center"]
        if e is None:
            self.gallery[sid] = {"center": cx, "vel": (0.0, 0.0), "team": t.get("team", 0),
                                 "color": t.get("color"), "jersey": t.get("jersey"),
                                 "pitch": t.get("pitch"),
                                 "last": fi, "active": True}
            return
        dt = max(1, fi - e["last"])
        observed_vel = (np.asarray(cx) - e["center"]) / dt
        e["vel"] = tuple(0.5 * np.asarray(e["vel"]) + 0.5 * observed_vel)
        e["center"] = cx
        e["pitch"] = t.get("pitch")
        if t.get("team"):
            e["team"] = t["team"]
        if t.get("color") is not None:
            e["color"] = (np.asarray(t["color"]) if e.get("color") is None
                          else 0.8 * np.asarray(e["color"]) + 0.2 * np.asarray(t["color"]))
        if t.get("jersey") is not None:
            e["jersey"] = t["jersey"]
        e["last"] = fi
        e["active"] = True
