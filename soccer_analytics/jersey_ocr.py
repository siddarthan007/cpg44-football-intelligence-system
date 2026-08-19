"""Jersey-number OCR → automatic wearable binding.

Per-frame OCR of small, blurred jersey numbers is unreliable, so this module
leans on **temporal majority voting**: read the number on a player's shirt every
few frames, accumulate votes per track id, and only trust a number once it wins a
clear majority over enough reads. That turns a noisy per-frame reader into a
confident track→number map over a clip.

For the wearable demo we only need to find which vision track wears each known
jersey (the 1–2 wearers). Given a roster ``{jersey_number: player_id}``, the
:class:`AutoBinder` resolves track→number→player and binds it into the
``SensorVideoSync`` automatically — no manual ``--roster track:player`` needed.

EasyOCR is imported lazily (GPU). If it is unavailable, the reader degrades to a
no-op and the pipeline falls back to manual binding.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Optional

import numpy as np


class JerseyReader:
    """Reads a 1–2 digit jersey number from a player crop (EasyOCR, GPU)."""

    def __init__(self, device: str = "", min_conf: float = 0.4, upscale: int = 4):
        self.min_conf = min_conf
        self.upscale = upscale
        self._reader = None
        self._ok = True
        gpu = not (device == "cpu")
        try:
            import easyocr
            self._reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        except Exception as e:   # missing dep / no model / no GPU
            print(f"[jersey] EasyOCR unavailable ({e}); auto-OCR disabled")
            self._ok = False

    @property
    def available(self) -> bool:
        return self._ok

    def read(self, frame, bbox) -> Optional[int]:
        if not self._ok:
            return None
        import cv2
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        h, w = y2 - y1, x2 - x1
        if h < 20 or w < 12:
            return None
        # torso / back-number region: upper-middle of the bbox
        cy1, cy2 = y1 + int(0.18 * h), y1 + int(0.58 * h)
        cx1, cx2 = x1 + int(0.20 * w), x1 + int(0.80 * w)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None
        crop = cv2.resize(crop, None, fx=self.upscale, fy=self.upscale,
                          interpolation=cv2.INTER_CUBIC)
        best, best_conf = None, self.min_conf
        for (_, text, conf) in self._reader.readtext(crop, allowlist="0123456789"):
            t = text.strip()
            if t.isdigit() and 1 <= len(t) <= 2 and conf >= best_conf:
                n = int(t)
                if 0 <= n <= 99:
                    best, best_conf = n, conf
        return best


class JerseyVoteTracker:
    """Accumulates per-track number votes and returns confident numbers."""

    def __init__(self, min_votes: int = 4, min_fraction: float = 0.5):
        self._votes: Dict[int, Counter] = defaultdict(Counter)
        self.min_votes = min_votes
        self.min_fraction = min_fraction

    def add(self, track_id: int, number: Optional[int]):
        if number is not None:
            self._votes[track_id][number] += 1

    def confident_number(self, track_id: int) -> Optional[int]:
        c = self._votes.get(track_id)
        if not c:
            return None
        ranked = c.most_common(2)
        num, votes = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0
        total = sum(c.values())
        # strict majority AND a clear margin over the runner-up, so a 2–2 tie
        # (decided only by insertion order) is never treated as confident.
        if total >= self.min_votes and votes / total > self.min_fraction and votes > runner:
            return num
        return None

    def resolve(self) -> Dict[int, int]:
        out = {}
        for tid in self._votes:
            n = self.confident_number(tid)
            if n is not None:
                out[tid] = n
        return out


class AutoBinder:
    """Ties it together: OCR jersey numbers, vote, and bind tracks to wearable
    players via a ``{jersey_number: player_id}`` roster."""

    def __init__(self, reader: JerseyReader, roster_numbers: Dict[int, int],
                 every: int = 5):
        self.reader = reader
        self.roster = {int(k): int(v) for k, v in roster_numbers.items()}
        self.votes = JerseyVoteTracker()
        self.every = every
        self.bound: Dict[int, int] = {}          # track_id -> player_id (auto-bound)
        self.bound_players: Dict[int, int] = {}  # player_id -> track_id (uniqueness)

    def step(self, frame_idx: int, frame, players, sync) -> Dict[int, int]:
        """``players`` = iterable of (track_id, bbox). OCRs on a cadence, votes,
        and binds any track whose confident number is in the roster. Returns the
        newly-bound {track_id: player_id} this call."""
        if not self.reader.available or frame_idx % self.every != 0:
            return {}
        newly = {}
        for (tid, bbox) in players:
            # leave already-bound tracks alone — including manual --roster binds,
            # which are authoritative (never clobber them with an OCR guess).
            if tid in self.bound or sync.player_of(tid) is not None:
                continue
            self.votes.add(tid, self.reader.read(frame, bbox))
            num = self.votes.confident_number(tid)
            if num is None or num not in self.roster:
                continue
            pid = self.roster[num]
            # one wearable per player: after a track-id switch, move the player's
            # binding to the fresh track instead of double-binding.
            old = self.bound_players.get(pid)
            if old is not None and old != tid:
                sync.unbind(old)
                self.bound.pop(old, None)
            sync.bind(tid, pid)
            self.bound[tid] = pid
            self.bound_players[pid] = tid
            newly[tid] = pid
            print(f"[jersey] track {tid} → #{num} → player {pid} (auto-bound)")
        return newly
