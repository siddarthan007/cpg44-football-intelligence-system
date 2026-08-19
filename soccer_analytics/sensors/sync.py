"""Time-align the wearable stream to video frames and map vision tracks to
players.

Two responsibilities:
1. **Temporal sync** — keep a short ring buffer of each player's recent wearable
   samples; for a given video timestamp, return the nearest sample per player
   within a tolerance (handles the two streams running at different rates).
2. **Identity mapping** — bind a vision ``track_id`` to a real ``player_id``
   (jersey/roster). Until bound, wearable data can't be attributed; binding can
   be manual (roster) or, later, driven by jersey-number OCR.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Optional

from .schema import SensorSample


class SensorVideoSync:
    def __init__(self, tolerance_s: float = 0.5, buffer_per_player: int = 200):
        self.tolerance = tolerance_s
        self._buf: Dict[int, deque] = defaultdict(lambda: deque(maxlen=buffer_per_player))
        # vision track_id -> real player_id
        self._track2player: Dict[int, int] = {}

    # ---- identity mapping ---- #
    def bind(self, track_id: int, player_id: int):
        self._track2player[int(track_id)] = int(player_id)

    def bind_many(self, mapping: Dict[int, int]):
        for t, p in mapping.items():
            self.bind(t, p)

    def unbind(self, track_id: int):
        self._track2player.pop(int(track_id), None)

    def player_of(self, track_id: int) -> Optional[int]:
        return self._track2player.get(int(track_id))

    # ---- temporal sync ---- #
    def ingest(self, samples):
        for s in samples:
            self._buf[s.player_id].append(s)

    def sample_at(self, player_id: int, t: float) -> Optional[SensorSample]:
        """Nearest wearable sample for ``player_id`` within tolerance of time ``t``."""
        buf = self._buf.get(player_id)
        if not buf:
            return None
        best, bestdt = None, self.tolerance
        for s in reversed(buf):            # recent first
            dt = abs(s.t - t)
            if dt <= bestdt:
                best, bestdt = s, dt
            elif s.t < t - self.tolerance:
                break                      # older than tolerance; stop scanning
        return best

    def latest(self, player_id: int) -> Optional[SensorSample]:
        buf = self._buf.get(player_id)
        return buf[-1] if buf else None
