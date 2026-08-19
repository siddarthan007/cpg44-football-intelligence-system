"""Jersey-colour team assignment (robust, self-correcting).

Pipeline:
  1. **Jersey colour per player** — a TIGHT central-torso patch is KMeans(2)-split
     into two colours; the JERSEY is the *larger* cluster (it fills a tight torso
     crop), so stray grass/skin at the edges is rejected. This beats both the
     top-half + corner-vote KMeans (mislabels green-shirt-on-grass) and a plain
     median (grass contamination flips white players green).
  2. **Two team centroids** fit over a BATCH of colours (warm-up + refit as more
     samples arrive), clustered on **a,b chroma** (shadow-invariant; L added back
     only for same-chroma kits like white-vs-black).
  3. **Continuous, confidence-gated temporal voting** — every frame each track
     casts a vote ONLY if its colour is unambiguously nearer one centroid; the
     team is the rolling-window majority. No freeze, so an early wrong guess
     self-corrects; ambiguous frames are ignored.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

from .core import PLAYER, Detections


class TeamAssigner:
    def __init__(self, vote_window: int = 25, min_crop: int = 8, conf_ratio: float = 0.82):
        self.team_colors: Optional[np.ndarray] = None
        self._kmeans_team = None
        self._votes: Dict[int, deque] = defaultdict(lambda: deque(maxlen=vote_window))
        self._final: Dict[int, int] = {}
        self.min_crop = min_crop
        self.conf_ratio = conf_ratio          # vote only if nearer/farther dist ratio < this
        self._use_L = False
        self._colorbuf: deque = deque(maxlen=4000)   # accumulated jersey colours for refit

    def needs_color(self, track_id: int) -> bool:
        return True                            # always re-read → continuous rechecking

    # ---- jersey colour of a single player ------------------------------- #
    def shirt_color(self, frame, bbox) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        w, h = x2 - x1, y2 - y1
        if w < self.min_crop or h < self.min_crop:
            return None
        # central upper-torso patch (avoids head, shorts, arms, grass at edges)
        px1, px2 = x1 + int(0.28 * w), x1 + int(0.72 * w)
        py1, py2 = y1 + int(0.12 * h), y1 + int(0.50 * h)
        patch = frame[py1:py2, px1:px2]
        if patch.shape[0] < 3 or patch.shape[1] < 3:
            return None
        small = cv2.resize(patch, (min(patch.shape[1], 16), min(patch.shape[0], 16)),
                           interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
        if len(lab) < 4:
            return None
        km = KMeans(n_clusters=2, n_init=1, random_state=0).fit(lab[:, 1:3])  # cluster on chroma
        labels = km.labels_
        n0 = int((labels == 0).sum())
        jersey = 0 if n0 >= len(labels) - n0 else 1     # larger cluster = jersey
        return lab[labels == jersey].mean(axis=0)       # mean full-Lab of the jersey cluster

    # ---- team model ----------------------------------------------------- #
    def _team_feat(self, colors) -> np.ndarray:
        arr = np.asarray(colors, float).reshape(-1, 3)
        if self._use_L:
            feat = np.column_stack([arr[:, 0] * 0.5, arr[:, 1], arr[:, 2]])
        else:
            feat = arr[:, 1:3]                 # a,b chroma (shadow-invariant)
        # sklearn's predict() requires a C-contiguous array; a column slice is not
        return np.ascontiguousarray(feat, dtype=np.float64)

    def fit(self, colors: List[np.ndarray]):
        colors = [c for c in colors if c is not None]
        if len(colors) < 2:
            raise ValueError("need ≥2 shirt-colour samples to fit teams")
        A = np.asarray(colors, float)
        self._use_L = False
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(self._team_feat(A))
        if np.linalg.norm(km.cluster_centers_[0] - km.cluster_centers_[1]) < 16:
            self._use_L = True                 # near-identical chroma → add lightness
            km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(self._team_feat(A))
        self._kmeans_team = km
        self.team_colors = km.cluster_centers_
        self._colorbuf.extend(colors)

    def refit(self, min_samples: int = 150):
        """Re-fit the two team centroids on ALL accumulated colours (call
        periodically). Preserves centroid→team order so cached teams stay valid."""
        if self._kmeans_team is None or len(self._colorbuf) < min_samples:
            return
        prev = self._kmeans_team.cluster_centers_.copy()
        A = np.asarray(list(self._colorbuf), float)
        km = KMeans(n_clusters=2, n_init=6, random_state=0).fit(self._team_feat(A))
        # keep label order aligned to the previous model (KMeans labels are arbitrary)
        c = km.cluster_centers_
        if np.linalg.norm(c[0] - prev[0]) + np.linalg.norm(c[1] - prev[1]) > \
           np.linalg.norm(c[0] - prev[1]) + np.linalg.norm(c[1] - prev[0]):
            # c[::-1] has a negative stride → NOT C-contiguous, which breaks the
            # next KMeans.predict(); copy to a contiguous array.
            km.cluster_centers_ = np.ascontiguousarray(c[::-1])
            km.labels_ = 1 - km.labels_
        self._kmeans_team = km
        self.team_colors = km.cluster_centers_

    def collect(self, frame, players: Detections) -> List[np.ndarray]:
        out = []
        for bbox in players.of_class(PLAYER).xyxy:
            c = self.shirt_color(frame, bbox)
            if c is not None:
                out.append(c)
        return out

    # ---- prediction + continuous voting --------------------------------- #
    def predict_team(self, color: np.ndarray) -> int:
        if self._kmeans_team is None:
            raise RuntimeError("TeamAssigner.fit() must be called first")
        return int(self._kmeans_team.predict(self._team_feat(color))[0]) + 1

    def _predict_conf(self, color) -> Tuple[int, bool]:
        """(team, confident): confident only when the colour is clearly nearer one
        centroid — filters ambiguous/noisy frames out of the vote."""
        feat = self._team_feat(color)[0]
        c = self._kmeans_team.cluster_centers_
        d0 = float(np.linalg.norm(feat - c[0]))
        d1 = float(np.linalg.norm(feat - c[1]))
        near, nd, fd = (0, d0, d1) if d0 <= d1 else (1, d1, d0)
        confident = fd > 1e-6 and (nd / fd) < self.conf_ratio
        return near + 1, confident

    def assign_from_color(self, track_id: int, color: Optional[np.ndarray]) -> int:
        """Continuous, confidence-gated vote → rolling-window majority team."""
        if color is not None and self._kmeans_team is not None:
            self._colorbuf.append(color)
            team, conf = self._predict_conf(color)
            if conf:
                self._votes[track_id].append(team)
        votes = self._votes[track_id]
        if votes:
            self._final[track_id] = int(np.bincount(list(votes)).argmax())
        return self._final.get(track_id, 0)

    def assign(self, track_id: int, frame, bbox) -> int:
        return self.assign_from_color(track_id, self.shirt_color(frame, bbox))
