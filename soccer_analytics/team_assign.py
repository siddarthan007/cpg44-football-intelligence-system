"""Jersey-colour team assignment (calibrated, self-correcting).

Pipeline:
  1. **Jersey colour per player**, a TIGHT central-torso patch is KMeans(2)-split
     into two colours; the JERSEY is the *larger* cluster (it fills a tight torso
     crop), so stray grass/skin at the edges is rejected. This beats both the
     top-half + corner-vote KMeans (mislabels green-shirt-on-grass) and a plain
     median (grass contamination flips white players green).
  2. **Two team centroids** fit over a BATCH of colours (warm-up + refit as more
     samples arrive), clustered on **a,b chroma** (shadow-invariant; L added back
     only for same-chroma kits like white-vs-black). When the operator has
     calibrated kit colours on the review page, the clusters are labelled so
     team 1 is always the calibrated team 1 kit.
  3. **Direct calibrated matching** when seeds exist: each torso colour is
     compared against BOTH calibrated references with a CIELAB distance and a
     margin test, plus an absolute rejection radius for referees, goalkeepers in
     a third kit, and occlusion blends. This is more accurate than clustering
     alone because it uses operator knowledge instead of hoping the two kits
     separate cleanly in an unsupervised pass.
  4. **Continuous, confidence-gated temporal voting**, every frame each track
     casts a vote ONLY if its colour is unambiguously nearer one reference; the
     team is the rolling-window majority. No freeze, so an early wrong guess
     self-corrects; ambiguous frames are ignored.
"""

from __future__ import annotations

from collections import defaultdict, deque
import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

from .core import PLAYER, Detections


class TeamAssigner:
    def __init__(self, vote_window: int = 25, min_crop: int = 8,
                 conf_ratio: float = 0.82, seed_max_dist: float = 42.0,
                 seed_min_margin: float = 6.0):
        self.team_colors: Optional[np.ndarray] = None
        self._kmeans_team = None
        self._votes: Dict[int, deque] = defaultdict(lambda: deque(maxlen=vote_window))
        self._final: Dict[int, int] = {}
        self.min_crop = min_crop
        self.conf_ratio = conf_ratio          # vote only if nearer/farther dist ratio < this
        self._use_L = False
        self._colorbuf: deque = deque(maxlen=4000)   # accumulated jersey colours for refit
        # Calibrated kit references {1: Lab, 2: Lab} from the review desk.
        # When present they anchor the two centroids AND drive direct matching,
        # so cluster order can never swap the live Team 1 / Team 2 tags.
        self._seeds: Optional[Dict[int, np.ndarray]] = None
        # Direct-match gates: reject colours that are far from both kits or sit
        # almost midway between them (referee, third-kit keeper, blend).
        self.seed_max_dist = float(seed_max_dist)
        self.seed_min_margin = float(seed_min_margin)

    def seed_with_lab(self, seeds: Dict[int, Sequence[float]]) -> None:
        """Anchor the team model to calibrated kit colours (CIELAB, OpenCV scale).

        Call before ``fit``. Cluster labels are then aligned so team 1 is the
        cluster nearest ``seeds[1]``, which keeps live team tags stable and
        correct even when the two kits are similar."""
        cleaned = {}
        for team, value in seeds.items():
            if team in (1, 2) and value is not None and len(value) == 3 and np.isfinite(value).all():
                cleaned[int(team)] = np.asarray(value, float)
        if len(cleaned) == 2:
            self._seeds = cleaned

    def needs_color(self, track_id: int) -> bool:
        return True                            # always re-read to continuous rechecking

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
        small = cv2.resize(patch, (min(patch.shape[1], 24), min(patch.shape[0], 24)),
                           interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)

        # Grass mask: Hue 35-85, Saturation > 35, Value > 30
        grass_mask = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] >= 30)

        # If non-grass pixels exist, isolate player kit; otherwise use entire torso
        non_grass_lab = lab[~grass_mask].astype(float)
        if len(non_grass_lab) >= max(8, int(0.65 * grass_mask.size)):
            pts = non_grass_lab
        else:
            pts = lab.reshape(-1, 3).astype(float)

        if len(pts) < 4:
            return None
        # This patch contains at most 576 pixels. A small deterministic Lloyd
        # loop avoids launching sklearn/OpenMP for every shirt on every update.
        chroma = pts[:, 1:3]
        centre = chroma.mean(axis=0)
        a = chroma[np.argmax(np.sum((chroma - centre) ** 2, axis=1))]
        b = chroma[np.argmax(np.sum((chroma - a) ** 2, axis=1))]
        centres = np.stack([a, b])
        labels = np.zeros(len(chroma), dtype=int)
        for _ in range(8):
            labels = np.argmin(np.sum((chroma[:, None] - centres) ** 2, axis=2), axis=1)
            updated = np.stack([chroma[labels == k].mean(axis=0)
                                if np.any(labels == k) else centres[k] for k in range(2)])
            if np.allclose(updated, centres, atol=0.1):
                break
            centres = updated
        n0 = int((labels == 0).sum())
        jersey = 0 if n0 >= len(labels) - n0 else 1     # dominant cluster = jersey
        return pts[labels == jersey].mean(axis=0)       # mean full-Lab of the jersey cluster

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
            self._use_L = True                 # near-identical chroma to add lightness
            km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(self._team_feat(A))
        km = self._align_to_seeds(km)
        self._kmeans_team = km
        self.team_colors = km.cluster_centers_
        self._colorbuf.extend(colors)

    def _align_to_seeds(self, km: KMeans) -> KMeans:
        """Relabel clusters so cluster 0 sits nearest the calibrated team 1 kit."""
        if self._seeds is None:
            return km
        c = km.cluster_centers_
        s1 = self._team_feat(self._seeds[1])[0]
        s2 = self._team_feat(self._seeds[2])[0]
        straight = float(np.linalg.norm(c[0] - s1) + np.linalg.norm(c[1] - s2))
        swapped = float(np.linalg.norm(c[0] - s2) + np.linalg.norm(c[1] - s1))
        if swapped < straight:
            km.cluster_centers_ = np.ascontiguousarray(c[::-1])
            km.labels_ = 1 - km.labels_
        return km

    def refit(self, min_samples: int = 150):
        """Re-fit the two team centroids on ALL accumulated colours (call
        periodically). Preserves centroid to team order so cached teams stay valid."""
        if self._kmeans_team is None or len(self._colorbuf) < min_samples:
            return
        prev = self._kmeans_team.cluster_centers_.copy()
        A = np.asarray(list(self._colorbuf), float)
        km = KMeans(n_clusters=2, n_init=6, random_state=0).fit(self._team_feat(A))
        # keep label order aligned to the previous model (KMeans labels are arbitrary)
        c = km.cluster_centers_
        if np.linalg.norm(c[0] - prev[0]) + np.linalg.norm(c[1] - prev[1]) > \
           np.linalg.norm(c[0] - prev[1]) + np.linalg.norm(c[1] - prev[0]):
            # c[::-1] has a negative stride to NOT C-contiguous, which breaks the
            # next KMeans.predict(); copy to a contiguous array.
            km.cluster_centers_ = np.ascontiguousarray(c[::-1])
            km.labels_ = 1 - km.labels_
        km = self._align_to_seeds(km)
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
        """Best-guess team for one colour sample (no gating). With calibrated
        seeds this is the nearest calibrated kit; otherwise the cluster model."""
        if self._seeds is not None:
            d1 = float(np.linalg.norm(np.asarray(color, float) - self._seeds[1]))
            d2 = float(np.linalg.norm(np.asarray(color, float) - self._seeds[2]))
            return 1 if d1 <= d2 else 2
        if self._kmeans_team is None:
            raise RuntimeError("TeamAssigner.fit() must be called first")
        return int(self._kmeans_team.predict(self._team_feat(color))[0]) + 1

    def _predict_conf(self, color) -> Tuple[int, bool]:
        """(team, confident) for one colour sample.

        With calibrated seeds: match against the two reference kits directly.
        A vote counts only when the colour is inside the acceptance radius of
        one kit AND clearly closer to it than to the other. Everything else
        (referee black, goalkeeper third kit, occlusion blends, shadow shifts)
        abstains instead of guessing.

        Without seeds: fall back to the confidence-ratio test against the two
        fitted centroids."""
        color = np.asarray(color, float)
        if self._seeds is not None:
            d1 = float(np.linalg.norm(color - self._seeds[1]))
            d2 = float(np.linalg.norm(color - self._seeds[2]))
            near, nd, fd = (1, d1, d2) if d1 <= d2 else (2, d2, d1)
            confident = (
                nd <= self.seed_max_dist              # close enough to a real kit
                and (fd - nd) >= self.seed_min_margin  # clearly nearer than the other
            )
            return near, confident
        feat = self._team_feat(color)[0]
        c = self._kmeans_team.cluster_centers_
        d0 = float(np.linalg.norm(feat - c[0]))
        d1 = float(np.linalg.norm(feat - c[1]))
        near, nd, fd = (0, d0, d1) if d0 <= d1 else (1, d1, d0)
        confident = fd > 1e-6 and (nd / fd) < self.conf_ratio
        return near + 1, confident

    def assign_from_color(self, track_id: int, color: Optional[np.ndarray]) -> int:
        """Continuous, confidence-gated vote to rolling-window majority team."""
        if color is not None and np.isfinite(color).all() and (self._kmeans_team is not None or self._seeds is not None):
            self._colorbuf.append(color)
            team, conf = self._predict_conf(color)
            if conf:
                self._votes[track_id].append(team)
        votes = self._votes[track_id]
        if votes:
            counts = np.bincount(list(votes), minlength=3)
            top = int(counts.argmax())
            # Require a clear majority inside the window before committing;
            # a 50/50 split stays unassigned rather than flickering.
            if counts[top] >= max(2, math.ceil(0.6 * counts.sum())):
                self._final[track_id] = top
            elif track_id in self._final and counts.sum() > 0:
                prev = self._final[track_id]
                if counts[prev] / counts.sum() < 0.4:
                    del self._final[track_id]   # evidence moved away from old tag
        return self._final.get(track_id, 0)

    def reset_votes(self) -> None:
        """Drop per-track votes after a seed alignment so stale team tags clear."""
        self._votes.clear()
        self._final.clear()

    def assign(self, track_id: int, frame, bbox) -> int:
        return self.assign_from_color(track_id, self.shirt_color(frame, bbox))
