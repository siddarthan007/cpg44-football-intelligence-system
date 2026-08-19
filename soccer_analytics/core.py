"""Shared lightweight data types + bbox geometry (NumPy only, no heavy deps)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# class ids — must match the trained detector / data.yaml order
BALL, GOALKEEPER, PLAYER, REFEREE = 0, 1, 2, 3


@dataclass
class Detections:
    """A frame's detections. Arrays are parallel, length N."""
    xyxy: np.ndarray                      # (N,4) float [x1,y1,x2,y2]
    class_id: np.ndarray                  # (N,)  int
    confidence: np.ndarray                # (N,)  float
    tracker_id: Optional[np.ndarray] = None  # (N,) int, after tracking

    def __len__(self) -> int:
        return int(len(self.xyxy))

    @classmethod
    def empty(cls) -> "Detections":
        return cls(np.zeros((0, 4)), np.zeros((0,), int), np.zeros((0,), float),
                   np.zeros((0,), int))

    def __getitem__(self, m) -> "Detections":
        return Detections(
            self.xyxy[m], self.class_id[m], self.confidence[m],
            None if self.tracker_id is None else self.tracker_id[m],
        )

    def of_class(self, *cls_ids) -> "Detections":
        if len(self) == 0:
            return self
        mask = np.isin(self.class_id, np.asarray(cls_ids))
        return self[mask]


def foot_points(xyxy: np.ndarray) -> np.ndarray:
    """Bottom-centre of each box — the point that touches the pitch. (N,2)."""
    xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
    x = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    y = xyxy[:, 3]
    return np.stack([x, y], axis=1)


def centers(xyxy: np.ndarray) -> np.ndarray:
    xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
    x = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    y = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
    return np.stack([x, y], axis=1)


def bbox_widths(xyxy: np.ndarray) -> np.ndarray:
    xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
    return xyxy[:, 2] - xyxy[:, 0]
