"""Pitch rendering, occupancy heatmaps, and a top-down radar/minimap.

- ``draw_pitch`` / ``generate_heatmap`` use matplotlib + scipy (lazy) for the
  static analytics figures.
- ``radar_frame`` renders a fast top-down minimap with NumPy/OpenCV (lazy cv2)
  for compositing onto each video frame.

Heatmaps consume **pitch-metre** positions, so they are a true bird's-eye view
(not the camera-distorted pixel heatmaps some reference repos produce).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")                     # headless-safe; figures are saved to disk
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from .config import PitchConfig


# --------------------------------------------------------------------------- #
# matplotlib pitch + heatmap
# --------------------------------------------------------------------------- #
def draw_pitch(ax, pc: PitchConfig, line_color: str = "white", bg: str = "#1a1a1a"):
    """Draw a to-scale football pitch onto a matplotlib Axes (metres)."""
    L, W = pc.length, pc.width
    ax.set_facecolor(bg)
    ax.plot([0, 0, L, L, 0], [0, W, W, 0, 0], color=line_color, lw=1.5)
    ax.plot([L / 2, L / 2], [0, W], color=line_color, lw=1.5)              # halfway
    ax.add_patch(patches.Circle((L / 2, W / 2), pc.centre_circle_radius,
                                fill=False, color=line_color, lw=1.5))
    cy = W / 2
    for x0, sign in [(0, 1), (L, -1)]:                                     # penalty + goal boxes
        pb, pbw = pc.penalty_box_length, pc.penalty_box_width
        gb, gbw = pc.goal_box_length, pc.goal_box_width
        ax.plot([x0, x0 + sign * pb, x0 + sign * pb, x0],
                [cy - pbw / 2, cy - pbw / 2, cy + pbw / 2, cy + pbw / 2], color=line_color, lw=1.2)
        ax.plot([x0, x0 + sign * gb, x0 + sign * gb, x0],
                [cy - gbw / 2, cy - gbw / 2, cy + gbw / 2, cy + gbw / 2], color=line_color, lw=1.2)
    ax.set_xlim(-3, L + 3)
    ax.set_ylim(-3, W + 3)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def generate_heatmap(positions: List[Tuple[float, float]], pc: PitchConfig,
                     out_path: str, title: str = "", bins: int = 50, sigma: float = 1.5,
                     cmap: str = "hot"):
    """Save a smoothed occupancy heatmap over the pitch. ``positions`` in metres."""
    pts = np.asarray([(x, y) for (x, y) in positions
                      if x is not None and not np.isnan(x)], dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    draw_pitch(ax, pc)
    if len(pts):
        heat, xe, ye = np.histogram2d(
            pts[:, 0], pts[:, 1], bins=bins,
            range=[[0, pc.length], [0, pc.width]])
        heat = gaussian_filter(heat, sigma=sigma)
        ax.imshow(heat.T, origin="lower", extent=[0, pc.length, 0, pc.width],
                  cmap=cmap, alpha=0.6, aspect="equal")
    if title:
        ax.set_title(title, color="white")
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# fast top-down radar for per-frame overlay
# --------------------------------------------------------------------------- #
def radar_frame(players_by_team: Dict[int, List[Tuple[float, float]]],
                ball: Optional[Tuple[float, float]], pc: PitchConfig,
                width_px: int = 400, team_bgr=((0, 140, 255), (255, 90, 0)),
                control_grid=None, pred_paths=None):
    """Render a top-down minimap (BGR uint8) with player dots + ball. If
    ``control_grid`` (Gy×Gx of team ids) is given, the pitch is lightly shaded by
    which team controls each Voronoi cell."""
    scale = width_px / pc.length
    h = int(pc.width * scale)
    img = np.full((h, width_px, 3), (40, 100, 40), np.uint8)   # green

    def to_px(x, y):
        return int(x * scale), int(y * scale)

    if control_grid is not None:
        shade = np.zeros((h, width_px, 3), np.uint8)
        gy, gx = control_grid.shape
        big = cv2.resize(control_grid.astype(np.uint8), (width_px, h),
                         interpolation=cv2.INTER_NEAREST)
        shade[big == 1] = team_bgr[0]
        shade[big == 2] = team_bgr[1]
        img = cv2.addWeighted(img, 0.7, shade, 0.3, 0)

    white = (255, 255, 255)
    cv2.rectangle(img, (0, 0), (width_px - 1, h - 1), white, 1)
    cv2.line(img, to_px(pc.length / 2, 0), to_px(pc.length / 2, pc.width), white, 1)
    cv2.circle(img, to_px(pc.length / 2, pc.width / 2),
               int(pc.centre_circle_radius * scale), white, 1)

    for team, pts in players_by_team.items():
        col = team_bgr[(team - 1) % 2] if team in (1, 2) else (200, 200, 200)
        for (x, y) in pts:
            if x is None or np.isnan(x):
                continue
            cv2.circle(img, to_px(x, y), 5, col, -1)
            cv2.circle(img, to_px(x, y), 5, (0, 0, 0), 1)
    if pred_paths is not None:                       # LSTM-predicted paths on pitch
        for path in pred_paths:
            pts = [to_px(x, y) for (x, y) in path
                   if x is not None and not np.isnan(x)]
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i + 1], (255, 255, 0), 1, cv2.LINE_AA)

    if ball is not None and not (ball[0] is None or np.isnan(ball[0])):
        cv2.circle(img, to_px(*ball), 4, (255, 255, 255), -1)
        cv2.circle(img, to_px(*ball), 4, (0, 0, 0), 1)
    return img


class LiveHeatmap:
    """Fast, incrementally-accumulating occupancy heatmap for the live dashboard
    (the matplotlib ``generate_heatmap`` is for final figures, too slow per-frame)."""

    def __init__(self, pc: PitchConfig, width_px: int = 480, bins: int = 60, decay: float = 1.0):
        self.pc = pc
        self.width_px = width_px
        self.bins = bins
        self.decay = decay
        self.grid = np.zeros((bins, bins), np.float32)

    def add(self, positions):
        for (x, y) in positions:
            if x is None or np.isnan(x):
                continue
            gx = int(np.clip(x / self.pc.length * (self.bins - 1), 0, self.bins - 1))
            gy = int(np.clip(y / self.pc.width * (self.bins - 1), 0, self.bins - 1))
            self.grid[gy, gx] += 1.0
        if self.decay < 1.0:
            self.grid *= self.decay

    def render(self):
        g = np.log1p(self.grid)
        g = (g / g.max() * 255).astype(np.uint8) if g.max() > 0 else g.astype(np.uint8)
        h = int(self.pc.width / self.pc.length * self.width_px)
        g = cv2.resize(g, (self.width_px, h), interpolation=cv2.INTER_LINEAR)
        g = cv2.GaussianBlur(g, (0, 0), 3)
        color = cv2.applyColorMap(g, cv2.COLORMAP_JET)
        pitch = np.full_like(color, (40, 100, 40))
        out = cv2.addWeighted(pitch, 0.35, color, 0.65, 0)
        cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), (255, 255, 255), 1)
        cv2.line(out, (out.shape[1] // 2, 0), (out.shape[1] // 2, out.shape[0]), (255, 255, 255), 1)
        return out
