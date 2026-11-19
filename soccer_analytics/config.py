"""Pitch geometry + pipeline configuration.

All pitch coordinates are in **metres**, origin at the top-left corner of the
pitch, x to along the length (goal to goal), y to along the width. Default is a
FIFA-standard 105 × 68 m pitch. These landmark points double as:
  * the destination points for a homography fit from detected field keypoints, and
  * the template drawn in the 2-D radar / heatmap view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class PitchConfig:
    length: float = 105.0          # x, metres (touchline to touchline the long way)
    width: float = 68.0            # y, metres
    penalty_box_length: float = 16.5
    penalty_box_width: float = 40.32
    goal_box_length: float = 5.5
    goal_box_width: float = 18.32
    centre_circle_radius: float = 9.15
    penalty_spot_dist: float = 11.0
    goal_width: float = 7.32
    kind: str = "football"         # "football" (IFAB) or "futsal" (FIFA futsal laws)

    @classmethod
    def for_kind(cls, kind: str = "football", length: float = 105.0, width: float = 68.0) -> "PitchConfig":
        """Markings for a pitch type. Futsal (FIFA Futsal Laws, Law 1): 3 m goal,
        6 m penalty mark, 3 m centre circle, and a penalty area made of two 6 m
        quarter circles joined by a 3.16 m straight line in front of the goal."""
        if kind == "futsal":
            return cls(length=length, width=width, penalty_box_length=6.0, penalty_box_width=3.16,
                       goal_box_length=0.0, goal_box_width=0.0, centre_circle_radius=3.0,
                       penalty_spot_dist=6.0, goal_width=3.0, kind="futsal")
        return cls(length=length, width=width)

    def named_points(self) -> Dict[str, Tuple[float, float]]:
        """Well-known pitch landmarks (metres). Keys are stable identifiers that a
        keypoint model or a manual-click tool can map image points onto."""
        L, W = self.length, self.width
        pb, pbw = self.penalty_box_length, self.penalty_box_width
        gb, gbw = self.goal_box_length, self.goal_box_width
        cy = W / 2.0
        pts = {
            # corners
            "tl_corner": (0.0, 0.0),
            "tr_corner": (L, 0.0),
            "bl_corner": (0.0, W),
            "br_corner": (L, W),
            # halfway line
            "half_top": (L / 2, 0.0),
            "half_bottom": (L / 2, W),
            "centre": (L / 2, cy),
            # centre-circle cardinal points (radius = 9.15 m), visible on kickoff
            # views where box/corner landmarks are not, so they enable calibration
            "circle_left": (L / 2 - self.centre_circle_radius, cy),
            "circle_right": (L / 2 + self.centre_circle_radius, cy),
            "circle_top": (L / 2, cy - self.centre_circle_radius),
            "circle_bottom": (L / 2, cy + self.centre_circle_radius),
            # left penalty box
            "l_pen_top": (pb, cy - pbw / 2),
            "l_pen_bottom": (pb, cy + pbw / 2),
            "l_pen_top_line": (0.0, cy - pbw / 2),
            "l_pen_bottom_line": (0.0, cy + pbw / 2),
            "l_pen_spot": (self.penalty_spot_dist, cy),
            # right penalty box
            "r_pen_top": (L - pb, cy - pbw / 2),
            "r_pen_bottom": (L - pb, cy + pbw / 2),
            "r_pen_top_line": (L, cy - pbw / 2),
            "r_pen_bottom_line": (L, cy + pbw / 2),
            "r_pen_spot": (L - self.penalty_spot_dist, cy),
            # goal boxes
            "l_goal_top": (gb, cy - gbw / 2),
            "l_goal_bottom": (gb, cy + gbw / 2),
            "r_goal_top": (L - gb, cy - gbw / 2),
            "r_goal_bottom": (L - gb, cy + gbw / 2),
        }
        if self.kind == "futsal":
            # no goal area; the penalty-area corners on the goal line are not
            # painted (the arcs meet the goal line 6 m either side instead)
            for k in ("l_goal_top", "l_goal_bottom", "r_goal_top", "r_goal_bottom",
                      "l_pen_top_line", "l_pen_bottom_line", "r_pen_top_line", "r_pen_bottom_line"):
                pts.pop(k)
            r = self.penalty_box_length
            pts.update({"l_arc_top_line": (0.0, cy - pbw / 2 - r), "l_arc_bottom_line": (0.0, cy + pbw / 2 + r),
                        "r_arc_top_line": (L, cy - pbw / 2 - r), "r_arc_bottom_line": (L, cy + pbw / 2 + r)})
        return pts


@dataclass
class PipelineConfig:
    weights: str = "runs/detect/soccernet/weights/best.pt"   # best measured held-out mAP
    conf: float = 0.3
    iou: float = 0.5
    imgsz: int = 1280
    tracker: str = "imm"           # measured best; "bytetrack" and "botsort" remain available
    track_low_conf: float = 0.1    # retained for second-stage association, not new births
    track_buffer: int = 90
    device: str = ""               # "" -> auto, "0" -> cuda:0, "cpu"
    fps: float = 25.0              # source fps; overridden by the video if readable
    # team assignment
    team_sample_frames: int = 30   # frames used to fit the team colour model
    predict_trajectories: bool = False
    # metrics
    max_speed_mps: float = 12.0    # clamp implausible per-frame speeds (amateur jitter)
    smooth_window: int = 5         # frames for position smoothing
    # pitch
    pitch: PitchConfig = field(default_factory=PitchConfig)
