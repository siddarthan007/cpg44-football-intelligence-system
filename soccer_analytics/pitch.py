"""Build a :class:`~soccer_analytics.view.ViewTransformer` (image px → pitch m).

Two ways to calibrate, so this works on ANY footage (not one hardcoded clip like
the reference repos):

1. **Static calibration file** (recommended for a fixed university camera):
   a YAML mapping named pitch landmarks to their pixel positions. Create it with
   ``python -m soccer_analytics.calibrate``. One file per camera view.

2. **Keypoint model** (optional): if you have a pitch-keypoint detector (e.g. the
   Roboflow ``sports`` package), pass its detected image points + the matching
   landmark names to :func:`view_from_named_points` per frame.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import yaml

from .config import PitchConfig
from .view import ViewTransformer


def known_lines(pc: PitchConfig) -> Dict[str, tuple]:
    """Pitch lines usable as calibration constraints: name → (axis, value)."""
    if pc.kind == "futsal":
        # only full-length straight lines; the penalty area is mostly arcs
        return {"far_touchline": ("y", 0.0), "near_touchline": ("y", pc.width),
                "halfway": ("x", pc.length / 2.0), "left_goal_line": ("x", 0.0), "right_goal_line": ("x", pc.length)}
    return {
        "far_touchline": ("y", 0.0),
        "near_touchline": ("y", pc.width),
        "halfway": ("x", pc.length / 2.0),
        "left_goal_line": ("x", 0.0),
        "right_goal_line": ("x", pc.length),
        # penalty-area and goal-area edges: visible in most attacking views
        "left_box_front": ("x", pc.penalty_box_length),
        "right_box_front": ("x", pc.length - pc.penalty_box_length),
        "left_goal_area_front": ("x", pc.goal_box_length),
        "right_goal_area_front": ("x", pc.length - pc.goal_box_length),
        "box_far_edge": ("y", pc.width / 2 - pc.penalty_box_width / 2),
        "box_near_edge": ("y", pc.width / 2 + pc.penalty_box_width / 2),
        "goal_area_far_edge": ("y", pc.width / 2 - pc.goal_box_width / 2),
        "goal_area_near_edge": ("y", pc.width / 2 + pc.goal_box_width / 2),
    }


def view_from_named_points(named_px: Dict[str, Sequence[float]],
                           pc: PitchConfig,
                           named_lines: Optional[Dict[str, Sequence]] = None,
                           circle_pts: Optional[Sequence] = None
                           ) -> ViewTransformer:
    """Fit a homography from {landmark_name: [px_x, px_y]} plus optional
    {line_name: [[px_x,px_y], ...]} points on known pitch lines and optional
    ``circle_pts`` = [[px_x,px_y], ...] points ON the centre circle. Line and
    circle constraints anchor the fit across the frame, vital when the point
    landmarks only cover a small region (and hand-clicked "circle top/bottom"
    are biased under perspective)."""
    from .view import fit_homography_lines

    template = pc.named_points()
    has_constraints = bool(named_lines) or bool(circle_pts)
    circle_names = {"circle_left", "circle_right", "circle_top", "circle_bottom"}
    img_pts, pitch_pts = [], []           # hard correspondences
    init_img, init_pitch = [], []         # bootstrap-only (biased under perspective)
    extra_circle = []
    unknown = []
    for name, px in named_px.items():
        if name not in template:
            unknown.append(name)
            continue
        p = [float(px[0]), float(px[1])]
        if has_constraints and name in circle_names:
            # clicked "circle extremes" are the ELLIPSE's extremes, biased as
            # point correspondences. Use them to bootstrap + as on-circle
            # constraints (which is what they truly are).
            init_img.append(p); init_pitch.append(list(template[name]))
            extra_circle.append(p)
        else:
            img_pts.append(p); pitch_pts.append(list(template[name]))
    if unknown:
        raise ValueError(f"unknown pitch landmark(s): {unknown}. "
                         f"valid names: {sorted(template)}")
    if len(img_pts) + len(init_img) < 4:
        raise ValueError(f"need ≥4 known landmarks, got {len(img_pts) + len(init_img)}")

    constraints = []
    lines = known_lines(pc)
    for lname, pts in (named_lines or {}).items():
        if lname not in lines:
            raise ValueError(f"unknown pitch line {lname!r}; valid: {sorted(lines)}")
        axis, value = lines[lname]
        for p in pts:
            constraints.append(((float(p[0]), float(p[1])), axis, value))
    circle = (pc.length / 2.0, pc.width / 2.0, pc.centre_circle_radius)
    for p in list(circle_pts or []) + extra_circle:
        constraints.append(((float(p[0]), float(p[1])), "circle", circle))

    H = fit_homography_lines(img_pts, pitch_pts, constraints,
                             init_src=init_img, init_dst=init_pitch)
    return ViewTransformer(H=H, pitch_length=pc.length, pitch_width=pc.width)


def calibration_meta(path: str) -> dict:
    """Pitch size, lens curvature and camera mode stored with a calibration.

    Older files without these keys mean: FIFA 105 x 68 m, pinhole lens,
    camera mode decided by the pipeline ("auto")."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    pitch = data.get("pitch") or {}
    lens = data.get("lens") or {}
    return {
        "pitch_length": float(pitch.get("length", 105.0)),
        "pitch_width": float(pitch.get("width", 68.0)),
        "pitch_kind": str(pitch.get("kind", "football")),
        "k1": float(lens.get("k1", 0.0)),
        "camera": str(data.get("camera", "auto")),
        "image_size": data.get("image_size"),
    }


def load_calibration(path: str, pc: Optional[PitchConfig] = None) -> ViewTransformer:
    """Load a calibration YAML (CLI tool or the web Pitch setup).

    Format::

        points: {tl_corner: [110, 1035], ...}       # clicked pixels (as seen)
        lines:  {halfway: [[x, y], ...]}            # optional constraints
        pitch:  {length: 100, width: 64}            # optional, metres
        lens:   {k1: -0.12}                         # optional radial curvature
        homography: [[...], [...], [...]]           # optional, image -> pitch
        camera: fixed | panning | auto

    When ``homography`` is present (e.g. after snapping to painted lines) it
    is used directly; it refers to the *undistorted* image when k1 != 0.
    """
    meta = calibration_meta(path)
    if pc is None or (pc.length, pc.width) != (meta["pitch_length"], meta["pitch_width"]):
        pc = PitchConfig.for_kind(meta["pitch_kind"], meta["pitch_length"], meta["pitch_width"])
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data.get("homography") is not None:
        H = np.asarray(data["homography"], float)
        if H.shape == (3, 3) and np.isfinite(H).all():
            return ViewTransformer(H=H / H[2, 2], pitch_length=pc.length, pitch_width=pc.width)
    pts = dict(data.get("points", data))
    lines = data.get("lines")
    circle = data.get("circle")
    k1 = meta["k1"]
    if k1 and meta["image_size"]:
        from .lens import undistort_points
        w, h = meta["image_size"]
        und = lambda v: undistort_points([v], k1, w, h)[0].tolist()
        pts = {k: und(v) for k, v in pts.items()}
        lines = {k: [und(v) for v in vs] for k, vs in (lines or {}).items()} or None
        circle = [und(v) for v in circle] if circle else None
    return view_from_named_points(pts, pc, lines, circle)


def save_calibration(path: str, named_px: Dict[str, Sequence[float]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump({"points": {k: [float(v[0]), float(v[1])]
                                   for k, v in named_px.items()}}, fh, sort_keys=False)
