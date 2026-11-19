"""Pre-kick-off camera check: will this view support accurate tracking?

Most tracking error is decided before the match starts, by where the camera is
and how it is set. This module scores one frame against the conditions the
rest of the pipeline needs and returns plain-language fixes:

* resolution (small players and the ball need pixels),
* sharpness (variance of the Laplacian; motion/focus blur kills ball recall),
* exposure (clipped highlights/shadows hide kits and lines),
* pitch coverage (share of grass in the frame),
* visible line structure (straight white segments for calibration),
* apparent player size, especially on the far side (detector small-object limit),
* number of people the detector finds.

Thresholds are engineering defaults for this detector at 1080p; they are
reported with the measured value so an operator can judge borderline cases.
"""

from __future__ import annotations


import cv2
import numpy as np


def _grade(ok: bool, warn: bool) -> str:
    return "good" if ok else "check" if warn else "fix"


def check_frame(frame: np.ndarray, detector=None) -> dict:
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    items = []

    items.append({"key": "resolution", "label": "Resolution", "value": f"{W}x{H}",
                  "grade": _grade(H >= 1080, H >= 720),
                  "advice": None if H >= 1080 else "Record at 1080p or higher; the ball is only a few pixels wide."})

    small = cv2.resize(gray, (960, int(960 * H / W)))
    sharp = float(cv2.Laplacian(small, cv2.CV_64F).var())
    items.append({"key": "sharpness", "label": "Sharpness", "value": round(sharp, 1),
                  "grade": _grade(sharp >= 120, sharp >= 60),
                  "advice": None if sharp >= 120 else "Image is soft: fix focus, clean the lens, use a faster shutter or more light."})

    hi = float((gray >= 250).mean())
    lo = float((gray <= 8).mean())
    items.append({"key": "exposure", "label": "Exposure", "value": f"{hi:.1%} blown, {lo:.1%} crushed",
                  "grade": _grade(hi < 0.02 and lo < 0.05, hi < 0.06 and lo < 0.12),
                  "advice": None if hi < 0.02 and lo < 0.05 else "Avoid pointing into the sun or floodlights; lock exposure on the grass."})

    grass = (hsv[..., 0] >= 30) & (hsv[..., 0] <= 90) & (hsv[..., 1] >= 40) & (hsv[..., 2] >= 35)
    cover = float(grass.mean())
    items.append({"key": "pitch", "label": "Pitch in frame", "value": f"{cover:.0%}",
                  "grade": _grade(cover >= 0.45, cover >= 0.3),
                  "advice": None if cover >= 0.45 else "Point the camera more at the pitch; too much sky, stands or fence."})

    # Painted lines are thin, low-saturation ridges brighter than the grass
    # around them: a white top-hat on the value channel finds them under any
    # lighting, where a fixed "white" threshold fails on shaded turf.
    v = hsv[..., 2]
    ridge = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    white = ((ridge >= 22) & (hsv[..., 1] <= 110)).astype(np.uint8) * 255
    white = cv2.bitwise_and(white, white, mask=cv2.dilate(grass.astype(np.uint8) * 255, np.ones((15, 15), np.uint8)))
    lines = cv2.HoughLinesP(white, 1, np.pi / 180, 80, minLineLength=int(0.08 * W), maxLineGap=12)
    n_lines = 0 if lines is None else len(lines)
    items.append({"key": "lines", "label": "Pitch lines visible", "value": n_lines,
                  "grade": _grade(n_lines >= 6, n_lines >= 3),
                  "advice": None if n_lines >= 6 else "Calibration needs at least four landmarks: frame a penalty box, the halfway line or the centre circle."})

    people = np.zeros((0, 4))
    if detector is not None:
        try:
            from .core import BALL
            d = detector.detect(frame)
            people = d.xyxy[d.class_id != BALL]
        except Exception:
            people = np.zeros((0, 4))
    if len(people):
        heights = people[:, 3] - people[:, 1]
        top_band = people[:, 3] < np.percentile(people[:, 3], 30)
        far_h = float(np.median(heights[top_band])) if top_band.any() else float(np.median(heights))
        items.append({"key": "players", "label": "People detected", "value": int(len(people)),
                      "grade": _grade(len(people) >= 6, len(people) >= 2), "advice": None})
        items.append({"key": "far_size", "label": "Far-side player height", "value": f"{far_h:.0f} px",
                      "grade": _grade(far_h >= 45, far_h >= 28),
                      "advice": None if far_h >= 45 else "Far players are tiny: raise the camera, move closer to halfway or use a second camera for the far half."})
    elif detector is not None:
        items.append({"key": "players", "label": "People detected", "value": 0, "grade": "check",
                      "advice": "No players found; fine for an empty pitch, otherwise check framing."})

    # Depiction of what was measured: painted lines in amber, people boxed,
    # the small far-side players highlighted.
    vis = (frame * 0.55).astype(np.uint8)
    vis[cv2.dilate(white, np.ones((3, 3), np.uint8)) > 0] = (5, 183, 242)
    if len(people):
        heights = people[:, 3] - people[:, 1]
        for (x1, y1, x2, y2), hh in zip(people.astype(int), heights):
            col = (46, 112, 72) if hh >= 45 else (10, 101, 168) if hh >= 28 else (46, 58, 178)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
    small = cv2.resize(vis, (960, int(960 * H / W)))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 82])
    import base64
    overlay = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None

    worst = "fix" if any(i["grade"] == "fix" for i in items) else "check" if any(i["grade"] == "check" for i in items) else "good"
    return {"verdict": worst, "items": items, "frame": {"width": W, "height": H}, "overlay": overlay,
            "summary": {"good": "Ready to record.", "check": "Usable; fix the flagged items if you can.",
                        "fix": "Fix the flagged items before kick-off or accuracy will suffer."}[worst]}
