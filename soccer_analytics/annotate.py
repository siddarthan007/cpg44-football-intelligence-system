"""Frame annotation overlays (OpenCV, imported lazily).

Draws team-coloured ellipses under players, track ids, a ball marker, a
possession bar, and composites the top-down radar minimap.
"""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

TEAM_BGR = {1: (114, 107, 68), 2: (84, 95, 161), 0: (185, 181, 173)}
REFEREE_BGR = (111, 182, 211)


def draw_player(frame, xyxy, team: int, label: str = "", is_ref: bool = False,
                wearable: bool = False):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    color = REFEREE_BGR if is_ref else TEAM_BGR.get(team, TEAM_BGR[0])
    cx, w = (x1 + x2) // 2, x2 - x1
    axes = (max(int(w * 0.6), 8), max(int(w * 0.25), 4))
    cv2.ellipse(frame, (cx, y2), axes, 0, -45, 235, color, 2, cv2.LINE_AA)
    if label:
        text = str(label)
        font_scale = 0.52
        (text_w, text_h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        pad_x, pad_y = 6, 5
        left = cx - (text_w + pad_x * 2) // 2
        top = y2 + 4
        cv2.rectangle(
            frame,
            (left, top),
            (left + text_w + pad_x * 2, top + text_h + pad_y * 2),
            color,
            -1,
        )
        cv2.putText(
            frame,
            text,
            (left + pad_x, top + text_h + pad_y - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if wearable:
        # A small white ring marks a player with a paired wearable.
        cv2.circle(frame, (cx, y1 - 10), 6, (245, 245, 240), 2, cv2.LINE_AA)
    return frame


def draw_prediction(frame, pts, color=(150, 174, 142)):
    """Draw an LSTM-predicted future path (list of pixel points) as a fading line."""
    if pts is None or len(pts) < 2:
        return frame
    p = np.asarray(pts, np.int32)
    for i in range(len(p) - 1):
        cv2.line(frame, tuple(p[i]), tuple(p[i + 1]), color, 2, cv2.LINE_AA)
    cv2.circle(frame, tuple(p[-1]), 4, color, -1, cv2.LINE_AA)
    return frame


def draw_ball(frame, center):
    if center is None:
        return frame
    x, y = int(center[0]), int(center[1])
    pts = np.array([[x, y - 12], [x - 9, y - 24], [x + 9, y - 24]], np.int32)
    cv2.fillPoly(frame, [pts], (255, 255, 255))
    cv2.polylines(frame, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def draw_possession(frame, pct: Dict[int, float]):
    h, w = frame.shape[:2]
    bar_w, bar_h, x0, y0 = 300, 26, 20, 20
    p1 = pct.get(1, 0.0)
    split = int(bar_w * p1 / 100.0)
    cv2.rectangle(frame, (x0, y0), (x0 + split, y0 + bar_h), TEAM_BGR[1], -1)
    cv2.rectangle(frame, (x0 + split, y0), (x0 + bar_w, y0 + bar_h), TEAM_BGR[2], -1)
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 1)
    cv2.putText(frame, f"{p1:4.0f}%", (x0 + 4, y0 + 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{pct.get(2,0):4.0f}%", (x0 + bar_w - 46, y0 + 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def overlay_radar(frame, radar):
    if radar is None:
        return frame
    h, w = frame.shape[:2]
    rh, rw = radar.shape[:2]
    x0, y0 = w - rw - 20, h - rh - 20
    if x0 < 0 or y0 < 0:
        return frame
    roi = frame[y0:y0 + rh, x0:x0 + rw]
    frame[y0:y0 + rh, x0:x0 + rw] = (0.35 * roi + 0.65 * radar).astype(np.uint8)
    return frame
