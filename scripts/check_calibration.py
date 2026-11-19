"""Overlay a calibration's pitch model on its frame to verify it by eye.

    PYTHONPATH=. python scripts/check_calibration.py demo/calib/SNMOT-074.yaml \
        demo/clips/SNMOT-074_750f.mp4 out.jpg
Lines drawn in yellow are the pitch template projected through the fitted
homography; they should sit on the painted lines everywhere they are visible.
"""
import sys

import cv2
import numpy as np

from soccer_analytics.config import PitchConfig
from soccer_analytics.pitch import load_calibration
from soccer_analytics.view import apply_homography


def main(calib, video, out, frame_index=0):
    pc = PitchConfig()
    view = load_calibration(calib, pc)
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    Hinv = np.linalg.inv(view.H)
    L, W, cy = pc.length, pc.width, pc.width / 2

    def poly(pts):
        px = apply_homography(Hinv, pts)
        px = px[np.isfinite(px).all(1)]
        cv2.polylines(frame, [px.astype(np.int32)], False, (0, 230, 255), 2, cv2.LINE_AA)

    def seg(a, b, n=60):
        poly(np.linspace(a, b, n))
    for a, b in [((0, 0), (L, 0)), ((L, 0), (L, W)), ((L, W), (0, W)), ((0, W), (0, 0)), ((L / 2, 0), (L / 2, W))]:
        seg(a, b)
    for x0, d in ((0, 1), (L, -1)):
        for depth, half in ((16.5, 20.16), (5.5, 9.16)):
            seg((x0, cy - half), (x0 + d * depth, cy - half)); seg((x0 + d * depth, cy - half), (x0 + d * depth, cy + half))
            seg((x0 + d * depth, cy + half), (x0, cy + half))
    a = np.linspace(0, 2 * np.pi, 120)
    poly(np.column_stack((L / 2 + 9.15 * np.cos(a), cy + 9.15 * np.sin(a))))
    cv2.imwrite(out, frame)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
