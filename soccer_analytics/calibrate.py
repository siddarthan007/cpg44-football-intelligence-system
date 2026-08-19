"""Interactive pitch calibration: click known landmarks on a video frame to build
the image→pitch homography calibration file.

Usage:
    python -m soccer_analytics.calibrate --video match.mp4 --out camera1.yaml

Click at least 4 of the listed landmarks (more = more accurate). Press:
    n  -> skip to the next landmark in the list
    u  -> undo the last click
    s  -> save and quit
    q  -> quit without saving

The set of landmark names comes from PitchConfig.named_points(); pick the ones
clearly visible in your footage (corners, halfway-line ends, penalty-box corners,
penalty spots, centre).
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import cv2

from .config import PitchConfig
from .pitch import save_calibration

# a sensible click order — the most useful, easiest-to-see landmarks first
SUGGESTED_ORDER = [
    "tl_corner", "tr_corner", "br_corner", "bl_corner",
    "half_top", "half_bottom", "centre",
    "l_pen_top", "l_pen_bottom", "r_pen_top", "r_pen_bottom",
    "l_pen_spot", "r_pen_spot",
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Click pitch landmarks to calibrate homography.")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True, help="output calibration .yaml")
    p.add_argument("--frame", type=int, default=0, help="which frame to calibrate on")
    args = p.parse_args(argv)

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("could not read frame")
        return 2

    pc = PitchConfig()
    valid = pc.named_points()
    order = [n for n in SUGGESTED_ORDER if n in valid]
    picked: Dict[str, Tuple[float, float]] = {}
    idx = [0]

    def redraw():
        disp = frame.copy()
        for name, (x, y) in picked.items():
            cv2.circle(disp, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(disp, name, (int(x) + 6, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cur = order[idx[0]] if idx[0] < len(order) else "(done — press s to save)"
        cv2.putText(disp, f"CLICK: {cur}   [{len(picked)} set]  n=skip u=undo s=save q=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("calibrate", disp)

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and idx[0] < len(order):
            picked[order[idx[0]]] = (float(x), float(y))
            idx[0] += 1
            redraw()

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_mouse)
    redraw()
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == ord("n") and idx[0] < len(order):
            idx[0] += 1; redraw()
        elif k == ord("u") and picked:
            last = list(picked)[-1]; picked.pop(last)
            idx[0] = max(0, idx[0] - 1); redraw()
        elif k == ord("s"):
            break
        elif k == ord("q"):
            cv2.destroyAllWindows()
            print("quit without saving")
            return 1
    cv2.destroyAllWindows()

    if len(picked) < 4:
        print(f"only {len(picked)} landmarks — need ≥4 for a homography; not saved")
        return 2
    save_calibration(args.out, picked)
    print(f"saved {len(picked)} landmarks to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
