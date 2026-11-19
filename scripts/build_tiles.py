"""Native-resolution tiles for slicing-aided fine-tuning (SAHI, Akyon et al. 2022).

Broadcast frames are 1920x1080. Training at 1280 px shrinks a 12 px ball to
8 px; at 640 px to 4 px. Cutting each frame into 640x640 tiles at native scale
lets the detector learn the ball at its real size; at inference the fully
convolutional model runs on the full 1920 px frame, where objects have the
same scale it was trained on.

Tiles: x0 in {0, 640, 1280}, y0 in {0, 440} (6 per frame, 200 px vertical
overlap). Boxes are clipped to the tile; a box is kept when at least 45 % of
its area is inside (a cut player is still a player), the ball when its centre
is inside.

    PYTHONPATH=. python scripts/build_tiles.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

SRC = Path.home() / "SoccerNet" / "yolo"
DST = Path.home() / "SoccerNet" / "yolo_tiles640"
TILE = 640
XS, YS = (0, 640, 1280), (0, 440)


def tile_split(split: str, stride: int):
    imgs = sorted((SRC / "images" / split).glob("*.jpg"))[::stride]
    (DST / "images" / split).mkdir(parents=True, exist_ok=True)
    (DST / "labels" / split).mkdir(parents=True, exist_ok=True)
    n = 0
    for p in imgs:
        im = cv2.imread(str(p))
        H, W = im.shape[:2]
        lab = SRC / "labels" / split / f"{p.stem}.txt"
        rows = np.loadtxt(lab, ndmin=2) if lab.is_file() and lab.stat().st_size else np.zeros((0, 5))
        cls = rows[:, 0].astype(int)
        cx, cy, bw, bh = rows[:, 1] * W, rows[:, 2] * H, rows[:, 3] * W, rows[:, 4] * H
        x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        for x0 in XS:
            for y0 in YS:
                crop = im[y0:y0 + TILE, x0:x0 + TILE]
                if crop.shape[0] < TILE or crop.shape[1] < TILE:
                    crop = cv2.copyMakeBorder(crop, 0, TILE - crop.shape[0], 0, TILE - crop.shape[1],
                                              cv2.BORDER_CONSTANT, value=(114, 114, 114))
                cx1, cy1 = np.clip(x1 - x0, 0, TILE), np.clip(y1 - y0, 0, TILE)
                cx2, cy2 = np.clip(x2 - x0, 0, TILE), np.clip(y2 - y0, 0, TILE)
                inside = (cx2 - cx1) * (cy2 - cy1) / np.maximum(bw * bh, 1e-6)
                ccx, ccy = cx - x0, cy - y0
                keep = np.where(cls == 0, (ccx >= 0) & (ccx < TILE) & (ccy >= 0) & (ccy < TILE), inside >= 0.45)
                keep &= (cx2 - cx1 >= 2) & (cy2 - cy1 >= 2)
                out = []
                for k in np.nonzero(keep)[0]:
                    out.append(f"{cls[k]} {(cx1[k] + cx2[k]) / 2 / TILE:.6f} {(cy1[k] + cy2[k]) / 2 / TILE:.6f} "
                               f"{(cx2[k] - cx1[k]) / TILE:.6f} {(cy2[k] - cy1[k]) / TILE:.6f}")
                name = f"{p.stem}_x{x0}_y{y0}"
                cv2.imwrite(str(DST / "images" / split / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                (DST / "labels" / split / f"{name}.txt").write_text("\n".join(out))
                n += 1
    print(f"[tiles] {split}: {len(imgs)} frames -> {n} tiles", flush=True)


def main():
    names = yaml.safe_load((SRC / "data.yaml").read_text())["names"]
    tile_split("train", 2)       # every 10th original frame (source is already every 5th)
    tile_split("val", 3)
    (DST / "data.yaml").write_text(yaml.safe_dump({"path": str(DST), "train": "images/train",
                                                   "val": "images/val", "names": names}))


if __name__ == "__main__":
    main()
