"""Train a small-object detector for football: YOLO26 with a stride-4 (P2) head.

Why this architecture (see docs/TRACKING_BENCHMARK.md, section 6):
- the ball is 6-15 px at 1280 px input; a stride-8 head (YOLOv8/11 default)
  places it on a coarse grid where it often vanishes, while the extra P2/4
  head keeps a 4 px grid for it;
- YOLO26 is NMS-free (end-to-end), which avoids suppressing overlapping
  players in clusters, and uses small-target-aware label assignment;
- the s/m scales fit an 8 GB laptop GPU at 1280 px with AMP.

Data: SoccerNet-Tracking converted to YOLO (48 training clips, 9 validation
clips for checkpoint selection; the 49 test clips are only used by `eval`).
Training resumes automatically from last.pt after an interruption.

    PYTHONPATH=. python scripts/train_detector_p2.py train --scale s --epochs 40
    PYTHONPATH=. python scripts/train_detector_p2.py eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "SoccerNet" / "yolo" / "data.yaml"              # full frames (test eval)
TILES = Path.home() / "SoccerNet" / "yolo_tiles640" / "data.yaml"   # native-scale tiles (training)
OUT = ROOT / "data" / "benchmarks" / "detector_compare.json"


def train(a):
    from ultralytics import YOLO
    from soccer_analytics.data_split import validate_training_data
    validate_training_data(TILES)
    name = f"yolo26{a.scale}_p2_soccernet"
    last = ROOT / "runs" / "detect" / name / "weights" / "last.pt"
    if last.is_file() and not a.fresh:
        print(f"[train] resuming {last}", flush=True)
        YOLO(str(last)).train(resume=True)
        return
    model = YOLO(f"yolo26{a.scale}-p2.yaml").load(f"yolo26{a.scale}.pt")   # transfer matching layers
    model.train(
        data=str(TILES), imgsz=640, epochs=a.epochs, batch=a.batch, device=0, workers=3,
        amp=True, cache=False, project=str(ROOT / "runs" / "detect"), name=name, exist_ok=True,
        plots=False, patience=100, cos_lr=True, close_mosaic=8,
        # gentler zoom-out keeps the ball above a few pixels; small rotations
        # and blur mimic panning broadcast and fixed campus cameras
        scale=0.3, translate=0.1, degrees=2.0, mixup=0.0, fraction=a.fraction,
    )


def evaluate(a):
    from ultralytics import YOLO
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    cands = {
        "yolov8m soccernet (previous best)": ROOT / "runs/detect/soccernet/weights/best.pt",
        "yolo26s-p2 soccernet": ROOT / "runs/detect/yolo26s_p2_soccernet/weights/best.pt",
        "yolo26m-p2 soccernet": ROOT / "runs/detect/yolo26m_p2_soccernet/weights/best.pt",
    }
    sizes = {"yolov8m soccernet (previous best)": 1280}
    for name, w in cands.items():
        if not w.is_file() or (name in results and not a.force):
            continue
        m = YOLO(str(w))
        sz = sizes.get(name, 1920)          # P2 tile models run on the native 1920 px frame
        r = m.val(data=str(DATA), split="test", imgsz=sz, batch=4, plots=False, verbose=False,
                  project=str(ROOT / "runs" / "detect" / "val"), name=w.parts[-3], exist_ok=True)
        results[name] = {
            "weights": str(w.relative_to(ROOT)), "imgsz": sz,
            "map50_95": float(r.box.map), "map50": float(r.box.map50),
            "per_class": {m.names[int(c)]: {"precision": float(r.box.p[i]), "recall": float(r.box.r[i]),
                                            "ap50": float(r.box.ap50[i]), "ap50_95": float(r.box.maps[int(c)])}
                          for i, c in enumerate(r.box.ap_class_index)},
            "ms_per_image": r.speed,
            "params_m": round(sum(p.numel() for p in m.model.parameters()) / 1e6, 2),
        }
        OUT.write_text(json.dumps(results, indent=2))
        print(name, results[name]["map50_95"], results[name]["per_class"].get("ball"), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("train", "eval"))
    p.add_argument("--scale", default="s", choices=("n", "s", "m"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    (train if a.cmd == "train" else evaluate)(a)


if __name__ == "__main__":
    main()
