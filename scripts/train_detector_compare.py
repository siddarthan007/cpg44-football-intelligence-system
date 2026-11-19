"""Equal-data detector comparison: YOLOv8m continuation vs RT-DETR-L fine-tune.

Both models train on the 48 SoccerNet-Tracking training clips (every 5th frame,
/home/siddartha/SoccerNet/yolo images/train), select checkpoints on the 9
held-out validation clips (images/val), and are scored once on the 49 official
test clips (images/test). Results go to data/benchmarks/detector_compare.json.

    conda activate soccer
    PYTHONPATH=. python scripts/train_detector_compare.py rtdetr --epochs 8 --imgsz 960
    PYTHONPATH=. python scripts/train_detector_compare.py yolo --epochs 12 --imgsz 1280
    PYTHONPATH=. python scripts/train_detector_compare.py eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "SoccerNet" / "yolo" / "data.yaml"
OUT = ROOT / "data" / "benchmarks" / "detector_compare.json"


def train_rtdetr(a):
    from ultralytics import RTDETR
    from soccer_analytics.data_split import validate_training_data
    validate_training_data(DATA)
    m = RTDETR("rtdetr-l.pt")
    m.train(data=str(DATA), imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, device=0, workers=2,
            amp=True, cache=False, project=str(ROOT / "runs" / "detect"), name="rtdetr_l_soccernet",
            exist_ok=True, plots=False, patience=50, close_mosaic=2, lr0=1e-4, optimizer="AdamW")


def train_yolo(a):
    from ultralytics import YOLO
    from soccer_analytics.data_split import validate_training_data
    validate_training_data(DATA)
    m = YOLO(str(ROOT / "runs/detect/soccernet/weights/best.pt"))
    # Continue the undertrained checkpoint. Less aggressive down-scaling keeps
    # the few-pixel ball visible; close mosaic for the last epochs.
    m.train(data=str(DATA), imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, device=0, workers=2,
            amp=True, cache=False, project=str(ROOT / "runs" / "detect"), name="soccernet_v4",
            exist_ok=True, plots=False, lr0=0.002, lrf=0.1, cos_lr=True, warmup_epochs=0.5,
            scale=0.35, mosaic=1.0, close_mosaic=4, patience=50)


def evaluate(a):
    from ultralytics import YOLO, RTDETR
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    cands = {
        "yolov8m_soccernet (epoch 10)": (YOLO, ROOT / "runs/detect/soccernet/weights/best.pt", 1280),
        "yolov8m_soccernet_v4 (continued)": (YOLO, ROOT / "runs/detect/soccernet_v4/weights/best.pt", 1280),
        "rtdetr_l_soccernet": (RTDETR, ROOT / "runs/detect/rtdetr_l_soccernet/weights/best.pt", a.imgsz),
    }
    for name, (cls, w, sz) in cands.items():
        if not w.is_file() or (name in results and not a.force):
            continue
        m = cls(str(w))
        r = m.val(data=str(DATA), split="test", imgsz=sz, batch=4, plots=False, verbose=False,
                  project=str(ROOT / "runs" / "detect" / "val"), name=w.parts[-3], exist_ok=True)
        results[name] = {
            "weights": str(w.relative_to(ROOT)), "imgsz": sz,
            "map50_95": float(r.box.map), "map50": float(r.box.map50),
            "per_class_map50_95": {m.names[i]: float(v) for i, v in enumerate(r.box.maps)},
            "per_class": {m.names[int(c)]: {"precision": float(r.box.p[i]), "recall": float(r.box.r[i]),
                                            "ap50": float(r.box.ap50[i])}
                          for i, c in enumerate(r.box.ap_class_index)},
            "ms_per_image": r.speed,
        }
        OUT.write_text(json.dumps(results, indent=2))
        print(name, results[name]["map50_95"], results[name]["per_class"].get("ball"), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("rtdetr", "yolo", "eval"))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    {"rtdetr": train_rtdetr, "yolo": train_yolo, "eval": evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
