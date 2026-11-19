"""Adapt the detector to a specific ground (campus fine-tuning loop).

Broadcast training data (SoccerNet) differs from a campus tripod camera in
viewpoint, player size, turf, kits and light. This module runs the standard
human-in-the-loop adaptation recipe:

1. ``add_video``   sample diverse frames (uniform in time, near-duplicates
                   skipped by a frame-difference test);
2. ``prelabel``    the current detector proposes boxes (model labels);
3. people review in the web annotator (accept, fix class, delete, add),
   and ``save_labels`` stores the reviewed frame;
4. ``build_dataset`` converts reviewed frames to native-scale 640 px tiles,
   holds out the *last* 20 % of reviewed frames in time as campus validation
   (no temporal leakage), and mixes in SoccerNet tiles so the model does not
   forget broadcast cases (replay, cf. continual learning practice);
5. ``train`` fine-tunes from the current best checkpoint (low LR, few epochs);
6. ``evaluate`` scores base and adapted models on the campus validation
   frames; the adapted model is recommended only if it is better.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data" / "adapt"
CLASSES = ["ball", "goalkeeper", "player", "referee"]          # SoccerNet/yolo order
SN_TILES = Path.home() / "SoccerNet" / "yolo_tiles640"
SAFE = re.compile(r"^[a-z0-9-]{1,40}$")


def _proj(name: str) -> Path:
    if not SAFE.match(name or ""):
        raise ValueError("project names use lowercase letters, digits and dashes")
    d = WORK / name
    (d / "frames").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(parents=True, exist_ok=True)
    return d


def add_video(project: str, video: str, n: int = 60, min_diff: float = 6.0) -> int:
    d = _proj(project)
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        raise ValueError("could not read the video")
    stem = re.sub(r"[^a-z0-9]+", "-", Path(video).stem.lower())[:24] or "clip"
    picks = np.linspace(0, total - 1, min(n * 3, total)).astype(int)
    kept, prev = 0, None
    for f in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, frame = cap.read()
        if not ok:
            continue
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)
        if prev is not None and float(np.mean(np.abs(small - prev))) < min_diff:
            continue                                   # near-duplicate of the last kept frame
        prev = small
        cv2.imwrite(str(d / "frames" / f"{stem}_{int(f):07d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 93])
        kept += 1
        if kept >= n:
            break
    cap.release()
    return kept


def frames(project: str) -> List[dict]:
    d = _proj(project)
    out = []
    for p in sorted((d / "frames").glob("*.jpg")):
        lab = d / "labels" / f"{p.stem}.json"
        info = json.loads(lab.read_text()) if lab.is_file() else {}
        out.append({"id": p.stem, "reviewed": bool(info.get("reviewed")), "boxes": len(info.get("boxes", []))})
    return out


def prelabel(project: str, weights: str, imgsz: int = 1280, device: str = "") -> int:
    from .tracker import Detector
    d = _proj(project)
    if not device:
        try:                                   # share the 8 GB card politely with a running training job
            import torch
            free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
            device = "0" if free > 1.5e9 else "cpu"
        except Exception:
            device = "cpu"
    det = Detector(weights, imgsz=imgsz, device=device, person_conf=0.25, ball_conf=0.15,
                   track_low_conf=0.25, remap_goalkeeper=False)
    names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}   # canonical ids in core.py
    n = 0
    for p in sorted((d / "frames").glob("*.jpg")):
        lab = d / "labels" / f"{p.stem}.json"
        if lab.is_file() and json.loads(lab.read_text()).get("reviewed"):
            continue
        r = det.detect(cv2.imread(str(p)))
        boxes = [{"cls": names.get(int(c), "player"), "box": [round(float(v), 1) for v in b], "score": round(float(s), 3), "source": "model"}
                 for b, c, s in zip(r.xyxy, r.class_id, r.confidence)]
        lab.write_text(json.dumps({"boxes": boxes, "reviewed": False}))
        n += 1
    return n


def get_labels(project: str, frame_id: str) -> dict:
    d = _proj(project)
    lab = d / "labels" / f"{frame_id}.json"
    img = d / "frames" / f"{frame_id}.jpg"
    if not img.is_file():
        raise KeyError("frame not found")
    h, w = cv2.imread(str(img)).shape[:2]
    data = json.loads(lab.read_text()) if lab.is_file() else {"boxes": [], "reviewed": False}
    return {**data, "width": w, "height": h}


def save_labels(project: str, frame_id: str, boxes: List[dict]) -> dict:
    d = _proj(project)
    if not (d / "frames" / f"{frame_id}.jpg").is_file():
        raise KeyError("frame not found")
    clean = []
    for b in boxes:
        if b.get("cls") not in CLASSES:
            continue
        x1, y1, x2, y2 = (float(v) for v in b["box"])
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            clean.append({"cls": b["cls"], "box": [x1, y1, x2, y2], "source": b.get("source", "human")})
    (d / "labels" / f"{frame_id}.json").write_text(json.dumps({"boxes": clean, "reviewed": True, "at": time.time()}))
    return {"saved": len(clean)}


def _tile(img, boxes, out_img: Path, out_lab: Path, stem: str) -> int:
    H, W = img.shape[:2]
    xs = list(range(0, max(1, W - 640) + 1, 640)) or [0]
    if xs[-1] + 640 < W:
        xs.append(W - 640)
    ys = list(range(0, max(1, H - 640) + 1, 440)) or [0]
    if ys[-1] + 640 < H:
        ys.append(max(0, H - 640))
    n = 0
    for x0 in xs:
        for y0 in ys:
            crop = img[y0:y0 + 640, x0:x0 + 640]
            if crop.shape[0] < 640 or crop.shape[1] < 640:
                crop = cv2.copyMakeBorder(crop, 0, 640 - crop.shape[0], 0, 640 - crop.shape[1], cv2.BORDER_CONSTANT, value=(114, 114, 114))
            rows = []
            for b in boxes:
                x1, y1, x2, y2 = b["box"]
                cx1, cy1, cx2, cy2 = np.clip([x1 - x0, y1 - y0, x2 - x0, y2 - y0], 0, 640)
                area = (x2 - x1) * (y2 - y1)
                inside = (cx2 - cx1) * (cy2 - cy1) / max(area, 1e-6)
                keep = (0 <= (x1 + x2) / 2 - x0 < 640 and 0 <= (y1 + y2) / 2 - y0 < 640) if b["cls"] == "ball" else inside >= 0.45
                if keep and cx2 - cx1 >= 2 and cy2 - cy1 >= 2:
                    rows.append(f"{CLASSES.index(b['cls'])} {(cx1 + cx2) / 1280:.6f} {(cy1 + cy2) / 1280:.6f} {(cx2 - cx1) / 640:.6f} {(cy2 - cy1) / 640:.6f}")
            name = f"{stem}_x{x0}_y{y0}"
            cv2.imwrite(str(out_img / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out_lab / f"{name}.txt").write_text("\n".join(rows))
            n += 1
    return n


def build_dataset(project: str, replay_tiles: int = 2500, seed: int = 44) -> Path:
    d = _proj(project)
    reviewed = [p for p in sorted((d / "frames").glob("*.jpg"))
                if (d / "labels" / f"{p.stem}.json").is_file() and json.loads((d / "labels" / f"{p.stem}.json").read_text()).get("reviewed")]
    if len(reviewed) < 10:
        raise ValueError(f"review at least 10 frames first ({len(reviewed)} reviewed)")
    cut = int(len(reviewed) * 0.8)
    ds = d / "dataset"
    if ds.exists():
        shutil.rmtree(ds)
    for split in ("train", "val"):
        (ds / "images" / split).mkdir(parents=True)
        (ds / "labels" / split).mkdir(parents=True)
    for i, p in enumerate(reviewed):
        split = "train" if i < cut else "val"
        boxes = json.loads((d / "labels" / f"{p.stem}.json").read_text())["boxes"]
        _tile(cv2.imread(str(p)), boxes, ds / "images" / split, ds / "labels" / split, p.stem)
    # full-frame copies of the held-out frames for the base vs adapted comparison
    full = ds / "fullval"
    (full / "images" / "val").mkdir(parents=True)
    (full / "labels" / "val").mkdir(parents=True)
    for p in reviewed[cut:]:
        img = cv2.imread(str(p)); H, W = img.shape[:2]
        shutil.copy(p, full / "images" / "val" / p.name)
        rows = [f"{CLASSES.index(b['cls'])} {(b['box'][0] + b['box'][2]) / 2 / W:.6f} {(b['box'][1] + b['box'][3]) / 2 / H:.6f} "
                f"{(b['box'][2] - b['box'][0]) / W:.6f} {(b['box'][3] - b['box'][1]) / H:.6f}"
                for b in json.loads((d / "labels" / f"{p.stem}.json").read_text())["boxes"]]
        (full / "labels" / "val" / f"{p.stem}.txt").write_text("\n".join(rows))
    (full / "data.yaml").write_text(yaml.safe_dump({"path": str(full), "train": "images/val", "val": "images/val", "names": dict(enumerate(CLASSES))}))
    # replay of broadcast tiles to avoid forgetting
    if (SN_TILES / "images" / "train").is_dir():
        pool = sorted((SN_TILES / "images" / "train").glob("*.jpg"))
        random.Random(seed).shuffle(pool)
        for p in pool[:replay_tiles]:
            os.symlink(p, ds / "images" / "train" / f"sn_{p.name}")
            os.symlink(SN_TILES / "labels" / "train" / f"{p.stem}.txt", ds / "labels" / "train" / f"sn_{p.stem}.txt")
    (ds / "data.yaml").write_text(yaml.safe_dump({"path": str(ds), "train": "images/train", "val": "images/val", "names": dict(enumerate(CLASSES))}))
    return ds / "data.yaml"


def status(project: str) -> dict:
    d = _proj(project)
    f = d / "status.json"
    return json.loads(f.read_text()) if f.is_file() else {"state": "idle"}


def start_training(project: str, base: str, epochs: int = 20) -> dict:
    d = _proj(project)
    data = build_dataset(project)
    log = d / "train.log"
    cmd = [sys.executable, "-m", "soccer_analytics.adapt", "train", project, base, str(data), str(epochs)]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONPATH": str(ROOT)})
    st = {"state": "training", "pid": proc.pid, "started": time.time(), "base": base, "epochs": epochs}
    (d / "status.json").write_text(json.dumps(st))
    return st


def _train_and_evaluate(project: str, base: str, data: str, epochs: int) -> None:
    from ultralytics import YOLO
    d = _proj(project)
    st = json.loads((d / "status.json").read_text()) if (d / "status.json").is_file() else {}
    try:
        YOLO(base).train(data=data, imgsz=640, epochs=epochs, batch=16, device=0, workers=3, amp=True, lr0=0.002,
                         lrf=0.2, warmup_epochs=1, cos_lr=True, close_mosaic=3, scale=0.3, plots=False,
                         project=str(d), name="run", exist_ok=True)
        best = d / "run" / "weights" / "best.pt"
        full = d / "dataset" / "fullval" / "data.yaml"
        res = {}
        for name, w in (("base", base), ("adapted", str(best))):
            r = YOLO(w).val(data=str(full), split="val", imgsz=1920 if "p2" in Path(w).as_posix() or name == "adapted" else 1280,
                            batch=2, plots=False, verbose=False, project=str(d), name=f"val_{name}", exist_ok=True)
            ball = {int(c): i for i, c in enumerate(r.box.ap_class_index)}.get(0)
            res[name] = {"map50_95": float(r.box.map), "map50": float(r.box.map50),
                         "ball_recall": float(r.box.r[ball]) if ball is not None else None, "weights": w}
        res["recommend"] = "adapted" if res["adapted"]["map50_95"] > res["base"]["map50_95"] else "base"
        st.update(state="done", finished=time.time(), results=res)
    except Exception as exc:                       # surface the failure in the UI
        st.update(state="failed", error=str(exc))
    (d / "status.json").write_text(json.dumps(st))


if __name__ == "__main__":
    if len(sys.argv) >= 6 and sys.argv[1] == "train":
        _train_and_evaluate(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
