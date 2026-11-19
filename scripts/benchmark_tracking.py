"""Reproducible tracking/ball benchmark on SoccerNet-Tracking sequences.

Step 1 caches detector output once per sequence (boxes, classes, scores, a
colour-histogram embedding per person box and the masked camera-motion warp),
so every tracker is compared on *identical* detections (guide EXP-P1).

Step 2 runs trackers on the cache and scores them with soccer_analytics.mot_eval
(HOTA/IDF1/MOTA) and the ball path with ball_tracking.ball_path_metrics.

    conda activate soccer
    PYTHONPATH=. python scripts/benchmark_tracking.py cache --split test --limit 12
    PYTHONPATH=. python scripts/benchmark_tracking.py run --split test --limit 12

Tune on ``--split train`` (or the 9 held-out val sequences) and report test once.
"""

from __future__ import annotations

import argparse
import configparser
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOCCERNET = Path.home() / "SoccerNet" / "tracking"
CACHE = ROOT / "data" / "benchmarks" / "dets"
VAL_SEQS = ["SNMOT-071", "SNMOT-076", "SNMOT-099", "SNMOT-102", "SNMOT-112",
            "SNMOT-115", "SNMOT-165", "SNMOT-166", "SNMOT-169"]


def sequences(split: str, limit: int):
    if split == "val":
        seqs = [SOCCERNET / "train" / s for s in VAL_SEQS]
    else:
        seqs = sorted((SOCCERNET / split).iterdir())
        if split == "train":
            seqs = [s for s in seqs if s.name not in VAL_SEQS]
    return seqs[:limit] if limit else seqs


def gt_for(seq: Path):
    from soccer_analytics.mot_eval import read_mot
    info = configparser.ConfigParser()
    info.read(seq / "gameinfo.ini")
    ball_ids, roles = set(), {}
    for key, value in info["Sequence"].items():
        if key.startswith("trackletid_"):
            tid = int(key.split("_")[1])
            roles[tid] = value.strip()
            if value.strip().startswith("ball"):
                ball_ids.add(tid)
    people = read_mot(seq / "gt" / "gt.txt", keep_ids=[t for t in roles if t not in ball_ids])
    ball = read_mot(seq / "gt" / "gt.txt", keep_ids=ball_ids) if ball_ids else {}
    ball_centres = {f - 1: (b[0, 0] + b[0, 2] / 2, b[0, 1] + b[0, 3] / 2) for f, (_, b) in ball.items()}
    return people, ball_centres, roles


def cache(args):
    import cv2
    from soccer_analytics.tracker import Detector
    from soccer_analytics.imm_tracker import CameraMotion, appearance_embedding
    from soccer_analytics.core import BALL

    CACHE.mkdir(parents=True, exist_ok=True)
    det = Detector(args.weights, imgsz=args.imgsz, device="0", person_conf=0.05,
                   ball_conf=0.03, half=True, track_low_conf=0.05, remap_goalkeeper=True)
    from soccer_analytics.reid_model import ReIDEncoder
    enc = ReIDEncoder(device="cuda")
    tag = Path(args.weights).parts[-3]
    for seq in sequences(args.split, args.limit):
        out = CACHE / f"{seq.name}_{tag}_{args.imgsz}.npz"
        if out.exists() and not args.force:
            if not enc.available or "reid" in np.load(out).files:
                print(f"[cache] {out.name} exists")
                continue
        images = sorted((seq / "img1").glob("*.jpg"))
        cam = CameraMotion()
        rows, warps, embs, reids = [], [], [], []
        t0 = time.time()
        for f, path in enumerate(images):
            frame = cv2.imread(str(path))
            d = det.detect(frame)
            people = d.class_id != BALL
            emb = np.zeros((len(d), 64), np.float32)
            if people.any():
                emb[people] = appearance_embedding(frame, d.xyxy[people])
            warps.append(cam.update(frame, d.xyxy[people]))
            rid = np.zeros((len(d), 512), np.float16)
            if enc.available and people.any():
                rid[people] = enc(frame, d.xyxy[people]).astype(np.float16)
            reids.append(rid)
            for k in range(len(d)):
                rows.append([f, *d.xyxy[k], d.class_id[k], d.confidence[k]])
            embs.append(emb)
        np.savez_compressed(out, dets=np.asarray(rows, np.float32).reshape(-1, 7),
                            emb=np.concatenate(embs) if embs else np.zeros((0, 64)),
                            warps=np.asarray(warps, np.float32), n=len(images),
                            reid=np.concatenate(reids) if reids else np.zeros((0, 512), np.float16))
        print(f"[cache] {seq.name}: {len(images)} frames, {len(rows)} dets, {time.time() - t0:.1f}s")


def load_cache(seq: Path, tag: str, appearance: str = "hist"):
    data = np.load(CACHE / f"{seq.name}_{tag}.npz")
    emb = data["emb"]
    if appearance == "reid":
        if "reid" not in data.files:
            raise FileNotFoundError("cache has no ReID embeddings")
        emb = data["reid"].astype(np.float32)
    return data["dets"], emb, data["warps"], int(data["n"])


def to_xywh(boxes):
    b = np.asarray(boxes, float).reshape(-1, 4)
    return np.column_stack((b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]))


def run_bytetrack(dets, n, **_):
    from soccer_analytics.tracker import SoccerTracker
    from soccer_analytics.core import Detections, BALL
    tr = SoccerTracker(frame_rate=25, lost_track_buffer=90, track_activation_threshold=0.3,
                       backend="bytetrack", track_low_conf=0.1)
    out = {}
    for f in range(n):
        d = dets[(dets[:, 0] == f) & (dets[:, 5] != BALL)]
        r = tr.update(Detections(d[:, 1:5], d[:, 5].astype(int), d[:, 6]))
        out[f] = [(int(i), b) for i, b in zip(r.tracker_id, r.xyxy)]
    return out


def run_imm(dets, n, emb=None, warps=None, **kw):
    from soccer_analytics.imm_tracker import IMMTracker
    from soccer_analytics.core import BALL
    tr = IMMTracker(**kw)
    out = {}
    for f in range(n):
        m = (dets[:, 0] == f) & (dets[:, 5] != BALL)
        d = dets[m]
        e = emb[m] if emb is not None else None
        w = warps[f] if warps is not None else None
        res = tr.update(d[:, 1:5], d[:, 6], e, w)
        out[f] = [(tid, box) for tid, box, _ in res]
    return out, tr


def det_embeddings(per_frame, dets, emb):
    """Map each output box back to its detection's embedding (output = detection box)."""
    out = {}
    for f, items in per_frame.items():
        m = dets[:, 0] == f
        boxes, em = dets[m][:, 1:5], emb[m]
        for tid, b in items:
            if len(boxes) == 0:
                continue
            k = int(np.argmin(np.abs(boxes - b).sum(1)))
            if np.abs(boxes[k] - b).sum() < 1e-3:
                out.setdefault(tid, {})[f] = em[k]
    return out


def link(per_frame, dets, emb, **kw):
    from soccer_analytics.tracklet_link import TrackletLinker, tracklets_from_frames
    tracklets = tracklets_from_frames(per_frame, emb=det_embeddings(per_frame, dets, emb) if emb is not None else None)
    res = TrackletLinker(**kw).link(tracklets)
    return {f: list(zip(ids, boxes)) for f, (ids, boxes, _) in res.frames.items()}, res


def to_eval(per_frame):
    out = {}
    for f, items in per_frame.items():
        if items:
            out[f + 1] = (np.asarray([i for i, _ in items], int), to_xywh([b for _, b in items]))
    return out


def ball_eval(dets, n, gt_ball, **kw):
    from soccer_analytics.ball_tracking import select_ball_path, ball_path_metrics, BallPoint
    from soccer_analytics.core import BALL
    balls = dets[dets[:, 5] == BALL]
    cands, top1 = {}, []
    for f in range(n):
        b = balls[balls[:, 0] == f]
        if len(b):
            c = np.column_stack(((b[:, 1] + b[:, 3]) / 2, (b[:, 2] + b[:, 4]) / 2, b[:, 6]))
            cands[f] = c
            k = int(np.argmax(c[:, 2]))
            if c[k, 2] >= 0.1:                  # previous production rule: best box >= 0.1
                top1.append(BallPoint(f, c[k, 0], c[k, 1], c[k, 2], False, 0))
    path = select_ball_path(cands, n, **kw)
    return ball_path_metrics(top1, gt_ball, n), ball_path_metrics(path, gt_ball, n)


def run(args):
    from soccer_analytics.mot_eval import evaluate, combine
    tag = f"{Path(args.weights).parts[-3]}_{args.imgsz}"
    # Operating points were selected on the 9 validation clips (docs/TRACKING_BENCHMARK.md).
    variants = {
        "bytetrack (production)": (lambda d, e, r, w, n: run_bytetrack(d, n), None),
        "imm-eiou+cmc+hist": (lambda d, e, r, w, n: run_imm(d, n, e, w, use_imm=False)[0],
                              dict(max_gap=60, motion_gate=0.6, app_gate=0.15, interpolate_gap=20)),
        "imm-eiou+cmc+reid": (lambda d, e, r, w, n: run_imm(d, n, r, w, use_imm=False, app_veto=0.65, app_weight=0.5)[0],
                              dict(max_gap=750, motion_gate=1.2, app_gate=0.5, interpolate_gap=20)),
        "imm(2-mode)+eiou+cmc+reid": (lambda d, e, r, w, n: run_imm(d, n, r, w, use_imm=True, app_veto=0.65, app_weight=0.5)[0],
                                      dict(max_gap=750, motion_gate=1.2, app_gate=0.5, interpolate_gap=20)),
    }
    if args.only:
        variants = {k: v for k, v in variants.items() if any(o in k for o in args.only.split(","))}
    results = {k: [] for k in variants}
    results.update({k + " +link": [] for k, (_, lk) in variants.items() if lk})
    ball_rows = {"top1>=0.1 (production)": [], "dp-path": []}
    timing = {k: [] for k in variants}
    for seq in sequences(args.split, args.limit):
        try:
            dets, emb, warps, n = load_cache(seq, tag)
            reid = load_cache(seq, tag, "reid")[1]
        except FileNotFoundError:
            print(f"[skip] no cache for {seq.name}")
            continue
        gt, gt_ball, _ = gt_for(seq)
        for name, (fn, link_kw) in variants.items():
            t0 = time.perf_counter()
            pf = fn(dets, emb, reid, warps, n)
            timing[name].append((time.perf_counter() - t0) / n * 1000)
            results[name].append(evaluate(gt, to_eval(pf)))
            if link_kw:
                linked, _ = link(pf, dets, reid if "reid" in name else emb, **link_kw)
                results[name + " +link"].append(evaluate(gt, to_eval(linked)))
        b0, b1 = ball_eval(dets, n, gt_ball)
        ball_rows["top1>=0.1 (production)"].append(b0)
        ball_rows["dp-path"].append(b1)
        print(f"[run] {seq.name}: " + ", ".join(
            f"{k.split(' ')[0]}={results[k][-1]['HOTA']:.3f}" for k in list(results)[:2]), flush=True)

    summary = {"split": args.split, "detector": tag, "sequences": len(results[next(iter(variants))]),
               "tracking": {k: combine(v) for k, v in results.items() if v},
               "tracker_ms_per_frame": {k: float(np.mean(v)) for k, v in timing.items() if v},
               "ball": {k: {m: float(np.mean([r[m] for r in v])) for m in ("precision", "recall", "f1")}
                        for k, v in ball_rows.items() if v}}
    out = ROOT / "data" / "benchmarks" / f"tracking_{args.split}_{tag}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("cache", "run"))
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--weights", default=str(ROOT / "runs/detect/soccernet_v2/weights/best.pt"))
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--force", action="store_true")
    p.add_argument("--only", default="")
    args = p.parse_args()
    cache(args) if args.cmd == "cache" else run(args)


if __name__ == "__main__":
    main()
