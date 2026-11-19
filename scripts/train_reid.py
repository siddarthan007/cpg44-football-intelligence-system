"""Train the football ReID embedding on SoccerNet-Tracking training clips.

    conda activate soccer
    PYTHONPATH=. python scripts/train_reid.py build     # crops -> data/reid/*.npy (memmap)
    PYTHONPATH=. python scripts/train_reid.py train     # -> runs/reid/soccer_reid_r18.pt
    PYTHONPATH=. python scripts/train_reid.py eval      # within-clip retrieval on held-out val clips

Splits: the 9 validation clips used by scripts/benchmark_tracking.py are
excluded from training; the official test clips are never read here.
Identities are only meaningful *within* a clip (the same player in another
clip gets a different label), so batches draw several identities from the
same clip and retrieval is evaluated per clip: query = first half of the clip,
gallery = second half.
"""

from __future__ import annotations

import argparse
import configparser
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SN = Path.home() / "SoccerNet" / "tracking"
OUT = ROOT / "data" / "reid"
VAL = ["SNMOT-071", "SNMOT-076", "SNMOT-099", "SNMOT-102", "SNMOT-112",
       "SNMOT-115", "SNMOT-165", "SNMOT-166", "SNMOT-169"]


def _ball_ids(seq: Path):
    info = configparser.ConfigParser()
    info.read(seq / "gameinfo.ini")
    return {int(k.split("_")[1]) for k, v in info["Sequence"].items()
            if k.startswith("trackletid_") and v.strip().startswith("ball")}


def build(args):
    import cv2
    from soccer_analytics.reid_model import crop_boxes, INPUT_HW
    OUT.mkdir(parents=True, exist_ok=True)
    for split, names in (("train", [s.name for s in sorted((SN / "train").iterdir()) if s.name not in VAL]),
                         ("val", VAL)):
        stride = args.stride if split == "train" else 10
        rows = []
        for si, name in enumerate(names):
            seq = SN / "train" / name
            gt = np.loadtxt(seq / "gt" / "gt.txt", delimiter=",", ndmin=2)
            balls = _ball_ids(seq)
            gt = gt[~np.isin(gt[:, 1], list(balls))]
            gt = gt[(gt[:, 5] >= 24) & (gt[:, 4] >= 8)]          # skip tiny/cut boxes
            for f in np.unique(gt[:, 0]).astype(int)[::stride]:
                rows.append((si, f, gt[gt[:, 0] == f]))
        n = sum(len(r[2]) for r in rows)
        crops = np.lib.format.open_memmap(OUT / f"{split}_crops.npy", mode="w+", dtype=np.uint8,
                                          shape=(n, INPUT_HW[0], INPUT_HW[1], 3))
        meta = np.zeros((n, 3), np.int32)                         # seq, track id, frame
        k = 0
        for si, f, boxes in rows:
            frame = cv2.imread(str(SN / "train" / names[si] / "img1" / f"{f:06d}.jpg"))
            xyxy = np.column_stack((boxes[:, 2], boxes[:, 3], boxes[:, 2] + boxes[:, 4], boxes[:, 3] + boxes[:, 5]))
            crops[k:k + len(boxes)] = crop_boxes(frame, xyxy)
            meta[k:k + len(boxes)] = np.column_stack((np.full(len(boxes), si), boxes[:, 1], np.full(len(boxes), f)))
            k += len(boxes)
        crops.flush()
        np.save(OUT / f"{split}_meta.npy", meta)
        (OUT / f"{split}_sequences.json").write_text(json.dumps(names))
        print(f"[build] {split}: {n} crops from {len(names)} clips")


class Sampler:
    """Yield index batches of P identities x K crops, identities from 2 clips."""

    def __init__(self, meta, P=16, K=4, seed=0):
        self.rng = np.random.default_rng(seed)
        self.P, self.K = P, K
        labels = meta[:, 0].astype(np.int64) * 10000 + meta[:, 1]
        self.labels = labels
        self.by_label = {}
        for i, l in enumerate(labels):
            self.by_label.setdefault(int(l), []).append(i)
        self.by_seq = {}
        for l, idx in self.by_label.items():
            if len(idx) >= 2:
                self.by_seq.setdefault(l // 10000, []).append(l)

    def batches(self, n):
        seqs = list(self.by_seq)
        for _ in range(n):
            chosen = []
            for s in self.rng.choice(seqs, 2, replace=False):
                ids = self.by_seq[s]
                take = min(len(ids), self.P // 2)
                chosen += list(self.rng.choice(ids, take, replace=False))
            idx = []
            for l in chosen:
                pool = self.by_label[l]
                idx += list(self.rng.choice(pool, self.K, replace=len(pool) < self.K))
            yield np.asarray(idx)


def train(args):
    import torch
    import torch.nn.functional as F
    from soccer_analytics.reid_model import build_model, preprocess
    crops = np.load(OUT / "train_crops.npy", mmap_mode="r")
    meta = np.load(OUT / "train_meta.npy")
    sampler = Sampler(meta, P=args.P, K=args.K)
    uniq = {l: i for i, l in enumerate(sorted(set(sampler.labels.tolist())))}
    y_all = np.asarray([uniq[int(l)] for l in sampler.labels])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(uniq)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3.5e-4, weight_decay=5e-4)
    total = args.iters
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3.5e-4, total_steps=total, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    rng = np.random.default_rng(1)
    t0 = time.time()
    model.train()
    for it, idx in enumerate(sampler.batches(total)):
        x = np.asarray(crops[np.sort(idx)])
        order = np.argsort(np.argsort(idx))                      # restore sampler order
        x = x[order]
        # augmentation: flip, small translation, random erasing, brightness
        flip = rng.random(len(x)) < 0.5
        x[flip] = x[flip, :, ::-1]
        x = preprocess(x)
        for i in range(len(x)):
            dy, dx = rng.integers(-8, 9), rng.integers(-4, 5)
            x[i] = np.roll(x[i], (dy, dx), axis=(1, 2))
            if rng.random() < 0.5:
                h, w = rng.integers(16, 48), rng.integers(8, 24)
                y0, x0 = rng.integers(0, 128 - h), rng.integers(0, 64 - w)
                x[i, :, y0:y0 + h, x0:x0 + w] = rng.normal(0, 1, (3, h, w))
            x[i] *= rng.uniform(0.85, 1.15)
        xb = torch.from_numpy(x).to(dev)
        yb = torch.from_numpy(y_all[idx]).to(dev)
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            feat, logits = model(xb)
            ce = F.cross_entropy(logits, yb, label_smoothing=0.1)
        feat = feat.float()
        d = torch.cdist(feat, feat)
        same = yb[:, None] == yb[None]
        hardest_pos = (d * same).max(1).values
        hardest_neg = (d + 1e5 * same).min(1).values
        tri = F.relu(hardest_pos - hardest_neg + 0.3).mean()
        loss = ce + tri
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if it % 200 == 0:
            print(f"[train] it {it}/{total} ce {ce.item():.3f} tri {tri.item():.3f} "
                  f"{time.time() - t0:.0f}s", flush=True)
    out = ROOT / "runs" / "reid"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "input_hw": [128, 64], "arch": "resnet18-bnneck",
                "train_clips": json.loads((OUT / "train_sequences.json").read_text()),
                "excluded_val_clips": VAL, "iters": total, "batch": args.P * args.K},
               out / "soccer_reid_r18.pt")
    print(f"[train] saved {out / 'soccer_reid_r18.pt'}")


def _retrieval(emb, meta):
    """Within-clip mAP / rank-1: queries from the first half of the clip."""
    aps, r1 = [], []
    for s in np.unique(meta[:, 0]):
        m = meta[:, 0] == s
        e, ids, fr = emb[m], meta[m, 1], meta[m, 2]
        mid = np.median(fr)
        q, g = fr < mid, fr >= mid
        if not q.any() or not g.any():
            continue
        sim = e[q] @ e[g].T
        for i, qid in enumerate(ids[q]):
            rel = ids[g] == qid
            if not rel.any():
                continue
            order = np.argsort(-sim[i])
            hits = rel[order]
            r1.append(float(hits[0]))
            prec = np.cumsum(hits) / (np.arange(len(hits)) + 1)
            aps.append(float((prec * hits).sum() / hits.sum()))
    return {"mAP": float(np.mean(aps)), "rank1": float(np.mean(r1)), "queries": len(aps)}


def evaluate(args):
    from soccer_analytics.reid_model import ReIDEncoder
    from soccer_analytics.imm_tracker import appearance_embedding
    crops = np.load(OUT / "val_crops.npy", mmap_mode="r")
    meta = np.load(OUT / "val_meta.npy")
    x = np.asarray(crops)
    hist = np.stack([appearance_embedding(c, np.array([[0, 0, 64, 128]]))[0] for c in x])
    res = {"colour_histogram": _retrieval(hist, meta)}
    enc = ReIDEncoder()
    if enc.available:
        res["reid_r18"] = _retrieval(enc.embed_crops(x), meta)
    print(json.dumps(res, indent=2))
    (ROOT / "data" / "benchmarks").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "benchmarks" / "reid_val.json").write_text(json.dumps(res, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("build", "train", "eval"))
    p.add_argument("--stride", type=int, default=15)
    p.add_argument("--iters", type=int, default=6000)
    p.add_argument("--P", type=int, default=16)
    p.add_argument("--K", type=int, default=4)
    a = p.parse_args()
    {"build": build, "train": train, "eval": evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
