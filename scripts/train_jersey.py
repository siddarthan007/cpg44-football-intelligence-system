"""Train and evaluate the jersey-number reader on SoccerNet Jersey-2023.

    PYTHONPATH=. python scripts/train_jersey.py train      # ~25 min on an 8 GB laptop GPU
    PYTHONPATH=. python scripts/train_jersey.py eval       # tracklet accuracy on the test split

Model selection uses 10 % of the *training* tracklets; the test split is
scored once, by ``eval``, and written to data/benchmarks/jersey_test.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from soccer_analytics.jersey_model import DEFAULT_WEIGHTS, H, W, MEAN, STD, TrackVote, build_net, number_probs  # noqa: E402

DATA = Path.home() / "SoccerNet" / "jersey-2023"


def tracklets(split: str):
    gt = json.loads((DATA / split / f"{split}_gt.json").read_text())
    root = DATA / split / "images"
    out = []
    for tid, num in gt.items():
        files = sorted((root / tid).glob("*.jpg"))
        if files:
            out.append((tid, int(num), files))
    return out


class CropSet:
    def __init__(self, items, per_track: int, train: bool, seed: int = 0):
        import torch
        self.torch = torch
        rng = random.Random(seed)
        self.rows = []
        for tid, num, files in items:
            pick = files if len(files) <= per_track else rng.sample(files, per_track)
            self.rows += [(f, num) for f in pick]
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import cv2
        f, num = self.rows[i]
        img = cv2.imread(str(f))
        if self.train:
            h, w = img.shape[:2]
            # scale/shift jitter, no horizontal flip (a flipped 2 is not a 2)
            s = random.uniform(0.9, 1.1)
            M = np.float32([[s, 0, random.uniform(-0.06, 0.06) * w], [0, s, random.uniform(-0.05, 0.05) * h]])
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            if random.random() < 0.3:
                k = random.choice((3, 5))
                img = cv2.GaussianBlur(img, (k, k), 0)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + random.uniform(-12, 12)) % 180     # kit colours vary
            hsv[..., 1:] *= np.float32([random.uniform(0.7, 1.3), random.uniform(0.7, 1.3)])
            img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        x = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
        x = ((x[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
        leg = float(num >= 0)
        tens = (num // 10) if num >= 10 else 0
        units = num % 10 if num >= 0 else -100
        tens = tens if num >= 0 else -100
        return self.torch.from_numpy(np.ascontiguousarray(x)), leg, tens, units


def train(a):
    import torch
    import torch.nn.functional as F
    import torchvision
    items = tracklets("train")
    random.Random(0).shuffle(items)
    nval = len(items) // 10
    val, tr = items[:nval], items[nval:]
    print(f"[jersey] {len(tr)} train / {len(val)} val tracklets", flush=True)
    net = build_net()
    sd = torchvision.models.resnet18(weights="IMAGENET1K_V1").state_dict()
    net.backbone.load_state_dict({k: v for k, v in sd.items() if not k.startswith("fc.")}, strict=False)
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(tr) * a.per_track // a.batch + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    best = -1.0
    for ep in range(a.epochs):
        ds = CropSet(tr, a.per_track, True, seed=ep)
        dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                                         drop_last=True, persistent_workers=False)
        net.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, leg, tens, units in dl:
            x, leg, tens, units = x.to(dev, non_blocking=True), leg.float().to(dev), tens.to(dev), units.to(dev)
            with torch.autocast(dev, enabled=dev == "cuda"):
                pl, pt, pu = net(x)
                loss = F.binary_cross_entropy_with_logits(pl, leg) + \
                    F.cross_entropy(pt, tens, ignore_index=-100, label_smoothing=0.05) + \
                    F.cross_entropy(pu, units, ignore_index=-100, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(x); n += len(x)
        acc = evaluate(net, val, dev, a.eval_per_track)["accuracy"]
        print(f"[jersey] epoch {ep + 1}/{a.epochs} loss {tot / max(n, 1):.3f} val tracklet acc {acc:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if acc > best:
            best = acc
            DEFAULT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": net.state_dict(), "val_accuracy": acc, "epoch": ep + 1}, DEFAULT_WEIGHTS)
    print(f"[jersey] best val tracklet accuracy {best:.3f} -> {DEFAULT_WEIGHTS}", flush=True)


def evaluate(net, items, dev, per_track: int = 0, vote_kw=None):
    """Tracklet accuracy, the Jersey-2023 challenge metric (-1 counts as a class)."""
    import torch
    net.eval()
    vote = TrackVote(**(vote_kw or {}))
    correct = 0
    by = {"legible_correct": 0, "legible": 0, "illegible_correct": 0, "illegible": 0}
    for k, (tid, num, files) in enumerate(items):
        pick = files if not per_track or len(files) <= per_track else files[:: max(1, len(files) // per_track)]
        ds = CropSet([(tid, num, pick)], 10 ** 9, False)
        xs = torch.stack([ds[i][0] for i in range(len(ds))])
        with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
            pl, pt, pu = net(xs.to(dev))
        vote.add(k, torch.sigmoid(pl.float()).cpu().numpy(), number_probs(pt.float().cpu().numpy(), pu.float().cpu().numpy()))
        pred, _ = vote.decide(k)
        pred = -1 if pred is None else pred
        ok = pred == num
        correct += ok
        key = "legible" if num >= 0 else "illegible"
        by[key] += 1
        by[key + "_correct"] += ok
    return {"accuracy": correct / max(len(items), 1), "tracklets": len(items),
            "legible_accuracy": by["legible_correct"] / max(by["legible"], 1),
            "illegible_accuracy": by["illegible_correct"] / max(by["illegible"], 1),
            "legible": by["legible"], "illegible": by["illegible"]}


def evaluate_test(a):
    import torch
    net = build_net()
    state = torch.load(DEFAULT_WEIGHTS, map_location="cpu")
    net.load_state_dict(state["model"])
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    net.to(dev)
    # decision thresholds were chosen on the validation tracklets, not here
    res = evaluate(net, tracklets("test"), dev, a.eval_per_track)
    res.update({"model": "ResNet-18, legible/tens/units heads, track vote", "val_accuracy": state.get("val_accuracy"),
                "per_track_crops": a.eval_per_track or "all", "dataset": "SoccerNet Jersey-2023 test"})
    out = ROOT / "data" / "benchmarks" / "jersey_test.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("train", "eval"))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--per-track", type=int, default=64, help="crops sampled per tracklet per epoch")
    p.add_argument("--eval-per-track", type=int, default=0, help="0 = every crop")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    train(a) if a.cmd == "train" else evaluate_test(a)


if __name__ == "__main__":
    main()
