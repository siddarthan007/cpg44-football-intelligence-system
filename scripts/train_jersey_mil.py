"""Train the attention-MIL jersey reader on SoccerNet Jersey-2023.

    PYTHONPATH=. python scripts/train_jersey_mil.py train
    PYTHONPATH=. python scripts/train_jersey_mil.py eval

Why MIL: labels are per tracklet, and in most crops of a tracklet the number
is not visible. Training a crop classifier on the tracklet label (the first
reader, ``train_jersey.py``) taught it mostly wrong labels: 77 % on held-out
training tracklets but 39.6 % on the test split. Here each training example
is a bag of crops from one tracklet with that tracklet's label, and the
network learns which crops to trust (Ilse et al. 2018).

Validation tracklets are held out from the training games, so validation is
optimistic (the same players appear in other tracklets); only the test split,
scored once by ``eval``, is reported.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from soccer_analytics.jersey_model import MEAN, STD, TH, TW, build_mil, number_probs, torso  # noqa: E402

DATA = Path.home() / "SoccerNet" / "jersey-2023"
ERASE = os.environ.get("JERSEY_ERASE", "0") == "1"
OUT = Path(os.environ.get("JERSEY_MIL_OUT") or ROOT / "models" / "jersey_mil_r18.pt")


def tracklets(split):
    gt = json.loads((DATA / split / f"{split}_gt.json").read_text())
    out = []
    for tid, num in gt.items():
        files = sorted((DATA / split / "images" / tid).glob("*.jpg"))
        if files:
            out.append((tid, int(num), files))
    return out


def load(path, train):
    img = cv2.imread(str(path))
    if train:
        h, w = img.shape[:2]
        s = random.uniform(0.92, 1.08)
        M = np.float32([[s, 0, random.uniform(-0.05, 0.05) * w], [0, s, random.uniform(-0.04, 0.04) * h]])
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if random.random() < 0.3:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-15, 15)) % 180          # kit colours vary
        hsv[..., 1:] *= np.float32([random.uniform(0.7, 1.3), random.uniform(0.7, 1.3)])
        img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    x = cv2.resize(torso(img), (TW, TH), interpolation=cv2.INTER_CUBIC)
    x = ((x[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
    if train and ERASE and random.random() < 0.3:          # random erasing (Zhong et al. 2020)
        h, w = TH // random.randint(3, 6), TW // random.randint(3, 6)
        y0, x0 = random.randint(0, TH - h), random.randint(0, TW - w)
        x[:, y0:y0 + h, x0:x0 + w] = 0.0
    return x


class Bags:
    def __init__(self, items, k, train):
        import torch
        self.torch, self.items, self.k, self.train = torch, items, k, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        tid, num, files = self.items[i]
        pick = random.choices(files, k=self.k) if len(files) < self.k else random.sample(files, self.k)
        x = np.stack([load(f, self.train) for f in pick])
        tens = (num // 10 if num >= 10 else 0) if num >= 0 else -100
        units = num % 10 if num >= 0 else -100
        return self.torch.from_numpy(x), float(num >= 0), tens, units


def run_eval(net, items, dev, per_track=48, thr=0.5):
    import torch
    net.eval()
    rows = []
    for tid, num, files in items:
        pick = files if len(files) <= per_track else [files[int(j)] for j in np.linspace(0, len(files) - 1, per_track)]
        x = torch.from_numpy(np.stack([load(f, False) for f in pick]))[None].to(dev)
        with torch.no_grad(), torch.autocast("cuda", enabled=dev == "cuda"):
            pl, pt, pu, _ = net(x, torch.ones(1, len(pick), dtype=torch.bool, device=dev))
        P = number_probs(pt.float().cpu().numpy(), pu.float().cpu().numpy())[0]
        rows.append((num, float(torch.sigmoid(pl.float())[0]), int(P.argmax()), float(P.max())))
    return rows


def score(rows, thr):
    ok = leg_ok = leg = ill_ok = ill = 0
    for num, pleg, n, _ in rows:
        pred = n if pleg >= thr else -1
        ok += pred == num
        if num >= 0:
            leg += 1; leg_ok += pred == num
        else:
            ill += 1; ill_ok += pred == -1
    return {"accuracy": ok / max(len(rows), 1), "legible_accuracy": leg_ok / max(leg, 1),
            "illegible_accuracy": ill_ok / max(ill, 1), "tracklets": len(rows), "legible": leg, "illegible": ill}


def train(a):
    import torch
    import torch.nn.functional as F
    import torchvision
    items = tracklets("train")
    random.Random(0).shuffle(items)
    nval = len(items) // 10
    val, tr = items[:nval], items[nval:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_mil()
    sd = torchvision.models.resnet18(weights="IMAGENET1K_V1").state_dict()
    net.backbone.load_state_dict({k: v for k, v in sd.items() if not k.startswith("fc.")}, strict=False)
    net.to(dev)
    dl = torch.utils.data.DataLoader(Bags(tr, a.k, True), batch_size=a.batch, shuffle=True, num_workers=a.workers,
                                     drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl), pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    best, best_thr = -1.0, 0.5
    for ep in range(a.epochs):
        net.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, leg, tens, units in dl:
            x, leg, tens, units = x.to(dev, non_blocking=True), leg.float().to(dev), tens.to(dev), units.to(dev)
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=dev)
            with torch.autocast(dev, enabled=dev == "cuda"):
                pl, pt, pu, _ = net(x, mask)
                loss = F.binary_cross_entropy_with_logits(pl.float(), leg) + \
                    F.cross_entropy(pt.float(), tens, ignore_index=-100, label_smoothing=0.05) + \
                    F.cross_entropy(pu.float(), units, ignore_index=-100, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(x); n += len(x)
        if (ep + 1) % 2 == 0 or ep + 1 == a.epochs:
            rows = run_eval(net, val, dev, per_track=32)
            thr, res = max(((t, score(rows, t)) for t in np.arange(0.3, 0.75, 0.05)), key=lambda r: r[1]["accuracy"])
            print(f"[mil] epoch {ep + 1}/{a.epochs} loss {tot / max(n, 1):.3f} val acc {res['accuracy']:.3f} "
                  f"(legible {res['legible_accuracy']:.3f}, thr {thr:.2f}) {time.time() - t0:.0f}s", flush=True)
            if res["accuracy"] > best:
                best, best_thr = res["accuracy"], float(thr)
                OUT.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": net.state_dict(), "arch": "mil", "threshold": best_thr, "val_accuracy": best,
                            "epoch": ep + 1, "input": [TH, TW], "torso": True}, OUT)
        else:
            print(f"[mil] epoch {ep + 1}/{a.epochs} loss {tot / max(n, 1):.3f} {time.time() - t0:.0f}s", flush=True)
    print(f"[mil] best val {best:.3f} at threshold {best_thr:.2f} -> {OUT}", flush=True)


def evaluate(a):
    import torch
    state = torch.load(OUT, map_location="cpu")
    net = build_mil()
    net.load_state_dict(state["model"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(dev)
    rows = run_eval(net, tracklets("test"), dev, per_track=a.per_track)
    res = score(rows, state["threshold"])
    res.update({"model": "ResNet-18 attention MIL, torso crops", "threshold_from_val": state["threshold"],
                "val_accuracy": state["val_accuracy"], "crops_per_track": a.per_track,
                "dataset": "SoccerNet Jersey-2023 test", "previous_crop_classifier": 0.396})
    out = Path(os.environ.get("JERSEY_RESULT") or ROOT / "data" / "benchmarks" / "jersey_test.json")
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("train", "eval"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=16, help="tracklets per step")
    p.add_argument("--k", type=int, default=16, help="crops per tracklet bag")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--per-track", type=int, default=48)
    a = p.parse_args()
    train(a) if a.cmd == "train" else evaluate(a)


if __name__ == "__main__":
    main()
