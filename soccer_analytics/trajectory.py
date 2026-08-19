"""LSTM player-trajectory prediction (CPG44 Objective 3 / DL course / ref [8] Honda).

Predicts each player's future path from their recent trajectory — the sequential
deep-learning component the report calls for ("LSTM networks for analysing
sequential multimodal data", and Honda et al. [8], who fuse video with LSTM
player trajectories for pass-receiver prediction).

Design: translation-invariant. The LSTM ingests the last K velocity steps
(Δposition, normalised pitch units) and directly predicts the next H velocities;
positions are recovered by integrating from the last observed point. Trains on
the SoccerNet Tracking ground-truth tracklets (which we already have), reporting
ADE/FDE (average / final displacement error) — the standard trajectory metrics.

    python -m soccer_analytics.trajectory train --src ~/SoccerNet/tracking/train --epochs 15
    # → runs/trajectory/traj_lstm.pt  (+ ADE/FDE on a held-out split)
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

K_IN, H_OUT = 15, 10          # observe 15 frames, predict 10
IMW, IMH = 1920.0, 1080.0     # SoccerNet frame size for normalisation


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class TrajectoryLSTM(nn.Module):
    """Encode K velocity steps → predict H future velocity steps."""

    def __init__(self, hidden: int = 64, layers: int = 1, horizon: int = H_OUT):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(2, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, horizon * 2)

    def forward(self, vel_seq: torch.Tensor) -> torch.Tensor:
        # vel_seq: (B, K, 2) → (B, H, 2) predicted velocities
        _, (h, _) = self.lstm(vel_seq)
        return self.head(h[-1]).view(-1, self.horizon, 2)

    @torch.no_grad()
    def predict_positions(self, recent_xy: np.ndarray) -> np.ndarray:
        """recent_xy: (>=K+1, 2) normalised positions → (H, 2) future positions."""
        self.eval()
        xy = np.asarray(recent_xy, dtype=np.float32)
        vel = np.diff(xy[-(K_IN + 1):], axis=0)          # (K, 2)
        if len(vel) < K_IN:
            vel = np.pad(vel, ((K_IN - len(vel), 0), (0, 0)))
        t = torch.from_numpy(vel[None])                   # (1, K, 2)
        pred_vel = self(t)[0].cpu().numpy()               # (H, 2)
        return xy[-1] + np.cumsum(pred_vel, axis=0)


# --------------------------------------------------------------------------- #
# dataset from SoccerNet gt
# --------------------------------------------------------------------------- #
def _tracklet_positions(gt_path: str) -> List[np.ndarray]:
    """Return per-id arrays of (frame-ordered) normalised centre positions."""
    by_id: dict = {}
    with open(gt_path, "r", errors="replace") as fh:
        for line in fh:
            p = line.split(",")
            if len(p) < 6:
                continue
            try:
                fr, tid = int(float(p[0])), int(float(p[1]))
                l, t, w, h = (float(p[i]) for i in range(2, 6))
            except ValueError:
                continue
            cx, cy = (l + w / 2) / IMW, (t + h / 2) / IMH
            by_id.setdefault(tid, []).append((fr, cx, cy))
    out = []
    for tid, rows in by_id.items():
        rows.sort()
        out.append(np.array([[r[1], r[2]] for r in rows], dtype=np.float32))
    return out


def build_windows(src: str, max_seqs: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide (K_IN input velocities, H_OUT target positions, last position) windows
    over every tracklet in every sequence under ``src``."""
    gts = sorted(glob.glob(os.path.join(src, "*", "gt", "gt.txt")))
    if max_seqs:
        gts = gts[:max_seqs]
    Xv, Yp, P0 = [], [], []
    for gt in gts:
        for pos in _tracklet_positions(gt):
            n = len(pos)
            if n < K_IN + H_OUT + 1:
                continue
            vel = np.diff(pos, axis=0)                    # (n-1, 2)
            for i in range(0, n - K_IN - H_OUT - 1, 3):   # stride 3
                Xv.append(vel[i:i + K_IN])
                last = pos[i + K_IN]
                Yp.append(pos[i + K_IN + 1:i + K_IN + 1 + H_OUT] - last)  # future offsets
                P0.append(last)
    if not Xv:
        raise RuntimeError(f"no trajectory windows built from {src}")
    return np.asarray(Xv, np.float32), np.asarray(Yp, np.float32), np.asarray(P0, np.float32)


# --------------------------------------------------------------------------- #
# train / eval
# --------------------------------------------------------------------------- #
def _ade_fde(pred_off: np.ndarray, true_off: np.ndarray) -> Tuple[float, float]:
    # de-normalise to metres-ish via pitch scale (approx: full frame ≈ pitch), in px
    d = np.linalg.norm(pred_off - true_off, axis=-1)      # (N, H) normalised units
    scale = math.hypot(IMW, IMH)                          # normalised → pixels
    return float(d.mean() * scale), float(d[:, -1].mean() * scale)


def train(src: str, epochs: int = 15, batch: int = 256, hidden: int = 64,
          device: str = "", max_seqs: int = 0, out: str = "runs/trajectory/traj_lstm.pt"):
    from .device import resolve_device
    dev = resolve_device(device).device
    print(f"[traj] building windows from {src} …")
    Xv, Yp, P0 = build_windows(src, max_seqs=max_seqs)
    print(f"[traj] {len(Xv)} windows | device {dev}")

    # deterministic split (no RNG needed)
    n = len(Xv)
    idx = np.arange(n)
    cut = int(n * 0.85)
    tr, va = idx[:cut], idx[cut:]

    model = TrajectoryLSTM(hidden=hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    Xv_t = torch.from_numpy(Xv).to(dev)
    Yp_t = torch.from_numpy(Yp.reshape(n, -1)).to(dev)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr))
        tot = 0.0
        for i in range(0, len(tr), batch):
            b = torch.as_tensor(tr[perm[i:i + batch].numpy()])
            xb, yb = Xv_t[b], Yp_t[b]
            opt.zero_grad()
            # integrate predicted velocities → offsets for a position-space loss
            off = torch.cumsum(model(xb), dim=1).reshape(len(b), -1)
            loss = lossf(off, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        # eval
        model.eval()
        with torch.no_grad():
            pv = model(Xv_t[torch.as_tensor(va)])
            off = torch.cumsum(pv, dim=1).cpu().numpy()
        ade, fde = _ade_fde(off, Yp[va])
        print(f"[traj] epoch {ep+1}/{epochs} loss {tot/len(tr):.5f} | ADE {ade:.1f}px FDE {fde:.1f}px")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "hidden": hidden,
                "K_IN": K_IN, "H_OUT": H_OUT}, out)
    print(f"[traj] saved → {out}  (ADE {ade:.1f}px, FDE {fde:.1f}px on holdout)")
    return model


def load(path: str, device: str = "cpu") -> TrajectoryLSTM:
    ckpt = torch.load(path, map_location=device)
    m = TrajectoryLSTM(hidden=ckpt.get("hidden", 64))
    m.load_state_dict(ckpt["state"])
    m.eval()
    return m


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="LSTM player-trajectory prediction.")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--src", required=True, help="SoccerNet tracking train/ dir")
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--max-seqs", type=int, default=0)
    t.add_argument("--device", default="")
    t.add_argument("--out", default="runs/trajectory/traj_lstm.pt")
    a = p.parse_args(argv)
    if a.cmd == "train":
        train(a.src, epochs=a.epochs, max_seqs=a.max_seqs, device=a.device, out=a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
