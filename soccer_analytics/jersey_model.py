"""Jersey-number recognition: a trained reader plus per-track voting.

A generic OCR engine (the earlier EasyOCR reader) is built for printed text,
not for 15-pixel, motion-blurred, folded shirt numbers. This reader is trained
on SoccerNet Jersey-2023 (Giancola et al.), 1,400 player tracklets from
broadcast football:

* backbone: ResNet-18 (ImageNet initialisation), input 128 x 64 crops;
* three heads: *legible* (is a number readable in this crop), *tens* digit
  (none, 1..9) and *units* digit (0..9). Digit heads let rare numbers borrow
  from common ones (Gerke, Mueller & Schaefer 2015 use the same factorisation);
* labels are per tracklet, so a crop where the number is turned away still
  carries the number. The legibility head and the track-level vote absorb
  that noise: a crop only votes in proportion to how legible it looks, and a
  track answers "unknown" when no crop is legible enough (Koshkina & Elder
  2024 show that legibility filtering is the largest single gain).

``TrackVote`` keeps the running evidence for each track id so the live
pipeline and the offline review use the same decision rule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

H, W = 128, 64
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "jersey_r18.pt"


def build_net():
    import torch.nn as nn
    import torchvision

    class JerseyNet(nn.Module):
        def __init__(self):
            super().__init__()
            m = torchvision.models.resnet18(weights=None)
            m.fc = nn.Identity()
            self.backbone = m
            self.drop = nn.Dropout(0.2)
            self.legible = nn.Linear(512, 1)
            self.tens = nn.Linear(512, 10)        # 0 = no tens digit
            self.units = nn.Linear(512, 10)

        def forward(self, x):
            f = self.drop(self.backbone(x))
            return self.legible(f).squeeze(1), self.tens(f), self.units(f)

    return JerseyNet()


# Torso band of a player box: the shirt number sits in the upper half of the
# body; cropping it roughly doubles the pixels per digit at a fixed input size.
TORSO = (0.05, 0.60, 0.05, 0.95)          # y0, y1, x0, x1 as fractions of the box
TH, TW = 96, 64


def torso(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    y0, y1, x0, x1 = TORSO
    return img[int(y0 * h):max(int(y1 * h), int(y0 * h) + 2), int(x0 * w):max(int(x1 * w), int(x0 * w) + 2)]


def build_mil():
    """Attention multiple-instance network (Ilse, Tomczak & Welling, ICML 2018).

    A track is a bag of crops with one label. Each crop is embedded, a gated
    attention scores how informative it is, and the heads read the weighted
    mean. Crops where the number is turned away get low weight instead of
    being trained on a label they do not show."""
    import torch
    import torch.nn as nn
    import torchvision

    class JerseyMIL(nn.Module):
        def __init__(self):
            super().__init__()
            m = torchvision.models.resnet18(weights=None)
            m.fc = nn.Identity()
            self.backbone = m
            self.att_v = nn.Linear(512, 128)
            self.att_u = nn.Linear(512, 128)
            self.att_w = nn.Linear(128, 1)
            self.drop = nn.Dropout(0.2)
            self.legible = nn.Linear(512, 1)
            self.tens = nn.Linear(512, 10)
            self.units = nn.Linear(512, 10)

        def embed(self, x):                        # (n,3,H,W) -> (n,512)
            return self.backbone(x)

        def pool(self, f, mask):                   # f (B,K,512), mask (B,K) bool
            a = self.att_w(torch.tanh(self.att_v(f)) * torch.sigmoid(self.att_u(f))).squeeze(-1)
            a = a.masked_fill(~mask, -1e4)
            w = torch.softmax(a, dim=1)
            z = self.drop((w.unsqueeze(-1) * f).sum(1))
            return self.legible(z).squeeze(-1), self.tens(z), self.units(z), w

        def forward(self, x, mask):                # x (B,K,3,H,W)
            B, K = x.shape[:2]
            f = self.embed(x.flatten(0, 1)).view(B, K, -1)
            return self.pool(f, mask)

    return JerseyMIL()


def preprocess(crops: Sequence[np.ndarray]) -> np.ndarray:
    """BGR uint8 crops (any size) -> (n,3,H,W) float32 normalised RGB."""
    import cv2
    out = np.empty((len(crops), 3, H, W), np.float32)
    for i, c in enumerate(crops):
        x = cv2.resize(c, (W, H), interpolation=cv2.INTER_AREA if c.shape[0] > H else cv2.INTER_CUBIC)
        x = (x[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD
        out[i] = x.transpose(2, 0, 1)
    return out


def number_probs(tens_logits: np.ndarray, units_logits: np.ndarray) -> np.ndarray:
    """(n,10),(n,10) logits -> (n,100) probability of numbers 0..99."""
    def sm(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)
    pt, pu = sm(tens_logits), sm(units_logits)
    P = (pt[:, :, None] * pu[:, None, :]).reshape(len(pt), 100)   # index = 10*tens + units
    return P


class TrackVote:
    """Accumulate crop evidence per track; decide a number or abstain."""

    def __init__(self, min_legible: float = 0.6, min_frames: int = 2, min_share: float = 0.5):
        self.ev: Dict[int, np.ndarray] = {}
        self.leg: Dict[int, List[float]] = {}
        self.min_legible, self.min_frames, self.min_share = min_legible, min_frames, min_share

    def add(self, track: int, p_legible: np.ndarray, P: np.ndarray) -> None:
        w = np.clip((np.asarray(p_legible) - 0.5) * 2, 0, 1) ** 2      # only legible-looking crops vote
        self.ev[track] = self.ev.get(track, np.zeros(100)) + (w[:, None] * P).sum(0)
        self.leg.setdefault(track, []).extend(np.asarray(p_legible).tolist())

    def decide(self, track: int) -> Tuple[Optional[int], float]:
        leg = np.asarray(self.leg.get(track, []))
        e = self.ev.get(track)
        if e is None or (leg >= self.min_legible).sum() < self.min_frames:
            return None, 0.0
        share = float(e.max() / max(e.sum(), 1e-9))
        n = int(e.argmax())
        return (n if share >= self.min_share else None), share

    def resolve(self) -> Dict[int, int]:
        out = {}
        for t in self.ev:
            n, _ = self.decide(t)
            if n is not None:
                out[t] = n
        return out


class MILVote:
    """Per-track bag of crop embeddings, pooled by the MIL attention."""

    def __init__(self, reader: "JerseyReader", keep: int = 64):
        self.reader, self.keep = reader, keep
        self.bags: Dict[int, List[np.ndarray]] = {}

    def add(self, track: int, feats: np.ndarray, k: int) -> None:
        bag = self.bags.setdefault(track, [])
        if len(bag) < self.keep:
            bag.append(feats[k])
        else:                                  # reservoir: keep a spread over the whole match
            j = np.random.randint(0, len(bag) + 1)
            if j < self.keep:
                bag[j] = feats[k]

    def decide(self, track: int) -> Tuple[Optional[int], float]:
        bag = self.bags.get(track)
        if not bag or len(bag) < 3:
            return None, 0.0
        p_leg, P = self.reader.pool(np.stack(bag))
        n = int(P.argmax())
        return (n if p_leg >= self.reader.threshold else None), float(P.max() * p_leg)


class CropVote(TrackVote):
    """TrackVote with the same add signature as MILVote."""

    def add(self, track: int, out, k: int) -> None:          # type: ignore[override]
        leg, P = out
        super().add(track, leg[k:k + 1], P[k:k + 1])


class JerseyReader:
    """Shirt-number reader. Prefers the attention-MIL model when present.

    ``out = read_batch(frame, boxes)`` then ``vote.add(track, out, k)`` for the
    k-th box, and ``vote.decide(track)``, with ``vote = reader.vote()``."""

    def __init__(self, weights: Optional[str] = None, device: str = ""):
        self.available = False
        self.arch = "crop"
        self.threshold = 0.5
        mil = DEFAULT_WEIGHTS.with_name("jersey_mil_r18.pt")
        path = Path(weights) if weights else (mil if mil.is_file() else DEFAULT_WEIGHTS)
        if not path.is_file():
            return
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(path, map_location="cpu")
        self.arch = state.get("arch", "crop") if isinstance(state, dict) else "crop"
        self.net = build_mil() if self.arch == "mil" else build_net()
        self.net.load_state_dict(state["model"] if "model" in state else state)
        self.threshold = float(state.get("threshold", 0.5)) if isinstance(state, dict) else 0.5
        self.net.eval().to(self.device)
        self.available = True

    def vote(self):
        return MILVote(self) if self.arch == "mil" else CropVote()

    def _crops(self, frame: np.ndarray, boxes: np.ndarray):
        crops = []
        h, w = frame.shape[:2]
        for x1, y1, x2, y2 in np.asarray(boxes, float).reshape(-1, 4):
            x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
            crops.append(frame[y1:y2, x1:x2] if x2 - x1 >= 6 and y2 - y1 >= 12 else np.zeros((H, W, 3), np.uint8))
        return crops

    def read_batch(self, frame: np.ndarray, boxes: np.ndarray):
        crops = self._crops(frame, boxes)
        if self.arch == "mil":
            if not crops:
                return np.zeros((0, 512), np.float32)
            import cv2
            x = np.stack([((cv2.resize(torso(c), (TW, TH), interpolation=cv2.INTER_CUBIC)[:, :, ::-1].astype(np.float32) / 255.0
                            - MEAN) / STD).transpose(2, 0, 1) for c in crops])
            with self.torch.no_grad():
                return self.net.embed(self.torch.from_numpy(x).to(self.device)).float().cpu().numpy()
        if not crops:
            return np.zeros(0), np.zeros((0, 100))
        with self.torch.no_grad():
            x = self.torch.from_numpy(preprocess(crops)).to(self.device)
            leg, t, u = self.net(x)
        return self.torch.sigmoid(leg).cpu().numpy(), number_probs(t.cpu().numpy(), u.cpu().numpy())

    def pool(self, feats: np.ndarray) -> Tuple[float, np.ndarray]:
        """MIL only: (p_legible, P over 0..99) for a bag of crop embeddings."""
        t = self.torch.from_numpy(np.asarray(feats, np.float32))[None].to(self.device)
        with self.torch.no_grad():
            leg, tt, uu, _ = self.net.pool(t, self.torch.ones(t.shape[:2], dtype=self.torch.bool, device=self.device))
        return float(self.torch.sigmoid(leg)[0]), number_probs(tt.cpu().numpy(), uu.cpu().numpy())[0]
