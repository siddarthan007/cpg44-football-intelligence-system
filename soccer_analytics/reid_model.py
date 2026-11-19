"""Compact football player re-identification embedding.

Architecture follows the "Bag of Tricks" strong ReID baseline (Luo et al.,
CVPR-W 2019): ImageNet ResNet-18 trunk with last-stage stride 1, global average
pooling, and a BatchNorm neck whose output is the L2-normalised embedding.
It is trained by ``scripts/train_reid.py`` on SoccerNet-Tracking *training*
sequences with identity cross-entropy plus batch-hard triplet loss, sampling
identities from the same clip so the hard negatives are team-mates in the same
kit, which is exactly the confusion colour histograms cannot resolve.

``ReIDEncoder`` is the inference wrapper used by the tracker and the offline
linker. If the checkpoint is missing it reports ``available = False`` and the
caller falls back to colour histograms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

INPUT_HW = (128, 64)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "runs" / "reid" / "soccer_reid_r18.pt"


def build_model(num_classes: int = 0, pretrained: bool = True):
    import torch
    import torch.nn as nn
    import torchvision

    class ReIDNet(nn.Module):
        def __init__(self):
            super().__init__()
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            r = torchvision.models.resnet18(weights=weights)
            # last-stride 1 keeps a 8x4 map for 128x64 input (finer part cues)
            r.layer4[0].conv1.stride = (1, 1)
            r.layer4[0].downsample[0].stride = (1, 1)
            self.trunk = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool,
                                       r.layer1, r.layer2, r.layer3, r.layer4)
            self.neck = nn.BatchNorm1d(512)
            self.neck.bias.requires_grad_(False)
            self.classifier = nn.Linear(512, num_classes, bias=False) if num_classes else None

        def forward(self, x):
            f = self.trunk(x).mean(dim=(2, 3))       # global feature (triplet)
            b = self.neck(f)                          # BNNeck feature (ID loss / inference)
            if self.training and self.classifier is not None:
                return f, self.classifier(b)
            return nn.functional.normalize(b, dim=1)

    return ReIDNet()


def preprocess(crops_bgr: np.ndarray) -> np.ndarray:
    """(N,H,W,3) uint8 BGR at INPUT_HW -> (N,3,H,W) float32 normalised RGB."""
    x = crops_bgr[..., ::-1].astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def crop_boxes(frame, boxes_xyxy) -> np.ndarray:
    import cv2
    H, W = frame.shape[:2]
    out = np.zeros((len(boxes_xyxy), INPUT_HW[0], INPUT_HW[1], 3), np.uint8)
    for i, (x1, y1, x2, y2) in enumerate(np.asarray(boxes_xyxy, float).reshape(-1, 4)):
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(W, x2)), int(min(H, y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        out[i] = cv2.resize(frame[y1:y2, x1:x2], (INPUT_HW[1], INPUT_HW[0]), interpolation=cv2.INTER_LINEAR)
    return out


class ReIDEncoder:
    def __init__(self, weights: Optional[str] = None, device: str = "", half: bool = True):
        self.available = False
        path = Path(weights) if weights else DEFAULT_WEIGHTS
        if not path.is_file():
            return
        try:
            import torch
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            self.model = build_model(0, pretrained=False)
            state = {k: v for k, v in ckpt["model"].items() if not k.startswith("classifier")}
            self.model.load_state_dict(state)
            self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
            self.half = bool(half and self.device.type == "cuda")
            self.model.eval().to(self.device)
            if self.half:
                self.model.half()
            self.dim = 512
            self.available = True
            self.meta = {k: v for k, v in ckpt.items() if k != "model"}
        except Exception as exc:  # pragma: no cover - checkpoint/runtime issues
            print(f"[reid] encoder unavailable: {exc}")

    def __call__(self, frame, boxes_xyxy) -> np.ndarray:
        if not self.available or len(boxes_xyxy) == 0:
            return np.zeros((len(boxes_xyxy), 512), np.float32)
        return self.embed_crops(crop_boxes(frame, boxes_xyxy))

    def embed_crops(self, crops: np.ndarray, batch: int = 128) -> np.ndarray:
        import torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(crops), batch):
                x = torch.from_numpy(preprocess(crops[i:i + batch])).to(self.device)
                if self.half:
                    x = x.half()
                out.append(self.model(x).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)
