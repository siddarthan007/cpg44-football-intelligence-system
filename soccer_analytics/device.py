"""GPU / device management.

Central place to resolve the compute device and inference precision so every
stage (detection, tracking, future injury models) uses the GPU consistently.
Tuned for the target hardware: RTX 5060 Laptop (Blackwell, 8 GB VRAM).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DeviceInfo:
    device: str          # "cuda:0" or "cpu"
    name: str
    total_vram_gb: float
    capability: str      # e.g. "12.0" for Blackwell
    half_ok: bool        # safe to use FP16 inference


def resolve_device(prefer: str = "") -> DeviceInfo:
    """Pick the compute device. ``prefer`` may be "", "cpu", "cuda", "cuda:0", "0"."""
    want_cpu = prefer.lower() == "cpu"
    if not want_cpu and torch.cuda.is_available():
        idx = 0
        if prefer and prefer not in ("cuda", ""):
            idx = int(prefer.replace("cuda:", "").replace("cuda", "") or 0)
        props = torch.cuda.get_device_properties(idx)
        cap = f"{props.major}.{props.minor}"
        vram = props.total_memory / (1024 ** 3)
        return DeviceInfo(f"cuda:{idx}", props.name, round(vram, 1), cap, half_ok=True)
    return DeviceInfo("cpu", "cpu", 0.0, "-", half_ok=False)


def recommend_batch_imgsz(info: DeviceInfo, task: str = "train") -> dict:
    """Conservative starting points, not a guarantee for arbitrary checkpoints.

    Keep 1280 for small balls, reduce physical batch first, and measure peak
    allocated AND reserved memory before increasing it. Gradient accumulation
    retains a larger effective batch. AutoBatch is an explicit opt-in in CLI.
    """
    if info.device == "cpu":
        return {"imgsz": 640, "batch": 4, "model": "yolov8n.pt", "amp": False}
    v = info.total_vram_gb
    if task == "train":
        if v < 9:       # RTX 5060 8 GB
            return {"imgsz": 1280, "batch": 2, "model": "yolov8m.pt", "amp": True}
        if v < 17:      # 12-16 GB (e.g. 4070/4080)
            return {"imgsz": 1280, "batch": 10, "model": "yolov8m.pt", "amp": True}
        return {"imgsz": 1280, "batch": 16, "model": "yolov8l.pt", "amp": True}  # 24 GB+
    # inference
    return {"imgsz": 1280 if v >= 9 else 960, "batch": 1, "half": info.half_ok}


def describe() -> str:
    info = resolve_device()
    if info.device == "cpu":
        return "Compute: CPU (no CUDA GPU visible)"
    return (f"Compute: {info.name} | {info.total_vram_gb} GB | "
            f"CC {info.capability} | device {info.device}")
