"""YOLOv8 detection training + fine-tuning.

Two workflows:

1. **base**     — train the SoccerNet detector (from a COCO-pretrained checkpoint).
2. **finetune** — adapt that detector to YOUR university-level footage, which is
                  lower resolution / lower quality than SoccerNet. Uses two-stage
                  transfer learning (freeze backbone → unfreeze at low LR) plus
                  heavier photometric augmentation so the model generalises across
                  the domain gap (compression, blur, colour, camera differences).

Requires ``ultralytics`` (and ``albumentations`` for the low-quality photometric
augmentations — Ultralytics auto-applies Blur/CLAHE/ImageCompression when it is
installed).

CLI:
    python -m soccer_analytics.train base     --data /path/soccernet/data.yaml
    python -m soccer_analytics.train finetune --data /path/custom/data.yaml \
        --base runs/detect/soccernet/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from ultralytics import YOLO

from .device import resolve_device, recommend_batch_imgsz


def _load_yolo(weights: str):
    return YOLO(weights)


# Low-quality-footage photometric augmentations. Applied only if albumentations is
# available AND the installed Ultralytics accepts the `augmentations=` kwarg
# (newer releases); otherwise Ultralytics' own default Albumentations pipeline
# (Blur/MedianBlur/CLAHE/ImageCompression at low p) still kicks in automatically.
def _low_quality_transforms():
    try:
        import albumentations as A
    except ImportError:
        return None
    try:
        comp = A.ImageCompression(quality_range=(35, 90), p=0.5)
    except Exception:
        comp = A.ImageCompression(quality_lower=35, quality_upper=90, p=0.5)
    return [
        comp,                                                           # jpeg/stream artefacts
        A.Blur(blur_limit=5, p=0.3),
        A.MotionBlur(blur_limit=5, p=0.3),                              # camera / player motion
        A.RandomBrightnessContrast(p=0.3),
        A.CLAHE(clip_limit=3.0, p=0.2),
    ]


def train_base(
    data: str,
    model: str = "yolov8m.pt",
    epochs: int = 100,
    imgsz: int = 1280,
    batch: int = 8,
    device: str = "",
    project: Optional[str] = None,   # None → ultralytics default runs/detect/<name>
    name: str = "soccernet",
    **overrides,
):
    """Train the SoccerNet detector. imgsz≥1280 matters — the ball is ~12 px wide.

    ``batch`` may be an integer count, ``-1`` (Ultralytics auto-batch, ~60% VRAM),
    or a fraction in (0,1) → auto-batch targeting that VRAM fraction."""
    dev = resolve_device(device)
    print("[train]", dev.name, dev.device)
    batch = int(batch) if batch >= 1 else batch     # keep fraction/-1 as float
    yolo = _load_yolo(model)
    args = dict(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=dev.device,
        amp=dev.half_ok,      # mixed precision → fits 8 GB, ~1.5× faster
        name=name,
        exist_ok=True,
        patience=25,          # early stop
        # ball is tiny & rare -> keep mosaic, close it near the end for clean boxes
        mosaic=1.0,
        close_mosaic=10,
        # mild default photometric jitter
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    )
    if project:
        args["project"] = project
    args.update(overrides)
    res = yolo.train(**args)
    print(f"[train] weights: {getattr(res, 'save_dir', '?')}/weights/best.pt")
    return res


def finetune(
    data: str,
    base: str,
    epochs: int = 60,
    imgsz: int = 1280,
    batch: int = 8,
    device: str = "",
    project: Optional[str] = None,   # None → ultralytics default runs/detect/<name>
    name: str = "custom_finetune",
    freeze: int = 10,
    lr0: float = 0.005,
    two_stage: bool = True,
    stage1_epochs: int = 0,          # 0 → 1/3 of total epochs (min 1)
    low_quality_aug: bool = True,
    **overrides,
):
    """Fine-tune the SoccerNet detector on custom (lower-quality) footage.

    two_stage=True: Stage 1 freezes the backbone (``freeze`` layers) and trains the
    head/neck so it adapts to the new domain without wrecking pretrained features;
    Stage 2 unfreezes everything at a lower LR. This is the recommended recipe for a
    small custom dataset with a domain gap.
    """
    aug = _low_quality_transforms() if low_quality_aug else None
    dev = resolve_device(device)
    print("[finetune]", dev.name, dev.device)
    batch = int(batch) if batch >= 1 else batch     # keep fraction/-1 as float

    # augmentation profile geared to a resolution/quality domain gap
    common = dict(
        data=data, imgsz=imgsz, batch=batch, device=dev.device, amp=dev.half_ok,
        exist_ok=True, patience=20,
        hsv_h=0.02, hsv_s=0.7, hsv_v=0.5,   # stronger colour jitter for camera diff
        scale=0.5, translate=0.1, fliplr=0.5,
        mosaic=1.0, close_mosaic=10,
    )
    if project:
        common["project"] = project
    if aug is not None:
        common["augmentations"] = aug  # ignored gracefully by older ultralytics below
    common.update(overrides)

    if not two_stage:
        yolo = _load_yolo(base)
        return _safe_train(yolo, dict(common, epochs=epochs, freeze=freeze, lr0=lr0, name=name))

    if stage1_epochs <= 0:
        stage1_epochs = max(1, epochs // 3)     # stage 1 = 1/3 of the budget

    # ---- Stage 1: freeze backbone, adapt head/neck ----
    yolo = _load_yolo(base)
    s1 = _safe_train(
        yolo,
        dict(common, epochs=stage1_epochs, freeze=freeze, lr0=lr0, name=f"{name}_s1"),
    )
    # derive stage-1 weights from the actual save_dir (robust to ultralytics paths)
    stage1_best = Path(getattr(s1, "save_dir", f"runs/detect/{name}_s1")) / "weights" / "best.pt"

    # ---- Stage 2: unfreeze all, low LR ----
    yolo2 = _load_yolo(str(stage1_best) if stage1_best.is_file() else base)
    s2 = _safe_train(
        yolo2,
        dict(common, epochs=max(epochs - stage1_epochs, 1), lr0=lr0 / 5, name=f"{name}_s2"),
    )
    return s2


def _safe_train(yolo, args: dict):
    """Call yolo.train, dropping the `augmentations=` kwarg if the installed
    Ultralytics version does not support it (keeps compatibility across releases)."""
    try:
        return yolo.train(**args)
    except TypeError as e:
        if "augmentations" in str(e) and "augmentations" in args:
            args.pop("augmentations")
            print("[train] this ultralytics version has no `augmentations=` kwarg; "
                  "relying on its built-in Albumentations defaults instead.")
            return yolo.train(**args)
        raise


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Train / fine-tune the SoccerNet YOLOv8 detector.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("base", help="Train on SoccerNet.")
    b.add_argument("--data", required=True, help="Path to SoccerNet data.yaml.")
    b.add_argument("--model", default="yolov8m.pt")
    b.add_argument("--epochs", type=int, default=100)
    b.add_argument("--imgsz", type=int, default=1280)
    b.add_argument("--batch", type=float, default=4,
                   help="int count (default 4 fits 8 GB @1280), -1 (auto, cloud GPUs "
                        "only — OOMs on 8 GB), or a 0-1 VRAM fraction")
    b.add_argument("--device", default="")
    b.add_argument("--name", default="soccernet")
    b.add_argument("--auto", action="store_true",
                   help="Pick model/imgsz/batch to fill the detected GPU VRAM "
                        "(recommended: yolov8m@1280, auto-batch ~80%% of 8 GB).")

    f = sub.add_parser("finetune", help="Fine-tune on custom lower-quality footage.")
    f.add_argument("--data", required=True, help="Path to custom data.yaml.")
    f.add_argument("--base", required=True, help="SoccerNet best.pt to start from.")
    f.add_argument("--epochs", type=int, default=60)
    f.add_argument("--imgsz", type=int, default=1280)
    f.add_argument("--batch", type=float, default=4,
                   help="int count (default 4 fits 8 GB @1280), -1 (auto), or a 0-1 fraction")
    f.add_argument("--device", default="")
    f.add_argument("--name", default="custom_finetune")
    f.add_argument("--freeze", type=int, default=10)
    f.add_argument("--lr0", type=float, default=0.005)
    f.add_argument("--stage1-epochs", type=int, default=0, help="Epochs for stage 1 (0 -> 1/3 of total)")
    f.add_argument("--single-stage", action="store_true", help="Skip the freeze/unfreeze two-stage recipe.")
    f.add_argument("--no-lq-aug", action="store_true", help="Disable low-quality photometric augmentation.")

    args = p.parse_args(argv)
    if args.cmd == "base":
        model, imgsz, batch = args.model, args.imgsz, args.batch
        if args.auto:
            rec = recommend_batch_imgsz(resolve_device(args.device), task="train")
            model, imgsz, batch = rec["model"], rec["imgsz"], rec["batch"]
            print(f"[train] --auto → model={model} imgsz={imgsz} batch={batch}")
        train_base(args.data, model=model, epochs=args.epochs, imgsz=imgsz,
                   batch=batch, device=args.device, name=args.name)
    elif args.cmd == "finetune":
        finetune(args.data, base=args.base, epochs=args.epochs, imgsz=args.imgsz,
                 batch=args.batch, device=args.device, name=args.name, freeze=args.freeze,
                 lr0=args.lr0, stage1_epochs=args.stage1_epochs, two_stage=not args.single_stage,
                 low_quality_aug=not args.no_lq_aug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
