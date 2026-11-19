"""Subsample SoccerNet training sequences without using test frames for validation."""

import os
import json
import tempfile
from pathlib import Path
from ultralytics import YOLO
import yaml
from soccer_analytics.data_split import split_sequences

ROOT = Path(__file__).resolve().parents[1]


def build_fast_dataset(stride_train=8, stride_val=25, source=None):
    source = Path(source or ROOT.parent / "SoccerNet_YOLO").resolve()
    if min(stride_train, stride_val) < 1:
        raise ValueError("frame strides must be positive")
    train, val, manifest = split_sequences((source / "images/train").glob("*.jpg"))
    # Fresh output avoids deleting another run's prepared dataset.
    dataset_out = Path(tempfile.mkdtemp(prefix="cpg44-soccernet-"))
    for split, files, stride in (("train", train, stride_train), ("val", val, stride_val)):
        image_dir, label_dir = dataset_out / "images" / split, dataset_out / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for image in files[::stride]:
            label = source / "labels/train" / f"{image.stem}.txt"
            if not label.is_file():
                raise ValueError(f"missing annotation: {label}")
            os.symlink(image, image_dir / image.name)
            os.symlink(label, label_dir / label.name)
    names = yaml.safe_load((source / "data.yaml").read_text())["names"]
    (dataset_out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(dataset_out), "train": "images/train", "val": "images/val", "names": names,
    }), encoding="utf-8")
    (dataset_out / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ready] Sequence-disjoint dataset: {dataset_out}")
    return dataset_out


def run_training(
    base_weights="runs/detect/soccernet/weights/best.pt",
    epochs=25,
    batch=2,
    imgsz=1280,
    name="soccernet_apex",
):
    dataset_out = build_fast_dataset()

    base_path = Path(base_weights)
    if not base_path.is_file():
        base_path = "yolov8m.pt"
        print(f"[!] Base checkpoint not found, falling back to {base_path}")
    else:
        print(f"[+] Initializing from high-precision checkpoint: {base_path}")

    model = YOLO(str(base_path))

    print("\n=======================================================")
    print(" Starting YOLOv8 Apex Training on RTX 5060 GPU")
    print(f" Target Resolution: {imgsz}px | Batch: {batch} | Epochs: {epochs}")
    print(" Measure peak VRAM and validation metrics before promoting the checkpoint.")
    print(" Optimizations: AdamW, Cosine LR, High Box Gain (8.5), DFL (1.8), Mixed Precision AMP")
    print("=======================================================\n")

    results = model.train(
        data=str(dataset_out / "data.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,
        workers=2,
        cache=False,
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        cos_lr=True,
        box=8.5,             # Amplified bounding box loss gain (tight player silhouette)
        cls=0.6,             # Classification loss gain
        dfl=1.8,             # Distribution focal loss gain (sub-pixel boundary accuracy)
        mosaic=1.0,
        close_mosaic=8,      # Disable mosaic in last 8 epochs for clean box convergence
        amp=True,            # Mixed precision FP16 on CUDA
        name=name,
        exist_ok=True,
        patience=15,
        plots=False,
        verbose=True,
    )

    save_dir = getattr(results, "save_dir", f"runs/detect/{name}")
    print("\n=======================================================")
    print(" Apex Training Successfully Completed!")
    print(f" Best Weights: {save_dir}/weights/best.pt")
    print("=======================================================")


if __name__ == "__main__":
    run_training()
