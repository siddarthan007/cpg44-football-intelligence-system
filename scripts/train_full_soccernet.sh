#!/usr/bin/env bash
# Full High-Accuracy YOLOv8 Fine-Tuning on RTX 5060 GPU (8GB VRAM)
# Uses SoccerNet_YOLO dataset with 1280px resolution and AMP mixed precision
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate soccer
cd /home/siddartha/capstone

DATASET="/home/siddartha/SoccerNet_YOLO/data.yaml"
BASE_WEIGHTS="runs/detect/soccernet_v2/weights/best.pt"

if [ ! -f "$DATASET" ]; then
    echo "[-] Dataset not found at $DATASET"
    exit 1
fi

echo "======================================================="
echo " Starting YOLOv8 Fine-Tuning for High-Accuracy Weights"
echo " Dataset: $DATASET"
echo " Base Weights: $BASE_WEIGHTS"
echo " Target Resolution: 1280 px (Small ball & far player detection)"
echo " Device: CUDA:0 (NVIDIA RTX 5060)"
echo "======================================================="

# Run fine-tuning with 1280 resolution, batch 4, and mixed precision
python -m soccer_analytics.train finetune \
    --data "$DATASET" \
    --base "$BASE_WEIGHTS" \
    --imgsz 1280 \
    --epochs 30 \
    --stage1-epochs 10 \
    --batch 4 \
    --device 0 \
    --name soccernet_v3

echo "======================================================="
echo " Fine-Tuning Complete!"
echo " Best Weights Saved To: runs/detect/soccernet_v3/weights/best.pt"
echo "======================================================="
