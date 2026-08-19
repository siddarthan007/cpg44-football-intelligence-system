#!/usr/bin/env bash
# =============================================================================
# Full SoccerNet detector training on RunPod (cloud GPU).
#
# Use a RunPod **PyTorch 2.x** pod (torch preinstalled) with an A100 40 GB / A40 /
# RTX 4090. Open a terminal in /workspace and run:
#
#     bash runpod_train.sh
#
# Outputs the trained detector to
#     /workspace/capstone/runs/detect/soccernet_full/weights/best.pt
# Download it back to the laptop with `runpodctl` or the RunPod file browser, then
# fine-tune locally on campus footage.
# =============================================================================
set -e
WS=/workspace
DATA_ROOT=${DATA_ROOT:-$WS/SoccerNet/tracking}      # where train/ test/ live
MODEL=${MODEL:-yolov8m.pt}                          # m fits A100; use l/x on bigger cards
IMGSZ=${IMGSZ:-1280}
BATCH=${BATCH:-24}                                  # A100 40 GB @1280; lower for 4090 (~10)
EPOCHS=${EPOCHS:-100}
STRIDE=${STRIDE:-3}                                 # every 3rd frame → ~14k imgs

cd $WS
# --- code ---
if [ ! -d capstone ]; then
  echo "Upload/clone the capstone repo to $WS/capstone first (runpodctl send, or git clone)."
  exit 1
fi
cd capstone

# --- deps (torch already in the RunPod template) ---
pip install -q ultralytics supervision opencv-python-headless scikit-learn \
    xgboost pandas scipy matplotlib pyyaml tqdm

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),
'-', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# --- dataset ---
if [ ! -d "$DATA_ROOT/train" ]; then
  echo "SoccerNet tracking not found at $DATA_ROOT."
  echo "Either upload it, or download with your NDA password:"
  echo '  pip install SoccerNet'
  echo '  python -c "from SoccerNet.Downloader import SoccerNetDownloader as D; d=D(\"'$WS'/SoccerNet\"); d.password=input(\"pw: \"); d.downloadDataTask(task=\"tracking\", split=[\"train\",\"test\"])"'
  exit 1
fi

echo "== converting SoccerNet → YOLO =="
python soccernet_to_yolo.py --src "$DATA_ROOT" --dst "$WS/SoccerNet/yolo" \
    --stride "$STRIDE" --image-mode symlink

echo "== training $MODEL @${IMGSZ} batch ${BATCH} for ${EPOCHS} epochs =="
python -m soccer_analytics.train base \
    --data "$WS/SoccerNet/yolo/data.yaml" \
    --model "$MODEL" --imgsz "$IMGSZ" --batch "$BATCH" --epochs "$EPOCHS" \
    --name soccernet_full

echo "DONE → runs/detect/soccernet_full/weights/best.pt"
echo "Download it, then locally:"
echo "  python -m soccer_analytics.train finetune --data ~/campus/data.yaml --base best.pt"
