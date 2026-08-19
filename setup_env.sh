#!/usr/bin/env bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate soccer
echo "=== python: $(python --version) ==="
echo "=== [1/3] torch cu128 (Blackwell sm_120) ==="
pip install --no-input torch torchvision --index-url https://download.pytorch.org/whl/cu128
echo "=== [2/3] vision + ml stack ==="
pip install --no-input ultralytics supervision opencv-python scikit-learn scipy pandas matplotlib albumentations pyyaml tqdm rich
echo "=== [3/3] wearable/injury + dashboard extras ==="
pip install --no-input xgboost pyserial fastapi "uvicorn[standard]" websockets
echo "=== verify torch cuda ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
    x=torch.randn(1024,1024,device="cuda"); print("cuda matmul ok", float((x@x).sum().abs()>0))
PY
echo "SETUP_DONE_OK"
