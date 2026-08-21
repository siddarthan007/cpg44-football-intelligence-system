#!/usr/bin/env bash
# Reproducible software smoke test using only measured demo video. It does not
# generate wearable readings or injury labels.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(mktemp -d /tmp/cpg44-smoke.XXXXXX)"
cleanup() {
  if [[ "$SMOKE_DIR" == /tmp/cpg44-smoke.* && -d "$SMOKE_DIR" ]]; then
    rm -rf -- "$SMOKE_DIR"
  fi
}
trap cleanup EXIT

cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR/backend/src:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$SMOKE_DIR/matplotlib"

echo "[1/5] Python API, synchronization and analytics tests"
pytest -q tests

echo "[2/5] React typecheck"
npm run lint --prefix frontend

echo "[3/5] React production bundle"
npm run build --prefix frontend

echo "[4/5] Calibrated vision/ByteTrack execution on measured footage"
python -m soccer_analytics.realtime \
  --video demo/sample_match.mp4 \
  --weights yolov8n.pt \
  --calibration demo/calib/SNMOT-060.yaml \
  --imgsz 320 \
  --max-frames 3 \
  --no-window \
  --stats "$SMOKE_DIR/stats.json" \
  --fused-out "$SMOKE_DIR/fused.ndjson"

python - "$SMOKE_DIR/stats.json" "$SMOKE_DIR/fused.ndjson" <<'PY'
import json
import sys
from pathlib import Path

stats_path, fused_path = map(Path, sys.argv[1:])
stats = json.loads(stats_path.read_text(encoding="utf-8"))
rows = [json.loads(line) for line in fused_path.read_text(encoding="utf-8").splitlines()]
assert stats.get("metric") is True
assert rows and rows[0]["data_quality"]["metric_calibration"] is True
print("vision snapshot quality:", rows[0]["data_quality"])
PY

echo "[5/5] Source and script checks"
python -m compileall -q backend/src soccer_analytics tests
bash -n scripts/*.sh
git diff --check

echo "CPG44 software smoke test passed. ESP32, reference-sensor, CUDA and VPS checks remain physical deployment gates."
