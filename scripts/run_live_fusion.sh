#!/usr/bin/env bash
# Vision dashboard fused with the already-running sensor hub.
# Usage: bash scripts/run_live_fusion.sh path/to/video.mp4 path/to/weights.pt [calib.yaml]
set -euo pipefail
cd "$(dirname "$0")/.."
VIDEO="${1:?video path}"
WEIGHTS="${2:?weights path}"
CALIB="${3:-}"
extra=()
if [[ -n "$CALIB" ]]; then extra+=(--calibration "$CALIB"); fi
exec python -m soccer_analytics.realtime \
  --video "$VIDEO" --weights "$WEIGHTS" \
  --sensor-hub http://127.0.0.1:8081 --player-id 7 --roster "7:7" \
  "${extra[@]}"
