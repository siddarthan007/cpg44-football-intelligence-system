#!/usr/bin/env bash
# Build the review-workspace demo analyses from the bundled SoccerNet clips.
# Only clips with a measured pitch calibration are processed in metric mode.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
WEIGHTS=${CPG44_MODEL_PATH:-runs/detect/soccernet/weights/best.pt}
for clip in demo/clips/*_750f.mp4; do
  name=$(basename "$clip" _750f.mp4)
  calib="demo/calib/${name}.yaml"
  out="demo/analysis/$(echo "$name" | tr 'A-Z' 'a-z')"
  args=(--video "$clip" --weights "$WEIGHTS" --out "$out" --name "SoccerNet ${name}")
  [ -f "$calib" ] && args+=(--calibration "$calib")
  echo "[demo] $name -> $out"
  PYTHONPATH=. "$PY" -m soccer_analytics.match_analysis "${args[@]}"
done
