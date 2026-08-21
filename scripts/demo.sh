#!/usr/bin/env bash
# One-command prototype demo. Picks a DIFFERENT SoccerNet sequence each run (set
# SEQ=SNMOT-060 to pin it). Uses metric mode when a per-sequence calibration
# exists in demo/calib/<SEQ>.yaml, else pixel mode (tracking + teams + ball +
# possession still shown; metres/tactics/injury need a calibration).
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate soccer

# rotating sequence: random each run unless SEQ is set
SEQ=${SEQ:-$(ls "$HOME/SoccerNet/tracking/train" | shuf -n1)}
SRC="$HOME/SoccerNet/tracking/train/$SEQ/img1"
[ -d "$SRC" ] || { echo "no img1 for $SEQ at $SRC"; exit 1; }
CLIP=demo/sample_match.mp4
CAL=demo/calib/$SEQ.yaml
FRAMES=${FRAMES:-750}          # full sequence → both teams get the ball (balanced possession)
OUT=demo_out
mkdir -p "$OUT"

echo "[demo] sequence: $SEQ"

# 1) build the clip fresh each run (sequence changes)
python - "$SRC" "$CLIP" "$FRAMES" <<'PY'
import cv2, glob, sys
src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
fr = sorted(glob.glob(src + "/*.jpg"))[:n]
assert fr, f"no frames in {src}"
h, w = cv2.imread(fr[0]).shape[:2]
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
for f in fr:
    vw.write(cv2.imread(f))
vw.release()
print(f"[demo] built {out}: {len(fr)} frames {w}x{h}")
PY

# 2) detector — prefer the TensorRT engine (≈2× faster inference) when exported
WEIGHTS=$(ls -t runs/detect/soccernet_v*/weights/best.engine 2>/dev/null | head -1 || true)
[ -z "$WEIGHTS" ] && WEIGHTS=$(ls -t runs/detect/soccernet_v*/weights/best.pt runs/detect/*/weights/best.pt 2>/dev/null | head -1 || true)
[ -n "$WEIGHTS" ] || { echo "[demo] no trained model — run: python -m soccer_analytics.train base --data ~/SoccerNet/yolo/data.yaml --auto"; exit 1; }
echo "[demo] detector: $WEIGHTS"

# 3) calibration → metric or pixel mode
if [ -f "$CAL" ]; then
  CALARG="--calibration $CAL"; echo "[demo] METRIC mode (calibration $CAL)"
else
  CALARG=""; echo "[demo] PIXEL mode (no demo/calib/$SEQ.yaml) — calibrate for metric analytics"
fi

# LIVE by default: shows the dashboard in a window (needs WSLg on WSL). q=quit,
# space=pause. Set HEADLESS=1 to skip the window (server) — it always records too.
WINDOWARG=""; [ -n "$HEADLESS" ] && WINDOWARG="--no-window"
[ -z "$HEADLESS" ] && echo "[demo] LIVE window — q=quit, space=pause  (HEADLESS=1 to disable)"

python -m soccer_analytics.realtime \
  --video "$CLIP" --weights "$WEIGHTS" $CALARG \
  --imgsz "${IMGSZ:-960}" $WINDOWARG \
  --out "$OUT/dashboard.mp4" --stats "$OUT/stats.json" \
  --fused-out "$OUT/fused_snapshots.ndjson"

echo ""
echo "[demo] DONE ($SEQ) →  $OUT/dashboard.mp4  (recording) + $OUT/stats.json"
