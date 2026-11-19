#!/usr/bin/env bash
# One-command React dashboard demo using real footage and the relay-backed stack.
# Supports selecting any SoccerNet sequence, custom video file, webcam, or calibration.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate soccer 2>/dev/null || true
fi

# Defaults
SEQ="${SEQ:-SNMOT-060}"
VIDEO_SOURCE="${VIDEO:-}"
FRAMES="${FRAMES:-750}"
FPS="${FPS:-25}"
CALIBRATION="${CALIBRATION:-}"
IMU_CALIBRATION="${IMU_CALIBRATION:-}"
WEIGHTS="${WEIGHTS:-}"
IMGSZ="${IMGSZ:-1280}"
PLAYER_ID="${PLAYER_ID:-1}"
MATCH_ID="${MATCH_ID:-live}"
ROSTER="${ROSTER:-}"
ROSTER_NUMBERS="${ROSTER_NUMBERS:-}"
HEADLESS=1
LOOP_FILE=1
SOCCERNET_DIR="${SOCCERNET_DIR:-$HOME/SoccerNet/tracking}"

usage() {
  cat <<'EOF'
CPG44 Soccer Analytics Demo Launcher
====================================
Runs the complete stack (Relay, Sensor Hub, Product API, Real-time Vision, Frontend Dashboard).

Usage:
  scripts/demo.sh [options] [SEQUENCE_OR_VIDEO]

Options:
  -s, --seq NAME          SoccerNet tracking sequence (e.g. SNMOT-060, SNMOT-061, SNMOT-070)
  -v, --video PATH|URL    Direct video path, mp4 file, or stream URL
  -n, --frames NUM        Max frames to compile from image sequence (default: 750)
  --fps NUM               Frame rate for compiled sequence video (default: 25)
  -c, --calib PATH        Pitch homography calibration YAML (default: auto-detect)
  --imu-calib PATH        Wearable IMU calibration JSON
  -w, --weights PATH      YOLO weights or TensorRT engine (default: auto-detect)
  --imgsz PX              Inference image resolution (default: 960)
  -p, --player-id ID      Target wearable player ID (default: 7)
  -m, --match-id ID       Match tag for relay (default: live)
  --roster MAP            Track:player pairs (e.g. "1:7,2:10")
  --jerseys MAP           Jersey:player pairs (e.g. "7:7,10:10")
  --window, --gui         Show OpenCV desktop visualization window
  --headless              Run without desktop OpenCV window (default)
  --no-loop               Play video once without looping (default: loop)
  -l, --list              List all available SoccerNet sequences found locally
  -h, --help              Show this help message

Examples:
  # 1. Run standard default demo (SNMOT-060 with metric calibration)
  ./scripts/demo.sh

  # 2. Demo a specific SoccerNet sequence
  ./scripts/demo.sh --seq SNMOT-065
  ./scripts/demo.sh SNMOT-070 -n 500

  # 3. Demo a custom video file directly
  ./scripts/demo.sh --video demo/sample_match.mp4
  ./scripts/demo.sh path/to/my_match.mp4 --calib demo/demo_calibration.yaml

  # 4. List all available SoccerNet sequences
  ./scripts/demo.sh --list
EOF
}

list_sequences() {
  echo "=== Available SoccerNet Tracking Sequences ==="
  found=0
  for split in train test challenge; do
    split_dir="$SOCCERNET_DIR/$split"
    if [[ -d "$split_dir" ]]; then
      for seq_dir in "$split_dir"/SNMOT-*; do
        if [[ -d "$seq_dir/img1" ]]; then
          seq_name="$(basename "$seq_dir")"
          count="$(find "$seq_dir/img1" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l)"
          cal_marker="[Pixel mode]"
          if [[ -f "demo/calib/$seq_name.yaml" ]]; then
            cal_marker="[Metric calib: demo/calib/$seq_name.yaml]"
          fi
          printf "  • %-12s (%s split, %4d frames) %s
" "$seq_name" "$split" "$count" "$cal_marker"
          found=1
        fi
      done
    fi
  done
  if [[ "$found" -eq 0 ]]; then
    echo "  No SoccerNet sequences found in $SOCCERNET_DIR"
  fi
  echo
}

# Parse CLI arguments
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--seq) SEQ="$2"; shift 2 ;;
    -v|--video) VIDEO_SOURCE="$2"; shift 2 ;;
    -n|--frames) FRAMES="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    -c|--calib|--calibration) CALIBRATION="$2"; shift 2 ;;
    --imu-calib|--imu-calibration) IMU_CALIBRATION="$2"; shift 2 ;;
    -w|--weights) WEIGHTS="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    -p|--player-id) PLAYER_ID="$2"; shift 2 ;;
    -m|--match-id) MATCH_ID="$2"; shift 2 ;;
    --roster) ROSTER="$2"; shift 2 ;;
    --jerseys) ROSTER_NUMBERS="$2"; shift 2 ;;
    --window|--gui) HEADLESS=0; shift ;;
    --headless) HEADLESS=1; shift ;;
    --no-loop) LOOP_FILE=0; shift ;;
    --loop) LOOP_FILE=1; shift ;;
    -l|--list) list_sequences; exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

# Handle positional argument if provided
if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
  arg="${POSITIONAL[0]}"
  if [[ -f "$arg" || "$arg" == *.mp4 || "$arg" == *.avi || "$arg" == *.mkv || "$arg" == *.mov ]]; then
    VIDEO_SOURCE="$arg"
  elif [[ "$arg" =~ ^SNMOT-[0-9]+$ ]]; then
    SEQ="$arg"
  else
    if [[ -z "$VIDEO_SOURCE" ]]; then
      SEQ="$arg"
    fi
  fi
fi

# Locate or build video source
if [[ -n "$VIDEO_SOURCE" ]]; then
  if [[ "$VIDEO_SOURCE" =~ ^[0-9]+$ ]]; then
    echo "[demo] Using camera device #$VIDEO_SOURCE"
  elif [[ ! -f "$VIDEO_SOURCE" && ! "$VIDEO_SOURCE" =~ ^(rtsp|http|https):// ]]; then
    echo "[demo] ERROR: Video file not found: $VIDEO_SOURCE"
    exit 1
  else
    echo "[demo] Using direct video source: $VIDEO_SOURCE"
  fi
  CLIP="$VIDEO_SOURCE"
else
  # Locate sequence directory across possible splits
  SRC=""
  for split in train test challenge; do
    candidate="$SOCCERNET_DIR/$split/$SEQ/img1"
    if [[ -d "$candidate" ]]; then
      SRC="$candidate"
      break
    fi
  done

  if [[ -z "$SRC" ]]; then
    echo "[demo] ERROR: Could not find sequence $SEQ in $SOCCERNET_DIR/{train,test,challenge}"
    echo "       Run 'scripts/demo.sh --list' to view available sequences."
    exit 1
  fi

  mkdir -p demo/clips
  CLIP="demo/clips/${SEQ}_${FRAMES}f.mp4"

  # Build video clip if not cached or if source is newer
  if [[ ! -f "$CLIP" ]]; then
    echo "[demo] Building clip for $SEQ ($FRAMES frames @ ${FPS}fps)..."
    python - "$SRC" "$CLIP" "$FRAMES" "$FPS" <<'PY'
import cv2, glob, sys, os
src, out, n, fps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
fr = sorted(glob.glob(os.path.join(src, "*.jpg")))[:n]
assert fr, f"No frames found in {src}"
h, w = cv2.imread(fr[0]).shape[:2]
os.makedirs(os.path.dirname(out), exist_ok=True)
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for f in fr:
    img = cv2.imread(f)
    if img is not None:
        vw.write(img)
vw.release()
print(f"[demo] Successfully built {out}: {len(fr)} frames ({w}x{h} @ {fps}fps)")
PY
  else
    echo "[demo] Using cached clip: $CLIP"
  fi
fi

# Detect detection weights in order of measured held-out SoccerNet test mAP
# (docs/TRACKING_BENCHMARK.md). Promote a new checkpoint with WEIGHTS=... after it wins.
if [[ -z "$WEIGHTS" ]]; then
  for candidate in runs/detect/soccernet/weights/best.pt \
                   runs/detect/soccernet_v3_s1/weights/best.pt runs/detect/soccernet_v2/weights/best.pt; do
    if [[ -f "$candidate" ]]; then WEIGHTS="$candidate"; break; fi
  done
  [ -z "$WEIGHTS" ] && WEIGHTS=$(ls -t runs/detect/soccernet_v*/weights/best.pt runs/detect/*/weights/best.pt 2>/dev/null | head -1 || true)
  [ -z "$WEIGHTS" ] && [ -f "yolov8m.pt" ] && WEIGHTS="yolov8m.pt"
  [ -z "$WEIGHTS" ] && [ -f "yolov8s.pt" ] && WEIGHTS="yolov8s.pt"
  [ -z "$WEIGHTS" ] && [ -f "yolov8n.pt" ] && WEIGHTS="yolov8n.pt"
fi

if [[ -z "$WEIGHTS" || ! -f "$WEIGHTS" ]]; then
  echo "[demo] ERROR: No trained weights or model found."
  echo "       Provide weights using --weights PATH"
  exit 1
fi
echo "[demo] Detector model: $WEIGHTS"

# Resolve pitch calibration
if [[ -z "$CALIBRATION" ]]; then
  if [[ -n "$SEQ" && -f "demo/calib/$SEQ.yaml" ]]; then
    CALIBRATION="demo/calib/$SEQ.yaml"
    echo "[demo] Metric Mode enabled (matched sequence calibration: $CALIBRATION)"
  elif [[ "$SEQ" == "SNMOT-060" && -f "demo/demo_calibration.yaml" ]]; then
    CALIBRATION="demo/demo_calibration.yaml"
    echo "[demo] Metric Mode enabled (SNMOT-060 kickoff calibration: $CALIBRATION)"
  else
    CALIBRATION=""
    echo "[demo] Adaptive Pitch Mode (dynamic full-span 105m x 68m projection & live heatmap enabled)"
  fi
else
  if [[ -f "$CALIBRATION" ]]; then
    echo "[demo] Metric Mode enabled (custom calibration: $CALIBRATION)"
  else
    echo "[demo] WARNING: Specified calibration file not found: $CALIBRATION (falling back to Pixel Mode)"
    CALIBRATION=""
  fi
fi

# Build arguments for start_system.sh
ARGS=(
  --video "$CLIP"
  --weights "$WEIGHTS"
  --imgsz "$IMGSZ"
  --player-id "$PLAYER_ID"
  --match-id "$MATCH_ID"
)

[[ -n "$CALIBRATION" ]] && ARGS+=(--calibration "$CALIBRATION")
[[ -n "$IMU_CALIBRATION" ]] && ARGS+=(--imu-calibration "$IMU_CALIBRATION")
[[ -n "$ROSTER" ]] && ARGS+=(--roster "$ROSTER")
[[ -n "$ROSTER_NUMBERS" ]] && ARGS+=(--jerseys "$ROSTER_NUMBERS")
[[ "$HEADLESS" == "1" ]] && ARGS+=(--headless)
[[ "$LOOP_FILE" == "1" ]] && ARGS+=(--loop)

echo
echo "================================================================="
echo " Starting CPG44 System Demo"
echo " Video / Sequence : ${VIDEO_SOURCE:-$SEQ ($CLIP)}"
echo " Weights          : $WEIGHTS"
echo " Calibration      : ${CALIBRATION:-None (Pixel Mode)}"
echo " Player ID        : $PLAYER_ID"
echo " Dashboard UI     : http://localhost:5173"
echo " API Docs         : http://localhost:8000/docs"
echo "================================================================="
echo

exec scripts/start_system.sh "${ARGS[@]}"
