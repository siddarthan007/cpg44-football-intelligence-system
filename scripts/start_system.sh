#!/usr/bin/env bash
# Start the field-PC stack. No generated telemetry or simulated match data is used.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${CPG44_ENV_FILE:-$ROOT_DIR/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi
SESSION_ID="${CPG44_SESSION_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SESSION_DIR="$ROOT_DIR/data/live_sessions/$SESSION_ID"
VIDEO_SOURCE="${CPG44_VIDEO_SOURCE:-}"
WEIGHTS="${CPG44_WEIGHTS:-}"
CALIBRATION="${CPG44_CALIBRATION:-}"
IMU_CALIBRATION="${CPG44_IMU_CALIBRATION:-}"
PLAYER_ID="${CPG44_PLAYER_ID:-7}"
MATCH_ID="${CPG44_MATCH_ID:-live}"
ROSTER="${CPG44_ROSTER:-}"
ROSTER_NUMBERS="${CPG44_ROSTER_NUMBERS:-}"
LOOP_FILE="${CPG44_LOOP_FILE:-0}"
IMGSZ="${CPG44_IMGSZ:-960}"
ANALYTICS_EVERY="${CPG44_ANALYTICS_EVERY:-10}"
SNAPSHOT_EVERY="${CPG44_SNAPSHOT_EVERY:-2}"
LATENCY_BUDGET_MS="${CPG44_LATENCY_BUDGET_MS:-240}"
API_HOST="${CPG44_API_HOST:-0.0.0.0}"
API_PORT="${CPG44_API_PORT:-8000}"
HUB_PORT="${CPG44_HUB_PORT:-8081}"
FRONTEND_PORT="${CPG44_FRONTEND_PORT:-5173}"
PIDS=()

if [[ -z "$WEIGHTS" ]]; then
  for candidate in \
    "$ROOT_DIR/runs/detect/soccernet/weights/best.pt" \
    "$ROOT_DIR/runs/detect/soccernet_v3_s1/weights/best.pt" \
    "$ROOT_DIR/runs/detect/soccernet_v2/weights/best.pt"; do
    if [[ -f "$candidate" ]]; then
      WEIGHTS="$candidate"
      break
    fi
  done
fi

usage() {
  cat <<'EOF'
Usage: scripts/start_system.sh [options]

  --video SOURCE      Camera index, stream URL, or match video path
  --weights PATH      Detection weights (default: local TensorRT engine)
  --imgsz PX          Inference input size (default: 960)
  --calibration PATH  Pitch homography YAML; required for metric analytics
  --imu-calibration PATH  Wearable IMU calibration JSON
  --player-id ID      Wearable roster ID (default: 7)
  --match-id ID       Match tag flashed to the wearable (default: live)
  --roster MAP        Stable track:player pairs, comma-separated
  --jerseys MAP       Jersey:player pairs, comma-separated
  --headless           Disable the OpenCV desktop window
  --loop               Repeat a recorded source until stopped
  -h, --help           Show this help

The same values may be set with CPG44_VIDEO_SOURCE,
CPG44_WEIGHTS, CPG44_CALIBRATION, CPG44_IMU_CALIBRATION, CPG44_PLAYER_ID,
CPG44_MATCH_ID, CPG44_ROSTER and CPG44_ROSTER_NUMBERS. CPG44_RELAY_URL and
CPG44_RELAY_TOKEN are required. The ESP32 is never addressed over the LAN.
EOF
}

HEADLESS="${CPG44_HEADLESS:-0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) VIDEO_SOURCE="$2"; shift 2 ;;
    --weights) WEIGHTS="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --calibration) CALIBRATION="$2"; shift 2 ;;
    --imu-calibration) IMU_CALIBRATION="$2"; shift 2 ;;
    --player-id) PLAYER_ID="$2"; shift 2 ;;
    --match-id) MATCH_ID="$2"; shift 2 ;;
    --roster) ROSTER="$2"; shift 2 ;;
    --jerseys) ROSTER_NUMBERS="$2"; shift 2 ;;
    --headless) HEADLESS=1; shift ;;
    --loop) LOOP_FILE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 2 ;;
  esac
done

cleanup() {
  trap - EXIT INT TERM
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

mkdir -p "$SESSION_DIR"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/backend/src:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CPG44_HUB_URL="http://127.0.0.1:$HUB_PORT"
export CPG44_PRODUCT_API="http://127.0.0.1:$API_PORT"

RELAY_URL="${CPG44_RELAY_URL:-}"
RELAY_URL="${RELAY_URL%/}"
RELAY_TOKEN="${CPG44_RELAY_TOKEN:-}"
if [[ "$RELAY_URL" != "https://cpg44.nivaspms.com" ]]; then
  echo "CPG44_RELAY_URL must be https://cpg44.nivaspms.com"
  exit 2
fi
if [[ ${#RELAY_TOKEN} -lt 32 ]]; then
  echo "CPG44_RELAY_TOKEN must contain at least 32 characters."
  exit 2
fi

HUB_CMD=(python -m soccer_analytics.sensors.hub
  --http-port "$HUB_PORT"
  --player-id "$PLAYER_ID"
  --match-id "$MATCH_ID"
  --relay-url "$RELAY_URL"
  --relay-token "$RELAY_TOKEN"
  --record "$SESSION_DIR/wearable_raw.ndjson")
if [[ -n "$IMU_CALIBRATION" ]]; then
  [[ -f "$IMU_CALIBRATION" ]] || {
    echo "IMU calibration file not found: $IMU_CALIBRATION"
    exit 2
  }
  HUB_CMD+=(--calibration "$IMU_CALIBRATION")
fi
"${HUB_CMD[@]}" &
PIDS+=("$!")
echo "Wearable processor: relay to http://127.0.0.1:$HUB_PORT"

python -m uvicorn cpg44_api.main:app --host "$API_HOST" --port "$API_PORT" &
PIDS+=("$!")

backend_ready=0
for _ in {1..30}; do
  if curl --fail --silent "http://127.0.0.1:$API_PORT/api/v1/health" >/dev/null; then
    backend_ready=1
    break
  fi
  sleep 0.5
done
if [[ "$backend_ready" -ne 1 ]]; then
  echo "The product API did not become healthy on port $API_PORT."
  exit 1
fi

if [[ -n "$VIDEO_SOURCE" ]]; then
  if [[ ! -f "$WEIGHTS" ]]; then
    echo "Detection weights not found: $WEIGHTS"
    exit 2
  fi
  VISION_CMD=(python -m soccer_analytics.realtime
    --video "$VIDEO_SOURCE"
    --weights "$WEIGHTS"
    --imgsz "$IMGSZ"
    --analytics-every "$ANALYTICS_EVERY"
    --snapshot-every "$SNAPSHOT_EVERY"
    --latency-budget-ms "$LATENCY_BUDGET_MS"
    --product-api "$CPG44_PRODUCT_API"
    --camera-id "${CPG44_CAMERA_ID:-primary}"
    --stats "$SESSION_DIR/stats.json"
    --fused-out "$SESSION_DIR/fused_snapshots.ndjson"
    --player-id "$PLAYER_ID")
  if [[ -n "$CALIBRATION" ]]; then
    [[ -f "$CALIBRATION" ]] || { echo "Calibration file not found: $CALIBRATION"; exit 2; }
    VISION_CMD+=(--calibration "$CALIBRATION")
  fi
  VISION_CMD+=(--sensor-hub "http://127.0.0.1:$HUB_PORT")
  [[ -n "$ROSTER" ]] && VISION_CMD+=(--roster "$ROSTER")
  [[ -n "$ROSTER_NUMBERS" ]] && VISION_CMD+=(--roster-numbers "$ROSTER_NUMBERS")
  [[ "$HEADLESS" == "1" ]] && VISION_CMD+=(--no-window)
  [[ "$LOOP_FILE" == "1" ]] && VISION_CMD+=(--loop)
  [[ -n "${CPG44_RECORD_VIDEO:-}" ]] && VISION_CMD+=(--out "$SESSION_DIR/dashboard.mp4")
  "${VISION_CMD[@]}" &
  PIDS+=("$!")
  echo "Vision/fusion source: $VIDEO_SOURCE"
else
  echo "Vision/fusion: not started; provide --video for live markings and metrics."
fi

(
  cd "$ROOT_DIR/frontend"
  npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT"
) &
PIDS+=("$!")

echo "Dashboard: http://localhost:$FRONTEND_PORT"
echo "API documentation: http://localhost:$API_PORT/docs"
echo "Session evidence: $SESSION_DIR"
if [[ -z "$CALIBRATION" ]]; then
  echo "Metric speed, distance and load remain withheld until --calibration is supplied."
fi
echo "Press Ctrl+C to stop all local services."

status=0
wait -n "${PIDS[@]}" || status=$?
if [[ "$status" -ne 0 ]]; then
  echo "A service exited with status $status."
fi
exit "$status"
