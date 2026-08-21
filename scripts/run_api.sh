#!/usr/bin/env bash
# Product API (FastAPI) + React dashboard.
# Terminal 1: this script's API. Terminal 2: npm run dev --prefix frontend
# Start scripts/run_hub.sh first so the API can read processed relay telemetry.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:backend/src:."
exec python -m uvicorn cpg44_api.main:app --host 0.0.0.0 --port 8000
