#!/usr/bin/env bash
# Start the relay-backed wearable processor in WSL.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CPG44_RELAY_URL:?Set CPG44_RELAY_URL to https://cpg44.nivaspms.com}"
: "${CPG44_RELAY_TOKEN:?Set CPG44_RELAY_TOKEN}"
exec python -m soccer_analytics.hub \
  --relay-url "$CPG44_RELAY_URL" \
  --relay-token "$CPG44_RELAY_TOKEN" \
  --http-port 8081 \
  --player-id "${CPG44_PLAYER_ID:-7}" \
  --match-id "${CPG44_MATCH_ID:-live}"
