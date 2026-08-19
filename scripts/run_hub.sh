#!/usr/bin/env bash
# Start the wearable receiver in WSL.
# Usage: bash scripts/run_hub.sh 192.168.43.12
set -euo pipefail
cd "$(dirname "$0")/.."
ESP32_IP="${1:?Pass the ESP32 IP from the Arduino Serial Monitor}"
exec python -m soccer_analytics.hub --esp32 "$ESP32_IP" --http-port 8081 --player-id "${2:-7}"
