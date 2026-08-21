#!/usr/bin/env bash
# Start the stateless CPG44 relay. Caddy is configured separately.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$ROOT_DIR/docker-compose.server.yml"
PROJECT_NAME="cpg44-telemetry"
RELAY_PORT="${CPG44_RELAY_PORT:-18081}"

cd "$ROOT_DIR"

if [[ -z "${CPG44_RELAY_TOKEN:-}" ]]; then
  echo "Set CPG44_RELAY_TOKEN to a private value of at least 32 characters."
  echo "Generate one with: openssl rand -hex 32"
  exit 1
fi
if [[ ${#CPG44_RELAY_TOKEN} -lt 32 ]]; then
  echo "CPG44_RELAY_TOKEN must contain at least 32 characters."
  exit 1
fi
if ! [[ "$RELAY_PORT" =~ ^[0-9]+$ ]] || (( RELAY_PORT < 1024 || RELAY_PORT > 65535 )); then
  echo "CPG44_RELAY_PORT must be an unused port from 1024 to 65535."
  exit 1
fi

command -v docker >/dev/null 2>&1 || { echo "Docker is required."; exit 1; }
docker compose version >/dev/null

if command -v ss >/dev/null 2>&1 && ss -H -ltn "sport = :$RELAY_PORT" | grep -q .; then
  existing="$(docker compose -p "$PROJECT_NAME" -f "$COMPOSE_FILE" ps --quiet relay 2>/dev/null || true)"
  if [[ -z "$existing" ]]; then
    echo "Loopback port $RELAY_PORT is already in use. Choose another CPG44_RELAY_PORT."
    exit 1
  fi
fi

docker compose -p "$PROJECT_NAME" -f "$COMPOSE_FILE" up -d --build relay

healthy=0
for _ in {1..30}; do
  if curl --fail --silent --show-error "http://127.0.0.1:$RELAY_PORT/health" >/dev/null; then
    healthy=1
    break
  fi
  sleep 1
done
if [[ "$healthy" -ne 1 ]]; then
  echo "Relay did not become healthy."
  docker compose -p "$PROJECT_NAME" -f "$COMPOSE_FILE" logs --tail=100 relay
  exit 1
fi

echo "Stateless relay is healthy on 127.0.0.1:$RELAY_PORT."
echo
echo "Add this separate site block to the existing Caddyfile after reviewing it:"
echo
echo "cpg44.nivaspms.com {"
echo "    encode zstd gzip"
echo "    reverse_proxy 127.0.0.1:$RELAY_PORT"
echo "}"
echo
echo "Then validate and reload Caddy using the same method as the existing deployment."
echo "Public clients must use https://cpg44.nivaspms.com, never the VPS IP."
