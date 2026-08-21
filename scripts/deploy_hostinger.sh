#!/usr/bin/env bash
# Compatibility entry point for the hardened, relay-only VPS deployment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/deploy_hostinger_relay.sh" "$@"
