#!/usr/bin/env bash
# Compatibility entry point. The supported VPS topology is relay-only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/deploy_hostinger_relay.sh" "$@"
