#!/usr/bin/env bash
# Save the currently verified Cloudflare certificate chain for ESP32 TLS checks.
set -euo pipefail

HOST="cpg44.nivaspms.com"
OUTPUT="${1:-configs/relay_ca.pem}"
TEMP_DIR="$(mktemp -d)"

cleanup() {
  find "$TEMP_DIR" -maxdepth 1 -type f -delete
  rmdir "$TEMP_DIR"
}
trap cleanup EXIT INT TERM

command -v openssl >/dev/null 2>&1 || {
  echo "OpenSSL is required."
  exit 1
}

openssl s_client \
  -connect "$HOST:443" \
  -servername "$HOST" \
  -verify_hostname "$HOST" \
  -verify_return_error \
  -showcerts \
  </dev/null >"$TEMP_DIR/connection.txt" 2>&1

grep -q "Verify return code: 0 (ok)" "$TEMP_DIR/connection.txt" || {
  echo "The public certificate for $HOST could not be verified."
  exit 1
}

awk '
  /-----BEGIN CERTIFICATE-----/ {copy=1}
  copy {print}
  /-----END CERTIFICATE-----/ {copy=0}
' "$TEMP_DIR/connection.txt" >"$TEMP_DIR/relay_ca.pem"

CERT_COUNT="$(grep -c "BEGIN CERTIFICATE" "$TEMP_DIR/relay_ca.pem")"
if [[ "$CERT_COUNT" -lt 1 ]]; then
  echo "No certificate was returned by $HOST."
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
install -m 600 "$TEMP_DIR/relay_ca.pem" "$OUTPUT"
echo "Saved $CERT_COUNT verified certificate(s) to $OUTPUT"
