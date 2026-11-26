#!/usr/bin/env bash
# Build the root certificates the vest and this laptop trust for the relay.
#
# Cloudflare issues the relay's edge certificate from Google Trust Services,
# Let's Encrypt or SSL.com and may move between them; it advises against pinning
# intermediates. So the file holds the self-signed roots of those three
# authorities (from the Mozilla bundle shipped with certifi), not the chain the
# server happens to present today. The script then checks, with no fallback to
# the system store, that the live relay verifies against exactly this file.
set -euo pipefail

HOST="cpg44.nivaspms.com"
OUTPUT="${1:-configs/relay_ca.pem}"
PY="${PYTHON:-python}"
TEMP="$(mktemp)"
trap 'rm -f "$TEMP"' EXIT INT TERM

command -v openssl >/dev/null 2>&1 || { echo "OpenSSL is required."; exit 1; }

"$PY" - "$TEMP" <<'EOF'
import re, sys
import certifi
from cryptography import x509
from cryptography.x509.oid import NameOID

ROOTS = [
    "GTS Root R1", "GTS Root R2", "GTS Root R3", "GTS Root R4",            # Google Trust Services
    "ISRG Root X1", "ISRG Root X2",                                       # Let's Encrypt
    "SSL.com TLS RSA Root CA 2022", "SSL.com TLS ECC Root CA 2022",       # SSL.com
    "SSL.com Root Certification Authority RSA", "SSL.com Root Certification Authority ECC",
    "Certum Trusted Network CA", "Certum Trusted Network CA 2",           # cross-signs SSL.com
]
found = {}
for pem in re.findall(r"-----BEGIN CERTIFICATE-----.+?-----END CERTIFICATE-----", open(certifi.where()).read(), re.S):
    cert = x509.load_pem_x509_certificate(pem.encode())
    cn = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    if cn and cn[0].value in ROOTS and cert.subject == cert.issuer:
        found[cn[0].value] = pem
with open(sys.argv[1], "w") as fh:
    for name in ROOTS:
        if name in found:
            fh.write(f"# {name}\n{found[name]}\n")
print("roots:", ", ".join(n for n in ROOTS if n in found))
missing = [n for n in ROOTS if n not in found]
if missing:
    print("not in this bundle:", ", ".join(missing))
EOF

RESULT="$(openssl s_client -connect "$HOST:443" -servername "$HOST" -verify_hostname "$HOST" \
  -CAfile "$TEMP" -no-CApath -no-CAstore -verify_return_error </dev/null 2>&1 || true)"
grep -q "Verify return code: 0 (ok)" <<<"$RESULT" || {
  echo "$HOST does not verify against these roots:"
  grep -E "verify error|Verify return code" <<<"$RESULT" || true
  exit 1
}

mkdir -p "$(dirname "$OUTPUT")"
install -m 644 "$TEMP" "$OUTPUT"
echo "Saved $(grep -c 'BEGIN CERTIFICATE' "$OUTPUT") root certificates to $OUTPUT; $HOST verifies against them."
