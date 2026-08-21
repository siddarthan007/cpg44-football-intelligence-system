# Relay setup for cpg44.nivaspms.com

The relay is a small FastAPI container with a bounded memory cache. It has no
Postgres, Redis, MinIO or file volume. Docker exposes it only on
`127.0.0.1:18081`, and Caddy publishes the hostname.

```text
ESP32 and project PC -> Cloudflare -> Caddy -> 127.0.0.1:18081 -> relay
```

## 1. Cloudflare

Create a proxied `A` record:

```text
Type: A
Name: cpg44
Target: <VPS public IP>
Proxy: Proxied
```

Set SSL/TLS mode to `Full (strict)`. Bypass cache for `/api/*`, `/ws/*` and
`/health`. Clients use the hostname, never the VPS IP.

## 2. Copy and configure the relay

Place this project at `/opt/cpg44` on the VPS. Create a private environment
file:

```bash
cd /opt/cpg44
umask 077
TOKEN="$(openssl rand -hex 32)"
printf 'CPG44_RELAY_TOKEN=%s\nCPG44_RELAY_PORT=18081\nCPG44_RELAY_CACHE_BYTES=4194304\n' "$TOKEN" > .env
unset TOKEN
```

Keep `.env` outside version control. Load it and start the relay:

```bash
set -a
. ./.env
set +a
scripts/deploy_hostinger_relay.sh
```

Check the loopback service:

```bash
docker compose -p cpg44-telemetry -f docker-compose.server.yml ps
curl --fail http://127.0.0.1:18081/health
ss -ltnp | grep 18081
```

The listening address must be `127.0.0.1:18081`, not `0.0.0.0:18081`.

## 3. Caddy

Add this site block:

```caddyfile
cpg44.nivaspms.com {
    encode zstd gzip
    reverse_proxy 127.0.0.1:18081
}
```

Validate and reload Caddy:

```bash
caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

## 4. Public check

```bash
curl --fail https://cpg44.nivaspms.com/health

set -a
. /opt/cpg44/.env
set +a
curl --fail \
  -H "X-Auth: $CPG44_RELAY_TOKEN" \
  https://cpg44.nivaspms.com/api/v1/sensors/latest
```

The first command is public health only. Sensor routes require `X-Auth`.

## 5. Project PC and firmware

Copy the same token to the project-PC shell and prepare the verified TLS chain
after DNS is active:

```bash
cd /home/siddartha/capstone
scripts/prepare_relay_ca.sh configs/relay_ca.pem
export CPG44_RELAY_URL="https://cpg44.nivaspms.com"
export CPG44_RELAY_TOKEN="<token from /opt/cpg44/.env>"
export CPG44_RELAY_CA_FILE="$PWD/configs/relay_ca.pem"
```

Start the backend with these variables, then flash the ESP32 from the Hardware
page. The token is placed in the ignored build header and is not sent to the
browser.

## Reconnection

Each raw sample has a stable event ID, source timestamp, boot ID, player ID and
match ID. The relay adds its sequence and receive time. On reconnect, the local
processor requests samples after its last relay sequence. `cache_gap=true`
means an older part has left the memory cache and that interval must be marked
incomplete.

## Commands

```bash
docker compose -p cpg44-telemetry -f docker-compose.server.yml logs --tail=200 relay
docker compose -p cpg44-telemetry -f docker-compose.server.yml restart relay
docker compose -p cpg44-telemetry -f docker-compose.server.yml stop relay
```

Restarting clears the relay cache. Long-term session evidence stays on the
project PC.
