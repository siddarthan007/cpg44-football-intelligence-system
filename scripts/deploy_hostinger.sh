#!/usr/bin/env bash
# Deploy CPG44 Sensor Relay on Hostinger KVM 2 VPS (Ubuntu 22.04 / 24.04)
set -e

echo "=== CPG44 Hostinger KVM 2 Deployment ==="

# Check root/sudo
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root or with sudo"
  exit 1
fi

# Update system
apt update && apt upgrade -y
apt install -y python3 python3-pip python3-venv git curl ufw

# Open required firewall ports
ufw allow 22/tcp     # SSH
ufw allow 80/tcp     # HTTP
ufw allow 443/tcp    # HTTPS
ufw allow 8081/tcp   # Sensor Relay REST + WS
ufw allow 9000/tcp   # ESP32 TCP Stream
ufw allow 9001/udp   # Telemetry UDP
ufw --force enable

# Setup deployment dir
INSTALL_DIR="/opt/cpg44-relay"
mkdir -p "$INSTALL_DIR"
cp -r backend/src "$INSTALL_DIR/"
mkdir -p "$INSTALL_DIR/data"

# Setup virtual environment
python3 -m venv "$INSTALL_DIR/venv"
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip
"$INSTALL_DIR/venv/bin/pip" install fastapi uvicorn[standard] websockets pydantic

# Create Systemd Service
cat << 'EOF' > /etc/systemd/system/cpg44-sensor-relay.service
[Unit]
Description=CPG44 Hostinger Sensor Relay Daemon
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/cpg44-relay
Environment=PYTHONPATH=/opt/cpg44-relay/src
Environment=PORT=8081
Environment=CPG44_DB_PATH=/opt/cpg44-relay/data/hostinger_telemetry.db
ExecStart=/opt/cpg44-relay/venv/bin/python -m cpg44_api.hostinger_sensor_relay
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable cpg44-sensor-relay
systemctl restart cpg44-sensor-relay

echo "=== CPG44 Sensor Relay Deployed & Running ==="
echo "Check status: systemctl status cpg44-sensor-relay"
echo "Public Health Check: http://$(curl -s ifconfig.me):8081/health"
