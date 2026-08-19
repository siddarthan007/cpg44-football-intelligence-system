# ⚽ CPG44 — Predictive & Tactical Football Intelligence Platform

An end-to-end AI-powered soccer analytics platform combining **Computer Vision** (single/multi-camera tracking, 2D top-down pitch projection, tactical modeling) with **ESP32 Wearable Telemetry** (PPG heart rate, SpO2, IMU PlayerLoad, GPS) and a **Hostinger KVM 2 Cloud Ingestion Relay**.

```
 [ ESP32-S3 Vest ] ──TCP:9000/UDP──▶ [ Hostinger KVM 2 VPS ] ──WebSocket──▶ [ React Web Dashboard ]
 (IMU 100Hz, PPG 25Hz, GPS 1Hz)       (Lightweight Relay Daemon)                     ▲
                                                  │                                  │
                                          Sync & Buffer API                          │ REST / WS
                                                  │                                  │
 [ Camera Sources ] ──RTSP/WebRTC/Upload──▶ [ Local WSL2 Engine ] ───────────────────┘
 (Phone/IP/GoPro/Uploads)                  - YOLOv8 Detector + ByteTrack
                                           - CIELAB Team Assigner + EasyOCR
                                           - 2D Homography Pitch Projection
                                           - Ball Kalman Tracker + Interpolation
                                           - di Prampero Metabolic Power & Load
                                           - Tactical Engine (Voronoi, Pressing)
                                           - LSTM Trajectory & Pass Predictor
                                           - XGBoost Injury Risk & Substitution
```

---

## 🌟 Key Capabilities

### 1. 🎯 Computer Vision & Deep Learning Tracking
- **Detector**: YOLOv8 fine-tuned on SoccerNet (classes: `player`, `ball`, `goalkeeper`, `referee`).
- **Multi-Object Tracking (MOT)**: ByteTrack association + Kalman filter smoothing with occlusion coasting.
- **Accurate Team Tagging**: CIELAB color-space clustering invariant to grass-green kits and shadow variations.
- **Ball Tracking**: Low-confidence thresholding (0.10), parabolic trajectory Kalman interpolation, and staleness recovery.
- **Predictive Trajectories**: LSTM player path prediction (ADE 14 px / FDE 26 px on SoccerNet).

### 2. 🗺️ 2D Top-Down Projection & Tactical Engine
- **Pitch Homography**: 105m × 68m standard pitch coordinate transformation with camera motion compensation.
- **Voronoi Space Control**: Real-time pitch control zone estimation.
- **Collective Dynamics**: Team centroid, width, depth, stretch index, surface area (convex hull), and pressing intensity.
- **Formation Detection**: Automated 4-3-3, 4-4-2, 3-5-2, 4-2-3-1 classification.
- **Expected Goals (xG)**: Geometric shot model with penalty and distance weighting.

### 3. 🫀 Multimodal Wearable Fusion & Injury Prevention
- **Firmware**: ESP32-S3 streaming MPU6050 IMU (100 Hz), MAX30102 PPG (25 Hz), and NEO-6M GPS (1 Hz) over TCP port 9000.
- **Time Synchronization**: Sub-millisecond linear regression aligning wearable packets to host video frame clocks.
- **Metabolic Power**: di Prampero metabolic power ($W/kg$) and Catapult speed zones (walking, jogging, running, HSR, sprinting).
- **Injury Risk Model**: XGBoost / Random Forest evaluator monitoring Acute:Chronic Workload Ratio (ACWR safe zone 0.8–1.3), cardiac drift, and fatigue substitution alerts.

### 4. ☁️ Hostinger KVM 2 Cloud Ingestion Relay (2 vCPU, 8 GB RAM)
- Dedicated async daemon handling TCP:9000 stream, UDP packets, and WebSocket distribution (`/ws/sensors`).
- In-memory circular buffer (5,000 samples/player) + persistent SQLite time-series database.
- Low footprint (<5% CPU, ~120 MB RAM) running 24/7.

---

## 🚀 4 Operational Modes

1. **Demo Showcase Mode**: One-click instant showcase using bundled match footage with simulated real-time playback, 2D radar, dynamic heatmaps, and simulated wearable telemetry.
2. **Upload Match Video Mode**: Upload university or college match footage (`.mp4`), calibrate 4 pitch corners, and run full inference with progress tracking and downloadable annotated video + JSON statistics.
3. **Live Camera Stream Mode**: Connect RTSP/WebRTC streams or mobile phone cameras (`/camera`) for near-real-time match analysis (~25+ FPS on RTX 5060).
4. **Training & Fine-Tuning Mode**: 2-stage transfer learning fine-tuning YOLO on college footage with synthetic low-quality augmentations.

---

## 🛠️ Quick Start Guide

### 1. Python Environment Setup (WSL2 / Linux)
```bash
# PyTorch cu128 for Blackwell RTX 5060 (sm_120)
conda create -n soccer python=3.11 -y && conda activate soccer
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 2. Start Backend & Sensor Hub
```bash
# Start FastAPI & WebSocket Backend (Port 8000)
PYTHONPATH=backend/src:. python -m uvicorn cpg44_api.main:app --host 0.0.0.0 --port 8000

# (Optional) Start Local Sensor Hub for ESP32 (Port 8081 & TCP Port 9000)
python -m soccer_analytics.hub --esp32 <ESP32_IP> --http-port 8081
```

### 3. Start Modern Web Dashboard
```bash
cd frontend
npm install
npm run dev
```
Open **http://127.0.0.1:5173** to view the live dashboard.

---

## ☁️ Deploying to Hostinger KVM 2 Server

To deploy the always-running cloud sensor relay on your Hostinger VPS:

```bash
# On your Hostinger VPS (Ubuntu 22.04/24.04):
git clone https://github.com/siddarthan007/cpg44-football-intelligence-system.git /opt/cpg44
cd /opt/cpg44
sudo bash scripts/deploy_hostinger.sh
```

Or deploy using Docker:
```bash
docker-compose -f docker-compose.server.yml up -d
```

---

## 📱 Mobile Camera Streaming

Turn any smartphone into a sideline match camera:
1. Open `http://<YOUR_SERVER_IP>:8000/camera` on the phone browser.
2. Allow camera access.
3. Frames stream directly into the gateway.

---

## 🧪 Testing

```bash
PYTHONPATH=backend/src:. pytest tests/
```

---

## 📄 License
MIT License. Built for Capstone Project **CPG44**.
