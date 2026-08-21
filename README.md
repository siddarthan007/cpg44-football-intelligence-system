# CPG44 football intelligence platform

CPG44 joins football video with an ESP32-S3 wearable. Player and ball tracking,
pitch mapping, sensor processing, fusion, analytics, training and the React
dashboard run on this PC. Wearable traffic always passes through the small
memory-only relay at `cpg44.nivaspms.com`.

## Working path

```text
ESP32-S3 -> HTTPS relay -> local sensor processor -> fusion/API -> dashboard
camera or footage -> YOLO + ByteTrack + pitch map -----------^
```

The supported firmware is
`firmware/wearable_stream/wearable_stream.ino`. It sends timestamped raw IMU,
PPG and GPS batches. The local processor calculates movement and quality-gated
heart-rate and SpO2 estimates. It does not connect to the ESP32 over the LAN.

The vision path uses real detections and ByteTrack results. Metric speed,
distance and tactical geometry are shown only after pitch calibration. Team and
jersey binding can abstain when the evidence is weak.

## Start

```bash
cd /home/siddartha/capstone
conda activate soccer
export CPG44_RELAY_URL="https://cpg44.nivaspms.com"
export CPG44_RELAY_TOKEN="<same token as the VPS>"
export CPG44_RELAY_CA_FILE="$PWD/configs/relay_ca.pem"

scripts/start_system.sh \
  --video /path/to/campus-footage.mp4 \
  --weights /path/to/best.pt \
  --calibration /path/to/campus-pitch.yaml \
  --player-id 7 \
  --match-id live \
  --roster "12:7" \
  --jerseys "7:7"
```

Open `http://localhost:5173`. The system uses real sources and leaves missing or
invalid values blank.

## Guides

- [Full demonstration tutorial](docs/DEMO_TUTORIAL.md)
- [Relay server setup](docs/VPS_RELAY.md)
- [Wearable build](HARDWARE.md)
- [Operating manual](docs/OPERATING_MANUAL.md)
- [Accuracy protocol](docs/CAPSTONE_ACCURACY_PROTOCOL.md)

## Checks

```bash
PYTHONPATH=backend/src:. pytest -q tests
npm run lint --prefix frontend
npm run build --prefix frontend
python -m compileall -q backend/src soccer_analytics
conda run -n soccer arduino-cli compile \
  --build-path /tmp/cpg44-wearable-build \
  --fqbn esp32:esp32:esp32s3 firmware/wearable_stream
```

The workload warning is coaching support, not a diagnosis. MAX30102 SpO2 is an
uncalibrated estimate and should be checked mainly during still or recovery
windows.
