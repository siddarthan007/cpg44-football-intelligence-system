# Running & inference guide (CPG44)

End-to-end operating manual: from raw video to the live dashboard, including the
**1–2 wearable demo** flow. Everything runs in the `soccer` conda env on the RTX
5060 (WSL2).

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate soccer
cd ~/capstone
```

## ⚡ Zero-setup demo (no campus footage / calibration / wearable needed)

```bash
bash scripts/demo.sh
```
Builds a clip from the local SoccerNet data, runs the whole pipeline with a
bundled approximate calibration + a **simulated** wearable stream, and writes a
**composite dashboard video** you can just play:

- `demo_out/dashboard.mp4` — match (tagged players, ball, possession) + radar +
  live heatmap + the full analytics panel (team performance, formation, shots/xG,
  player table, injury risk, tactical recommendations).
- `demo_out/stats.json` — the numbers.

Speed: **~26 FPS live** at `--imgsz 960` (FP16 inference + per-frame homography
tracking + LSTM path prediction, all on GPU). Recording the composite mp4 is
slower (encode-bound). Override res with `IMGSZ=1280 bash scripts/demo.sh`.

The live dashboard also draws each player's **LSTM-predicted path** (cyan) and
uses a **per-frame tracked homography** that follows the camera pan — so speeds
stay physically real (no 12 m/s clamping) instead of drifting like a static
calibration.

Prototype caveats (honest): the bundled calibration is eyeballed and the SoccerNet
camera pans, so absolute metres/speeds and the team-split are approximate on this
clip; the wearable panel uses simulated data. Real accuracy needs a fixed campus
camera calibration + the full-dataset detector. It demonstrates the end-to-end
system working, which is the point of a prototype demo.

---

## 0. One-time: verify GPU
```bash
python -c "from soccer_analytics.device import describe; print(describe())"
# → Compute: NVIDIA GeForce RTX 5060 Laptop GPU | 8.0 GB | CC 12.0 | device cuda:0
```

## 1. Dataset → 2. Train → 3. Calibrate
```bash
# dataset (SoccerNet MOT → YOLO)
python soccernet_to_yolo.py --src ~/SoccerNet/tracking --dst ~/SoccerNet/yolo --stride 5

# train detector — --auto fits the card: yolov8m@1280 batch 4, AMP (~6.4 GB / 2.5 it/s
# on the 8 GB 5060; batch 8 spills to system RAM → ~100× slower, so don't raise it)
python -m soccer_analytics.train base --data ~/SoccerNet/yolo/data.yaml --auto
# OR full training on a cloud GPU (A100): upload the repo to a RunPod pod, then
#   bash scripts/runpod_train.sh      # → runs/detect/soccernet_full/weights/best.pt

# fine-tune on your campus footage (two-stage, low-quality aug)
python -m soccer_analytics.train finetune --data ~/campus/data.yaml \
    --base runs/detect/soccernet/weights/best.pt

# calibrate the pitch ONCE per fixed camera (click ≥4 landmarks)
python -m soccer_analytics.calibrate --video campus_match.mp4 --out campus.yaml
```
> Metric analytics (metres, speed, metabolic power, load, injury, tactics) require
> a calibration. Without it the system still tracks + does possession, in pixels.

**Accuracy tip:** add `lines:` and `circle:` sections to the calibration YAML
(see `demo/calib/SNMOT-060.yaml`) — points ON the far touchline / halfway line
and ON the centre circle anchor the homography across the whole frame. A
circle-only calibration compresses sideline play toward the centre in the
heatmap/radar (fixed constraint-based fit handles the perspective tangency bias).

**TensorRT (≈2× inference):**
```bash
python -c "from ultralytics import YOLO; YOLO('runs/detect/soccernet_v2/weights/best.pt').export(format='engine', half=True, imgsz=960, device=0)"
```
`scripts/demo.sh` prefers `best.engine` automatically when present.

## 4. Offline analysis (report + heatmaps + annotated video)
```bash
python -m soccer_analytics.pipeline --video campus_match.mp4 \
    --weights runs/detect/soccernet/weights/best.pt \
    --calibration campus.yaml --render --out out/
```
Produces `out/stats.json` (per-player load + injury + team tactics), `out/heatmap_team{1,2}.png`,
and `out/annotated.mp4`.

## 5. Live dashboard (near real-time)
```bash
python -m soccer_analytics.realtime --video 0 \
    --weights runs/detect/soccernet/weights/best.pt --calibration campus.yaml
```
`--video 0` = webcam (or a file, or an RTSP URL). Four windows: **Match**
(tagged players, ball, possession), **Radar** (top-down + Voronoi pitch-control
shading), **Heatmap** (live), **Analytics** (possession, team performance, player
table, injury risk, tactical recommendations, FPS). Keys: `q` quit, `space` pause.

---

## 6. Wearable demo with 1–2 units

Only 1–2 players wear a device; **everyone else is analysed by vision** (distance,
speed, metabolic power, load, injury all still computed). The wearable adds
HR/SpO2 + true IMU PlayerLoad for the tagged players, drawn with a cyan marker.

### 6.1 Network (pick one; see [HARDWARE.md](HARDWARE.md))
- **Phone hotspot (recommended):** ESP32 + laptop both join it. Find the laptop IP:
  ```bash
  hostname -I        # e.g. 192.168.43.50
  ```
  Set `ENDPOINT="http://192.168.43.50:8000/ingest"` in the firmware.
- **Cloud relay:** run `python -c "from soccer_analytics.sensors.server import run_relay; run_relay(port=8000)"`
  on a VPS; point both the ESP32 and (a subscriber on) the laptop at it.

### 6.2 Flash the wearable (TCP stream — current hardware)

Open [`firmware/wearable_stream/wearable_stream.ino`](firmware/wearable_stream/wearable_stream.ino)
in Arduino IDE. Set `WIFI_SSID` / `WIFI_PASS`. Board: **ESP32S3 Dev Module**.
Serial monitor prints `ESP32 IP` and `RAW STREAM: tcp://…:9000`.

Then in WSL:

```bash
python -m soccer_analytics.hub --esp32 192.168.x.x --player-id 7
# browser: http://127.0.0.1:8081/
python -m soccer_analytics.realtime --video campus_match.mp4 \
    --weights runs/detect/soccernet_v2/weights/best.pt --calibration campus.yaml \
    --sensor-hub http://127.0.0.1:8081 --player-id 7 --roster "3:7"
```

Beginner walkthrough: [docs/BEGINNER_TECHNICAL_REPORT.md](docs/BEGINNER_TECHNICAL_REPORT.md).

### 6.3 Bind wearers to vision tracks — two ways

**Automatic (jersey-number OCR, recommended):** give the wearers' shirt numbers →
player ids; the system OCRs jerseys, votes over frames, and auto-binds:
```bash
python -m soccer_analytics.realtime --video 0 \
    --weights runs/detect/soccernet/weights/best.pt --calibration campus.yaml \
    --wearable-endpoint 8000 --roster-numbers "7:7,10:10"
#            jersey #7 → player 7,  jersey #10 → player 10
```
Prints `[jersey] track N → #7 → player 7 (auto-bound)` once confident. Needs
`easyocr` (installed) and readable numbers; clearer/closer footage binds faster.

**Manual:** note the on-screen **track ids** of the wearers, then:
```bash
    ... --wearable-endpoint 8000 --roster "3:7,5:10"    # track 3→player 7, 5→10
```
The endpoint starts in-process (`http://0.0.0.0:8000/ingest`); the ESP32s stream
to it; bound players get a cyan marker + HR/SpO2 metrics. (Track ids can switch on
long occlusions — auto-bind re-resolves; manual needs re-binding.)

### 6.4 Dry-run without hardware
Simulate the wearables to rehearse the whole flow:
```bash
python -m soccer_analytics.realtime --video campus_match.mp4 \
    --weights runs/detect/soccernet/weights/best.pt --calibration campus.yaml \
    --simulate-sensors --players 2 --roster "3:1,5:2"
```

## 7. Timestamp sync (how vision + wearable align)

Each wearable sample carries epoch time `t`; the live pipeline stamps every frame
with wall-clock time and `SensorVideoSync` matches the nearest sample per player
within ±0.5 s. Both sides run live → they share the wall clock. For post-hoc
analysis of a recorded video + a recorded wearable log, align by the capture
offset between the two recordings before feeding them in.

## 8. Reading the output

- **Dashboard / `stats.json` per player:** distance, top speed, HSR, sprints,
  accel/decel efforts, **metabolic power (W/kg)**, energy (kcal), PlayerLoad,
  speed-zone breakdown, and injury risk (0–1 + level + contributing factors).
- **Team tactics:** formation, Voronoi space-control %, width/depth/compactness,
  defensive-line height, pressing intensity, time in the attacking third,
  **shots + xG** (expected goals from shot location/angle via the Kalman-filtered
  ball), plus rule-based recommendations.
- **Vision vs wearable cross-check:** for tagged players, vision distance/speed can
  be compared against GPS, and the vision PlayerLoad proxy against the IMU value.

## 9. Campus-day checklist (footage + hardware, end-to-end)

Everything below is exercised by the test suite / dry runs — the day you get
campus footage and the wearable, the flow is:

1. **Record** a session from a fixed tripod camera (wider/higher = better).
2. **Label** ~200–500 frames (Roboflow or CVAT; classes: ball, goalkeeper,
   player, referee) → export YOLO format → `campus/data.yaml`.
3. **Fine-tune** (two-stage recipe, low-quality augmentation is automatic):
   `python -m soccer_analytics.train finetune --data campus/data.yaml --base runs/detect/soccernet_v2/weights/best.pt --epochs 60`
4. **Calibrate once**: `python -m soccer_analytics.calibrate --video session.mp4 --out campus.yaml`,
   then add `lines:` (touchlines/halfway visible from your camera) and `circle:`
   points for edge-to-edge accuracy (see `demo/calib/SNMOT-060.yaml`).
5. **Flash the wearable** (`firmware/soccer_wearable/`), point `ENDPOINT` at the
   laptop's hotspot IP, and run:
   `python -m soccer_analytics.realtime --video 0 --weights <best.pt|.engine> --calibration campus.yaml --wearable-endpoint 8000 --roster-numbers "7:7,10:10"`
6. Watch the live dashboard: possession, tactics, **injury risk**, and the
   **SUBSTITUTION WATCH** (fatigue-rate decline per player, with reasons).

## 10. Injury & substitution — how the predictions work

- **Injury risk** (`sensors/injury.py`): ACWR sweet-spot (Gabbett), HSR/sprint
  load, accel/decel efforts, metabolic load, and — with the wearable — cardiac
  drift and SpO2. Heuristic baseline today; `InjuryRiskModel.fit()` upgrades it
  to XGBoost the moment labelled injury data exists (the report's [5][6] method).
- **Substitution watch** (`substitution.py`): per-player **rate declines**
  between early and recent play (high-speed-running rate, work rate, sprint
  frequency, energy rate) + cardiac drift + injury risk + minutes played →
  ranked, explained substitution priorities. Appears in the dashboard
  recommendations ("SUB WATCH #7 …") and in `stats.json → substitution_watch`.
  `SubstitutionAdvisor.fit()`-style calibration on real substitution events is
  the natural extension once campus match logs exist.

## 11. Tests
```bash
PYTHONPATH=. python tests/test_pipeline_core.py
PYTHONPATH=backend/src:. python tests/test_hub_bridge.py
PYTHONPATH=backend/src:. python tests/test_web_api.py
```
