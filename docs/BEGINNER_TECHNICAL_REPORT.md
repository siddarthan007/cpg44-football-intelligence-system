# How CPG44 works — a beginner technical report

This is the same system in everyday language. You do not need to be a computer-vision expert to follow it. Think of three layers stacked like a sandwich: **a small computer on the player**, **a laptop program that understands the sensors**, and **a camera program that watches the match**.

---

## 1. What problem are we solving?

Coaches already know *where* players are if they watch the video. They cannot see:

- how hard the body is working on the inside (heart rate, blood oxygen)
- true impact load from the trunk (the IMU in the vest)
- how video speed and wearable heart-rate drift together into **injury risk**

Commercial kits (Catapult, STATSports) do this, but they are expensive. This capstone builds a **low-cost local version**: a campus camera + an ESP32 vest + software that runs in **WSL (Linux inside Windows)**.

---

## 2. The three machines (and who talks to whom)

```
  Player vest (ESP32-S3)
       │  Wi-Fi TCP port 9000
       │  one JSON line per sample (IMU 100 Hz, PPG 25 Hz, GPS 1 Hz)
       ▼
  Sensor hub  (Python in WSL)   http://127.0.0.1:8081/
       │  processed HR / SpO2 / motion / GPS
       │  Capstone join key: match_id + PLAYER_id + timestamp
       ▼
  Vision pipeline  (YOLO + ByteTrack + pitch map)
       │
       ▼
  Live dashboard: video + radar + load + injury advice
```

**Important WSL fact:** the vest is on your **Wi-Fi LAN**. The hub running in Ubuntu must be able to open a TCP connection to that IP. On Windows 11 with WSL2 *mirrored* networking this usually just works. If it does not, put the laptop and the ESP32 on the same phone hotspot, then use the IP printed in the Arduino Serial Monitor.

The laptop **connects out** to the ESP32. The ESP32 does not need to know the WSL IP. That is the opposite of the older HTTP-POST firmware sketch.

---

## 3. Layer A — the vest (firmware)

**File you flash:** `firmware/wearable_stream/wearable_stream.ino`  
(this matches `c:\Users\LENOVO\Downloads\wearable_stream\wearable_stream.ino`)

| Chip | What it measures | How often |
|------|------------------|-----------|
| MPU6050 | acceleration + rotation | ~100 times per second |
| MAX30102 | infrared + red light through the finger/arm (raw PPG) | 25 times per second |
| NEO-6M | GPS lat/lon/speed | about once per second |

The vest is **deliberately dumb**. It does **not** compute heart rate on the chip. It only:

1. Joins Wi-Fi
2. Opens TCP port **9000**
3. Sends one JSON object per line, with a device clock in **microseconds**
4. Answers `SYNC,...` messages so the laptop can line up clocks

Why not compute HR on the ESP32? The MAX30102 math is noisy while running. The laptop has more CPU, uses the Maxim/SparkFun method in floating point, and can **refuse** a reading when the IMU says the player was shaking too hard. That is more honest than a fake-stable number on the watch face.

Set `WIFI_SSID` / `WIFI_PASS` at the top of the `.ino`, flash it, and copy the printed `ESP32 IP`.

---

## 4. Layer B — the sensor hub (receiver)

**Run in WSL:**

```bash
cd ~/capstone
python -m soccer_analytics.hub --esp32 192.168.x.x --http-port 8081 --player-id 7
```

(`python /home/siddartha/sensor_hub.py --esp32 …` still works: it now points at this repo.)

The hub does four jobs:

### 4.1 Clock sync (so video and vest share a timeline)

The ESP32’s `millis()` / `esp_timer` is **not** wall-clock time. The hub sends:

`SYNC,<id>,<laptop_time_ns>`

The vest replies with the times it saw the message. After many round-trips the hub fits a line:

> host time ≈ slope × device_microseconds + offset

Then every IMU/PPG/GPS packet gets a **host timestamp**. Vision frames use `time.time()` on the same laptop. Fusion keeps samples that fall within **±0.5 seconds** of a video frame.

### 4.2 IMU processing (motion, not speed)

- Subtract gyro bias while the vest is still
- Mahony filter → roll/pitch (yaw is relative; this board has no magnetometer in the stream)
- Remove gravity → linear acceleration
- **Do not** integrate acceleration to get speed. That always drifts. Speed comes from **GPS** and from **the camera**.

### 4.3 PPG → heart rate and SpO2

- Need ~4 seconds of IR/red samples
- Peak detection (Maxim method) → beats per minute
- Ratio-of-ratios polynomial → SpO2 *estimate* (not a medical device)
- Gates: finger must be on the sensor, IR and red must move together, IMU must be fairly still, two HR estimators must agree

That is why numbers often say **motion_artifact** during sprints. Treat in-play HR as **recovery / warm-up** unless you upgrade to a chest-strap ECG.

### 4.4 HTTP API (what the rest of the app reads)

| URL | Meaning |
|-----|---------|
| http://127.0.0.1:8081/ | live wearable dashboard |
| `/health` | is the vest connected? |
| `/api/latest` | full processed JSON |
| `/api/stream` | same JSON, 10 times per second |
| `/api/v1/observations/wearable` | Capstone-style record: `match_id + PLAYER_7 + timestamp` |

The last URL is copied from the larger football-CV platform’s **wearable contract**. Later you can store rows in a database and join them with camera tracks using those three keys. You do not redesign the detector to know about heart rate.

---

## 5. Layer C — the camera brain (vision)

This is the original CPG44 pipeline in `soccer_analytics/`.

1. **Detect** people and the ball (YOLOv8).
2. **Track** them across frames (ByteTrack) so “player 12” stays 12 when they overlap.
3. **Team colour** so red vs blue is automatic.
4. **Homography**: click pitch landmarks once (`calibrate`). Pixels become **metres** on a 105×68 m field.
5. **Kalman filter** smooths jitter and estimates velocity and acceleration.
6. **Catapult-style load** from those metres: distance, high-speed running, sprints, metabolic power.
7. **Tactics**: formation, Voronoi space control, pressing, shots/xG.
8. **Injury heuristic**: ACWR-style load + optional HR drift / low SpO2 from the vest.

Without calibration you still get tracking and possession, but speeds in metres (and injury load) need a calibrated camera.

**Fuse with the real vest:**

```bash
# terminal 1 — hub
python -m soccer_analytics.hub --esp32 192.168.x.x --player-id 7

# terminal 2 — vision (WSLg windows)
python -m soccer_analytics.realtime \
  --video your_match.mp4 \
  --weights runs/detect/soccernet_v2/weights/best.pt \
  --calibration campus.yaml \
  --sensor-hub http://127.0.0.1:8081 \
  --player-id 7 --roster "12:7"
```

`--roster "12:7"` means: vision **track 12** is wearing **player 7’s** vest. Or use `--roster-numbers "7:7"` so jersey OCR binds automatically.

You can also start both from one command:

```bash
python -m soccer_analytics.realtime --video clip.mp4 --weights best.pt \
  --esp32 192.168.x.x --player-id 7
```

---

## 6. How fusion is “glued” (one sentence each)

| Piece | Job |
|-------|-----|
| `HubSensorSource` | every 0.1 s, GET `/api/latest`, convert to `SensorSample` |
| `SensorVideoSync` | keep a short buffer per `player_id`; pick the nearest sample to this video frame |
| `FusionEngine` | attach HR/SpO2/IMU to that player’s vision velocity |
| `LoadEngine` | update metabolic power + PlayerLoad |
| `HeuristicInjuryModel` | turn those numbers into low / moderate / high risk |
| Dashboard | cyan ring = this player has a vest; yellow line = live HR/SpO2 |

Units: the hub IMU is **m/s²**. Catapult PlayerLoad expects **g**. The bridge divides by 9.80665. Gyro is converted from rad/s to deg/s.

---

## 7. What we borrowed from the other Capstone folder

The Windows folder `Downloads/Capstone/Capstone` is a bigger **multi-camera** platform (phones as cameras, identity fusion, React dashboard). We did **not** copy that whole stack into WSL. We took the ideas that make a *product* rather than a pile of scripts:

- a **written contract** for wearable rows (`match_id + global_player_id + timestamp`)
- config in `configs/default.yaml` instead of hiding numbers only in code
- honest limits (optical HR is noisy when sprinting; GPS is worse than a calibrated camera on the pitch)
- a beginner-facing “how it works” page on the hub (`/how-it-works`)

Your working product today is: **this repo’s vision stack + the real TCP vest + the FastAPI hub**.

---

## 8. How to run a campus demo (checklist)

1. Phone hotspot. Laptop + ESP32 join it.
2. Flash `firmware/wearable_stream/wearable_stream.ino`. Copy IP from Serial.
3. In WSL: `python -m soccer_analytics.hub --esp32 <that-ip>`
4. Browser: http://127.0.0.1:8081/ — you should see IMU move when you shake the board. Finger on MAX30102 (still) for HR.
5. Point a tripod camera at the pitch, calibrate once, run `realtime` with `--sensor-hub` and `--roster`.
6. If you have no footage yet: `bash scripts/demo.sh` (simulated wearable, SoccerNet clip).

---

## 9. Honest limitations (say these in the viva)

- SpO2/HR from MAX30102 are **estimates**, not clinical.
- In-play optical HR is often rejected (motion). That is a feature.
- Yaw heading is relative (no magnetometer on this stream).
- IMU cannot replace GPS/camera for distance.
- Detector accuracy depends on trained weights and lighting; the software pipeline is real even when the model is still being trained.
- University Wi-Fi often blocks device-to-laptop traffic — use a hotspot.

---

## 11. Browser dashboard (Football CV inspiration)

We did **not** copy the multi-camera identity engine or PostgreSQL stack. We
copied the *product shape*:

- FastAPI in `backend/src/cpg44_api/` (health, matches, analytics, wearable POST, live WebSocket)
- React + Vite dashboard in `frontend/` (Live, Wearable, Players, Analytics, Settings)
- Dark pitch-side CSS, sidebar nav, tactical pitch in metres, Stat/Panel chips
- Wearable rows join on `match_id + global_player_id + timestamp`

```bash
PYTHONPATH=backend/src:. python -m uvicorn cpg44_api.main:app --port 8000
npm install --prefix frontend && npm run dev --prefix frontend
```

Then http://127.0.0.1:5173/ . Keep the sensor hub on :8081 if the vest is on.

---

## 12. File map

| Path | Role |
|------|------|
| `firmware/wearable_stream/wearable_stream.ino` | what you flash |
| `soccer_analytics/sensors/hub.py` | receiver + dashboard |
| `soccer_analytics/sensors/hub_bridge.py` | hub JSON → vision `SensorSample` |
| `soccer_analytics/realtime.py` | live match UI |
| `soccer_analytics/pipeline.py` | offline video → `stats.json` |
| `backend/src/cpg44_api/` | FastAPI product API (REST + live WS) |
| `frontend/` | React dashboard (Vite) |
| `RUNNING.md` / `HARDWARE.md` | operator + wiring detail |
