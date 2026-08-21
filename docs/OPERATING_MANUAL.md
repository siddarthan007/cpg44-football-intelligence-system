# CPG44 operating manual

## 1. Prepare the relay

Follow `docs/VPS_RELAY.md`. On the project PC, keep the same relay token in the
shell environment:

```bash
export CPG44_RELAY_URL="https://cpg44.nivaspms.com"
export CPG44_RELAY_TOKEN="<same token as the VPS>"
export CPG44_RELAY_CA_FILE="$PWD/configs/relay_ca.pem"
```

## 2. Flash the wearable

Use the tested `firmware/wearable_stream` sketch. In Dashboard > ESP32 setup,
select the WSL serial port and enter player ID, match ID, SSID and password. The
device sends only to the relay hostname.

Keep the match ID and player ID the same when starting the local stack.

## 3. Calibrate the camera

Use a fixed, elevated view. Measure the actual test area and create the pitch
mapping file:

```bash
python -m soccer_analytics.calibrate \
  --video /data/campus-session.mp4 \
  --out configs/campus-pitch.yaml
```

Recalibrate after moving or zooming the camera. Sample both team kits again if
lighting or jerseys change.

## 4. Start a session

```bash
scripts/start_system.sh \
  --video /data/campus-session.mp4 \
  --weights runs/detect/campus/weights/best.pt \
  --calibration configs/campus-pitch.yaml \
  --imu-calibration configs/wearable-7.json \
  --player-id 7 \
  --match-id live \
  --roster "12:7" \
  --jerseys "7:7"
```

Open:

- `http://localhost:5173` for the main dashboard
- `http://127.0.0.1:8000/docs` for the product API
- `http://127.0.0.1:8081` for sensor quality

The local session folder contains raw relay samples, fused frame rows and final
statistics. No sample is generated when a source is missing.

## 5. Field checks

- Confirm relay and wearable status are live.
- Confirm the match ID and player ID agree with the flashed values.
- Wait for valid contact before presenting BPM or SpO2.
- Confirm the pitch file matches the video.
- Review team and jersey labels before trusting player analytics.
- Keep frame age and packet age visible during the demo.

For a timing test, show an LED in the camera view while tapping the wearable.
Repeat the event across the session and compare the video time with the IMU
peak. Report the measured result, not a claimed value.

## 6. Training and evaluation

Split datasets by match, not by nearby frames. Report detector AP, ball recall,
tracking IDF1/HOTA and identity switches separately. Validate HR and SpO2 with
reference devices as described in `docs/CAPSTONE_ACCURACY_PROTOCOL.md`.

The injury-related model stays locked until enough reviewed player-sessions and
outcome labels exist. Until then, the dashboard shows explained workload
indicators only.

## 7. Stop

Press Ctrl+C in the launcher terminal and wait for the services to close. Then
run the checks listed in the root `README.md`.
