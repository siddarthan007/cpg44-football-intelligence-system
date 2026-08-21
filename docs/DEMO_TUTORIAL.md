# August 22 demonstration tutorial

Use a real prerecorded campus clip for a repeatable vision demo and a real
wearable for live sensor data. The software path remains the same as a camera
session. No sample or statistic is mocked.

For a prerecorded clip, the React view follows the clip clock. Live wearable
values prove the relay and sensor path, but they must not be described as the
body data of a player recorded earlier. A true fused player example must use
wearable data captured during the same recording or a live camera drill.

## Before the presentation

1. Deploy and test the relay using `docs/VPS_RELAY.md`.
2. Prepare one 2 to 3 minute 1080p clip with visible players, ball and pitch
   lines. The included 30-second clip is for a quick check, not the full panel
   demonstration.
3. Create a pitch calibration for that exact camera view.
4. Use weights that include player, goalkeeper, referee and ball classes.
5. Flash the wearable with match ID `live` and the correct player ID.
6. Check team colours and jersey binding using the same kits as the clip.
7. Run the full demo once and keep the resulting logs.

## Start the system

The repository already has a 30-second rehearsal clip, detector weights, and a
matching prototype calibration. Use this command for the first full rehearsal:

```bash
cd /home/siddartha/capstone
conda activate soccer
export CPG44_RELAY_URL="https://cpg44.nivaspms.com"
export CPG44_RELAY_TOKEN="<same token as the VPS>"
export CPG44_RELAY_CA_FILE="$PWD/configs/relay_ca.pem"

scripts/start_system.sh \
  --video demo/sample_match.mp4 \
  --weights runs/detect/soccernet_v2/weights/best.pt \
  --calibration demo/demo_calibration.yaml \
  --player-id 7 \
  --match-id live \
  --roster "7:7" \
  --jerseys "7:7" \
  --headless
```

This calibration is suitable for the prototype display. Do not present its
speed or distance values as the final campus accuracy result. Use a measured
campus calibration before reporting those values as evaluation evidence.

For another clip, replace the three file paths and confirm the visible track ID
before setting the roster pair:

```bash
cd /home/siddartha/capstone
conda activate soccer
export CPG44_RELAY_URL="https://cpg44.nivaspms.com"
export CPG44_RELAY_TOKEN="<same token as the VPS>"
export CPG44_RELAY_CA_FILE="$PWD/configs/relay_ca.pem"

scripts/start_system.sh \
  --video /path/to/demo-footage.mp4 \
  --weights /path/to/best.pt \
  --calibration /path/to/demo-pitch.yaml \
  --player-id 7 \
  --match-id live \
  --roster "12:7" \
  --jerseys "7:7" \
  --headless
```

Open `http://localhost:5173` and use full-screen mode. The React Live page labels
the source as processed footage while it publishes current tracking frames.

## What to show

1. Live page: annotated footage, readable player IDs, ball state and pitch
   markings.
2. Player tagging: team colour and the link between track, jersey and wearable.
3. Wearables: packet age, BPM, SpO2 estimate, motion and signal quality.
4. Analytics: distance, speed, team shape, possession evidence and workload.
5. Data quality: explain one blank or warning instead of hiding it.
6. Relay recovery: stop the local wearable processor briefly, start it again
   and show cached samples continuing in relay order.

## Ten-minute panel order

- 0:00 to 2:00: problem, scope and architecture
- 2:00 to 4:00: player, ball and pitch tracking
- 4:00 to 5:30: wearable and relay
- 5:30 to 7:30: tactical and workload analytics
- 7:30 to 8:30: training and accuracy plan
- 8:30 to 10:00: contribution, progress and next work

## Final check

```bash
curl --fail https://cpg44.nivaspms.com/health
curl --fail http://127.0.0.1:8081/health
curl --fail http://127.0.0.1:8000/api/v1/health
```

Keep a second local copy of the demo footage and calibration file. If the
wearable link fails during the panel, continue the real video pipeline and show
the recorded relay/session evidence from the completed rehearsal. Do not label
recorded evidence as live.
