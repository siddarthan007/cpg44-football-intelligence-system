# Running CPG44

The wearable path requires the public relay. There is no ESP32 LAN mode.

```bash
cd /home/siddartha/capstone
conda activate soccer
export CPG44_RELAY_URL="https://cpg44.nivaspms.com"
export CPG44_RELAY_TOKEN="<same token as the VPS>"

scripts/start_system.sh \
  --video <camera-index-or-video-path> \
  --weights <detector-weights.pt> \
  --calibration <pitch-calibration.yaml> \
  --player-id 7 \
  --match-id live \
  --roster "<track-id>:7" \
  --jerseys "7:7"
```

Open `http://localhost:5173`. Use prerecorded footage for a repeatable panel
demo or a camera index for a field test. The wearable still sends real data
through the relay in either case.

For video-only analysis:

```bash
python -m soccer_analytics.pipeline \
  --video campus-session.mp4 \
  --weights runs/detect/campus/weights/best.pt \
  --calibration configs/campus-pitch.yaml \
  --render \
  --out out/campus-session
```

See [DEMO_TUTORIAL.md](docs/DEMO_TUTORIAL.md) for the full walkthrough.
