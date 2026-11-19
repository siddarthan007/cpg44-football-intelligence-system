# CPG44: Predictive and Tactical Intelligence System for Athlete Performance Using Vision-Based Tracking and Wearable Data Analytics

Capstone CPG 44, Thapar Institute of Engineering and Technology (2026).

CPG44 follows every player and the ball from one fixed camera, pairs each
player's ESP32-S3 sensor vest (IMU, optical pulse, GNSS) with the right track,
and turns both into a short list of findings a coach can act on, each with the
evidence behind it. Wearable traffic always passes through the authenticated
memory-only relay at `cpg44.nivaspms.com`; all vision and fusion run locally.

```text
camera ─▶ detector ─▶ IMM-EIoU tracker + football ReID ─▶ offline linking ─▶ teams ─▶ pitch metres
   │                                     ball DP path ─▶ physics 3D arcs ──────────────┘    │
   └▶ camera check                                                                        ▼
ESP32-S3 vest ─▶ HTTPS relay ─▶ hub: NLMS pulse, PlayerLoad, clock map ─▶ fusion ─▶ coach engine
                                                                                        │
                       review workspace: footage, board, 3D, match strip, findings, load, to check
```

## What changed in the November 2026 pass (measured, not assumed)

| Area | Before | After | Where measured |
|---|---|---|---|
| Detector choice | `soccernet_v2` default, 0.27 mAP50-95 | `soccernet` 0.42 (ball AP50 0.27 → 0.43); continuation + RT-DETR comparison in `train_detector_compare.py` | 49 held-out test clips |
| Player tracking | ByteTrack | IMM-EIoU + masked camera compensation + trained football ReID + offline linking | [TRACKING_BENCHMARK.md](docs/TRACKING_BENCHMARK.md) |
| Re-identification | colour histogram, mAP 0.29 | ResNet-18 BNNeck trained on SoccerNet train, mAP 0.79 | 9 held-out clips |
| Team separation | frame voting, 77 % | identity-level trimmed 2-means, 97 % | 9 held-out clips |
| Ball | best box per frame, unbounded interpolation | dynamic-programming path, bounded gap fill, 3D arcs | test clips + synthetic 3D truth |
| Vest heart rate | optical pulse: no valid heart rate in any recorded session | AD8232 chest ECG, Pan-Tompkins beats and interval cleaning: 0.5 bpm mean error from rest to sprinting, 11.2 bpm for simple peak counting | 18 synthetic recordings, end-to-end check |
| IMU firmware | ±8 g, DLPF 44 Hz at 50 Hz sampling (aliasing) | ±16 g, ±2000 °/s, DLPF 21 Hz | data sheet |
| Vest delivery | 35 of 75 samples/s arrived; up to 106k dropped per boot; no valid heart rate in any recorded session | 96 samples per POST (relay limit 100), gap bridging under 0.32 s | 21 recorded sessions, [SENSOR_PIPELINE.md](docs/hardware/SENSOR_PIPELINE.md) |
| Vest and video clocks | laptop clock 27 to 767 ms behind the relay, uncorrected | NTP-style offset against the relay, plus lag measured by vest/track correlation | recorded sessions, synthetic test |
| IMU calibration | gyro bias absorbed motion; speed invented from acceleration | still-window gyro bias, accelerometer auto-calibration (Frosio 2009), standard PlayerLoad, no IMU speed | replay of a 37 min session, tests |
| GNSS | 1 Hz, distance from position differences | 5 Hz RMC+GGA, Kalman filter on Doppler speed, multipath gate | synthetic test |
| Who wears the vest | set by hand | found by correlating vest motion with every track (0.97 vs 0.29 next best, offset to 0.1 s) and cross-checked with the shirt number | synthetic test |
| Ball on the board | rolling ball 5.0 m from the nearest player (median) | 1.75 m; player within 2 m 10 % to 55 % of moments | SNMOT-074 |
| Fatigue | first half against second half | own peak 5 min window, net of the team's change, sprint speed, vest heart-rate factors | synthetic 30 min session |
| Phone camera | JPEG over plain HTTP (phone browsers refuse the camera) | WebRTC direct with relay signalling, relay fallback | end-to-end test with a simulated camera |
| Shirt numbers | generic OCR; first trained reader 39.6 % | attention MIL on torso crops, 53.3 % of test tracks (readable numbers 22 % to 48 %) | SoccerNet Jersey-2023 test |
| Heart rate | optical pulse, unusable while running | AD8232 chest ECG, Pan-Tompkins beats, within 2.5 bpm up to 185 bpm | synthetic ECG with step noise |
| Two cameras | one camera only | sound-synced fusion of two corner cameras, 35 % lower position error | simulated futsal game |
| Shots | a goal kick counted as a 573 km/h shot | speed capped at physics, direction over 5 frames, kicker must be attacking | SNMOT-074 |
| 3D ball | up to 309 m/s between frames, 5 Hz chords, phantom 14 m lobs | smoothed pitch mapping, per-frame cameras, flights rebuilt from take-off to landing; max 32 m/s on SNMOT-074 | full-rate ball on 2 demo clips |
| Live path | pieces tested alone | vest, relay, hub, API and phone camera checked together (`scripts/e2e_live.py`) | 11 of 11 stages |

All numbers live in `data/benchmarks/*.json` and are shown in the app's
**Evidence ledger**.

## Run the demo

```bash
cd /home/siddartha/capstone
conda activate soccer
scripts/build_demo_analysis.sh        # once: analyses the calibrated demo clips
scripts/demo.sh                       # API, relay hub, live vision and the UI
```

Open `http://localhost:5173`. The left rail follows a matchday: Camera check,
Pitch setup, Kits, Squad and vests, then Live, Matches and Add footage.

For your own ground (fixed camera, any pitch size, wide lens) follow
[docs/CAMPUS_GUIDE.md](docs/CAMPUS_GUIDE.md): calibrate in *Pitch setup*,
then optionally fine-tune the detector in *Adapt to ground*.

Each match opens with the footage beside a pitch board (Moment, Options,
Shape, Heat, Passes, 3D), a match strip, and tabs for Findings, Plan,
To check, Players, Teams and Shots. Options shows who controls each part of
the pitch and the pass and shot the decision engine rates
(`soccer_analytics/decision.py`); Plan shows each side's shape with and
without the ball and where it concedes space.

## Reproduce the measurements

```bash
PYTHONPATH=. python scripts/benchmark_tracking.py cache --split test --weights runs/detect/soccernet/weights/best.pt
PYTHONPATH=. python scripts/benchmark_tracking.py run   --split test --weights runs/detect/soccernet/weights/best.pt
PYTHONPATH=. python scripts/train_reid.py build && PYTHONPATH=. python scripts/train_reid.py train && PYTHONPATH=. python scripts/train_reid.py eval
PYTHONPATH=. python scripts/eval_team_separation.py
PYTHONPATH=. python scripts/evaluate_physics.py
PYTHONPATH=. python scripts/train_detector_compare.py eval
PYTHONPATH=. python -m soccer_analytics.sensors.session_report data/live_sessions/<session>
```

Tuning always uses the 9 validation clips; test clips are scored once.

## Checks

```bash
PYTHONPATH=backend/src:. python scripts/e2e_live.py --camera   # vest + relay + hub + API + phone camera
PYTHONPATH=backend/src:. pytest -q tests
npm run lint --prefix frontend && npm run build --prefix frontend
```

## Documents

- [Tracking, ReID, team and ball benchmark](docs/TRACKING_BENCHMARK.md)
- [Building the vest](docs/hardware/BUILD_GUIDE.md) and [wiring](HARDWARE.md)
- [Campus guide: camera, calibration, adaptation](docs/CAMPUS_GUIDE.md)
- [Interface design rationale](docs/DESIGN_RATIONALE.md)
- [References for the report](docs/REFERENCES.md)
- [Project atlas](docs/PROJECT_ATLAS.md), [accuracy protocol](docs/CAPSTONE_ACCURACY_PROTOCOL.md), [operating manual](docs/OPERATING_MANUAL.md)
- [RT-DETR and Replayfield study](docs/RTDETR_AND_REPLAYFIELD.md), [2026 research notes](docs/RESEARCH_2026.md)

Workload findings are coaching cues, not medical diagnoses. The AD8232 is a
monitoring front end, not a medical ECG.
