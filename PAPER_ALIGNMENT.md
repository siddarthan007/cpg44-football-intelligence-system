# CPG44 report ↔ implementation alignment

How the built system maps to the report's objectives, research gaps, and cited
literature (`CPG44_Report.pdf`). "Following the paper" concretely.

## Objectives → modules
| Report objective | Implementation |
|---|---|
| O1 Integrated vision + wearable data framework | `pipeline.py`, `realtime.py`, `sensors/` (fusion, sync, endpoint) |
| O2 Wearable HW (GPS/accel/pulse-ox) | `firmware/wearable_stream/` (ESP32-S3 TCP stream) + `sensors/hub.py` receiver, `HARDWARE.md` |
| O3 ML for performance trends + tactical decisions | `trajectory.py` (LSTM), `pass_prediction.py`, `catapult.py`, `sensors/injury.py`, `tactical_engine.py`, `sensors/recommend.py`, `shots.py` |
| O4 Evaluation on public datasets | SoccerNet converter + train/eval; trajectory ADE/FDE; detector mAP |

## Research gaps (report §4) → how addressed
- **"Very limited multimodal fusion (vision *or* wearable, not both)."** → `sensors/fusion.py` fuses vision external load (di Prampero metabolic power, speed zones, HSR, sprints, accel/decel) with wearable internal load (HR/SpO2/IMU), time-aligned per frame. Vision computes load for *all* players; wearable augments the tagged ones.
- **"Commercial platforms (Genius, Catapult) expensive/proprietary."** → open pipeline: Catapult-style PlayerLoad + metabolic power from vision alone (`catapult.py`), a €-cheap ESP32 wearable, all local.
- **"Insights only post-match; near-real-time decision support underdeveloped."** → `realtime.py` runs ~16–21 FPS with a live 4-window dashboard, wearable stream time-synced (wall-clock ±0.5 s), on-the-fly injury/tactical recommendations.

## Literature review (report references) → coverage
| Ref | Cited method | Implemented |
|---|---|---|
| [1] YOLO (Redmon 2016) | real-time detection | `tracker.py` YOLOv8, `train.py` (SoccerNet + 2-stage finetune) |
| [2] ByteTrack (Zhang 2022) | MOT by associating every box | `tracker.py` `sv.ByteTrack`, GK→player remap, ball-from-raw |
| [3] Yousef 2025 | YOLO possession; poor ball (36%) | possession (`metrics.py`); ball handled with conf=0.1 + Kalman + interpolation + staleness guard (the cited weakness) |
| [4] Zheng 2025 | CV-for-football review (occlusion, small-object) | Kalman occlusion-coast (`filters.py`), imgsz 1280 for the ~12 px ball, camera-motion comp (`view.py`) |
| [5] Pilka 2023 | GPS + **XGBoost** on **ACWR/HSR** injury | `sensors/injury.py` (XGBoost/RF), `catapult.py` (ACWR, HSR, player load) |
| [6] Leckey 2025 | tree-based (RF/XGBoost) injury risk | `sensors/injury.py` (`InjuryRiskModel`) + heuristic baseline |
| [7] Teixeira 2025 | AI for tactical/collective dynamics (CNN/LSTM, space-control, formation, centrality) | `tactical_engine.py`: Voronoi **space control**, **formation**, centroid, width/depth, **stretch index**, **surface area** (convex hull), pressing, thirds, phases |
| [8] Honda 2022 | **pass-receiver prediction** from video + **LSTM trajectories** | `trajectory.py` (**LSTM trajectory prediction**, trained on SoccerNet: ADE 14 px / FDE 26 px), `pass_prediction.py` (receiver likelihood: openness, lane clearness, progressiveness) |
| [9] SoccerNet (Giancola 2018) | dataset | `soccernet_to_yolo.py` converter; trajectory + detector training |

Extra (beyond the cited baselines): **xG / shot model** (`shots.py`), **jersey-OCR
auto-binding** (`jersey_ocr.py`), and a **Lab-space team assigner** (`team_assign.py`)
that fixes the green-jersey-on-grass failure all the referenced repos share.

## Honest limitations (matching the report's own framing)
- **Ball detection** is the weak link (ref [3]: ~36 % ball AP) — small/fast object;
  a full-dataset detector + tiling helps but it remains hard.
- **Homography accuracy** gates all metric analytics; a fixed calibrated camera is
  required. A per-frame **pitch-keypoint** model (planned) would remove this.
- **Pass/injury/xG models** are geometry/heuristic-grounded and *trainable* — they
  need labelled event/injury data to reach the accuracy of the cited DL models
  (Honda's 3D-CNN+LSTM+Transformer; Pilka's clinically-validated XGBoost).
- Optical HR/SpO2 (MAX30102) is unreliable under running motion — rest/recovery use.

## Trainable models delivered
| Model | Data | Metric |
|---|---|---|
| YOLOv8 detector | SoccerNet→YOLO | player mAP50 (train in progress) |
| Trajectory LSTM | SoccerNet gt tracklets (88 k windows) | ADE 14 px, FDE 26 px |
| Injury (XGBoost/RF) | `WorkloadFeatures`→label (bootstrap now; real later) | holdout acc ~0.69 |
| xG (logistic) | shot geometry (calibratable) | monotonic, penalty-plausible |
