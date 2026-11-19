# Tracking, identity, team and ball benchmark (12 Nov 2026)

Everything here is reproducible from `data/benchmarks/*.json` and the scripts
named in each section. Protocol: SoccerNet-Tracking [E1]; the detector and every
tracker setting were chosen on **9 validation clips** held out of training
(SNMOT-071, 076, 099, 102, 112, 115, 165, 166, 169); the **49 official test
clips** were scored once with the frozen configuration. All trackers receive
identical cached detections (`scripts/benchmark_tracking.py cache`), so
differences are due to association only. Metrics: HOTA/AssA/DetA [E4],
IDF1 [E5], MOTA/IDSW [E6], computed by `soccer_analytics/mot_eval.py`
(TrackEval procedure; unit-tested). Players, goalkeepers and referees are
evaluated together; the ball is evaluated separately.

## 1. Detector

| Checkpoint (held-out test split, 7,350 frames) | mAP50-95 | Ball AP50 | Ball recall | Player AP50 |
|---|---|---|---|---|
| `soccernet_v2` (previous default) | 0.272 | 0.274 | 0.260 | 0.863 |
| `soccernet` (epoch 10 of an interrupted run) | **0.420** | **0.434** | **0.467** | **0.949** |

`soccernet_v2` had been selected using the test images as validation, which
both leaks and, as measured, does not even generalise; it is no longer the
default anywhere. The equal-data YOLOv8m continuation vs RT-DETR-L comparison
is in section 6.

## 2. Player tracking: 49 held-out test clips, frozen configuration

`data/benchmarks/tracking_test_soccernet_1280.json` (36,750 frames, 1,146 ground-truth identities).

| Tracker (identical detections) | HOTA | DetA | AssA | IDF1 | MOTA | ID switches | Identities | ms/frame |
|---|---|---|---|---|---|---|---|---|
| ByteTrack (previous production) | 56.7 | 70.6 | 46.1 | 64.8 | 85.1 | 2,718 | 2,922 | 2.0 |
| IMM-EIoU + CMC + colour histogram | 63.5 | 72.7 | 56.0 | 75.1 | 87.8 | 1,634 | 1,872 | 2.0 |
| IMM-EIoU + CMC + football ReID | 65.6 | 72.9 | 59.3 | 77.9 | 87.8 | 1,464 | 1,897 | 2.1 |
| 2-mode IMM + EIoU + CMC + ReID | 65.3 | 72.9 | 58.8 | 77.4 | 87.9 | 1,451 | 1,911 | 3.0 |
| colour histogram + offline link | 64.1 | 72.7 | 56.9 | 76.2 | 87.8 | 1,502 | 1,802 | |
| **IMM-EIoU + CMC + ReID + offline link (adopted)** | **69.7** | 72.9 | **66.9** | **86.4** | 87.9 | **999** | **1,295** | 2.1 + link |
| 2-mode IMM variant + offline link | 69.5 | 72.9 | 66.5 | 86.0 | 88.0 | 989 | 1,300 | |

Against the previous production tracker on unseen test clips: **HOTA +13.0,
IDF1 +21.6, ID switches −63 %, spurious identities from 2.5x to 1.13x the
true count**. Tracker timing excludes ReID embedding and camera-motion
estimation, which are computed once per frame and cached here. The tracker is
evaluated on players, keepers and referees together. For context only
(different detectors and training data): published SoccerNet-Tracking results
for Deep-EIoU are 85.4 HOTA and GTA 83.1 HOTA with stronger detectors
[T3, T6]; DetA here (72.9) shows the detector is now the main limit, which is
why the detector experiments in section 6 matter.

## 3. How each component was chosen (validation clips)

| Step | Validation HOTA | IDF1 | Note |
|---|---|---|---|
| ByteTrack (production) | 0.648 | | baseline |
| IMM-EIoU + CMC + colour histogram | 0.666* | | *exploratory, 6 clips |
| IMM-EIoU + CMC + football ReID (veto 0.65, weight 0.5) | 0.712 | 0.861 | grid of 9 settings |
| + offline linking (gap 750 frames, appearance gate 0.5) | **0.751** | **0.928** | grid of 18 settings |

Findings worth stating in the report:

- The 2-mode IMM filter did **not** improve on a single constant-velocity
  filter once expansion IoU was used (identical HOTA within 0.1). Expansion IoU,
  masked camera-motion compensation and appearance carried the gain.
- The trained ReID changed linking from slightly harmful (colour histograms
  merged team-mates) to clearly helpful.
- Cost: the tracker is ~2.6 ms/frame on CPU without appearance; with ReID
  crops and camera compensation the live adapter measured 22 ms p50 per frame
  (GPU shared with another job). Use `--tracker bytetrack` live when frame
  rate matters more than identity.

## 4. Re-identification (validation clips, within-clip retrieval)

| Appearance model | mAP | Rank-1 |
|---|---|---|
| Colour histogram (HSV shirt + shorts) | 0.288 | 0.389 |
| ResNet-18 BNNeck, batch-hard triplet + ID loss, team-mate-hard sampling (`scripts/train_reid.py`, 6,000 iterations, ~5 min) | **0.789** | **0.884** |

Trained only on the 48 training clips; queries come from the first half of
each validation clip, the gallery from the second half, and the distractors
are mostly team-mates in identical kits.

## 5. Team (side) separation (validation clips, vs ground-truth left/right)

| Method | Accuracy | Coverage |
|---|---|---|
| Frame-level colour voting (previous) | 0.768 | 0.968 |
| Identity-level clustering on ReID embeddings (ablation) | 0.704 | 0.890 |
| **Identity-level trimmed 2-means on CIELAB (L/2, a, b)** | **0.973** | 0.881 |

Root cause of the old failures: three clips with dark-vs-white kits differ in
*lightness*, not chroma, and yellow referees/keepers captured one of the two
chroma clusters. Halving L keeps it as a cue without letting shadows dominate,
and trimming 20% of identities during fitting stops a third kit from taking a
cluster. Uncovered identities are mostly keepers and referees, which abstain
by design.

## 6. Detector experiments

| Candidate | What was run | Result |
|---|---|---|
| RT-DETR-L (transformer), COCO init, 960 px | 3 of 8 planned epochs before an unplanned machine restart | 0.099 mAP50-95 on validation after epoch 3, against 0.42 for the YOLOv8m baseline. DETR-style models need far longer schedules (around 72 epochs in the RT-DETR paper), which do not fit this laptop budget. Not adopted. |
| YOLO26s-P2, native-scale 640 px tiles (SAHI-style), full-frame 1920 px inference | 30 epochs (`scripts/train_detector_p2.py`) | On the 49 test clips: 0.427 mAP50-95 against 0.422 for YOLOv8m at 1280 px; ball AP50-95 0.161 against 0.158. Inference 36 ms per frame against 23 ms. The gain is within run-to-run noise and costs 57 % more time per frame, so YOLOv8m stays in use. A measured negative result: tiling helped training fit the ball, but the full-frame test score barely moved. |

Why YOLO26-P2 on tiles: the ball is 6-15 px. A stride-4 head keeps a 4 px
grid for it, YOLO26 is NMS-free (no suppression of overlapping players), and
training on native-resolution tiles avoids shrinking the ball by 1.5-3x as
whole-frame training at 1280 or 640 px does. Promotion stays manual and
evidence-based (`CPG44_MODEL_PATH` or the Adapt page's "Use" button, which
only appears when the new model wins on held-out frames).

## 6b. Shirt numbers (SoccerNet Jersey-2023, 1,211 test tracklets)

Labels are per tracklet: in most crops of a tracklet the number is turned
away, folded or blurred. The score is the challenge metric, the share of
tracklets whose number (or "not readable") is right.

| Reader | Test accuracy | Readable numbers | Unreadable correctly refused |
|---|---|---|---|
| ResNet-18 crop classifier, tracklet vote | 39.6 % | 22.2 % | 81.4 % |
| **ResNet-18 attention MIL on torso crops** (Ilse et al. 2018) | **53.3 %** | **48.4 %** | 65.4 % |
| Same, 60 epochs with random erasing | 53.4 % | 48.8 % | 64.5 % |

Training on bags of crops with the tracklet label roughly doubled readable
numbers. Held-out training tracklets scored 82 to 85 %, far above test,
because the same players appear in other training tracklets; only the test
split is reported. Published results reach about 87 % with pose-guided torso
crops and a scene-text recogniser (Koshkina & Elder 2024), the next step. In
the app a number is only shown once a whole track agrees, two players on one
side cannot share a number at the same time, and the squad page turns numbers
into names.

## 7. Ball, per-frame location within 12 px of ground truth

| Method | Validation P / R / F1 | Test P / R / F1 |
|---|---|---|
| Highest-confidence box ≥ 0.1 (previous) | 83.3 / 69.9 / 75.6 | 81.8 / 70.1 / 75.1 |
| Dynamic-programming path + bounded quadratic gap fill (adopted) | 85.1 / 75.0 / 79.4 | **83.4 / 75.1 / 78.6** |

The DP parameters (tau 0.2, birth 0.6, max skip 12, motion weight 0.25) were
the best of a 36-setting grid on validation. Spare balls near the touchline or
goal remain the main false-positive source; the shot detector therefore needs
a sustained strike next to a player.

## 8. Physics checks with known ground truth (`scripts/evaluate_physics.py`)

3D ball from one camera (40 random shots per noise level, broadcast-like camera
recovered from its homography):

| Pixel noise | Median 3D error | Peak-height error | Aerial arcs found | Ground-projection error |
|---|---|---|---|---|
| 0.5 px | 0.18 m | 0.07 m | 39/40 | 5.1 m |
| 1 px | 0.39 m | 0.15 m | 40/40 | 5.2 m |
| 2 px | 1.18 m | 0.44 m | 37/40 | 5.2 m |
| 3 px | 1.99 m | 0.78 m | 35/40 | 5.2 m |

Vest heart rate while running (20 synthetic cadence/heart-rate combinations):
band-pass only MAE 47.5 bpm (locks onto step cadence); 3-axis NLMS 2.9 bpm;
after the quality gate 1.2 bpm with 14/20 windows accepted.

These verify the methods. Field accuracy needs real references: labelled 3D
ball data (SoccerNet-v3D [B3]) and a chest-strap ECG for heart rate.

## 8b. Getting tracked players onto the board

The first version placed players through an online Kalman gate. Whenever a
re-linked identity or a calibration correction made a position jump, the
gate rejected it and the player vanished from the top-down view. The board
showed a median of 11 to 15 players per moment while about 20 were on
screen. Offline, each projected foot point is now judged against its own
+-6 frame neighbourhood (Hampel-style rolling median), then smoothed
(Savitzky-Golay). On SNMOT-074: **13,635 of 15,113 player boxes placed**
(61 off the pitch, 69 outliers), **median 19 players per moment** on the board.
Referees are drawn as officials, players who just left the camera view fade
out over 4 s, and the camera's ground footprint is shaded, so a missing
player is explained, not hidden.

Spare balls: a motionless ball by the goal or touchline forms a perfectly
consistent track and can win the ball-path selection. A spot that never
moves and is seen in at least 60 % of the clip is now treated as a spare
ball. On the 9 validation and 49 test clips this changes nothing (F1 identical),
while fixing the SNMOT-074 demo where the spare ball had been chosen.

Ball on the board: the 3D ball fit recovers a pinhole camera from the pitch
homography. Clicked or line-refined homographies are not exact cameras, and
forcing the rotation to be orthonormal moved every ground point: 37 px on the
SNMOT-074 calibration and 105 px on SNMOT-060, several metres on the pitch.
Rolling balls were fitted through that camera, so the ball sat about 5 m from
the player controlling it and possession was almost never assigned. Now the
camera is refined against the homography (Zhang 2000, maximum-likelihood step;
residual 2.4 px and 6.5 px) and rolling balls are fitted through the
homography itself, the same map that places the players. On SNMOT-074 the
median distance from a rolling ball to the nearest player fell from 5.0 m to
1.75 m, and the share of ground-ball moments with a player within 2 m rose
from 10 % to 55 %. The synthetic 3D benchmark (section 8) is unchanged, as
expected for an exact camera (`test_rolling_ball_agrees_with_a_non_physical_homography`).

Goalkeepers wear a third kit, so colour clustering leaves them without a
side. The side is now read from the offside law: in each moment the side
whose deepest outfield player is nearest the keeper's goal defends it, and the
keeper joins that side when at least 70 % of moments agree.

### 3D ball: from 309 m/s to physics (November 2026, second pass)

Measured at full frame rate, the ball on the demo clips "moved" up to
309 m/s between frames (99th percentile 134 m/s on SNMOT-074, 105 m/s on
SNMOT-060), where a hard shot is about 30 m/s. Four causes, four fixes:

| Cause | Fix |
|---|---|
| Line snapping moved the pitch mapping by 4 to 9 m in single frames near the far goal | Offline smoothing of the mapping: a grid of image points followed through every frame's mapping, Hampel and Savitzky-Golay filtered, homography refitted per frame (median correction 0.26 m, 95th percentile 1.06 m) |
| Flights fitted to 6 frames invented 12 to 14 m apexes (towards the lens and upwards look alike) | A flight needs 0.4 s of observations, a clear absolute gain over the ground model, launch speed under 36 m/s and an apex reachable with the vertical speed seen |
| All pixels were warped into one reference frame, which only holds for a perfect rotating camera | Each observation is fitted through its own frame's camera; cameras of a panning shot share one position (pan-tilt-zoom model) |
| An airborne ball projected onto the grass seems to race | A stretch faster than any rolling ball is rebuilt as a flight between take-off and landing (constant horizontal speed, z = g/2 (t - t0)(t1 - t)). A landing needs 6 calm frames, because the shadow of a ball moving away and back also stops for a moment at the top of its flight. Anything still impossible is withheld rather than invented |

| Clip | 99th percentile speed | Maximum | Frames over 40 m/s | Positions withheld |
|---|---|---|---|---|
| SNMOT-074 | 134 to **31.6 m/s** | 309 to **32.2 m/s** | 62 to **0** | 0 of 722 |
| SNMOT-060 | 105 to **26.9 m/s** | 191 to **46.4 m/s** | 55 to **1** | 0 of 674 |

The SNMOT-074 goal kick is now one 3.4 s flight peaking at 14.5 m, the
SNMOT-060 high balls peak at 8.0 and 6.5 m. The 3D view reads the ball at
every frame (`ball_track`), not the 5 Hz board samples, and draws it at its
real size.

## 9. Known limits

- SoccerNet is broadcast footage with a moving camera; the campus setting is a
  fixed camera, which should be easier for calibration but has smaller players.
  Campus clips must be labelled to confirm transfer.
- Speed and distance depend on calibration quality; demo top speeds of
  33-36 km/h for two players show residual calibration error during pans.
- The tracklet linker uses offline context; the live path uses the online
  tracker only.

## 10. Transformer trackers: studied, not adopted (yet)

| Method | Idea | Reported sports result | Fit for CPG44 now |
|---|---|---|---|
| TransT [T10] | Transformer feature fusion for *single*-object tracking | n/a (SOT) | Not a multi-player tracker; the ball path problem is better served by temporal detection + physics (sections 7-8) |
| TrackFormer [T9], MOTR [T11] | Track queries carried through a DETR decoder; detection and association learned jointly | below tracking-by-detection on SportsMOT | Needs end-to-end training on long clips; heavy for an 8 GB laptop |
| MeMOTR [T12], MOTIP [T13], SambaMOTR [T14], MATR [T15] | Memory, ID-as-classification, state-space models, motion-aware queries | SportsMOT HOTA ≈ 70-75 (MATR 72.2-72.7) | Strongest of the family, still below tuned tracking-by-detection |
| Deep-EIoU [T3], GTA [T6], ARC-SORT [T4] | Detector + association with expansion IoU, appearance, global linking | SportsMOT HOTA 77-82; SoccerNet 79-85 | Adopted (sections 2-3) |

Decision: the SportsMOT leaderboard and the SoccerNet 2025 challenge report
[E7] show tracking-by-detection with sports ReID and offline association ahead
of end-to-end transformer trackers, at a fraction of the compute. CPG44 keeps
the transformer where it pays off cheaply, in the detector (RT-DETR comparison,
section 6), and uses physics and global association for identity. A
transformer tracker becomes worth revisiting once labelled campus clips exist
to fine-tune on, using the same cached-detection protocol for a fair test.
