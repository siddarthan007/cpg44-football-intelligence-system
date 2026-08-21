# Capstone report to implementation alignment

The proposal identifies three practical gaps: vision and wearable systems are
usually isolated, commercial platforms are expensive and closed, and useful
near-real-time decision support is limited. The implementation addresses those
gaps with local, inspectable processing and a required memory-only relay.

| Report objective | Current implementation | Evidence still required |
|---|---|---|
| Integrate vision and wearable data | source timestamps, relay replay order, roster/jersey binding, frame-time fusion and aligned NDJSON | campus synchronization error distribution |
| Build a wearable with GPS, acceleration and pulse oximetry | tested ESP32 raw-stream sketch plus host PPG/IMU/GPS processing | per-unit reference validation and field reliability |
| Track players and ball | YOLO detector, ByteTrack, ReID layer, ball low-confidence recovery and stale-coast guard | held-out campus AP, HOTA/IDF1 and ball recall |
| Convert footage into tactical information | pitch homography, possession, passes, width/depth/centroid, line height, space control, shots/xG | event labels and coach-review agreement |
| Support performance and prevention decisions | load features, work-rate trend and explained substitution watch | prospective outcome labels; calibrated model evaluation |
| Evaluate public and local data | SoccerNet conversion/training tools and campus upload/training paths | frozen test split and published result table |

## Claims boundary

- YOLO, ByteTrack and the sensor processing paths execute real code. Their
  accuracy is not assumed from architecture names; it must be measured on the
  held-out footage.
- Physical metrics require a valid pitch calibration. Pixel-mode output is not
  labelled as metres.
- Team/jersey assignment includes an abstention path. Coverage must be reported
  beside identity accuracy.
- MAX30102 SpO2 is an uncalibrated estimate and optical readings are motion
  sensitive. Invalid windows remain unavailable.
- The current load warning is a heuristic coaching indicator. Supervised strain
  training is locked until sufficient independent outcome labels exist.
- ACWR can be a feature but is not treated as a diagnosis or a universal safe
  zone.

## Novel product contribution

The capstone's useful contribution is not a new detector name. It is the
evidence-preserving link from player track to synchronized body signal and from
that link to an explainable coaching view:

1. Tracklets persist across short occlusions and can be bound to a roster by
   manual confirmation or multi-frame jersey evidence.
2. The wearable uses an SNTP anchor, and the relay preserves each source time for the local video timeline.
3. External-load geometry and internal-load readings are kept with provenance,
   freshness and validity rather than merged into an unexplained score.
4. Coach prompts describe observed tactical/load changes and why they were
   raised; a missing sensor or calibration reduces the available claims.
5. Raw and fused session logs are retained so every dashboard statement can be
   replayed and evaluated later.

The evaluation design and acceptance measurements are defined in
`docs/CAPSTONE_ACCURACY_PROTOCOL.md`.
