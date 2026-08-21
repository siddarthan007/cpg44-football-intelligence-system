# CPG44 measurement and accuracy protocol

This protocol turns the platform into an assessable capstone rather than a
screen demo. Freeze model weights, calibration files and software revision
before each evaluation. Keep training, validation and test sessions separate by
match and date.

## 1. Required provenance for every session

Record:

- session ID, date, pitch dimensions and weather/lighting;
- camera model, resolution, frame rate, position, height and zoom;
- detector weights hash, confidence threshold and ByteTrack configuration;
- pitch and team-colour calibration file hashes;
- wearable serial/player ID, placement and IMU calibration hash;
- raw wearable log, fused snapshot log and manual ground-truth annotations;
- software commit and whether PyTorch used CPU or CUDA.

Never replace unavailable readings with a typical value. Store `null`, the
rejection reason and the relevant quality score.

## 2. Vision evaluation

Use held-out campus footage with representative daylight, shadow, occlusion,
motion blur and kit combinations.

| Component | Ground truth | Report |
|---|---|---|
| Detection | frame-level boxes by class | AP50-95, AP50, precision, recall per class; ball recall separately |
| Tracking | identity-labelled tracklets | HOTA, IDF1, MOTA, identity switches and fragmentation |
| Ball | labelled ball centres/absence | recall, centre error in pixels, false coast duration |
| Pitch mapping | surveyed landmarks | median and p95 reprojection error in pixels and metres |
| Speed/distance | timed runs or trusted reference | MAE, bias and p95 absolute error by speed band |
| Team assignment | manual kit labels | accuracy and coverage; abstentions reported separately |
| Jersey identity | tracklet jersey labels | tracklet accuracy, coverage, wrong-bind count and time-to-bind |

An identity method is not improved if it raises accuracy by silently classifying
only easy players. Report accuracy and coverage together. Jersey binding should
require multi-frame votes, a roster constraint and an abstention path.

For live demonstrations, report measured processing FPS, end-to-end frame age
and dropped-frame percentage on this PC. Do not infer FPS from the video file's
nominal frame rate.

## 3. Sensor evaluation

### Heart rate

Compare the MAX30102 output with an ECG chest strap during stillness, warm-up
and post-exercise recovery. Align clocks first. Report valid-window coverage,
MAE, median absolute error, bias, 95% limits of agreement and error by motion
band. Running windows rejected by the motion/optical gate count against coverage
but not as invented measurements.

### SpO2

Compare only in still or low-motion conditions against a certified pulse
oximeter. Report bias, MAE, 95% limits of agreement and coverage across the
observed saturation range. The platform must continue to label MAX30102 SpO2 as
an uncalibrated estimate unless a device-specific validation justifies otherwise.

### IMU

Use six static orientations to estimate bias and axis scale. Confirm each axis
measures approximately one g in its positive and negative orientation. Use a
stationary run for gyro bias, then controlled impacts/turns for repeatability.
Report axis error, magnitude error, noise standard deviation and drift. Do not
obtain match distance by double-integrating acceleration.

### GPS

Survey the pitch corners and repeat stationary fixes. Report horizontal scatter,
fix availability, HDOP distribution and speed error on straight timed runs.
NEO-6M position is a cross-check; calibrated camera geometry is the primary
on-pitch position source.

## 4. Wearable/video synchronization

Perform at least 20 synchronization events across the beginning, middle and end
of a session. Use a visible LED transition paired with a sharp IMU tap. Measure
absolute camera-to-IMU offset after source-time mapping and report median, p95
and maximum. Also report relay delay, delay variation, sequence gaps and replayed
sample count.

The acceptance threshold should be tied to the camera frame period. At 25 fps,
one frame is 40 ms; a practical target is median offset no greater than one
frame and p95 no greater than two frames.

## 5. Tactical and coaching validation

For a selected set of phases, have two football-knowledgeable reviewers label
possession, passes, shots, team shape and pressure events independently. Resolve
disagreements only after recording inter-rater agreement. Report event precision,
recall and timing error before presenting derived recommendations.

Useful, defensible live outputs include:

- team width, depth, centroid, line height, stretch and attacking-third presence;
- possession/pass/shot evidence with explicit ball-observation quality;
- individual distance, high-speed running, sprint count and repeat-sprint decline;
- external-load versus valid recovery-heart-rate mismatch;
- substitution-review priority with measured reasons and data freshness.

## 6. Outcome-model protocol

An injury or high-strain model needs prospectively defined labels from qualified
staff. A rule-derived score cannot be used as its own training label. Require at
least 100 labelled sessions and 10 cases per class before the training action is
enabled; larger cohorts are strongly preferred.

Use a chronological or leave-one-player-out test design. Report class prevalence,
ROC-AUC, average precision, Brier score, calibration curve, sensitivity,
specificity and confidence intervals. Compare with simple baselines. Acute to
chronic workload ratio may be one feature, but must not be the diagnosis or the
sole decision rule.

Until this evidence exists, name the current output `load indicator` or
`substitution watch`, show its contributing factors and state that it is not a
medical prediction.

## 7. Demo acceptance checklist

A session is demo-ready only when:

- the dashboard source says `live`, not recorded fallback;
- camera frame age and wearable last-seen age are visible and acceptable;
- pitch and team calibrations match the current setup;
- identity fragmentation and ball-observation warnings are reviewed;
- invalid PPG values are blank rather than carried forward indefinitely;
- the raw and fused NDJSON files are growing;
- stopping the launcher closes every process and writes final statistics;
- disconnecting the wearable or relay produces an offline state without
  breaking the video path or dashboard.
