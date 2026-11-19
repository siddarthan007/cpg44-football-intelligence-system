# RT-DETR and Replayfield: feasibility and evidence, 11 November 2026

**Recommendation:** evaluate RT-DETR as an interchangeable detector, retain the tracking/fusion contracts, and build an evidence-labelled 3D replay as a separate presentation path. RT-DETR is a plausible 8 GB inference candidate; no local RT-DETR speed, memory or accuracy result has yet been measured. A convincing replay does not establish metric tracking accuracy.

## What the public Replayfield material establishes

[Replayfield](https://replayfield.com/) describes a prelaunch, server-processed five-a-side upload workflow, elevated camera guidance, multi-camera support and review of uncertain goals. Its public HTML animates pitch players with JavaScript random phases and sine/cosine motion; its 3D examples embed `assets/replay-3d.webm`/`.mp4`. No model specification, calibrated position-error benchmark or independently validated accuracy result was found. These observations describe the landing-page demonstration, not its undisclosed backend. The displayed sample statistics are not a benchmark for CPG44.

The HTML was inspected directly; a temporary copy is in ignored `data/research_validation/replayfield-page.html`. No private API, account, uploaded match or proprietary server code was accessed. Do not infer that its backend uses RT-DETR, Gaussian splatting, NeRF, a particular pose network or a specific tracker from its marketing output.

## RT-DETR in this implementation

The current `tracker.Detector` constructs an Ultralytics `YOLO` wrapper. The downstream contract already consists of boxes, canonical roles and confidence arrays, so changing the detector need not replace ByteTrack/BoT-SORT, calibrated projection, identity binding, sensor fusion, workload or the UI. A small backend factory in `tracker.py`, selected explicitly in `PipelineConfig` and CLI, is the appropriate integration seam.

There are two different experiments:

| Candidate | Integration and benefit | Constraint to verify |
|---|---|---|
| Ultralytics RT-DETR-L | The lowest integration friction: use `RTDETR(...)` and convert its `Results.boxes` to the existing detection contract | Football fine-tuning, confidence calibration, ball/distant-player recall and square input cost; current YOLO constructor is not a validated universal loader |
| Official RT-DETRv4-S | More interesting compact student experiment for the 8 GB target | Native `.pth` checkpoints/configuration require the authors' runtime or an exported-model adapter; they are not interchangeable with Ultralytics RT-DETR weights |

The [official Ultralytics documentation](https://docs.ultralytics.com/models/rtdetr/) describes NMS-free detection and adjustable decoder evaluation depth. Context7 confirms separate `RTDETR` loading, export and query settings. Reducing decoder layers or queries can trade accuracy for latency; do not copy those settings without evaluating crowded players and small balls. ByteTrack's low/high thresholds must be recalibrated to the new confidence distribution rather than inherited blindly.

[RT-DETRv4](https://github.com/AIMIA-PKU/RT-DETRv4) is an ECCV 2026 work: the [paper](https://arxiv.org/abs/2510.25257) was first submitted on 29 October 2025 and revised on 8 October 2026. It uses foundation-model feature distillation during training to improve the deployed student. The released benchmark is COCO/T4, not football on an RTX 5060 Laptop. The [small-model configuration](https://github.com/AIMIA-PKU/RT-DETRv4/blob/main/configs/rtv4/rtv4_hgnetv2_s_coco.yml) references a DINOv3 teacher, so teacher-assisted training and student inference have different memory budgets. Verify a supported fine-tuning configuration before assuming that the full training recipe fits 8 GB.

The inspected [TensorRT reference inference script](https://github.com/AIMIA-PKU/RT-DETRv4/blob/main/tools/inference/trt_inf.py) defaults dynamic bindings to a maximum batch of 32 and its example resizes to 640 square. Our live adapter should allocate batch 1 at the explicitly tested input shape and reuse buffers. Its PIL conversion, annotation and video writing are demonstration work, not detector-only latency. Do not adopt its allocator or reported speed unexamined.

For an initial deployment experiment: student model, batch 1, FP16, fixed shape, no teacher in the live process, bounded optional ball ROI and sparse appearance computation. Start with a shape the checkpoint supports, then test greater resolution where small objects require it. Training should begin with AMP and physical batch 1-2, no RAM image cache, conservative workers and measured headroom; these settings are starting points, not a fit guarantee. Profile total device usage with rendering enabled. Do not load two full-frame detectors on every frame by default.

The ball needs its own experiment. A player detector can improve person AP yet make football analytics worse by missing a few-pixel ball. Compare one unified detector against a player-focused transformer plus a selectively invoked small ball/crop detector, at matched compute and matched resolution. The latter is a hypothesis, not an established advantage.

## How multiple replay angles can be constructed

This is a proposed CPG44 design, not a claim about Replayfield's private implementation.

| Layer | What is measured or estimated | What a new viewpoint adds |
|---|---|---|
| Calibrated ground tracking | Player contact point mapped to pitch X/Y, with support and uncertainty | Positions can be displayed from above or on a perspective field |
| Avatar replay | Ground positions plus modelled height, body animation and estimated movement heading | A virtual camera can orbit, sit behind the goal or follow a player |
| Pose reconstruction | Estimated 3D joints/body orientation, preferably constrained across time/views | Body motion becomes more faithful, with substantial occlusion and compute challenges |
| Ball flight reconstruction | Estimated or triangulated X/Y/Z and contact events | Height, flight and impact can be replayed and evaluated |
| Volumetric reconstruction | View-dependent appearance/geometry from sufficient imagery | Photorealistic novel views, at a different capture and compute cost |

A renderer can move its virtual camera around a simple avatar scene without recovering photorealistic backs of players. For the first replay version, use trusted pitch X/Y, standard avatar shapes and explicitly estimated heading. Heading from velocity is not measured torso direction, head orientation or gaze; a keeper-view camera does not imply knowledge of what the keeper actually saw.

For geometry, camera projection has the form `s [u,v,1] = K [R|t] [X,Y,Z,1]`. A pitch homography handles the ground plane `Z=0`. It does not recover an airborne ball's height. Calibration intrinsics/extrinsics and multiple synchronized views allow triangulation; a single observation leaves depth ambiguous. See the [OpenCV calibration and triangulation reference](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).

For one camera, make the first milestone a ground-based replay, with clear source support, not an asserted full 3D reconstruction. For two cameras, seek overlapping coverage, a useful baseline and different sightlines, then measure synchronization and reprojection errors. Adding a second near-identical angle can contribute little depth information. Camera placement and lens distortion should be validated before increasing model size.

For shot angles on the ground, use the angle between vectors from the shot point to the two goalposts: `atan2(abs(cross(a,b)), dot(a,b))`, using measured goal geometry. This is different from camera viewing angle, body orientation or ball launch elevation. A precise-looking numeric angle inherits position/calibration error.

## A particularly relevant 2026 3D research direction

[Physics-Based 3D Ball Trajectory Reconstruction from Monocular Soccer Video](https://lukaszgrad.github.io/soccer-ball3d/), CVsports 2026, and its [source](https://github.com/lukaszgrad/ball3d) compare contact-segmented flight models fitted through reprojection. It finds that observation noise and single-view ambiguity can matter more than additional physical model complexity. The published pipeline also uses annotated pivot/contact information; an automatic live version would need a separately evaluated contact detector. Treat this as an offline reconstruction experiment with 3D ground truth, not an immediately deployable guarantee.

CPG44's 1 Hz low-cost GPS and a single 6-axis IMU do not independently provide accurate 3D player position, absolute heading or gaze. IMU integration drifts; player contact location should remain anchored to calibrated vision and appropriately qualified spatial sensors. Wearable motion and valid pulse remain valuable evidence for effort and response rather than an excuse to invent missing 3D coordinates.

## Three controlled next experiments

1. **Detector substitution:** same held-out matches, class ontology, calibration, tracker and identity configuration; compare existing YOLO, fine-tuned RT-DETR-L and a properly adapted RT-DETRv4-S. Report player and ball recall by object size, false detections, HOTA/IDF1, accepted-pin coverage, position error, warm p50/p95, dropped frames and peak total VRAM. Keep initialization separate. Training budget and labels must be comparable.
2. **Calibration/contact ablation:** current box-bottom point versus ankle/segmentation-supported ground contact, with measured pitch landmarks, distortion correction and calibration uncertainty. Evaluate position error and coverage jointly. Do not loosen innovation gates merely to show more dots.
3. **Honest replay:** export an observed 10-20 second event with timestamps, identities, calibration version, position support and covariance; render simple avatars and an orbit camera. Interpolation may smooth display, but it adds no measured heatmap time or physical evidence. Add airborne-ball reconstruction only when its error can be evaluated against synchronized views or labelled 3D trajectories.

The first decision is therefore not an unconditional YOLO replacement. RT-DETRv4-S is a worthwhile compact detector experiment; calibrated geometry, identity continuity, reliable small-ball observations and evidence-aware replay are the larger system requirements.
