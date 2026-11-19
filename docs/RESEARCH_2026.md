# Research and experiment plan, 11 November 2026

The useful goal is accurate, inspectable football evidence linked to wearables.
A newer model name alone cannot establish that result. This review connects
primary papers and actual source code to the failures found in this repository.
Implementation and measurement status are kept separate below.

## Recent work and its relevance

| Work and verified date | What to investigate here | Evidence and limitation |
|---|---|---|
| [Sports motion feasibility by quantile regression](https://arxiv.org/html/2610.36183), 26 October 2026 preprint | A small heading-relative time-to-arrival network for tactical reachability, after reliable kinematics exist | Learns lower-tail arrival time; transfer and reachable-set calibration require evaluation. It is not a tracking detector. |
| [ARC-SORT](https://arxiv.org/html/2610.22306), 15 October 2026 preprint; [code](https://github.com/Athlitix/arcsort) | IMM motion, uncertainty-adaptive association regions and delayed resolution of ambiguous crossings | Strong next tracker experiment. Its offline merger and benchmark embedding stack are distinct from a small live tracker. Paper results are not campus results. |
| [Cue-level fusion for soccer tracking](https://www.mdpi.com/2313-433X/12/10/512), 14 October 2026 | Appearance/motion/shape vetoes and adaptive offline tracklet linking | Supports component ablations. Its reported benchmark must not be substituted for evaluation of this code. |
| [ContextReID](https://arxiv.org/html/2610.13259), 9 October 2026 preprint | Compare contextual multi-crop player embeddings against compact single-crop sports ReID | Retrieval improvement does not automatically imply better end-to-end tracking. The paper explicitly reports greater compute/memory needs, making it an offline experiment first. |
| [Field Converter](https://arxiv.org/html/2609.10498), 9 September 2026 preprint | Geometry-initialized temporal residual correction for player ground/root position | Requires calibrated footage and pose supervision; airborne motion remains a difficulty. Do not deploy a randomly initialized correction network. |
| [ELASTIC](https://arxiv.org/html/2608.30227), 31 August 2026 preprint | Align reviewed ball events with reliable trajectories using candidate touches and ordered sequence alignment | Addresses event/tracking synchronization. It does not establish camera-to-ESP32 clock synchronization. |
| [Training-free off-screen player imputation](https://arxiv.org/abs/2607.11548), 13 July 2026 preprint | Quantify the effect of partial visibility on spatial control | An inferred off-screen player must remain separate from an observed player pin and heatmap. |
| [SoccerMaster](https://arxiv.org/html/2512.11016), 7 May 2026 revision, CVPR 2026 | Soccer-specific shared spatial/temporal representations, identification and learned field cues | Reviewed the SigLIP backbone and temporal blocks in [source](https://github.com/haolinyang-hlyang/SoccerMaster/blob/main/codes/SoccerMaster/models/soccer_master.py). It remains a separate research baseline: large-scale pretraining and jersey/goalkeeper limitations are explicit in the paper. |
| [Broadcast2Pitch](https://openaccess.thecvf.com/content/WACV2026/html/Oo_Broadcast2Pitch_Game_State_Reconstruction_from_Unconstrained_Soccer_Videos_WACV_2026_paper.html), WACV March 2026; [authors' repository](https://github.com/yinmayoo185/SoccernetGSR) | Dense field cues, homography optimization, learned ReID and identity-aware retrospective refinement | Reviewed the repository workflow and `kpts.py` optimization. The conference PDF returned an access error here; no claim of complete PDF review is made. The authors warn their field registration differs from the older challenges report. |
| [GTATrack](https://arxiv.org/html/2602.00484), January 2026 preprint of a 2025 workshop method | Deep-EIoU/OSNet tracklets followed by global association | Keep the online/offline distinction. The linked [repository](https://github.com/ron941/GTATrack-STC2025) describes execution-level reproduction, not a new unified implementation. |

The search also found a 30 October 2026 [conference submission](https://sciforum.net/paper/36461)
on single-camera analytics. Its abstract is insufficient to choose a replacement
model, so it is a discovery lead rather than an implementation dependency.

## Source-code findings

| Source inspected | Concrete lesson for this code |
|---|---|
| [SoccerNet PnLCalib adapter](https://github.com/SoccerNet/sn-gamestate/blob/main/sn_gamestate/calibration/pnlcalib.py) | Separate point and line heatmaps, confidence thresholds and geometry optimization. Calibration is a model-backed measurement process, not assumed frame corners. |
| [SoccerNet tracklet team clustering](https://github.com/SoccerNet/sn-gamestate/blob/main/sn_gamestate/team/tracklet_team_clustering_api.py) | Aggregate embeddings by tracklet; keep cluster labels separate from semantic home/away identity. |
| [Roboflow team classifier](https://github.com/roboflow/sports/blob/main/sports/common/team.py) | SigLIP embeddings, UMAP and clustering are a candidate for kit separation. They do not by themselves identify individual teammates. |
| [ByteTrack association code](https://github.com/ifzhang/ByteTrack/blob/main/yolox/tracker/byte_tracker.py) | Preserve low-score detections for a second association pass and keep births conservative. This directly exposed the early filtering defect in the active detector. |
| [Ultralytics tracker configuration](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/botsort.yaml) | Camera-motion compensation and neural ReID are independent switches. The added local adapter enables CMC and does not claim a trained appearance encoder. |

Context7 was used for current Ultralytics tracker contracts and React effect
cleanup. Installed code was then inspected because this environment is pinned to
Ultralytics 8.4.123 and Supervision 0.29.1; current online documentation can describe
newer constructor signatures. The local adapter was tested against those installed
versions instead of copying an incompatible example.

## Ordered experiment programme

| Experiment | Controlled comparison | Measurements and promotion rule |
|---|---|---|
| E0: evidence correctness | Corrected pipeline versus frozen original outputs | Confirm calibration abstention, units, role mapping, team coverage, identity conflicts and source-time joins with regression cases. Implemented here. |
| E1: detector | Existing v2/v3 checkpoints, then a fine-tuned YOLO26 small/medium candidate, at fixed 960/1280 input | Per-class AP50-95, ball recall/false positives, latency distribution, peak VRAM and small-object error analysis on the same held-out sessions. A generic pretrained YOLO26 file is not a football accuracy upgrade. |
| E2: tracker | ByteTrack versus added BoT-SORT CMC, then ARC-SORT in an isolated adapter | Use identical cached detections, frame timestamps and identity labels. Report HOTA/IDF1, switches, fragmentation and processing latency. Keep the best measured operating point, not the newest name. |
| E3: calibration | Manual measured points/lines plus supported propagation versus PnLCalib landmark updates | Held-out surveyed points; median/p95 pixel and metre error, failure/coverage, pan/zoom/cut recovery and lens distortion. Tune update frequency against the measured GPU budget. |
| E4: identity and teams | Current colour/motion gates versus sports-trained OSNet/PRTReID; then contextual embeddings offline | Teammate retrieval, crossing/re-entry switches, wrong roster binds, time-to-bind, accuracy and coverage. Test opponents wearing the same number and keep team-qualified roster keys. |
| E5: position/heatmap | Bottom-centre versus confident ankle/ground-contact point; causal filter versus offline smoothing on gap-free runs | Surveyed/pair-camera position errors; speed/distance error by motion band; occupancy player-seconds and missing-coverage map. Keep predictions separate from observed occupancy. |
| E6: tactical meaning | Visible-player nearest-cell baseline versus validated time-to-arrival control | Event precision/recall, calibration of pass-option scores, sensitivity to off-screen players and coach agreement. A Voronoi area share must not be labelled a calibrated possession probability. |
| E7: multimodal alignment | Source-time anchored recording or live capture versus independently observed sync events | Offset median/p95, clock drift, delay, dropped samples and valid optical-window coverage. Recorded footage without an epoch anchor must not inherit today's wearable samples. |

Run each ablation independently before combining improvements. Freeze match/date
splits, annotations, seeds, checkpoint hashes, calibration files and source hashes.
Use a validation set for tuning, then run the untouched test once. Since earlier
training used `images/test` for validation, those historical test results cannot
be presented as untouched model-selection evidence. Sequence separation alone
also does not guarantee match separation if multiple clips come from one match.

## Data and compute choice

[SoccerTrack v2](https://arxiv.org/html/2508.01802v1) is especially relevant to the
campus setting: panoramic university matches with pitch coordinates, identity,
roles and teams. Combine public development data with genuinely held-out campus
sessions rather than treating broadcast performance as automatic transfer.

The observed runtime is Python 3.11.15, PyTorch 2.11.0+cu128, CUDA 12.8,
Ultralytics 8.4.123, Supervision 0.29.1, and an RTX 5060 Laptop GPU. Keep compact
live models and move contextual embedding extraction/global relinking to a
separate offline budget. Measure total process/device memory, not just PyTorch
allocated tensors. Exported TensorRT engines also require their actual fixed
input resolution and runtime compatibility to be respected.

The installed firmware uses a 20,000 microsecond IMU period, or 50 Hz, while
older hardware prose said 100 Hz. The hardware guide now follows the source.
Keep the source-time relay design and the 250 Hz chest ECG while validating
packet loss, sample intervals, electrode attachment and reference measurements
in the field.

The [u-blox NEO-6 data sheet](https://content.u-blox.com/sites/default/files/products/documents/NEO-6_DataSheet_%28GPS.G6-HW-09005%29.pdf)
lists metre-scale horizontal position accuracy under specified conditions.
This supports using the current GNSS as a surveyed cross-check rather than
claiming sub-metre player identity from GPS alone. [Analog Devices' optical
noise guidance](https://www.analog.com/cn/resources/design-notes/how-common-noise-and-error-sources-affect-optical-biosensing.html)
explains motion contamination and the value of independent motion references;
the current IMU quality gates must be validated, not replaced with default
confidence values.

## Run the new checks

```bash
conda activate soccer
cd /home/siddartha/capstone
PYTHONPATH=backend/src:. pytest -q tests
npm run lint --prefix frontend
npm run build --prefix frontend
python scripts/map_project.py

# Both tracker backends, real frames, no publication to the active session:
PYTHONPATH=. python scripts/verify_vision_pipeline.py \
  --video demo/clips/SNMOT-060_750f.mp4 \
  --weights runs/detect/soccernet_v3_s1/weights/best.pt \
  --calibration demo/calib/SNMOT-060.yaml \
  --frames 40 --out data/research_validation/my-new-run

# Direct live selection; no trained neural ReID is implied by this switch:
python -m soccer_analytics.realtime --video /path/to/campus.mp4 \
  --weights /path/to/validated.pt --calibration /path/to/camera.yaml \
  --tracker botsort --no-window

# Team-qualified OCR roster, with source epoch when recorded streams are aligned:
# Add --roster-numbers '1/7:101,2/7:202' --video-start-epoch CAPTURE_EPOCH_SECONDS

# One clip, independently annotated pitch-space truth:
python -m soccer_analytics.evaluation \
  --truth /path/to/labelled-frames.ndjson \
  --predictions /path/to/recorded-snapshots.ndjson \
  --gate-m 2 --out /path/to/results.json
```

The evaluation input is one JSON object per annotated frame:

```json
{"frame_index": 0, "metric": true, "players": [{"track_id": 7, "player_id": 101, "team": 1, "x": 35.2, "y": 18.4}]}
```

Ground-truth `track_id` must persist for the same person throughout that clip.
Prediction IDs need not equal truth IDs. The evaluator matches positions with a
fixed distance gate, computes a global identity contingency assignment, records
switches between matched observations, and reports both accuracy and coverage.
Its `pitch_idf1_at_gate` is explicitly a pitch-space diagnostic. Use the official
[SoccerNet GSR toolkit](https://github.com/SoccerNet/sn-gamestate) and TrackEval for
comparable HOTA/IDF1/GS-HOTA; this tool does not replace them. It evaluates only
the frames that have ground truth, rejects duplicate frame indices, and rejects
invalid ground-truth coordinates. Missing predicted pins count against recall.

Prepare future fast-training validation from training sequences only:

```bash
PYTHONPATH=. python -c 'from scripts.train_apex import build_fast_dataset; print(build_fast_dataset())'
```

This creates a fresh temporary symlink dataset and a split manifest. It does not
train a model or delete an earlier prepared dataset. Use the returned `data.yaml`
for a controlled experiment, and retain the manifest with the run.

## Verification and remaining evidence

See `research/2026-11-11/verification.json` for real-footage smoke diagnostics.
The scoped [UI review](research/2026-11-11/ui-review.md) records the design reasons,
contrast calculations, keyboard check and responsive observations.
Both tracker backends and the offline analysis path execute. An uncalibrated
real-footage check produced zero metric pins and no heatmap. Browser review
exercised actual annotated footage, occupancy mode and the missing-data state.
The live 750-source-frame preview dropped 209 frames to keep up with playback;
that is a latency/coverage tradeoff, not proof of source-rate inference. The
short smoke runs include initialization, run in a fixed backend order and are
not a fair speed ranking. Reported PyTorch memory excludes non-PyTorch allocations.

These are functional checks. No before/after campus AP, identity score or position
error improvement is claimed. New detector/ReID/field-model training, full-match
labelled comparisons, field synchronization and hardware reference validation
remain the substantive next experiments. Model promotion requires independent
accuracy measurements, coverage, and the observed deployment budget.

## Follow-up: relay, paired response and constrained computation

[McByte++ (16 August 2026)](https://arxiv.org/abs/2608.15688) combines mask propagation, conditional camera compensation and online ReID. Its reported benchmark improvements are not results for this system. The inspected [ReID tracker source](https://github.com/tstanczyk95/McBytePlusPlus/blob/main/yolox/tracker/mcbyteplusplus_tracker__with_reid.py) keeps bounded per-track appearance history, normalises the average prototype, separates exported identities from tracker identities, and checks camera warp scale/finite values. These are useful architectural comparisons. Its torchreid/checkpoint and mask-propagation stack still require a separate environment, licence and 8 GB deployment benchmark before adoption.

[Selective Mask Propagation for Multi-Object Tracking (June 2026, July revision)](https://arxiv.org/html/2606.13033) directs expensive mask processing toward uncertain intervals. The useful experiment is a bounded difficult-frame path with an explicit duty-cycle budget, evaluated on crossings and re-entry. Running a large video segmentation model continuously would compete with the detector and preview on this GPU. No segmentation checkpoint was installed or silently made a new default.

The inspected [ARC-SORT orchestration](https://github.com/Athlitix/arcsort/blob/main/arcsort/tracker.py) exposes independent IMM, camera compensation, adaptive expansion, Mahalanobis gating, ReID and crossing-arbitration switches. It distinguishes detection boxes from filtered output. The code also warns about hybrid scores allowing non-overlapping matches without a motion gate. A fair next comparison caches identical detections and changes one association component at a time; enabling every flag would hide the source of gains and cost. This is source review, not a reproduction of its paper.

The [2026 collegiate fatigue study](https://pubmed.ncbi.nlm.nih.gov/41788876/) monitored 48 male players over 12 weeks and used GPS, pulse and subjective/functional measurements. It reports XGBoost AUC 0.895 in ordinary cross-validation and 0.847 with athlete exclusion. Wellness and morning HRV were important inputs. CPG44 does not currently collect equivalent labels or all these features, so these scores cannot transfer to its MAX30102/IMU system. A compact tabular baseline with athlete-disjoint evaluation is a credible supervised starting point once outcome labels exist; a neural network is not automatically the strongest small-data choice.

The implemented `ReadinessEngine` is an explicitly untrained coaching comparison. It requires a roster binding, an accepted metric position, synchronized pulse, quality >=0.6 and alignment <=250 ms. It retains at most 600 approximately 1 Hz paired observations, requires three minutes and sufficient early/recent coverage, compares pulse within shared 1 m/s speed bands, and suggests a check-in only when both average movement declines >=25% and comparable-speed pulse rises >=10 bpm. These thresholds are engineering hypotheses, not validated fatigue cutoffs. Missing or incomparable evidence produces an explanation instead of a probability. Tactical changes can also reduce running; this is a review cue, not a substitution command.

Context7 was used for current [websockets buffer/backpressure documentation](https://websockets.readthedocs.io/en/stable/topics/memory.html) and [Ultralytics training memory options](https://docs.ultralytics.com/modes/train/). The live hub bounds the incoming frame queue to one; the 8 MiB message ceiling accommodates legacy relay replay during hub-first upgrade. The new relay sends replay pages of 256 observations and up to 100 ingest observations per broadcast batch; eight queued batches trigger an explicit recovery marker and disconnect instead of silent loss. Training defaults are physical batch 2, AMP, two workers and no image RAM cache. AutoBatch remains an explicit choice and arbitrary checkpoints still need peak-memory measurement. No per-frame `empty_cache()` is used.

The follow-up local relay test is reproducible with `scripts/verify_streaming.py`: 512 synthetic observations arrived in order, all 512 recovered in two replay pages, and 16 repeated events were suppressed. Local HTTP-to-WebSocket batch age measured p50 3.48 / p95 5.01 / p99 5.30 ms; the cache held 299,871 serialized bytes. These exclude Wi-Fi, TLS/WAN, physical acquisition and vision. See [relay evidence](research/2026-11-11/relay-verification.json). They do not establish field end-to-end latency.

For 120 real source frames, the later ByteTrack run measured stage p50/p95 (ms): detection 18.84/29.41; ball search plus camera mapping 10.81/29.68; identity, positions and tactics 3.89/13.49; fusion 0.06/0.19. This identifies detector and mapping work as the current larger costs. In a separate paced preview, 199/750 source frames were dropped to stay near the source clock. Run order, GPU contention, warmup and source cadence prevent treating the small difference from the earlier 209-frame drop as a demonstrated speed gain. Position acceptance also varies later in the clip; higher pin counts alone would not establish lower position error.

## Deployment and verification status of the follow-up

The active suite passes 93 tests; type checking, production frontend build and Python compilation pass. New regression contracts cover replay consumption, slow clients, unclamped acquisition age, stale pulse, physiology deduplication, bounded cumulative distance and paired-response abstention. The UI was checked with actual inference at desktop and 390px, including keyboard player selection, occupancy mode and source notes. Browser captures and the design review are separate from model accuracy evidence.

The public VPS was **not** changed. Upgrade the local sensor hub first (it accepts both legacy `wearable_sample` and new `wearable_batch`), then deploy the relay; old hubs do not understand the new batch message. Validate authenticated stream recovery before the field session. Source timestamps and signed age are preserved, including old replay and future-clock failures. Delay is no longer fabricated by subtracting assumed clock skew or capped at 250 ms.

Firmware batching was changed from 8 samples / 100 ms / 32 queue slots with a count-based backlog discard to at most 32 samples / 25 ms / 256 slots with a two-second age horizon. Existing backlog drains without another full batching wait. Expired or rejected samples increment the drop counter. This is intended to preserve contiguous PPG through normal HTTP round trips. The local Arduino CLI lacks the ESP32 platform, so this new firmware has **not** compiled or run on the vest. Verify compilation, heap, actual acquisition rates, continuity and WAN age before flashing. No latency promise is inferred from queue size.

