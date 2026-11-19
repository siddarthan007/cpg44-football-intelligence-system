# CPG44 reference list

Kept for the final report. IDs (e.g. [T3]) are used in code docstrings and the
docs. Status column:

- **core**: established, peer-reviewed, details stable.
- **seen**: located during the 12 Nov 2026 research pass (paper page, arXiv
  listing, repository or search result).
- **guide**: cited by `CPG44_Football_Intelligence_Enhancement_Guide.pdf`;
  open the source and confirm authors/venue before citing.

Where a component in this repository implements or adapts an idea, the file is
named in the last column.

## Detection

| ID | Reference | Status | Used in |
|---|---|---|---|
| D1 | Jocher, G., Chaurasia, A., Qiu, J. *Ultralytics YOLOv8*, software, 2023. https://github.com/ultralytics/ultralytics | core | `tracker.py` |
| D2 | Carion, N. et al. "End-to-End Object Detection with Transformers (DETR)." ECCV 2020. arXiv:2005.12872 | core | background |
| D3 | Zhao, Y. et al. "DETRs Beat YOLOs on Real-time Object Detection (RT-DETR)." CVPR 2024. arXiv:2304.08069 | core | `scripts/train_detector_compare.py` |
| D4 | Lv, W. et al. "RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer." arXiv:2407.17140, 2024 | core | background |
| D5 | RT-DETRv4 (foundation-model distillation), arXiv:2510.25257, ECCV 2026; https://github.com/AIMIA-PKU/RT-DETRv4 | seen (earlier pass) | `docs/RTDETR_AND_REPLAYFIELD.md` |
| D7 | Akyon, F. C., Altinuc, S. O., Temizel, A. "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection (SAHI)." ICIP 2022. arXiv:2202.06934 | core | native-scale tile training, `scripts/build_tiles.py` |
| D8 | Ultralytics YOLO26 (end-to-end, NMS-free; small-target-aware label assignment), docs.ultralytics.com/models/yolo26, 2025-2026; `yolo26-p2.yaml` stride-4 head | seen (installed package) | `scripts/train_detector_p2.py` |
| D6 | Komorowski, J., Kurzejamski, G., Sarwas, G. "FootAndBall: Integrated Player and Ball Detector." VISAPP 2020. arXiv:1912.05445 | core | ball ROI design |

## Multi-object tracking and re-identification

| ID | Reference | Status | Used in |
|---|---|---|---|
| T1 | Zhang, Y. et al. "ByteTrack: Multi-Object Tracking by Associating Every Detection Box." ECCV 2022. arXiv:2110.06864 | core | baseline; two-stage association in `imm_tracker.py` |
| T2 | Aharon, N., Orfaig, R., Bobrovsky, B.-Z. "BoT-SORT: Robust Associations Multi-Pedestrian Tracking." arXiv:2206.14651, 2022 | core | camera-motion compensation, box state |
| T3 | Huang, H.-W. et al. "Iterative Scale-Up ExpansionIoU and Deep Features Association for Multi-Object Tracking in Sports (Deep-EIoU)." WACV Workshops 2024. arXiv:2306.13074 | core / seen | expansion IoU in `imm_tracker.py` |
| T4 | Adari, G. K. "ARC-SORT: Adaptive, Robust, Compensated Multi-Object Tracking with Interacting Multiple Models." arXiv:2610.22306, Oct 2026 | seen (aggregator) / guide | IMM + uncertainty-adaptive expansion |
| T5 | Blom, H. A. P., Bar-Shalom, Y. "The Interacting Multiple Model Algorithm for Systems with Markovian Switching Coefficients." IEEE TAC 33(8), 1988 | core | `imm_tracker.py` |
| T6 | Sun, J., Huang, H.-W., Yang, C.-Y., Jiang, Z., Hwang, J.-N. "GTA: Global Tracklet Association for Multi-Object Tracking in Sports." ACCV Workshops 2024. arXiv:2411.08216 | seen | `tracklet_link.py` |
| T7 | Stanczyk, T., Yoon, S., Brémond, F. "Training-Free Long-Term Multi-Object Tracking for Sports Video Analytics (McByte++)." arXiv:2608.15688, Aug 2026 | guide | masked CMC rationale |
| T8 | Cao, J. et al. "Observation-Centric SORT." CVPR 2023. arXiv:2203.14360 | core | background |
| T9 | Meinhardt, T., Kirillov, A., Leal-Taixé, L., Feichtenhofer, C. "TrackFormer: Multi-Object Tracking with Transformers." CVPR 2022. arXiv:2101.02702 | core | transformer-tracking study |
| T10 | Chen, X. et al. "Transformer Tracking (TransT)." CVPR 2021. arXiv:2103.15436 (single-object tracking) | core | transformer-tracking study |
| T11 | Zeng, F. et al. "MOTR: End-to-End Multiple-Object Tracking with Transformer." ECCV 2022. arXiv:2105.03247 | core | transformer-tracking study |
| T12 | Gao, R., Wang, L. "MeMOTR: Long-Term Memory-Augmented Transformer for Multi-Object Tracking." ICCV 2023. arXiv:2307.15700 | core | transformer-tracking study |
| T13 | Gao, R. et al. "Multiple Object Tracking as ID Prediction (MOTIP)." CVPR 2025. arXiv:2403.16848 | core | transformer-tracking study |
| T14 | Segu, M. et al. "SambaMOTR: Samba-based Tracking with Synchronized State Space Models." ICLR 2025. arXiv:2410.01806 | core | transformer/SSM study |
| T15 | "Motion-Aware Transformer for Multi-Object Tracking (MATR)." arXiv:2509.21715, 2025 (72.2-72.7 HOTA SportsMOT) | seen | transformer-tracking study |
| T16 | Luo, H. et al. "Bag of Tricks and a Strong Baseline for Deep Person Re-identification." CVPR Workshops 2019. arXiv:1903.07071 | core | `reid_model.py` (BNNeck, last stride 1) |
| T17 | Hermans, A., Beyer, L., Leibe, B. "In Defense of the Triplet Loss for Person Re-Identification." arXiv:1703.07737, 2017 | core | batch-hard triplet in `scripts/train_reid.py` |
| T18 | He, K. et al. "Deep Residual Learning for Image Recognition." CVPR 2016 | core | ReID backbone |
| T19 | Zhou, K. et al. "Omni-Scale Feature Learning for Person Re-Identification (OSNet)." ICCV 2019. arXiv:1905.00953 | core | comparison baseline |
| T20 | Kalman, R. E. "A New Approach to Linear Filtering and Prediction Problems." J. Basic Eng. 82, 1960 | core | `filters.py`, `imm_tracker.py` |

## Datasets and evaluation

| ID | Reference | Status | Used in |
|---|---|---|---|
| E1 | Cioppa, A. et al. "SoccerNet-Tracking: Multiple Object Tracking Dataset and Benchmark in Soccer Videos." CVPR Workshops 2022. arXiv:2204.06918 | core / seen | all benchmarks |
| E2 | Cui, Y. et al. "SportsMOT: A Large Multi-Object Tracking Dataset in Multiple Sports Scenes." ICCV 2023. arXiv:2304.05170 | core | literature numbers |
| E3 | Somers, V. et al. "SoccerNet Game State Reconstruction: End-to-End Athlete Tracking and Identification on a Minimap." CVPR Workshops 2024. arXiv:2404.11335 | core | evaluation design |
| E4 | Luiten, J. et al. "HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking." IJCV 129, 2021. arXiv:2009.07736 | core | `mot_eval.py` |
| E5 | Ristani, E. et al. "Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking." ECCV Workshops 2016. arXiv:1609.01775 | core | IDF1 in `mot_eval.py` |
| E6 | Bernardin, K., Stiefelhagen, R. "Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics." EURASIP JIVP 2008 | core | MOTA/IDSW in `mot_eval.py` |
| E7 | SoccerNet 2025 Challenges results. arXiv:2508.19182 | seen | literature context |
| E8 | SoccerNet 2026 Challenges results. arXiv:2607.07320 | guide | literature context |
| E9 | Scott, A. et al. "SoccerTrack v2" (panoramic university matches). arXiv:2508.01802 | seen (earlier pass) | future campus data |

## Ball and 3D

| ID | Reference | Status | Used in |
|---|---|---|---|
| B1 | Grad, Ł., Czajkowski, K. M., Varashylau, A. "Physics-Based 3D Ball Trajectory Reconstruction from Monocular Soccer Video: A Multi-Model Benchmark." CVPR Workshops (CVsports) 2026, pp. 9940-9948. https://lukaszgrad.github.io/soccer-ball3d/ ; code https://github.com/lukaszgrad/ball3d (CC-BY-NC-SA-4.0) | seen | `ball3d.py` (method re-implemented, no code copied) |
| B2 | "Where Is The Ball: 3D Ball Trajectory Estimation From 2D Monocular Tracking." CVPR Workshops 2025. arXiv:2506.05763 | seen | related work |
| B3 | "SoccerNet-v3D: Leveraging Sports Broadcast Replays for 3D Scene Understanding." arXiv:2504.10106, 2025 | seen | 3D ball ground truth for future evaluation |
| B4 | "TOTNet: Occlusion-Aware Temporal Tracking for Robust Ball Detection in Sports Videos." arXiv:2508.09650, 2025 | seen | temporal ball detection, future work |
| B5 | Huang, Y.-C. et al. "TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications." AVSS 2019. arXiv:1907.03698 | core | temporal consistency idea in `ball_tracking.py` |
| B6 | Kim, T. et al. "Physics-Based 3D Position Analysis of a Soccer Ball from Monocular Image Sequences." ICCV 1998 | seen | historical baseline for parabola fitting |
| B7 | Kienzle, D. et al. "Towards Learning Monocular 3D Object Localization From 2D Labels using the Physical Laws of Motion." arXiv:2310.17462, 2023 | seen | related work |

## Camera calibration and geometry

| ID | Reference | Status | Used in |
|---|---|---|---|
| C1 | Zhang, Z. "A Flexible New Technique for Camera Calibration." IEEE TPAMI 22(11), 2000 | core | `ball3d.camera_from_homography` |
| C2 | Hartley, R., Zisserman, A. *Multiple View Geometry in Computer Vision*, 2nd ed., Cambridge, 2004 | core | homography, DLT |
| C3 | Gutiérrez-Pérez, M., Agudo, A. "No Bells, Just Whistles: Sports Field Registration by Leveraging Geometric Properties" / "PnLCalib: Sports Field Registration via Points and Lines Optimization." arXiv:2404.08401 (CVIU 2026 per guide) | seen / guide | next calibration experiment |
| C4 | Theiner, J., Ewerth, R. "TVCalib: Camera Calibration for Sports Field Registration in Soccer." WACV 2023 | core / seen | related work |
| C5 | Magera, F. et al. "BroadTrack: Broadcast Camera Tracking for Soccer." arXiv:2412.01721, 2024 | guide | PTZ-constrained propagation (future) |
| C6 | Sha, L. et al. "End-to-End Camera Calibration for Broadcast Videos." CVPR 2020 | seen | related work |
| C8 | Brown, D. C. "Close-Range Camera Calibration." Photogrammetric Engineering 37(8), 1971 (radial distortion model) | core | `lens.py` |
| C9 | Cuesta-Albertos, J. A., Gordaliza, A., Matrán, C. "Trimmed k-means: an attempt to robustify quantizers." Annals of Statistics 25(2), 1997 | core | team clustering in `match_analysis._assign_teams` |
| C10 | Hampel, F. R. "The Influence Curve and its Role in Robust Estimation." JASA 69, 1974 (rolling-median outlier filter) | core | `match_analysis._robust_track_mask` |
| C7 | Savitzky, A., Golay, M. J. E. "Smoothing and Differentiation of Data by Simplified Least Squares Procedures." Analytical Chemistry 36(8), 1964 | core | `match_analysis._smooth_kinematics` |

## Wearables and physiology

| ID | Reference | Status | Used in |
|---|---|---|---|
| W1 | Widrow, B. et al. "Adaptive Noise Cancelling: Principles and Applications." Proc. IEEE 63(12), 1975 | core | `sensors/motion_artifact.py` |
| W2 | Zhang, Z., Pi, Z., Liu, B. "TROIKA: A General Framework for Heart Rate Monitoring Using Wrist-Type PPG Signals During Intensive Physical Exercise." IEEE TBME 62(2), 2015 | core | spectral tracking with cadence suppression |
| W3 | Ram, M. R. et al. "A Novel Approach for Motion Artifact Reduction in PPG Signals Based on AS-LMS Adaptive Filter." IEEE TIM 61(5), 2012 | core | NLMS rationale |
| W4 | Comparative study of adaptive filters (NLMS/RLS/LMS) for MAX30102 motion artefacts, PMC10881614 | guide | NLMS choice |
| W5 | Per-axis NLMS for treadmill/cycling PPG, University of Essex EMBC paper, https://repository.essex.ac.uk/24790/ | seen | multi-axis reference |
| W6 | Maxim Integrated. *MAX30102 datasheet* and reference SpO2 algorithm (RD117 / AN6409) | core | Maxim SpO2 curve in `sensors/hub.py` |
| W7 | Boyd, L. J., Ball, K., Aughey, R. J. "The Reliability of MinimaxX Accelerometers for Measuring Physical Activity in Australian Football." IJSPP 6(3), 2011 | core | PlayerLoad |
| W8 | Buchheit, M. "Monitoring Training Status with HR Measures: Do All Roads Lead to Rome?" Frontiers in Physiology 5:73, 2014 | core | `coach_engine` drift and recovery rules |
| W9 | Impellizzeri, F. M., Marcora, S. M., Coutts, A. J. "Internal and External Training Load: 15 Years On." IJSPP 14(2), 2019 | core | internal/external coupling |
| W10 | Impellizzeri, F. M. et al. "Acute:Chronic Workload Ratio: Conceptual Issues and Fundamental Pitfalls." IJSPP 15(6), 2020 | core | ACWR not used as a diagnosis |
| W11 | Mohr, M., Krustrup, P., Bangsbo, J. "Match Performance of High-Standard Soccer Players with Special Reference to Development of Fatigue." J Sports Sci 21(7), 2003 | core | HSR decline rule |
| W12 | Bradley, P. S., Noakes, T. D. "Match Running Performance Fluctuations in Elite Soccer: Indicative of Fatigue, Pacing or Situational Influences?" J Sports Sci 31(15), 2013 | core | caution on fatigue inference |
| W13 | Osgnach, C. et al. "Energy Cost and Metabolic Power in Elite Soccer: A New Match Analysis Approach." MSSE 42(1), 2010 | core | `catapult.py` metabolic power |
| W14 | Multi-sensor fusion for fatigue in university soccer players, Frontiers in Physiology 2026, doi:10.3389/fphys.2026.1775906; PubMed 41788876 | guide / earlier pass | supervised-model roadmap |
| W15 | "Towards Athlete Fatigue Assessment from Association Football Videos" (acceleration-speed profiles). arXiv:2604.05636, 2026 | guide | A, S profile (future card) |

## Tactics

| ID | Reference | Status | Used in |
|---|---|---|---|
| K1 | Fernández, J., Bornn, L. "Wide Open Spaces: A Statistical Technique for Measuring Space Creation in Professional Soccer." MIT Sloan Sports Analytics Conference 2018 | core | pitch control background |
| K2 | Spearman, W. "Beyond Expected Goals." MIT Sloan Sports Analytics Conference 2018 | core | pitch control / xG background |
| K6 | Pollard, R., Ensum, J., Taylor, S. "Estimating the probability of a shot resulting in a goal: the effects of distance, angle and space." Int. J. Soccer Science 2(1), 2004; Caley, M. expected-goals model notes, 2013-2015 | core | logistic xG in `decision.py` |
| K7 | Singh, K. "Introducing Expected Threat (xT)." 2019, karun.in/blog/expected-threat.html | core | pass value idea (threat gained) |
| K3 | Low, B. et al. "A Systematic Review of Collective Tactical Behaviours in Football Using Positional Data." Sports Medicine 50, 2020 | core | team length/width/centroid |
| K4 | Vogelbein, M., Nopp, S., Hökelmann, A. "Defensive Transition in Soccer, Are Prompt Possession Regains a Measure of Success?" J Sports Sci 32(11), 2014 | core | 5 s regain rule |
| K5 | Fradua, L. et al. "Designing Small-Sided Games for Training Tactical Aspects in Soccer: Extrapolating Pitch Sizes from Full-Size Professional Matches." J Sports Sci 31(6), 2013 | core | effective playing space |

## Vest and hardware

| ID | Reference | Status | Used in |
|---|---|---|---|
| V1 | Vest-worn 10 Hz GNSS (Catapult OptimEye S5) between the scapulae in English Championship match play; stadium stands degrade GNSS. Int J Performance Analysis in Sport, 2021, doi:10.1080/24748668.2021.1947017 | seen | `hardware/VEST_DESIGN.md` §2 |
| V2 | Catapult Sports, device wearing guidance (support documentation) | seen | pocket placement |
| V3 | Review noting scapular placement limits limb-level IMU detail. Frontiers in Physiology 2018, doi:10.3389/fphys.2018.00228 | seen | scope statement |
| V4 | Wrist PPG motion artefacts under running: W2 and the IEEE Signal Processing Cup 2015 dataset | core | why not the wrist |
| V5 | IFAB *Laws of the Game*, Law 4 (Players' Equipment, EPTS); FIFA Quality Programme for EPTS | core | safety/competition use |
| V6 | u-blox. *NEO-6 data sheet* (GPS.G6-HW-09005) | core | 1 Hz, metre-level accuracy |
| V7 | InvenSense/TDK. *MPU-6000/6050 Register Map and Descriptions* (DLPF, full-scale ranges) | core | firmware DLPF/range fix |
| V8 | Barrett, Midgley & Lovell (2014). PlayerLoad: reliability, convergent validity, and influence of unit position during treadmill running. *Int J Sports Physiol Perform* 9(6):945-952 | core | standard PlayerLoad in `sensors/calibration.py` |
| V9 | Frosio, Pedersini & Borghese (2009). Autocalibration of MEMS accelerometers. *IEEE Trans Instrum Meas* 58(6):2034-2041 | core | `AccelAutoCal` |
| V10 | Townshend, Worringham & Stewart (2008). Assessment of speed and position during human locomotion using nondifferential GPS. *Med Sci Sports Exerc* 40(1):124-132 | core | Doppler speed in the GNSS Kalman filter |
| V11 | Cristian (1989). Probabilistic clock synchronization. *Distributed Computing* 3(3):146-158 | core | relay clock offset |
| V12 | Teixeira, Jung & Savvides (2010). Tasking networked CCTV cameras and mobile phones to identify and localize multiple people. *UbiComp* | core | vest-to-track association |
| V13 | Maxim Integrated, MAX17048 data sheet (ModelGauge fuel gauge) | core | battery state in firmware |

## Fatigue, load and player identity

| ID | Reference | Status | Used in |
|---|---|---|---|
| F1 | Bradley & Noakes (2013). Match running performance fluctuations in elite soccer: indicative of fatigue, pacing or situational influences? *J Sports Sci* 31(15):1627-1638 | core | peak-window method in `fatigue.py` |
| F2 | Paul, Bradley & Nassis (2015). Factors affecting match running performance of elite soccer players. *Int J Sports Physiol Perform* 10(4):516-519 | core | team-relative decline |
| F3 | Mohr, Krustrup & Bangsbo (2003). Match performance of high-standard soccer players with special reference to development of fatigue. *J Sports Sci* 21(7):519-528 | core | temporary fatigue, sprint decline |
| F4 | Buchheit (2014). Monitoring training status with HR measures: do all roads lead to Rome? *Front Physiol* 5:73 | core | HR recovery and drift factors |
| F5 | Impellizzeri et al. (2020). Acute:chronic workload ratio: conceptual issues and fundamental pitfalls. *Int J Sports Physiol Perform* 15(6):907-913 | core | why no injury probability is claimed |
| J1 | SoccerNet Jersey Number Recognition 2023 (Cioppa et al., SoccerNet 2023 challenges report, arXiv:2309.06006) | core | training and test data for `jersey_model.py` |
| J2 | Koshkina & Elder (2024). A general framework for jersey number recognition in sports video. *CVPR Workshops (CVsports)* | core | legibility gating, track-level vote |
| J3 | Gerke, Mueller & Schaefer (2015). Soccer jersey number recognition using convolutional neural networks. *ICCV Workshops* | core | digit-factorised output |
| J4 | Ilse, Tomczak & Welling (2018). Attention-based deep multiple instance learning. *ICML* | core | `jersey_model.build_mil` |
| J5 | Zhong, Zheng, Kang, Li & Yang (2020). Random erasing data augmentation. *AAAI* | seen | jersey training, second run |
| E1 | Pan & Tompkins (1985). A real-time QRS detection algorithm. *IEEE Trans Biomed Eng* 32(3):230-236 | core | `sensors/ecg.py` |
| E2 | Task Force of the ESC and NASPE (1996). Heart rate variability: standards of measurement, physiological interpretation and clinical use. *Circulation* 93(5):1043-1065 | core | RMSSD and artefact rules |
| E3 | Shaffer & Ginsberg (2017). An overview of heart rate variability metrics and norms. *Front Public Health* 5:258 | core | short-term RMSSD windows |
| E4 | Analog Devices, AD8232 data sheet (single-lead heart rate monitor front end) | core | `docs/hardware/BUILD_GUIDE.md` |
| E5 | Texas Instruments, TPS63020 data sheet (buck-boost converter) | core | power design |
| M1 | Kuhn (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly* 2 | core | cross-camera matching, vest-to-track assignment |

## Products and context (not academic sources)

| ID | Reference | Use |
|---|---|---|
| P1 | Replayfield, https://replayfield.com (accessed 12 Nov 2026): pre-launch amateur-football video analysis; capture guide, match room, review of uncertain goals, 3D goal replay, player cards | Problem framing only. CPG44 does not copy its layout, copy or visuals; see `docs/DESIGN_RATIONALE.md` |
| P2 | `CPG44_Football_Intelligence_Enhancement_Guide.pdf` (project file, Nov 2026) | Engineering plan; several numbers in it are marked provisional by its author |
| P3 | Project proposal: Nepal, S., Agrawal, R., Thagunna, B. K., Sedhai, S., Yadav, O. N. *Predictive and Tactical Intelligence System for Athlete Performance Using Vision-Based Tracking and Wearable Data Analytics*, CPG 44, TIET Patiala, March 2026 | Objectives |
