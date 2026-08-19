"""
soccer_analytics
================

Vision pipeline for the CPG44 capstone (everything except wearable fusion):

    video ──▶ detect+track ──▶ team assignment ──▶ pitch homography
          ──▶ metrics (speed/distance/possession) ──▶ heatmaps ──▶ tactics

Modules
-------
- ``train``        YOLOv8 detection training + fine-tuning (SoccerNet, then custom footage)
- ``tracker``      YOLO + ByteTrack detection/tracking wrapper
- ``team_assign``  jersey-colour team clustering
- ``pitch``        field-keypoint / manual homography → pitch coordinates
- ``view``         perspective-transform helpers
- ``metrics``      speed, distance, ball possession
- ``heatmap``      per-player / team occupancy heatmaps + radar
- ``tactics``      formation estimation + rule-based recommendations
- ``pipeline``     end-to-end runner (annotated video + stats)
"""

__version__ = "0.1.0"

CLASS_NAMES = ["ball", "goalkeeper", "player", "referee"]
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3
