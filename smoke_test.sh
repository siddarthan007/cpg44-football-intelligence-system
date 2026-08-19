#!/usr/bin/env bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh; conda activate soccer
cd ~/capstone
echo "===== [1] tiny SoccerNet -> YOLO subset ====="
rm -rf /tmp/mini_sn /tmp/mini_yolo /tmp/smoke_out
mkdir -p /tmp/mini_sn/train /tmp/mini_sn/test
for s in SNMOT-060 SNMOT-061 SNMOT-062; do ln -s ~/SoccerNet/tracking/train/$s /tmp/mini_sn/train/$s; done
ln -s ~/SoccerNet/tracking/test/SNMOT-116 /tmp/mini_sn/test/SNMOT-116
python soccernet_to_yolo.py --src /tmp/mini_sn --dst /tmp/mini_yolo --stride 30 --image-mode symlink 2>&1 | tail -3

echo "===== [2] build test clip (SNMOT-060 frames 1..200) ====="
python - <<PY
import cv2, glob
frames=sorted(glob.glob('/home/siddartha/SoccerNet/tracking/train/SNMOT-060/img1/*.jpg'))[:200]
h,w=cv2.imread(frames[0]).shape[:2]
vw=cv2.VideoWriter('/tmp/test_clip.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (w,h))
for f in frames: vw.write(cv2.imread(f))
vw.release(); print("wrote /tmp/test_clip.mp4", len(frames), "frames", w,"x",h)
PY

echo "===== [3] train yolov8n 6 epochs @640 on GPU ====="
python -m soccer_analytics.train base --data /tmp/mini_yolo/data.yaml --model yolov8n.pt \
  --epochs 6 --imgsz 640 --batch 8 --name smoke 2>&1 | tail -6

echo "===== [4] offline pipeline (pixel mode, headless) ====="
python -m soccer_analytics.pipeline --video /tmp/test_clip.mp4 \
  --weights runs/detect/smoke/weights/best.pt --out /tmp/smoke_out --imgsz 640 2>&1 | tail -8
echo "--- stats.json ---"; head -c 900 /tmp/smoke_out/stats.json

echo ""
echo "===== [5] realtime headless + simulated wearable (60 frames) ====="
python -m soccer_analytics.realtime --video /tmp/test_clip.mp4 \
  --weights runs/detect/smoke/weights/best.pt --imgsz 640 \
  --no-window --simulate-sensors --max-frames 60 2>&1 | tail -8

echo "===== [6] injury ML train+predict (XGBoost) ====="
python - <<PY
from soccer_analytics.sensors.injury import InjuryRiskModel, HeuristicInjuryModel, bootstrap_training_set
from soccer_analytics.sensors.schema import WorkloadFeatures
X,y=bootstrap_training_set(3000)
m=InjuryRiskModel().fit(X[:2400],y[:2400])
# eval accuracy on holdout
import numpy as np
pred=[1 if m.predict(f).risk>0.5 else 0 for f in X[2400:]]
acc=np.mean(np.array(pred)==np.array(y[2400:]))
print(f"backend={m._backend} holdout_acc={acc:.3f}")
r=m.predict(WorkloadFeatures(player_id=7, total_distance=11000, hsr_distance=1400,
    sprint_count=40, player_load=900, top_speed=9.2, avg_hr=182, hr_drift=28,
    min_spo2=90, acwr=1.8))
print("high-load player risk:", r.risk, r.level, r.factors)
PY
echo "SMOKE_DONE_OK"
