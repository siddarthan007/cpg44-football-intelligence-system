#!/usr/bin/env bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh; conda activate soccer
cd ~/capstone
echo "=== build 10-seq subset (stride 8) ==="
rm -rf /tmp/sn10 /tmp/yolo10
mkdir -p /tmp/sn10/train /tmp/sn10/test
i=0
for s in $(ls ~/SoccerNet/tracking/train | head -12); do
  ln -s ~/SoccerNet/tracking/train/$s /tmp/sn10/train/$s; i=$((i+1)); [ $i -ge 10 ] && break
done
for s in $(ls ~/SoccerNet/tracking/test | head -2); do ln -s ~/SoccerNet/tracking/test/$s /tmp/sn10/test/$s; done
python soccernet_to_yolo.py --src /tmp/sn10 --dst /tmp/yolo10 --stride 8 --image-mode symlink 2>&1 | tail -2
echo "=== train yolov8m@1280 batch4 30 epochs ==="
python -m soccer_analytics.train base --data /tmp/yolo10/data.yaml --model yolov8m.pt \
  --imgsz 1280 --batch 4 --epochs 30 --name soccernet_v2 2>&1 | tail -4
echo "TRAIN_V2_DONE"
