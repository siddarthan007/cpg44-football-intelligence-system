#!/usr/bin/env bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh; conda activate soccer
cd ~/capstone
echo "=== build 16-seq subset (stride 5, denser → more ball frames) ==="
rm -rf /tmp/sn16 /tmp/yolo16; mkdir -p /tmp/sn16/train /tmp/sn16/test
i=0; for s in $(ls ~/SoccerNet/tracking/train); do ln -s ~/SoccerNet/tracking/train/$s /tmp/sn16/train/$s; i=$((i+1)); [ $i -ge 16 ] && break; done
for s in $(ls ~/SoccerNet/tracking/test | head -3); do ln -s ~/SoccerNet/tracking/test/$s /tmp/sn16/test/$s; done
python soccernet_to_yolo.py --src /tmp/sn16 --dst /tmp/yolo16 --stride 5 --image-mode symlink 2>&1 | tail -2
echo "=== train yolov8m@1280 batch4 40 epochs ==="
python -m soccer_analytics.train base --data /tmp/yolo16/data.yaml --model yolov8m.pt \
  --imgsz 1280 --batch 4 --epochs 40 --name soccernet_v3 2>&1 | tail -3
echo "TRAIN_V3_DONE"
