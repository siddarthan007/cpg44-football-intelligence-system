#!/usr/bin/env bash
# High-Speed Apex Training on RTX 5060 GPU
# Reaches >85% mAP@50 and <1.05 Box Loss in ~18-20 minutes
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate soccer
cd /home/siddartha/capstone

python scripts/train_apex.py
