# SoccerNet Tracking → YOLOv8 dataset

`soccernet_to_yolo.py` converts the **SoccerNet Tracking** dataset (MOT Challenge
layout) into the Ultralytics YOLO detection format, splits by sequence into
train/val/test, and writes `data.yaml`.

## Why not JSON?

The SoccerNet *Tracking* release is **not** JSON (that is the *Action Spotting*
release). Each clip is an MOT-Challenge sequence:

```
SNMOT-060/
  seqinfo.ini      # imWidth=1920 imHeight=1080 seqLength=750 imExt=.jpg
  gameinfo.ini     # trackletID_<id> = <role>;<jersey>   ← gives the CLASS
  gt/gt.txt        # frame,id,bb_left,bb_top,bb_w,bb_h,conf,class,vis
  img1/000001.jpg  # zero-padded frames
```

Two quirks handled by the script:
1. `gt.txt` column 8 (class) is always `-1`. The real class is recovered from
   `gameinfo.ini` by mapping the tracklet **id** (gt col 2) → role.
2. Frame names collide across clips (every clip has `000001.jpg`), so outputs are
   namespaced `SNMOT-060_000001.jpg` / `.txt`.

## Classes

Team side is collapsed — team assignment is a downstream jersey-colour step, not a
detector class. Default index order matches the common Roboflow
`football-players-detection` weights used by the reference repos:

| idx | class      | from role |
|-----|------------|-----------|
| 0   | ball       | `ball` |
| 1   | goalkeeper | `goalkeeper(s) team {left,right}` |
| 2   | player     | `player team {left,right}` |
| 3   | referee    | `referee` |

`other` roles are dropped. Override with `--names` and/or `--class-map`.

## Usage

Run **inside WSL** (native symlinks, no Windows path issues):

```bash
python3 soccernet_to_yolo.py \
  --src /home/siddartha/SoccerNet/tracking \
  --dst /home/siddartha/SoccerNet/yolo \
  --val-frac 0.15 \
  --stride 5 \
  --image-mode symlink
```

Validate first without writing anything:

```bash
python3 soccernet_to_yolo.py --src ... --dst ... --stride 250 --dry-run
```

### Key flags

| flag | default | note |
|------|---------|------|
| `--val-frac` | 0.15 | fraction of **train sequences** held out for val (split by sequence → no frame leakage) |
| `--stride N` | 1 | keep every Nth annotated frame. 25 fps clips are highly redundant — `5`–`10` cuts dataset size with little loss |
| `--image-mode` | symlink | `symlink` (no disk dup) · `copy` · `hardlink` · `none` (labels only). Auto-falls back to copy if symlink is denied |
| `--negatives` | off | also emit empty `.txt` for boxless frames (background images) |
| `--min-visibility` | none | drop boxes below a visibility threshold (SoccerNet vis col is `-1`/unused here) |
| `--unknown-id-policy` | skip | gt id missing from gameinfo → `skip` or a class name |
| `--no-test` | off | skip building test split from `test/` |
| `--keep-existing` | off | by default `images/`+`labels/` under `--dst` are purged first so a rerun leaves no stale files; pass this to keep them |
| `--dry-run` | off | parse + report stats, write nothing |

`--dst` is resolved to an absolute path in `data.yaml` (Ultralytics resolves a
relative `path:` against its own `datasets_dir`, not your CWD).

Outputs `data.yaml`, `conversion_report.json`, and:

```
<dst>/images/{train,val,test}/<seq>_<frame>.jpg
<dst>/labels/{train,val,test}/<seq>_<frame>.txt
```

## Dataset facts (this copy)

- 57 train sequences, 49 test sequences, 750 frames each @ 25 fps, 1920×1080.
- test/ **includes** `gt.txt` (usable for evaluation).
- Verified: bbox round-trips exactly; no out-of-bounds/degenerate boxes in sampled frames.

## Train

```bash
yolo detect train model=yolov8m.pt data=/home/siddartha/SoccerNet/yolo/data.yaml \
  imgsz=1280 epochs=100 batch=8 name=soccernet_v8m
```

Notes: use `imgsz≥1280` — the ball is ~12 px wide, small at 640. Consider a
higher-res model or tiling if ball recall is weak (a known SoccerNet issue —
reference repos report ~36 % ball AP).

## Next pipeline stages (project objectives)

1. **Tracking** — feed YOLOv8 detections to ByteTrack (built into Ultralytics:
   `model.track(..., tracker="bytetrack.yaml")`) for persistent player/ball ids.
2. **Pitch homography** — map image coords → pitch coords (SoccerNet camera
   calibration, or the field-keypoint model from tryolabs/soccer-video-analytics).
3. **Heatmaps** — accumulate per-player pitch positions over time → 2D density.
4. **Tactical / predictive** — team assignment (jersey-colour clustering),
   possession, formation, then the fusion with wearable data per the report.
