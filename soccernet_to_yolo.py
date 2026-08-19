#!/usr/bin/env python3
"""
soccernet_to_yolo.py
====================

Convert the **SoccerNet Tracking** dataset (MOT Challenge layout) into the
Ultralytics YOLO detection format, build train/val/test splits, and emit the
``data.yaml`` consumed by ``ultralytics`` / YOLOv8.

Why this exists
---------------
The SoccerNet *Tracking* release is **not** distributed as JSON. Each clip is a
MOT-Challenge sequence::

    SNMOT-060/
        seqinfo.ini          # imWidth / imHeight / seqLength / imExt ...
        gameinfo.ini         # trackletID_<id> = <role>;<jersey>   -> gives the CLASS
        gt/gt.txt            # frame,id,bb_left,bb_top,bb_w,bb_h,conf,class,vis(,...)
                             #   NOTE: SoccerNet leaves class+visibility as -1 (unused)
        det/det.txt          # public detections (ignored here)
        img1/000001.jpg ...  # zero-padded frames

Two SoccerNet quirks this script handles:

1. The class column (col 8) in ``gt.txt`` is always ``-1``. The real object
   class must be recovered from ``gameinfo.ini`` by mapping the tracklet ``id``
   (col 2 of gt) to its role (``player``/``goalkeeper``/``referee``/``ball``).
2. Frame filenames collide across sequences (every clip has ``000001.jpg``), so
   output files are namespaced as ``<sequence>_<frame>.jpg`` / ``.txt``.

Output layout (Ultralytics standard)::

    <dst>/
        images/{train,val,test}/<seq>_<frame>.jpg
        labels/{train,val,test}/<seq>_<frame>.txt
        data.yaml
        conversion_report.json

Splitting is done **by sequence** (never by frame) so that near-identical
adjacent frames from one clip cannot leak between train and val.

Example
-------
    python soccernet_to_yolo.py \
        --src /home/siddartha/SoccerNet/tracking \
        --dst /home/siddartha/SoccerNet/yolo \
        --val-frac 0.15 --stride 5 --image-mode symlink

Dependencies: standard library only. ``Pillow`` and ``tqdm`` are used if present
but are optional.
"""

from __future__ import annotations

import argparse
import configparser
import json
import logging
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Optional dependencies (degrade gracefully if missing)
# --------------------------------------------------------------------------- #
try:  # only needed as a fallback when seqinfo.ini lacks image dimensions
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - optional
    Image = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - optional
    def tqdm(iterable: Iterable, **_kwargs):  # type: ignore
        return iterable


log = logging.getLogger("soccernet_to_yolo")


# --------------------------------------------------------------------------- #
# Class mapping
# --------------------------------------------------------------------------- #
# Default class vocabulary + index order. This matches the widely used Roboflow
# "football-players-detection" ordering, keeping the produced dataset compatible
# with the community pretrained weights referenced by the project's target repos
# (abdullahtarek/football_analysis, rajveersinghcse/Football-Analysis-using-YOLO).
DEFAULT_NAMES: List[str] = ["ball", "goalkeeper", "player", "referee"]


def default_role_to_class(role: str) -> Optional[str]:
    """Map a raw ``gameinfo.ini`` role string to a YOLO class name.

    SoccerNet roles seen in the data:
        "player team left", "player team right",
        "goalkeeper team right", "goalkeepers team left",   # note: plural variant
        "referee", "ball", "other"

    Team side (left/right) is intentionally collapsed into a single ``player``
    class -- team assignment is a downstream step (jersey-colour clustering),
    not an object-detection class. Returns ``None`` for roles that should be
    dropped (e.g. "other").
    """
    r = " ".join(role.strip().lower().split())  # collapse whitespace, lowercase
    if r.startswith("ball"):
        return "ball"
    if r.startswith("goalkeeper"):  # covers "goalkeeper" and "goalkeepers"
        return "goalkeeper"
    if r.startswith("player"):
        return "player"
    if r.startswith("referee"):
        return "referee"
    return None  # "other" / anything unrecognised -> dropped


# --------------------------------------------------------------------------- #
# Data holders
# --------------------------------------------------------------------------- #
@dataclass
class SeqInfo:
    name: str
    im_dir: str = "img1"
    im_ext: str = ".jpg"
    im_width: int = 0
    im_height: int = 0
    seq_length: int = 0


@dataclass
class Stats:
    sequences: int = 0
    sequences_skipped: int = 0
    frames_written: int = 0
    frames_missing_image: int = 0
    frames_empty: int = 0
    labels_written: int = 0
    boxes_by_class: Counter = field(default_factory=Counter)
    skipped_unknown_id: int = 0
    skipped_unmapped_role: int = 0
    skipped_zero_conf: int = 0
    skipped_low_visibility: int = 0
    skipped_degenerate_box: int = 0
    boxes_clipped: int = 0
    per_split_frames: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict:
        d = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in self.__dict__.items()}
        return d


# --------------------------------------------------------------------------- #
# INI parsing
# --------------------------------------------------------------------------- #
def _read_ini(path: Path) -> configparser.ConfigParser:
    cp = configparser.ConfigParser()
    # SoccerNet .ini files are plain "key=value" under a [Sequence] header;
    # keep case of keys as-is and be tolerant of odd spacing.
    cp.optionxform = str  # type: ignore[assignment]
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        cp.read_file(fh)
    return cp


def parse_seqinfo(seq_dir: Path) -> SeqInfo:
    """Read ``seqinfo.ini`` and, when dimensions are absent, fall back to the
    first image on disk (requires Pillow) so conversion can still proceed."""
    info = SeqInfo(name=seq_dir.name)
    ini = seq_dir / "seqinfo.ini"
    if ini.is_file():
        try:
            cp = _read_ini(ini)
        except configparser.Error as e:
            log.warning("[%s] malformed seqinfo.ini (%s); will try to probe image size", seq_dir.name, e)
            cp = configparser.ConfigParser()
        sec = cp["Sequence"] if cp.has_section("Sequence") else (cp[cp.sections()[0]] if cp.sections() else {})
        info.name = sec.get("name", seq_dir.name)
        info.im_dir = sec.get("imDir", "img1")
        info.im_ext = sec.get("imExt", ".jpg")
        info.im_width = int(sec.get("imWidth", 0) or 0)
        info.im_height = int(sec.get("imHeight", 0) or 0)
        info.seq_length = int(sec.get("seqLength", 0) or 0)
    else:
        log.warning("[%s] no seqinfo.ini; assuming defaults", seq_dir.name)

    if info.im_width <= 0 or info.im_height <= 0:
        w, h = _probe_image_size(seq_dir / info.im_dir, info.im_ext)
        if w and h:
            info.im_width, info.im_height = w, h
            log.info("[%s] dimensions probed from image: %dx%d", info.name, w, h)
    return info


def _probe_image_size(img_dir: Path, ext: str) -> Tuple[int, int]:
    if Image is None or not img_dir.is_dir():
        return 0, 0
    for p in sorted(img_dir.glob(f"*{ext}")):
        try:
            with Image.open(p) as im:
                return im.width, im.height
        except Exception:
            continue
    return 0, 0


def parse_gameinfo(seq_dir: Path, role_to_class) -> Dict[int, Optional[str]]:
    """Return ``{tracklet_id: class_name_or_None}`` from ``gameinfo.ini``.

    Lines look like ``trackletID_18= ball;1``. The numeric suffix is the tracklet
    id that appears in the ``id`` column of ``gt.txt``.
    """
    mapping: Dict[int, Optional[str]] = {}
    ini = seq_dir / "gameinfo.ini"
    if not ini.is_file():
        log.warning("[%s] no gameinfo.ini; all boxes will be treated as unknown id", seq_dir.name)
        return mapping
    try:
        cp = _read_ini(ini)
    except configparser.Error as e:
        log.warning("[%s] malformed gameinfo.ini (%s); treating all ids as unknown", seq_dir.name, e)
        return mapping
    sec = cp["Sequence"] if cp.has_section("Sequence") else (cp[cp.sections()[0]] if cp.sections() else {})
    for key, val in sec.items():
        if not key.startswith("trackletID_"):
            continue
        try:
            tid = int(key.split("_", 1)[1])
        except ValueError:
            continue
        role = val.split(";", 1)[0]  # strip the ";jersey" suffix
        mapping[tid] = role_to_class(role)
    return mapping


# --------------------------------------------------------------------------- #
# gt.txt parsing + conversion
# --------------------------------------------------------------------------- #
@dataclass
class Box:
    cls_idx: int
    xc: float
    yc: float
    w: float
    h: float

    def line(self) -> str:
        return f"{self.cls_idx} {self.xc:.6f} {self.yc:.6f} {self.w:.6f} {self.h:.6f}"


def convert_box(
    bb_left: float,
    bb_top: float,
    bb_w: float,
    bb_h: float,
    im_w: int,
    im_h: int,
) -> Tuple[Optional[Tuple[float, float, float, float]], bool]:
    """MOT top-left+wh (pixels) -> YOLO normalized center xywh.

    Clips the box to the image rectangle. Returns ``(xywh_or_None, was_clipped)``;
    ``None`` means the box is degenerate (zero/negative area) after clipping.
    """
    x1, y1 = bb_left, bb_top
    x2, y2 = bb_left + bb_w, bb_top + bb_h

    cx1, cy1 = max(0.0, x1), max(0.0, y1)
    cx2, cy2 = min(float(im_w), x2), min(float(im_h), y2)
    clipped = (cx1 != x1) or (cy1 != y1) or (cx2 != x2) or (cy2 != y2)

    cw, ch = cx2 - cx1, cy2 - cy1
    if cw <= 0.0 or ch <= 0.0:
        return None, clipped

    xc = (cx1 + cx2) / 2.0 / im_w
    yc = (cy1 + cy2) / 2.0 / im_h
    nw = cw / im_w
    nh = ch / im_h
    # numerical safety
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)
    return (xc, yc, nw, nh), clipped


def parse_gt(gt_path: Path) -> Dict[int, List[tuple]]:
    """Parse ``gt.txt`` into ``{frame: [(id, l, t, w, h, conf, vis), ...]}``.

    Malformed lines are skipped with a warning rather than aborting the run.
    """
    frames: Dict[int, List[tuple]] = defaultdict(list)
    if not gt_path.is_file():
        return frames
    with gt_path.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split(",")
            if len(parts) < 6:
                log.debug("%s:%d malformed gt line: %r", gt_path, lineno, raw)
                continue
            try:
                frame = int(float(parts[0]))
                tid = int(float(parts[1]))
                l = float(parts[2])
                t = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                conf = float(parts[6]) if len(parts) > 6 and parts[6] != "" else 1.0
                # visibility = standard MOT gt column 9 (0-indexed 8). SoccerNet
                # leaves this at -1 (unset), so --min-visibility is a no-op here
                # but works for MOT sets that populate it.
                vis = float(parts[8]) if len(parts) > 8 and parts[8] != "" else -1.0
            except ValueError:
                log.debug("%s:%d unparseable gt line: %r", gt_path, lineno, raw)
                continue
            frames[frame].append((tid, l, t, w, h, conf, vis))
    return frames


# --------------------------------------------------------------------------- #
# Filename padding detection
# --------------------------------------------------------------------------- #
def detect_pad_width(img_dir: Path, ext: str, default: int = 6) -> int:
    """Infer the zero-pad width of frame filenames (e.g. ``000001.jpg`` -> 6)."""
    if not img_dir.is_dir():
        return default
    for p in sorted(img_dir.glob(f"*{ext}")):
        stem = p.stem
        if stem.isdigit():
            return len(stem)
    return default


# --------------------------------------------------------------------------- #
# Image materialisation
# --------------------------------------------------------------------------- #
def materialise_image(src: Path, dst: Path, mode: str) -> None:
    """Place the source frame at ``dst`` using the requested strategy.

    ``symlink`` falls back to a copy if the OS/filesystem forbids symlinks
    (e.g. Windows without privilege). ``hardlink`` falls back likewise.
    """
    if dst.exists() or dst.is_symlink():
        return
    if mode == "none":
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    # default: symlink
    try:
        os.symlink(os.path.abspath(src), dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


# --------------------------------------------------------------------------- #
# Sequence discovery + splitting
# --------------------------------------------------------------------------- #
def list_sequences(split_dir: Path) -> List[Path]:
    if not split_dir.is_dir():
        return []
    seqs = [p for p in sorted(split_dir.iterdir()) if p.is_dir() and (p / "img1").is_dir()]
    # be tolerant of a non-standard imDir
    if not seqs:
        seqs = [p for p in sorted(split_dir.iterdir()) if p.is_dir() and (p / "seqinfo.ini").is_file()]
    return seqs


def plan_splits(
    src: Path,
    val_frac: float,
    seed: int,
    use_test_dir: bool,
) -> Dict[str, List[Path]]:
    """Return ``{"train": [...], "val": [...], "test": [...]}`` of sequence dirs.

    ``val`` is carved out of the SoccerNet *train* directory (SoccerNet ships no
    val split). ``test`` is the SoccerNet *test* directory when present.
    """
    train_seqs = list_sequences(src / "train")
    if not train_seqs:
        raise FileNotFoundError(
            f"No sequences under {src / 'train'} -- is --src the tracking root "
            f"that contains train/ and test/?"
        )

    rng = random.Random(seed)
    shuffled = train_seqs[:]
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_frac))
    # With >1 train sequence, always hold out at least one for val so a small
    # rounding (e.g. round(3 * 0.15) == 0) can't silently leave val empty.
    if len(shuffled) > 1:
        floor = 1 if val_frac > 0 else 0
        n_val = min(max(n_val, floor), len(shuffled) - 1)
    else:
        n_val = 0
    val_seqs = sorted(shuffled[:n_val], key=lambda p: p.name)
    tr_seqs = sorted(shuffled[n_val:], key=lambda p: p.name)

    splits = {"train": tr_seqs, "val": val_seqs, "test": []}
    if use_test_dir:
        splits["test"] = list_sequences(src / "test")
    return splits


# --------------------------------------------------------------------------- #
# Core conversion of one sequence
# --------------------------------------------------------------------------- #
def convert_sequence(
    seq_dir: Path,
    split: str,
    dst: Path,
    name_to_idx: Dict[str, int],
    role_to_class,
    args: argparse.Namespace,
    stats: Stats,
    write: bool = True,
) -> None:
    """Convert one MOT sequence. When ``write`` is False this is a stats-only
    (dry-run) pass: it parses and counts exactly as the real run but touches no
    disk, guaranteeing dry-run figures match the real conversion."""
    info = parse_seqinfo(seq_dir)
    if info.im_width <= 0 or info.im_height <= 0:
        log.error("[%s] unknown image size (install Pillow or fix seqinfo.ini); skipping", info.name)
        stats.sequences_skipped += 1
        return

    id_to_class = parse_gameinfo(seq_dir, role_to_class)
    gt = parse_gt(seq_dir / "gt" / "gt.txt")
    if not gt:
        log.warning("[%s] empty/missing gt/gt.txt; skipping sequence", info.name)
        stats.sequences_skipped += 1
        return

    img_dir = seq_dir / info.im_dir
    pad = detect_pad_width(img_dir, info.im_ext, default=6)

    img_out = dst / "images" / split
    lbl_out = dst / "labels" / split
    if write:
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

    frames_sorted = sorted(gt.keys())
    per_seq_written = 0
    for idx, frame in enumerate(frames_sorted):
        if args.stride > 1 and (idx % args.stride != 0):
            continue
        if args.max_frames_per_seq and per_seq_written >= args.max_frames_per_seq:
            break

        fname = f"{frame:0{pad}d}{info.im_ext}"
        src_img = img_dir / fname
        if not src_img.is_file():
            # ---- missing-frame handling: warn (rate-limited) and skip ----
            stats.frames_missing_image += 1
            if stats.frames_missing_image <= args.max_warn:
                log.warning("[%s] missing image for frame %d (%s); skipping",
                            info.name, frame, src_img)
            continue

        boxes: List[Box] = []
        for (tid, l, t, w, h, conf, vis) in gt[frame]:
            if conf == 0:
                stats.skipped_zero_conf += 1
                continue
            if args.min_visibility is not None and vis >= 0.0 and vis < args.min_visibility:
                stats.skipped_low_visibility += 1
                continue

            if tid in id_to_class:
                cls_name = id_to_class[tid]
                if cls_name is None:  # role mapped to "drop" (e.g. "other")
                    stats.skipped_unmapped_role += 1
                    continue
            else:
                # tracklet id absent from gameinfo.ini
                if args.unknown_id_policy == "skip":
                    stats.skipped_unknown_id += 1
                    continue
                cls_name = args.unknown_id_policy  # e.g. "player"

            if cls_name not in name_to_idx:
                stats.skipped_unmapped_role += 1
                continue

            xywh, clipped = convert_box(l, t, w, h, info.im_width, info.im_height)
            if clipped:
                stats.boxes_clipped += 1
            if xywh is None:
                stats.skipped_degenerate_box += 1
                continue
            boxes.append(Box(name_to_idx[cls_name], *xywh))
            stats.boxes_by_class[cls_name] += 1

        if not boxes and not args.negatives:
            # frame has an image but no usable box; skip unless negatives wanted
            stats.frames_empty += 1
            continue

        out_stem = f"{info.name}_{frame:0{pad}d}"
        if write:
            materialise_image(src_img, img_out / f"{out_stem}{info.im_ext}", args.image_mode)
            with (lbl_out / f"{out_stem}.txt").open("w", encoding="utf-8") as fh:
                fh.write("\n".join(b.line() for b in boxes))
                if boxes:
                    fh.write("\n")

        stats.frames_written += 1
        stats.labels_written += len(boxes)
        stats.per_split_frames[split] += 1
        per_seq_written += 1

    stats.sequences += 1
    log.info("[%s] split=%s frames=%d", info.name, split, per_seq_written)


# --------------------------------------------------------------------------- #
# data.yaml
# --------------------------------------------------------------------------- #
def write_data_yaml(dst: Path, names: List[str], splits_present: Dict[str, bool]) -> Path:
    """Write ``data.yaml`` (hand-serialised so PyYAML is not required)."""
    lines: List[str] = []
    lines.append("# Auto-generated by soccernet_to_yolo.py")
    # absolute path: Ultralytics resolves a relative `path` against its own
    # datasets_dir (e.g. ~/datasets), NOT the CWD, so relative would break.
    lines.append(f"path: {dst.resolve().as_posix()}")
    lines.append("train: images/train")
    lines.append("val: images/val" if splits_present.get("val") else "val: images/train")
    if splits_present.get("test"):
        lines.append("test: images/test")
    lines.append("")
    lines.append(f"nc: {len(names)}")
    lines.append("names:")
    for i, n in enumerate(names):
        lines.append(f"  {i}: {n}")
    lines.append("")
    out = dst / "data.yaml"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert SoccerNet Tracking (MOT format) to Ultralytics YOLO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--src", required=True, type=Path,
                   help="SoccerNet tracking root (contains train/ and test/).")
    p.add_argument("--dst", required=True, type=Path,
                   help="Output dataset root (created if missing).")
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="Fraction of TRAIN sequences held out for validation.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for the deterministic sequence shuffle.")
    p.add_argument("--stride", type=int, default=1,
                   help="Keep every Nth annotated frame per sequence (reduce redundancy).")
    p.add_argument("--max-frames-per-seq", type=int, default=0,
                   help="Cap frames written per sequence (0 = no cap).")
    p.add_argument("--image-mode", choices=["symlink", "copy", "hardlink", "none"],
                   default="symlink",
                   help="How to place images in the output tree.")
    p.add_argument("--no-test", action="store_true",
                   help="Do not build a test split from the SoccerNet test/ dir.")
    p.add_argument("--negatives", action="store_true",
                   help="Also write empty label files for frames with no boxes (background images).")
    p.add_argument("--min-visibility", type=float, default=None,
                   help="Drop boxes whose visibility (MOT gt col 9) is below this. "
                        "SoccerNet leaves it -1, so this is a no-op on SoccerNet data.")
    p.add_argument("--keep-existing", action="store_true",
                   help="Do NOT purge <dst>/images and <dst>/labels first. Default is to "
                        "clean them, so a rerun with different params leaves no stale files.")
    p.add_argument("--unknown-id-policy", default="skip",
                   help="What to do with a gt id missing from gameinfo.ini: "
                        "'skip' or a class name (e.g. 'player').")
    p.add_argument("--class-map", type=Path, default=None,
                   help="Optional JSON file: {\"role substring\": \"class\"} overriding defaults.")
    p.add_argument("--names", default=",".join(DEFAULT_NAMES),
                   help="Comma-separated class index order for the YOLO dataset.")
    p.add_argument("--max-warn", type=int, default=20,
                   help="Max per-category warnings to print before going quiet.")
    p.add_argument("--dry-run", action="store_true",
                   help="Parse and report stats without writing images/labels.")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return p


def load_custom_class_map(path: Path):
    """Return a role->class function backed by a user JSON of substring rules."""
    mapping = json.loads(path.read_text(encoding="utf-8"))
    rules = [(k.strip().lower(), v) for k, v in mapping.items()]

    def fn(role: str) -> Optional[str]:
        r = " ".join(role.strip().lower().split())
        for sub, cls in rules:
            if sub in r:
                return cls if cls else None
        return None

    return fn


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stdout,
    )

    if not args.src.is_dir():
        log.error("--src %s does not exist", args.src)
        return 2

    names = [n.strip() for n in args.names.split(",") if n.strip()]
    name_to_idx = {n: i for i, n in enumerate(names)}
    role_to_class = (
        load_custom_class_map(args.class_map) if args.class_map else default_role_to_class
    )
    # validate unknown-id-policy
    if args.unknown_id_policy != "skip" and args.unknown_id_policy not in name_to_idx:
        log.error("--unknown-id-policy %r is neither 'skip' nor one of %s",
                  args.unknown_id_policy, names)
        return 2

    try:
        splits = plan_splits(args.src, args.val_frac, args.seed, use_test_dir=not args.no_test)
    except FileNotFoundError as e:
        log.error("%s", e)
        return 2

    log.info("Split plan: train=%d val=%d test=%d sequences",
             len(splits["train"]), len(splits["val"]), len(splits["test"]))

    stats = Stats()
    args.dst = args.dst.resolve()  # absolute -> correct data.yaml + no CWD surprises

    if args.dry_run:
        log.info("--dry-run: no files will be written")
    else:
        args.dst.mkdir(parents=True, exist_ok=True)
        # Clean prior outputs so a rerun with different params can't leave stale
        # images/labels behind (which would corrupt splits / leak between them).
        if not args.keep_existing:
            for sub in ("images", "labels"):
                d = args.dst / sub
                if d.exists():
                    log.info("Cleaning existing %s", d)
                    shutil.rmtree(d, ignore_errors=True)
        else:
            log.warning("--keep-existing: not cleaning %s; stale files from a "
                        "previous run may remain and corrupt the dataset", args.dst)

    for split, seqs in splits.items():
        for seq_dir in tqdm(seqs, desc=f"{split}", unit="seq"):
            convert_sequence(seq_dir, split, args.dst, name_to_idx,
                             role_to_class, args, stats, write=not args.dry_run)

    # Presence is derived from what was actually WRITTEN, not merely planned:
    # a planned split whose sequences all got skipped must not appear in data.yaml.
    splits_present = {s: stats.per_split_frames.get(s, 0) > 0 for s in splits}
    if not splits_present.get("val"):
        log.warning("No validation frames were written; data.yaml will set "
                    "val=images/train, so validation would run on TRAINING images "
                    "(leakage). Add more sequences or raise --val-frac.")

    if not args.dry_run:
        yaml_path = write_data_yaml(args.dst, names, splits_present)
        report = {
            "src": str(args.src),
            "dst": str(args.dst),
            "names": names,
            "splits": {s: [p.name for p in seqs] for s, seqs in splits.items()},
            "splits_present": splits_present,
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "stats": stats.as_dict(),
        }
        (args.dst / "conversion_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        log.info("Wrote %s", yaml_path)

    _print_summary(stats, names)
    return 0


def _print_summary(stats: Stats, names: List[str]) -> None:
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("  sequences converted : %d (skipped %d)", stats.sequences, stats.sequences_skipped)
    log.info("  frames written      : %d", stats.frames_written)
    log.info("  labels (boxes)      : %d", stats.labels_written)
    for split, n in sorted(stats.per_split_frames.items()):
        log.info("    %-6s frames      : %d", split, n)
    log.info("  boxes by class      :")
    for n in names:
        log.info("    %-12s      : %d", n, stats.boxes_by_class.get(n, 0))
    log.info("  missing images      : %d", stats.frames_missing_image)
    log.info("  empty frames        : %d", stats.frames_empty)
    log.info("  boxes clipped       : %d", stats.boxes_clipped)
    log.info("  skipped unknown id  : %d", stats.skipped_unknown_id)
    log.info("  skipped role->drop  : %d", stats.skipped_unmapped_role)
    log.info("  skipped zero-conf   : %d", stats.skipped_zero_conf)
    log.info("  skipped low-vis     : %d", stats.skipped_low_visibility)
    log.info("  skipped degenerate  : %d", stats.skipped_degenerate_box)
    log.info("=" * 60)


if __name__ == "__main__":
    raise SystemExit(main())
