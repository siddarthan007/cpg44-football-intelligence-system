"""Pitch-space diagnostics on explicitly annotated frames, not a HOTA replacement.

Input NDJSON: one snapshot per frame_index, metric=true, players containing
track_id, x, y (metres), and optional team and player_id. Ground-truth track_id
must identify the same person throughout one clip. Evaluate clips separately.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def read_frames(path: Path) -> dict[int, dict]:
    frames = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            frame = int(row["frame_index"])
            if frame in frames:
                raise ValueError(f"{path}:{line_number}: duplicate frame_index {frame}; split repeated playback loops")
            if row.get("metric") is not True:
                raise ValueError(f"{path}:{line_number}: pitch evaluation requires metric=true")
            frames[frame] = row
    if not frames:
        raise ValueError(f"no annotated frames in {path}")
    return frames


def _players(frame, strict=False):
    result = []
    seen = set()
    for player in frame.get("players", []):
        tid = player["track_id"]
        if tid in seen:
            raise ValueError(f"duplicate track_id {tid} in one frame")
        seen.add(tid)
        xy = (player.get("x"), player.get("y"))
        if any(v is None for v in xy) or not np.isfinite(xy).all():
            if strict:
                raise ValueError("ground-truth positions must be finite metres; omit unannotated individuals explicitly")
            continue
        result.append(player)
    return result


def evaluate_frames(truth: dict[int, dict], predictions: dict[int, dict], gate_m: float = 2.0) -> dict:
    if not np.isfinite(gate_m) or gate_m <= 0:
        raise ValueError("gate_m must be positive and finite")
    counts = Counter()
    pairs = Counter()
    previous = {}
    errors = []
    for frame_idx in sorted(truth):
        gt = _players(truth[frame_idx], strict=True)
        pred = _players(predictions.get(frame_idx, {}))
        counts["ground_truth"] += len(gt)
        counts["predictions"] += len(pred)
        counts["team_labels"] += sum(p.get("team") in (1, 2) for p in gt)
        counts["roster_labels"] += sum(p.get("player_id") is not None for p in gt)
        if not gt or not pred:
            continue
        a = np.asarray([(p["x"], p["y"]) for p in gt])
        b = np.asarray([(p["x"], p["y"]) for p in pred])
        distance = np.linalg.norm(a[:, None] - b[None, :], axis=2)
        rows, cols = linear_sum_assignment(np.where(distance <= gate_m, distance, 1e9))
        for row, col in zip(rows, cols):
            if distance[row, col] > gate_m:
                continue
            g, p = gt[row], pred[col]
            counts["matched"] += 1
            errors.append(float(distance[row, col]))
            pairs[g["track_id"], p["track_id"]] += 1
            last = previous.get(g["track_id"])
            if last is not None and last != p["track_id"]:
                counts["id_switches"] += 1
            previous[g["track_id"]] = p["track_id"]
            if g.get("team") in (1, 2) and p.get("team") in (1, 2):
                counts["team_assigned"] += 1
                counts["team_correct"] += int(g["team"] == p["team"])
            if g.get("player_id") is not None and p.get("player_id") is not None:
                counts["roster_assigned"] += 1
                counts["roster_correct"] += int(g["player_id"] == p["player_id"])

    idtp = 0
    if pairs:
        gt_ids = sorted({key[0] for key in pairs})
        pred_ids = sorted({key[1] for key in pairs})
        matrix = np.asarray([[pairs[g, p] for p in pred_ids] for g in gt_ids])
        rows, cols = linear_sum_assignment(-matrix)
        idtp = int(matrix[rows, cols].sum())

    def ratio(n, d):
        return round(n / d, 6) if d else None

    return {
        "annotated_frames": len(truth), "gate_m": gate_m,
        "ground_truth_positions": counts["ground_truth"],
        "predicted_positions": counts["predictions"], "matched_positions": counts["matched"],
        "position_recall": ratio(counts["matched"], counts["ground_truth"]),
        "position_precision": ratio(counts["matched"], counts["predictions"]),
        "matched_position_mae_m": round(float(np.mean(errors)), 6) if errors else None,
        "matched_position_p95_m": round(float(np.percentile(errors, 95)), 6) if errors else None,
        "pitch_idf1_at_gate": ratio(2 * idtp, counts["ground_truth"] + counts["predictions"]),
        "identity_switches_between_matches": counts["id_switches"],
        "team_accuracy_when_assigned": ratio(counts["team_correct"], counts["team_assigned"]),
        "team_coverage": ratio(counts["team_assigned"], counts["team_labels"]),
        "roster_accuracy_when_bound": ratio(counts["roster_correct"], counts["roster_assigned"]),
        "roster_coverage": ratio(counts["roster_assigned"], counts["roster_labels"]),
        "scope": "Only ground-truth frame indices; invalid/unprojectable pins count as missed positions. "
                 "Pitch IDF1 uses this metric-distance gate. Use official TrackEval for benchmark HOTA/IDF1/GS-HOTA.",
    }


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gate-m", type=float, default=2.0)
    args = parser.parse_args(argv)
    result = evaluate_frames(read_frames(args.truth), read_frames(args.predictions), args.gate_m)
    result["inputs"] = {name: {"path": str(path.resolve()), "sha256": _sha256(path)}
                        for name, path in (("truth", args.truth), ("predictions", args.predictions))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
