"""Run real footage through both trackers without publishing to an active session."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import platform

import torch
import ultralytics
import supervision

from soccer_analytics.config import PipelineConfig
from soccer_analytics.pitch import load_calibration
from soccer_analytics.realtime import RealtimePipeline
from soccer_analytics.view import ViewTransformer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--calibration")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for backend in ("bytetrack", "botsort"):
        cfg = PipelineConfig(weights=args.weights, imgsz=args.imgsz, tracker=backend)
        view = load_calibration(args.calibration, cfg.pitch) if args.calibration else ViewTransformer()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        pipeline = RealtimePipeline(cfg, view, warmup=3, dashboard=False, snapshot_every=1)
        snapshot_path = args.out / f"{backend}.ndjson"
        if snapshot_path.exists():
            raise FileExistsError(f"choose a fresh output directory: {snapshot_path}")
        started = time.perf_counter()
        pipeline.run(args.video, max_frames=args.frames, fused_out=str(snapshot_path))
        elapsed = time.perf_counter() - started
        snapshots = [json.loads(line) for line in snapshot_path.read_text().splitlines()]
        if not snapshots:
            raise RuntimeError("pipeline emitted no snapshots")
        pins = sum(row["data_quality"]["positioned_players"] for row in snapshots)
        tracked = sum(len(row["players"]) for row in snapshots)
        if not args.calibration:
            assert all(not row["metric"] and row["heatmap"] is None for row in snapshots)
            assert pins == 0
        results.append({
            "tracker": backend, "frames": len(snapshots), "wall_seconds_including_warmup": round(elapsed, 3),
            "tracked_player_observations": tracked, "accepted_position_observations": pins,
            "calibration_supported_frames": sum(row["data_quality"]["calibration"]["valid"] for row in snapshots),
            "reidentifications": pipeline.reid.reids, "ambiguous_reidentifications": pipeline.reid.ambiguous,
            "peak_torch_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1) if torch.cuda.is_available() else None,
            "accuracy_claim": "Smoke diagnostics only. No manually labelled ground truth was used.",
        })
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report = {"video": args.video, "weights": args.weights,
              "weights_sha256": hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
              "calibration": args.calibration,
              "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                          "cuda": torch.version.cuda, "ultralytics": ultralytics.__version__,
                          "supervision": supervision.__version__}, "results": results}
    (args.out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
