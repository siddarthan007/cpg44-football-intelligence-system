"""Measure how completely and how quickly the vest's samples reach this laptop.

Reads the running API's vest status once a second for ``--seconds`` and saves
the result to ``data/benchmarks/vest_delivery_now.json``, which the Evidence
page shows next to the first prototype's bench numbers.

    PYTHONPATH=backend/src:. python scripts/measure_vest_delivery.py --seconds 60
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", default=str(ROOT / "data" / "benchmarks" / "vest_delivery_now.json"))
    args = ap.parse_args()

    imu, ecg, delay, dropped = [], [], [], []
    end = time.time() + args.seconds
    with httpx.Client(timeout=3.0) as client:
        while time.time() < end:
            s = client.get(f"{args.api}/api/v1/wearables/status").json()
            if s.get("connected"):
                got, want = s.get("rates_hz") or {}, s.get("expected_hz") or {}
                if want.get("imu"):
                    imu.append(min(1.0, (got.get("imu") or 0) / want["imu"]))
                if want.get("ecg"):
                    ecg.append(min(1.0, (got.get("ecg") or 0) / want["ecg"]))
                if s.get("transport_delay_ms") is not None:
                    delay.append(float(s["transport_delay_ms"]))
                if s.get("dropped_samples") is not None:
                    dropped.append(int(s["dropped_samples"]))
            time.sleep(1.0)
    if not imu:
        print("the vest was not heard from; switch it on and try again")
        return 1
    delay.sort()
    out = {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": args.seconds, "readings": len(imu),
        "imu_completeness_median": round(statistics.median(imu), 3),
        "ecg_completeness_median": round(statistics.median(ecg), 3) if ecg else None,
        "transport_delay_ms_p50": round(statistics.median(delay), 1) if delay else None,
        "transport_delay_ms_p90": round(delay[int(0.9 * (len(delay) - 1))], 1) if delay else None,
        "dropped_during_run": (dropped[-1] - dropped[0]) if len(dropped) > 1 else None,
        "note": "Live vest over Wi-Fi through the relay, read from the running app once a second.",
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
