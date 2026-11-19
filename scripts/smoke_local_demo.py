#!/usr/bin/env python3
"""Short local check for API, React, TensorRT tracking, and MJPEG delivery."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
API_PORT = 18000
UI_PORT = 15173


def get_json(url: str, timeout: float = 1.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, body: dict | None = None, timeout: float = 5.0) -> dict:
    payload = json.dumps(body or {}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} returned {exc.code}: {detail}") from exc


def wait_for(url: str, seconds: float = 20.0) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError(f"service did not become ready: {url}")


def stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def main() -> int:
    video = ROOT / "demo" / "sample_match.mp4"
    weights = ROOT / "runs" / "detect" / "soccernet_v2" / "weights" / "best.engine"
    calibration = ROOT / "demo" / "calib" / "SNMOT-060.yaml"
    for path in (video, weights, calibration):
        if not path.is_file():
            raise FileNotFoundError(path)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'backend' / 'src'}:{ROOT}"
    env["MPLCONFIGDIR"] = "/tmp/cpg44-mpl-smoke"
    env["VITE_API_BASE_URL"] = f"http://127.0.0.1:{API_PORT}"
    env["VITE_WS_BASE_URL"] = f"ws://127.0.0.1:{API_PORT}"
    env["CPG44_PRODUCT_API"] = f"http://127.0.0.1:{API_PORT}"
    env["CPG44_MODEL_PATH"] = str(weights)
    env["CPG44_HUB_URL"] = ""

    processes = []
    with tempfile.TemporaryDirectory(prefix="cpg44-smoke-") as tmp:
        tmp_path = Path(tmp)
        api_log = (tmp_path / "api.log").open("w", encoding="utf-8")
        ui_log = (tmp_path / "ui.log").open("w", encoding="utf-8")
        vision_started = False
        try:
            api_process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "cpg44_api.main:app", "--host", "127.0.0.1", "--port", str(API_PORT)],
                cwd=ROOT,
                env=env,
                stdout=api_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append(api_process)
            wait_for(f"http://127.0.0.1:{API_PORT}/api/v1/health")

            ui_process = subprocess.Popen(
                ["npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", str(UI_PORT)],
                cwd=ROOT / "frontend",
                env=env,
                stdout=ui_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append(ui_process)
            wait_for(f"http://127.0.0.1:{UI_PORT}")

            post_json(
                f"http://127.0.0.1:{API_PORT}/api/v1/cameras/register",
                {
                    "id": "smoke",
                    "name": "Smoke test footage",
                    "type": "file",
                    "source": str(video),
                    "calibration_path": str(calibration),
                },
            )
            started = post_json(
                f"http://127.0.0.1:{API_PORT}/api/v1/cameras/smoke/start"
            )
            if not started.get("ok"):
                raise RuntimeError("camera control did not start vision")
            vision_started = True

            snapshot = {}
            deadline = time.time() + 30
            while time.time() < deadline:
                snapshot = get_json(f"http://127.0.0.1:{API_PORT}/api/v1/live")
                if int(snapshot.get("frame_index") or 0) >= 100:
                    break
                status = get_json(f"http://127.0.0.1:{API_PORT}/api/v1/vision/status")
                if status.get("status") == "failed":
                    raise RuntimeError(status.get("error") or "vision process failed")
                time.sleep(0.2)
            if int(snapshot.get("frame_index") or 0) < 100:
                raise RuntimeError("vision process did not publish enough frames")

            with urllib.request.urlopen(
                f"http://127.0.0.1:{API_PORT}/api/v1/live/mjpeg", timeout=5
            ) as stream:
                mjpeg = stream.read(4096)
            if b"--frame" not in mjpeg or b"\xff\xd8" not in mjpeg:
                raise RuntimeError("MJPEG endpoint did not return a JPEG frame")

            result = {
                "ok": True,
                "frontend": f"http://127.0.0.1:{UI_PORT}",
                "frame_index": snapshot.get("frame_index"),
                "processing_fps": snapshot.get("processing_fps"),
                "source_fps": snapshot.get("source_fps"),
                "players": len(snapshot.get("players") or []),
                "ball_observed": (snapshot.get("data_quality") or {}).get("ball_observed"),
                "mjpeg_bytes_checked": len(mjpeg),
            }
            print(json.dumps(result, indent=2))
            return 0
        finally:
            for process in reversed(processes):
                if vision_started and process is processes[0]:
                    try:
                        post_json(f"http://127.0.0.1:{API_PORT}/api/v1/vision/stop")
                    except Exception:
                        pass
                stop(process)
            api_log.close()
            ui_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
