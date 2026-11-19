"""End-to-end check of the live path on one machine.

    PYTHONPATH=backend/src:. python scripts/e2e_live.py            # vest path only
    PYTHONPATH=backend/src:. python scripts/e2e_live.py --camera   # also a simulated phone camera

It starts a local copy of the relay, the sensor hub reading from it, and the
product API reading from the hub, then plays a simulated vest into the relay
exactly as the firmware sends it (batched HTTPS-style posts of IMU 50 Hz,
ECG 250 Hz in blocks of 50, GNSS 5 Hz and status every 2 s). It checks what
arrives at each stage: delivery rates, ECG heart rate against the simulated
one, GNSS speed, battery, clock offset. With ``--camera`` it also pairs a
Chromium tab with a fake camera through the relay and reads frames from it.

Exit code 0 means every stage passed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend" / "src"))
PY = sys.executable
TOKEN = secrets.token_hex(24)
RELAY, HUB, API = 18099, 18081, 18000
BPM, SPEED = 150.0, 4.0                    # what the simulated player does


def wait_http(url, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=1).read()
            return True
        except Exception:
            time.sleep(0.3)
    return False


def get(url):
    with urllib.request.urlopen(url, timeout=3) as r:
        return json.loads(r.read())


def post(url, body, headers=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


class FakeVest:
    """Produces firmware-shaped events on the real clock."""

    def __init__(self, player=1, match="e2e"):
        self.player, self.match, self.boot = player, match, secrets.randbits(31)
        self.seq = {"imu": 0, "ecg": 0, "gps": 0, "status": 0}
        self.t0 = time.time()
        self.stop = False
        self.sent = 0

    def event(self, kind, t, payload):
        self.seq[kind] += 1
        return {"event_id": f"{self.match}:{self.player}:{self.boot:08x}:{kind}:{self.seq[kind]}",
                "match_id": self.match, "player_id": self.player, "source": "wearable",
                "source_seq": self.seq[kind], "source_timestamp_ns": int(t * 1e9), "device_boot_id": str(self.boot),
                "sample_type": kind, "clock": {"valid": True, "method": "sntp_esp_timer_anchor"},
                "tags": {"session_id": self.match}, "payload": {"device_us": int((t - self.t0) * 1e6), **payload}}

    def ecg_block(self, t0):
        tt = t0 + np.arange(50) / 250.0
        phase = ((tt - self.t0) * BPM / 60.0) % 1.0
        w_r, w_t = 0.012 * BPM / 60, 0.045 * BPM / 60                # wave widths in beat phase
        x = 1.2 * np.exp(-0.5 * ((phase - 0.3) / w_r) ** 2) + 0.25 * np.exp(-0.5 * ((phase - 0.6) / w_t) ** 2)
        x += np.random.normal(0, 0.03, 50)
        return [int(v) for v in np.clip(2048 + 124 * x, 0, 4095)]

    def run(self, seconds):
        url = f"http://127.0.0.1:{RELAY}/api/v1/sensors/ingest"
        batch, next_imu, next_ecg, next_gps, next_st = [], self.t0, self.t0, self.t0, self.t0
        lat0, lon0 = 30.3525, 76.3712
        end = self.t0 + seconds
        while time.time() < end and not self.stop:
            now = time.time()
            while next_imu <= now:
                k = (next_imu - self.t0) * 2.8 * 2 * math.pi           # running cadence
                a = [0.4 * math.sin(k), 1.0, 9.81 + 3.0 * max(0.0, math.sin(k)) ** 3]
                batch.append(self.event("imu", next_imu, {"a": a, "g": [0.01, -0.02, 0.0], "temp_c": 31.0}))
                next_imu += 0.02
            while next_ecg <= now:
                batch.append(self.event("ecg", next_ecg, {"fs": 250, "v": self.ecg_block(next_ecg), "lead_off": False}))
                next_ecg += 0.2
            while next_gps <= now:
                d = SPEED * (next_gps - self.t0)
                batch.append(self.event("gps", next_gps, {"rx": True, "fix": True, "sat": 9, "chars": 1000,
                                                          "lat": lat0 + d / 111320.0, "lon": lon0, "speed_mps": SPEED,
                                                          "course_deg": 0.0, "hdop": 0.9}))
                next_gps += 0.2
            while next_st <= now:
                batch.append(self.event("status", next_st, {"rssi_dbm": -58, "heap": 180000, "mpu6050": True, "max30102": False,
                                                            "gps_rx": True, "imu_rate_hz": 50, "gps_rate_hz": 5.0, "ecg": True,
                                                            "ecg_rate_hz": 250.0, "battery_v": 3.95, "battery_pct": 82.0,
                                                            "dropped_samples": 0, "queued_samples": 0}))
                next_st += 2.0
            while batch:
                chunk, batch = batch[:96], batch[96:]
                try:
                    post(url, chunk, {"X-Auth": TOKEN})
                    self.sent += len(chunk)
                except Exception as exc:
                    print("  vest post failed:", exc)
            time.sleep(0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--camera", action="store_true")
    a = ap.parse_args()
    env = dict(os.environ, CPG44_RELAY_TOKEN=TOKEN, PYTHONPATH=f"{ROOT}/backend/src:{ROOT}")
    procs = []
    results = {}

    def start(cmd, extra=None):
        p = subprocess.Popen(cmd, env={**env, **(extra or {})}, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        procs.append(p)
        return p

    try:
        start([PY, "-m", "uvicorn", "cpg44_api.hostinger_sensor_relay:app", "--port", str(RELAY), "--log-level", "warning"])
        assert wait_http(f"http://127.0.0.1:{RELAY}/health"), "relay did not start"
        start([PY, "-m", "soccer_analytics.sensors.hub", "--http-port", str(HUB), "--player-id", "1", "--match-id", "e2e",
               "--relay-url", f"http://127.0.0.1:{RELAY}", "--relay-token", TOKEN, "--allow-local-relay"])
        assert wait_http(f"http://127.0.0.1:{HUB}/health"), "hub did not start"
        start([PY, "-m", "uvicorn", "cpg44_api.main:app", "--port", str(API), "--log-level", "warning"],
              {"CPG44_HUB_URL": f"http://127.0.0.1:{HUB}", "CPG44_SQUAD_FILE": "/tmp/cpg44-e2e-squad.json"})
        assert wait_http(f"http://127.0.0.1:{API}/api/v1/health"), "product API did not start"
        print("[e2e] relay, hub and API up")

        vest = FakeVest()
        th = threading.Thread(target=vest.run, args=(a.seconds,), daemon=True)
        th.start()
        time.sleep(min(a.seconds - 2, 35))
        hub = get(f"http://127.0.0.1:{HUB}/api/latest")
        st = get(f"http://127.0.0.1:{API}/api/v1/wearables/status")
        rates = st.get("rates_hz") or {}
        pulse = st.get("pulse") or {}
        gps = (st.get("gps") or {}).get("filtered") or {}
        checks = {
            "vest posts accepted": vest.sent > 0,
            "IMU arrives at >= 95 % of 50 Hz": rates.get("imu", 0) >= 47.5,
            "ECG blocks arrive at 5 per s": rates.get("ecg", 0) >= 4.5,
            "GNSS arrives at 5 Hz": rates.get("gps", 0) >= 4.5,
            f"ECG heart rate within 3 bpm of {BPM:.0f}": pulse.get("bpm") is not None and abs(pulse["bpm"] - BPM) <= 3,
            "heart rate source is the ECG": pulse.get("source") == "ecg",
            f"GNSS speed within 0.5 m/s of {SPEED}": gps.get("speed_mps") is not None and abs(gps["speed_mps"] - SPEED) <= 0.5,
            "battery reported": (st.get("battery") or {}).get("pct") == 82.0,
            "clock offset measured": st.get("clock_offset") is not None,
            "API sees the vest connected": st.get("connected") is True,
        }
        results.update(checks)
        print(f"[e2e] rates {rates}, pulse {pulse.get('bpm')} ({pulse.get('source')}), gnss {gps.get('speed_mps')}, "
              f"battery {(st.get('battery') or {}).get('pct')}, clock {st.get('clock_offset')}")
        vest.stop = True

        if a.camera:
            from playwright.sync_api import sync_playwright
            from soccer_analytics.phone_camera import PhoneCamera
            room = post(f"http://127.0.0.1:{RELAY}/api/v1/cam/rooms", {}, {"X-Auth": TOKEN})
            stop = threading.Event()

            def phone():
                with sync_playwright() as p:
                    b = p.chromium.launch(args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"])
                    pg = b.new_page()
                    pg.goto(f"http://127.0.0.1:{RELAY}{room['phone_path']}")
                    pg.click("#go")
                    stop.wait(60)
                    b.close()
            threading.Thread(target=phone, daemon=True).start()
            cam = PhoneCamera(f"phone://{room['room']}?key={room['key']}&relay=http://127.0.0.1:{RELAY}", first_frame_timeout_s=45)
            n, t1 = 0, time.time()
            while time.time() - t1 < 5:
                ok, f = cam.read()
                n += bool(ok)
            results["phone frames arrive over WebRTC"] = cam.mode == "direct" and n >= 25
            print(f"[e2e] phone camera {cam.mode}, {n / 5:.1f} fps read, frame {None if f is None else f.shape}")
            cam.release()
            stop.set()
    finally:
        for p in procs[::-1]:
            p.terminate()
        for p in procs:
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()

    width = max(len(k) for k in results) if results else 10
    for k, ok in results.items():
        print(f"  {'pass' if ok else 'FAIL'}  {k:{width}s}")
    return 0 if results and all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
