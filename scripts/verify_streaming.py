"""Exercise authenticated HTTP -> WebSocket relay and paginated recovery on loopback."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import socket
import sys
import time

import httpx
import numpy as np
from websockets.asyncio.client import connect


async def verify(output):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    token = "synthetic-relay-verification-token"
    env = dict(os.environ, CPG44_RELAY_TOKEN=token,
               PYTHONPATH=str(Path(__file__).resolve().parents[1] / "backend/src"))
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "uvicorn", "cpg44_api.hostinger_sensor_relay:app",
        "--host", "127.0.0.1", "--port", str(port), "--log-level", "error",
        env=env, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    base = f"http://127.0.0.1:{port}"
    headers = {"X-Auth": token}
    sequence, latency, received, started = 0, [], [], {}
    try:
        async with httpx.AsyncClient(base_url=base, headers=headers, trust_env=False) as client:
            for _ in range(100):
                try:
                    if (await client.get("/health")).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(.05)
            else:
                raise RuntimeError("temporary relay did not start")
            uri = f"ws://127.0.0.1:{port}/ws/sensors?match_id=verification&player_id=7"
            async with connect(uri, additional_headers=headers, compression=None, max_queue=2) as ws:
                assert json.loads(await ws.recv())["type"] == "replay"
                for batch in range(32):
                    rows = []
                    for _ in range(16):
                        sequence += 1
                        rows.append({"event_id": f"verification:{sequence}", "source": "wearable",
                                     "match_id": "verification", "player_id": 7, "source_seq": sequence,
                                     "source_timestamp_ns": time.time_ns(), "sample_type": "ppg",
                                     "payload": {"device_us": sequence * 40000, "red": 90000, "ir": 120000},
                                     "clock": {"valid": True}})
                        started[sequence] = time.perf_counter()
                    response = await client.post("/api/v1/sensors/ingest", json=rows)
                    response.raise_for_status()
                    message = json.loads(await asyncio.wait_for(ws.recv(), 2))
                    assert message["type"] == "wearable_batch"
                    for item in message["data"]:
                        received.append(item["source_seq"])
                        latency.append((time.perf_counter() - started[item["source_seq"]]) * 1000)
                    if batch == 0:
                        duplicate = await client.post("/api/v1/sensors/ingest", json=rows)
                        assert duplicate.json()["duplicates"] == 16
            assert received == list(range(1, 513))
            cursor, recovered, pages = 0, [], 0
            while True:
                async with connect(uri + f"&after_seq={cursor}", additional_headers=headers) as ws:
                    page = json.loads(await asyncio.wait_for(ws.recv(), 2))["data"]
                    recovered.extend(item["source_seq"] for item in page["items"])
                    cursor = page["next_seq"]
                    pages += 1
                    if not page["has_more"]:
                        break
            assert recovered == received
            health = (await client.get("/health")).json()
        report = {"scope": "synthetic loopback, excludes Wi-Fi, TLS/WAN, physical sensors and vision",
                  "samples": len(received), "duplicates_suppressed": 16, "recovered_samples": len(recovered),
                  "replay_pages": pages, "sample_loss": 0,
                  "http_to_websocket_batch_age_ms": {"p50": round(float(np.percentile(latency, 50)), 2),
                                                     "p95": round(float(np.percentile(latency, 95)), 2),
                                                     "p99": round(float(np.percentile(latency, 99)), 2)},
                  "cache": health["cache"]}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), 3)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    asyncio.run(verify(parser.parse_args().out))
