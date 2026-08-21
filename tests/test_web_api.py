"""Product API contract tests (no GPU, no Node)."""

import os
import sys
import asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))
os.environ.setdefault("CPG44_HUB_URL", "http://127.0.0.1:9")  # force hub miss

import httpx
from cpg44_api.main import STORE, create_app
from cpg44_api.store import ProductStore


def test_product_store_only_accepts_cpg44_relay_origin():
    store = ProductStore(hostinger_url="https://cpg44.nivaspms.com/")
    assert store.hostinger_url == "https://cpg44.nivaspms.com"

    try:
        ProductStore(hostinger_url="https://203.0.113.12")
    except ValueError as exc:
        assert "cpg44.nivaspms.com" in str(exc)
    else:
        raise AssertionError("direct-IP relay origin was accepted")


def test_health_and_wearable_join_key():
    async def check():
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            h = await client.get("/api/v1/health")
            assert h.status_code == 200
            assert h.json()["status"] == "ok"

            body = {
                "match_id": "MATCH_001",
                "global_player_id": "PLAYER_17",
                "timestamp": 1723984321.123,
                "source": "wearable",
                "metrics": {"gps_speed": 7.1, "acceleration": 1.9},
            }
            posted = await client.post("/api/v1/observations/wearable", json=body)
            assert posted.status_code == 200
            row = posted.json()
            assert row["match_id"] == "MATCH_001"
            assert row["global_player_id"] == "PLAYER_17"
            assert row["source"] == "wearable"

            listed = await client.get(
                "/api/v1/observations", params={"source": "wearable"}
            )
            assert listed.status_code == 200
            assert any(r["global_player_id"] == "PLAYER_17" for r in listed.json())

            matches = await client.get("/api/v1/matches")
            assert matches.status_code == 200
            assert matches.json()[0]["match_id"] == "live"

            live_body = {
                "match_id": "live",
                "timestamp": 1723984321.2,
                "source_kind": "recorded_file",
                "metric": True,
                "players": [{"global_player_id": "TRACK_7", "track_id": 7, "x": 41.2, "y": 22.4}],
                "ball": {"x": 52.5, "y": 34.0},
                "data_quality": {"status": "usable", "warnings": []},
            }
            assert STORE.ingest_live_frame(live_body)["ok"] is True

            jpeg = b"\xff\xd8CPG44-live-frame\xff\xd9"
            posted_frame = await client.post(
                "/api/v1/live/frame",
                content=jpeg,
                headers={"Content-Type": "image/jpeg"},
            )
            assert posted_frame.status_code == 200

            live = await client.get("/api/v1/live")
            assert live.status_code == 200
            assert live.json()["provenance"]["live"] is True
            assert live.json()["provenance"]["input_kind"] == "recorded_file"
            assert live.json()["players"][0]["track_id"] == 7

            frame = await client.get("/api/v1/live/frame")
            assert frame.status_code == 200
            assert frame.headers["content-type"] == "image/jpeg"
            assert frame.content == jpeg

    asyncio.run(check())


if __name__ == "__main__":
    test_health_and_wearable_join_key()
    print("web api tests passed")
