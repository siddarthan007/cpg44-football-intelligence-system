"""Product API contract tests (no GPU, no Node)."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))
os.environ.setdefault("CPG44_HUB_URL", "http://127.0.0.1:9")  # force hub miss

from fastapi.testclient import TestClient
from cpg44_api.main import create_app


def test_health_and_wearable_join_key():
    client = TestClient(create_app())
    h = client.get("/api/v1/health")
    assert h.status_code == 200
    assert h.json()["status"] == "ok"

    body = {
        "match_id": "MATCH_001",
        "global_player_id": "PLAYER_17",
        "timestamp": 1723984321.123,
        "source": "wearable",
        "metrics": {"gps_speed": 7.1, "acceleration": 1.9},
    }
    posted = client.post("/api/v1/observations/wearable", json=body)
    assert posted.status_code == 200
    row = posted.json()
    assert row["match_id"] == "MATCH_001"
    assert row["global_player_id"] == "PLAYER_17"
    assert row["source"] == "wearable"

    listed = client.get("/api/v1/observations", params={"source": "wearable"})
    assert listed.status_code == 200
    assert any(r["global_player_id"] == "PLAYER_17" for r in listed.json())

    matches = client.get("/api/v1/matches")
    assert matches.status_code == 200
    assert matches.json()[0]["match_id"] == "live"


if __name__ == "__main__":
    test_health_and_wearable_join_key()
    print("web api tests passed")
