import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from cpg44_api.flasher import ESP32Flasher
from cpg44_api.tagger import MatchTagger
from cpg44_api.main import create_app


def test_flasher_port_listing():
    flasher = ESP32Flasher()
    ports = flasher.list_ports()
    assert isinstance(ports, list)
    assert len(ports) > 0
    assert "device" in ports[0]


def test_flasher_config_generation(tmp_path):
    flasher = ESP32Flasher(firmware_dir=tmp_path)
    header = flasher.generate_config_header(player_id=10, wifi_ssid="TestNet", wifi_pass="Secret")
    assert Path(header).is_file()
    content = Path(header).read_text()
    assert "PLAYER_ID   = 10;" in content
    assert 'WIFI_SSID   = "TestNet";' in content


def test_tagger_color_classification():
    tagger = MatchTagger()
    # Team 1 is Royal Blue
    res_blue = tagger.classify_crop_team([30, 90, 230])
    assert res_blue["team"] == 1

    # Team 2 is Crimson Red
    res_red = tagger.classify_crop_team([130, 25, 25])
    assert res_red["team"] == 2


def test_tagger_event_logging():
    tagger = MatchTagger()
    ev = tagger.log_event({
        "type": "shot",
        "team": 1,
        "player_jersey": 17,
        "description": "Test Shot",
    })
    assert ev["type"] == "shot"
    assert ev["player_jersey"] == 17
    assert ev in tagger.events


def test_flasher_and_tagger_api_endpoints():
    app = create_app()
    client = TestClient(app)

    # Ports endpoint
    r = client.get("/api/v1/hardware/ports")
    assert r.status_code == 200
    assert len(r.json()) > 0

    # Teams endpoint
    r = client.get("/api/v1/tagging/teams")
    assert r.status_code == 200
    assert "team_1" in r.json()

    # Log event endpoint
    r = client.post("/api/v1/tagging/events", json={
        "type": "goal",
        "team": 1,
        "player_jersey": 27,
        "description": "Goal Tag API Test",
    })
    assert r.status_code == 200
    assert r.json()["type"] == "goal"
