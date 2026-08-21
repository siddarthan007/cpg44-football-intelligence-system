import asyncio
from pathlib import Path

import httpx

from cpg44_api.flasher import ESP32Flasher
from cpg44_api.tagger import MatchTagger
from cpg44_api.main import create_app


def test_flasher_port_listing():
    flasher = ESP32Flasher()
    ports = flasher.list_ports()
    assert isinstance(ports, list)
    # WSL correctly reports an empty list until usbipd attaches the ESP32.
    assert all("device" in port for port in ports)


def test_flasher_config_generation(tmp_path):
    ca_pem = "-----BEGIN CERTIFICATE-----\nVEVTVA==\n-----END CERTIFICATE-----\n"
    flasher = ESP32Flasher(
        firmware_dir=tmp_path,
        relay_token="r" * 32,
        relay_ca_pem=ca_pem,
    )
    header = flasher.generate_config_header(
        player_id=10, wifi_ssid="TestNet", wifi_pass="Secret123", match_id="trial-01"
    )
    assert Path(header).is_file()
    content = Path(header).read_text()
    assert "PLAYER_ID = 10;" in content
    assert 'MATCH_ID[] = "trial-01";' in content
    assert 'WIFI_SSID[] = "TestNet";' in content
    assert 'RELAY_ENDPOINT[] = "https://cpg44.nivaspms.com/api/v1/sensors/ingest";' in content
    assert 'RELAY_TOKEN[] = "rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr";' in content


def test_tagger_color_classification(tmp_path):
    tagger = MatchTagger(configs_dir=tmp_path)
    tagger.save_team_profiles({
        "team_1": {"name": "Home", "primary_color_rgb": [30, 90, 230]},
        "team_2": {"name": "Away", "primary_color_rgb": [130, 25, 25]},
    })
    res_blue = tagger.classify_crop_team([30, 90, 230])
    assert res_blue["team"] == 1

    res_red = tagger.classify_crop_team([130, 25, 25])
    assert res_red["team"] == 2


def test_tagger_event_logging(tmp_path):
    tagger = MatchTagger(configs_dir=tmp_path)
    ev = tagger.log_event({
        "timestamp_s": 72.5,
        "type": "shot",
        "team": 1,
        "player_jersey": 17,
        "description": "Test Shot",
    })
    assert ev["type"] == "shot"
    assert ev["player_jersey"] == 17
    assert ev in tagger.events


def test_flasher_and_tagger_api_endpoints():
    async def check():
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/hardware/ports")
            assert r.status_code == 200
            assert isinstance(r.json(), list)

            r = await client.get("/api/v1/tagging/teams")
            assert r.status_code == 200
            assert "team_1" in r.json()

            # A manual observation without a real match timestamp is rejected.
            r = await client.post("/api/v1/tagging/events", json={
                "type": "goal",
                "team": 1,
                "player_jersey": 27,
            })
            assert r.status_code == 422

    asyncio.run(check())
