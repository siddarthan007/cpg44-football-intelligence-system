"""Stateless relay contract tests without opening network sockets."""

import asyncio
import time

import httpx

import cpg44_api.hostinger_sensor_relay as relay_module
from cpg44_api.hostinger_sensor_relay import MemoryRelay


def valid_payload(sequence: int = 1):
    timestamp_ns = time.time_ns()
    return {
        "event_id": f"training-01:7:boot-4:{sequence}:{timestamp_ns}",
        "source_seq": sequence,
        "source_timestamp_ns": timestamp_ns,
        "player_id": 7,
        "device_boot_id": "boot-4",
        "match_id": "training-01",
        "source": "synchronized_local_hub",
        "hr": 154.0,
        "spo2": 97.1,
        "accel": [0.1, 0.2, 1.02],
        "accel_unit": "g",
        "gyro": [12.0, -2.0, 4.0],
        "gps": [30.12, 76.32, 280.0],
        "signal_quality": 0.86,
        "clock": {"valid": True, "drift_ppm": 3.2, "best_rtt_ms": 4.1, "samples": 12},
        "tags": {"session_id": "training-01", "jersey": 7},
    }


def test_relay_preserves_timestamps_tags_and_deduplicates():
    relay = MemoryRelay(max_bytes=4096)
    payload = valid_payload()
    envelope, duplicate = relay.register(payload)
    assert envelope is not None and duplicate is False
    assert envelope.source_timestamp_ns == payload["source_timestamp_ns"]
    assert envelope.relay_received_ns >= envelope.source_timestamp_ns
    assert envelope.tags["match_id"] == "training-01"
    assert envelope.tags["jersey"] == 7
    assert envelope.clock["valid"] is True

    same, duplicate = relay.register(payload)
    assert duplicate is True
    assert same.relay_seq == envelope.relay_seq
    assert relay.cache_status()["items"] == 1

    replay = relay.replay(after_seq=0)
    assert replay["items"][0]["event_id"] == payload["event_id"]
    assert replay["cache_gap"] is False

    missing_timestamp = valid_payload() | {"source_timestamp_ns": None, "timestamp": None}
    assert relay.register(missing_timestamp)[0] is None
    wrong_source = valid_payload() | {"source": "browser_claim"}
    assert relay.register(wrong_source)[0] is None


def test_memory_cache_is_byte_bounded():
    relay = MemoryRelay(max_bytes=1800)
    for sequence in range(1, 20):
        relay.register(valid_payload(sequence))
    status = relay.cache_status()
    assert status["bytes"] <= status["max_bytes"]
    assert status["items"] < 19
    assert relay.replay(after_seq=1)["cache_gap"] is True


def test_relay_preserves_validated_raw_samples():
    relay = MemoryRelay(max_bytes=4096)
    payload = valid_payload() | {
        "source": "wearable",
        "sample_type": "imu",
        "payload": {
            "device_us": 8_500_000,
            "a": [0.1, -0.2, 9.81],
            "g": [0.01, -0.02, 0.03],
            "temp_c": 31.2,
        },
    }
    envelope, duplicate = relay.register(payload)
    assert envelope is not None and duplicate is False
    assert envelope.sample_type == "imu"
    assert envelope.payload["device_us"] == 8_500_000
    assert envelope.payload["a"] == [0.1, -0.2, 9.81]

    bad = payload | {"event_id": "bad-raw", "payload": {"device_us": 4, "a": [1, 2, 3]}}
    assert relay.register(bad)[0] is None


def test_relay_http_ingest_requires_shared_token(monkeypatch):
    relay = MemoryRelay(max_bytes=4096)
    token = "a" * 32
    monkeypatch.setattr(relay_module, "RELAY", relay)
    monkeypatch.setattr(relay_module, "RELAY_TOKEN", token)

    async def check():
        transport = httpx.ASGITransport(app=relay_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            denied = await client.post("/api/v1/sensors/ingest", json=valid_payload())
            assert denied.status_code == 401

            accepted = await client.post(
                "/api/v1/sensors/ingest",
                json=valid_payload(),
                headers={"X-Auth": token},
            )
            assert accepted.status_code == 200
            assert accepted.json()["accepted"] == 1

            latest = await client.get(
                "/api/v1/sensors/latest", headers={"X-Auth": token}
            )
            assert latest.status_code == 200
            assert latest.json()["players"]["7"]["hr"] == 154.0

            history = await client.get(
                "/api/v1/sensors/history?after_seq=0", headers={"X-Auth": token}
            )
            assert history.status_code == 200
            assert len(history.json()["items"]) == 1

    asyncio.run(check())
