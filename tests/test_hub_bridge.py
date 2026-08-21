"""Tests for the ESP32 hub → vision fusion bridge (no GPU, no FastAPI server)."""

from soccer_analytics.sensors.hub import relay_envelope_to_packet
from soccer_analytics.sensors.hub_bridge import sample_to_observation, snapshot_to_sample


def test_snapshot_to_sample_converts_units_and_vitals():
    state = {
        "device": {"connected": True},
        "vitals": {
            "stable_15s": {"valid": True, "bpm": 148.0, "spo2_estimate_pct": 96.5},
            "rolling": {"valid": True, "bpm": 151.0, "spo2_estimate_pct": 97.0},
        },
        "imu": {
            "valid": True,
            "accel_body_mps2": [0.0, 0.0, 9.80665],
            "gyro_body_rads": [0.0, 0.0, 3.1415926535],
            "timestamp": {"host_unix_ns": 1_700_000_000_500_000_000},
        },
        "gps": {"fix": True, "lat": 30.1, "lon": 76.2, "alt_m": 312.0},
        "server": {"host_unix_ns": 1_700_000_000_500_000_000},
    }
    s = snapshot_to_sample(state, player_id=7)
    assert s is not None
    assert s.player_id == 7
    assert s.source == "esp32-hub"
    assert abs(s.hr - 148.0) < 1e-6
    assert abs(s.spo2 - 96.5) < 1e-6
    assert abs(s.accel[2] - 1.0) < 1e-6          # 1 g
    assert abs(s.gyro[2] - 180.0) < 0.01         # rad/s → deg/s
    assert s.gps == (30.1, 76.2)
    assert abs(s.t - 1_700_000_000.5) < 1e-6

    obs = sample_to_observation(s, match_id="MATCH_001")
    assert obs["match_id"] == "MATCH_001"
    assert obs["global_player_id"] == "PLAYER_7"
    assert obs["source"] == "wearable"
    assert obs["timestamp"] == s.t
    assert obs["metrics"]["heart_rate_bpm"] == 148.0


def test_empty_offline_snapshot_is_ignored():
    assert snapshot_to_sample({"device": {"connected": False}}, player_id=7) is None


def test_invalid_vitals_are_not_forwarded_as_measurements():
    state = {
        "device": {"connected": True},
        "vitals": {
            "stable_15s": {"valid": False, "bpm": 172.0, "spo2_estimate_pct": 88.0},
            "rolling": {"valid": False, "bpm": 170.0, "spo2_estimate_pct": 89.0},
        },
        "imu": {"valid": False},
        "gps": {"fix": False},
    }
    sample = snapshot_to_sample(state, player_id=7)
    assert sample is not None
    assert sample.hr is None
    assert sample.spo2 is None


def test_raw_relay_envelope_becomes_sensor_packet():
    packet = relay_envelope_to_packet({
        "relay_seq": 42,
        "match_id": "trial-01",
        "player_id": 7,
        "device_boot_id": "boot-4",
        "source_seq": 18,
        "source_timestamp_ns": 1_700_000_000_500_000_000,
        "relay_received_ns": 1_700_000_000_510_000_000,
        "sample_type": "ppg",
        "payload": {"device_us": 900_000, "red": 105_000, "ir": 122_000},
        "clock": {"valid": True, "method": "sntp_esp_timer_anchor"},
    })
    assert packet["t"] == "ppg"
    assert packet["red"] == 105_000
    assert packet["source_unix_ns"] == 1_700_000_000_500_000_000
    assert packet["relay_seq"] == 42


if __name__ == "__main__":
    test_snapshot_to_sample_converts_units_and_vitals()
    test_empty_offline_snapshot_is_ignored()
    print("hub_bridge tests passed")
