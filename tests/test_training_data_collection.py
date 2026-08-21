import sqlite3

from cpg44_api.training_manager import TrainingManager


def measured_snapshot(timestamp: float = 1_800_000_000.0) -> dict:
    return {
        "match_id": "campus-session-01",
        "timestamp": timestamp,
        "frame_index": 250,
        "metric": True,
        "players": [
            {
                "player_id": 7,
                "speed_mps": 6.4,
                "distance_m": 1540.2,
                "wearable_metrics": {
                    "hr": 166.0,
                    "spo2": 96.8,
                    "signal_quality": 0.91,
                },
                "load": {
                    "hsr_m": 212.4,
                    "sprints": 4,
                    "accel_efforts": 19,
                    "decel_efforts": 17,
                    "player_load": 33.1,
                    "player_load_imu": 31.8,
                    "metabolic_power_wkg": 12.2,
                    "hr_drift": 6.4,
                    "acwr": 1.12,
                },
            }
        ],
    }


def test_live_collection_is_calibrated_throttled_and_session_labelled(tmp_path):
    manager = TrainingManager(tmp_path)
    snapshot = measured_snapshot()

    assert manager.record_live_snapshot(snapshot) == 1
    assert manager.record_live_snapshot(snapshot | {"timestamp": snapshot["timestamp"] + 2}) == 0
    assert manager.record_live_snapshot(snapshot | {"timestamp": snapshot["timestamp"] + 6}) == 1
    assert manager.record_live_snapshot(snapshot | {"metric": False, "timestamp": snapshot["timestamp"] + 12}) == 0

    db_path = tmp_path / "data" / "multimodal_dataset.db"
    with sqlite3.connect(db_path) as connection:
        count, source_time, player_load = connection.execute(
            "SELECT COUNT(*), MIN(timestamp), MAX(player_load) FROM multimodal_samples"
        ).fetchone()
    assert count == 2
    assert source_time == snapshot["timestamp"]
    assert player_load == 31.8

    labelled = manager.label_player_session(
        "campus-session-01", 7, 0, "physio register 2026-08-20", 7
    )
    assert labelled["ok"] is True
    status = manager.get_status()["outcome_labels"]
    assert status["collected_samples"] == 2
    assert status["collected_player_sessions"] == 1
    assert status["labelled_samples"] == 1
    assert status["positive_outcomes"] == 0


def test_outcome_label_rejects_unmeasured_session(tmp_path):
    manager = TrainingManager(tmp_path)
    result = manager.label_player_session(
        "missing-session", 7, 1, "physio register 2026-08-20"
    )
    assert result["ok"] is False
    assert "no measured player-session" in result["error"]
