"""Recovery, source-time and bounded-history contracts for the joined stream."""

import asyncio
import json
import time
from dataclasses import replace

import numpy as np
import pytest

from cpg44_api.hostinger_sensor_relay import MemoryRelay
from soccer_analytics.metrics import MetricsEngine
from soccer_analytics.readiness import ReadinessEngine
from soccer_analytics.sensors.fusion import FusionEngine
from soccer_analytics.sensors.hub import TelemetryEngine, relay_reader
from soccer_analytics.sensors.hub_bridge import snapshot_to_sample
from soccer_analytics.sensors.schema import SensorSample


T0_NS = time.time_ns() - 60_000_000_000


def observation(index):
    """One 0.2 s ECG block, as the vest sends them."""
    return {"source": "wearable", "match_id": "stream-test", "player_id": 7,
            "source_seq": index, "source_timestamp_ns": T0_NS + index * 200_000_000,
            "device_boot_id": "test-boot", "sample_type": "ecg",
            "payload": {"device_us": index * 200000, "fs": 250, "v": [2048] * 50, "lead_off": False},
            "clock": {"valid": True}}


def test_replay_pages_do_not_skip_a_filtered_suffix():
    relay = MemoryRelay(max_bytes=1024 * 1024)
    for index in range(300):
        relay.register(observation(index) | {"player_id": 8 if index % 7 == 0 else 7})
    cursor, seen = 0, []
    while True:
        page = relay.replay(cursor, limit=19, player_id=7)
        seen.extend(item["relay_seq"] for item in page["items"])
        cursor = page["next_seq"]
        if not page["has_more"]:
            break
    assert seen == [index + 1 for index in range(300) if index % 7]
    assert len(seen) == len(set(seen))


def test_slow_client_gets_an_explicit_recovery_marker():
    async def check():
        relay = MemoryRelay()
        queue, _ = relay.subscribe(object(), 0, None, 7)
        for index in range(20):
            envelope, _ = relay.register(observation(index))
            await relay.broadcast(envelope)
        assert queue.qsize() == 1
        assert (await queue.get())["type"] == "stream_gap"
        assert len(relay.replay()["items"]) == 20
    asyncio.run(check())


def test_relay_reader_consumes_every_replayed_ecg_block(monkeypatch):
    import soccer_analytics.sensors.hub as hub
    relay = MemoryRelay()
    for index in range(180):
        relay.register(observation(index))
    message = json.dumps({"type": "replay", "data": relay.replay()})

    class Connection:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        def __aiter__(self): return self.messages()
        async def messages(self):
            yield message
            raise asyncio.CancelledError()

    monkeypatch.setattr(hub, "connect", lambda *args, **kwargs: Connection())
    engine = TelemetryEngine()
    async def check():
        with pytest.raises(asyncio.CancelledError):
            await relay_reader(engine, "https://cpg44.nivaspms.com", "test", 7, "stream-test")
        assert len(engine.ecg.v) == 180 * 50
        assert engine.state["device"]["last_relay_seq"] == 180
    asyncio.run(check())


@pytest.mark.parametrize("age_ms,valid", [(35, True), (9500, True), (-100, False)])
def test_source_epoch_and_unclamped_age_are_preserved(age_ms, valid):
    engine = TelemetryEngine()
    epoch = 1_800_000_000_000_000_000
    source = epoch - int(age_ms * 1e6)
    packet = engine.map_packet_timestamp({"source_unix_ns": source, "relay_clock": {"valid": True}}, 100_000_000_000, epoch)
    assert packet["host_unix_ns"] == source
    assert packet["transport_delay_ms"] == age_ms
    assert packet["clock_synced"] is valid


def test_stale_ecg_is_not_a_live_heart_rate():
    """Old ECG (the vest stopped sending) must not be shown as a current heart rate."""
    engine = TelemetryEngine()
    t0 = time.perf_counter_ns() / 1e9 - 20.0
    for k in range(50):
        engine.ecg.add_block(t0 + k * 0.2, 250.0, [2048.0 + 300.0 * ((i % 125) == 0) for i in range(50)], False)
    assert engine.calculate_vitals()["reason"] == "ecg_stale"


def test_old_pulse_is_not_attached_to_a_fresh_imu_sample():
    now = time.time_ns()
    state = {"device": {"connected": True}, "server": {"host_unix_ns": now},
             "imu": {"valid": True, "accel_body_mps2": [0, 0, 9.8],
                     "timestamp": {"host_unix_ns": now, "clock_synced": True}},
             "vitals": {"stable_15s": {"valid": True, "bpm": 150, "quality": .9,
                                       "window_end_unix_ns": now - 9_000_000_000}}}
    sample = snapshot_to_sample(state, 7)
    assert sample.accel is not None
    assert sample.hr is None


def test_camera_rate_does_not_multiply_pulse_observations():
    fusion = FusionEngine()
    fusion.sync.bind(1, 7)
    for index in range(20):
        t = 100 + index / 25
        fusion.ingest_sensors([SensorSample(7, t, hr=150, clock_synced=True, hr_timestamp=100)])
        fusion.step(1 / 25, t, {1: ((2., 0.), (0., 0.))})
    assert fusion.load._acc[1].hr_count == 1
    assert fusion.load.features(1).total_distance == 1.6


def test_live_distance_matches_offline_and_retains_session_totals():
    full = MetricsEngine(25)
    bounded = MetricsEngine(25, history_limit=18)
    for frame in range(2000):
        pos = (frame * .1, 20.) if frame % 13 else (np.nan, np.nan)
        for engine in (full, bounded):
            engine.update(frame, {1: pos}, {1: 1}, None)
    assert full.speed_distance() == bounded.speed_distance()
    assert bounded.speed_distance()[1]["distance"] > 190
    assert len(bounded.positions[1]) == 18
    assert bounded.speed_distance()[1]["distance"] == pytest.approx(199.8)


def test_paired_response_cue_requires_quality_identity_and_comparable_work():
    engine = ReadinessEngine()
    sample = SensorSample(7, 0., hr=130, signal_quality=.9, clock_synced=True)
    for t in range(241):
        # Same speed band enables a within-activity pulse comparison. Falling
        # average speed alone never establishes physiological fatigue.
        speed = 2.8 if t <= 120 else 2.0
        result = engine.update(1, t, speed, replace(sample, t=t, hr=130 if t <= 120 else 147),
                               player_id=7, offset_ms=0)
    assert result["status"] == "check_in"
    assert result["hr_drift_bpm"] == 17
    assert result["work_rate_change_pct"] < -25
    invalid = engine.update(1, 242, 2., replace(sample, clock_synced=False), player_id=7, offset_ms=0)
    assert invalid["status"] == "signal_limited"
    rebound = engine.update(1, 243, 2., sample, player_id=8, offset_ms=0)
    assert rebound["status"] == "building_baseline"


def test_work_rate_drop_without_higher_pulse_is_not_a_check_in():
    engine = ReadinessEngine()
    sample = SensorSample(7, 0., hr=130, signal_quality=.9, clock_synced=True)
    for t in range(241):
        result = engine.update(1, t, 2.8 if t <= 120 else 2., sample, player_id=7, offset_ms=0)
    assert result["status"] == "monitoring"
