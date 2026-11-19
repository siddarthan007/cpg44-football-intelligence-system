from types import SimpleNamespace

import numpy as np
import pytest

from soccer_analytics.core import Detections, PLAYER, REFEREE
from soccer_analytics.filters import TrackKalman, KalmanBank
from soccer_analytics.reid import ReIDManager
from soccer_analytics.view import ViewTransformer, fit_homography, apply_homography, PitchHomographyTracker


def track(bt, x=100, team=1, jersey=None):
    return {"bt_id": bt, "center": (float(x), 100.), "team": team,
            "jersey": jersey, "color": np.array([150., 100., 160.])}


def test_reid_rejects_same_number_across_teams_and_conflicting_numbers():
    manager = ReIDManager()
    sid = manager.update(0, [track(1, jersey=7)])[1]
    assert manager.update(1, [track(2, team=2, jersey=7)])[2] != sid
    assert manager.update(2, [track(3, jersey=9)])[3] != sid


def test_reid_abstains_on_indistinguishable_teammates_regardless_of_input_order():
    for incoming in ([track(3, 104), track(4, 106)], [track(4, 106), track(3, 104)]):
        manager = ReIDManager()
        old = set(manager.update(0, [track(1, 100), track(2, 110)]).values())
        result = manager.update(1, incoming)
        assert not old.intersection(result.values())
        assert len(set(result.values())) == 2


def test_reid_pitch_gate_and_missing_appearance():
    manager = ReIDManager()
    first = track(1, jersey=7) | {"pitch": (10., 10.)}
    sid = manager.update(0, [first])[1]
    teleported = track(2, jersey=7) | {"pitch": (90., 60.)}
    assert manager.update(1, [teleported])[2] != sid
    manager = ReIDManager()
    sid = manager.update(0, [track(1) | {"color": None}])[1]
    assert manager.update(1, [track(2) | {"color": None}])[2] != sid


def test_calibration_rejects_collinear_nonfinite_and_projective_horizon():
    with pytest.raises(ValueError):
        fit_homography([[0, 0], [1, 0], [2, 0], [3, 0]], [[0, 0], [1, 0], [1, 1], [0, 1]])
    with pytest.raises(ValueError):
        fit_homography([[0, 0], [1, 0], [1, 1], [0, np.inf]], [[0, 0], [1, 0], [1, 1], [0, 1]])
    assert np.isnan(apply_homography(np.array([[1, 0, 0], [0, 1, 0], [1, 0, -1.]]), [[1, 1]])).all()
    assert not ViewTransformer.from_field_span().is_metric


def test_kalman_rejects_teleport_and_nonfinite_without_corrupting_covariance():
    kalman = TrackKalman()
    for _ in range(30):
        kalman.step((10., 20.), 0.04)
    kalman.step((90., 60.), 0.04)
    assert not kalman.measurement_accepted
    assert np.linalg.norm(np.asarray(kalman.position) - (10., 20.)) < 1
    kalman.step((10., np.nan), 0.04)
    assert np.isfinite(kalman.position).all()
    assert np.linalg.eigvalsh(kalman.P).min() >= -1e-10


def test_kalman_bank_accounts_for_reentry_time_and_prunes_absent_tracks():
    bank = KalmanBank(max_missing=2)
    bank.update(1, (1., 1.), .04, time_s=0.)
    for _ in range(3):
        bank.prune([])
    assert 1 not in bank._filters and 1 not in bank._last_time


def test_heatmap_seconds_are_fps_independent_and_do_not_clamp_invalid_pins():
    from soccer_analytics.heatmap import LiveHeatmap
    from soccer_analytics.config import PitchConfig
    grids = []
    for fps in (10, 25):
        heat = LiveHeatmap(PitchConfig())
        for _ in range(fps):
            heat.add([(10., 20.), (-2., 30.), (50., np.nan), (1000., 0.)], dt=1 / fps)
        assert heat.grid.sum() == pytest.approx(1., abs=1e-5)
        snapshot = heat.grid_snapshot()
        assert sum(snapshot["seconds"]) == pytest.approx(1., abs=.002)
        grids.append(heat.grid)
    np.testing.assert_allclose(*grids, atol=1e-5)


@pytest.mark.parametrize("backend", ["bytetrack", "botsort"])
def test_tracker_recovers_weak_boxes_and_keeps_roles_separate(backend):
    from soccer_analytics.tracker import SoccerTracker
    tracker = SoccerTracker(backend=backend, minimum_consecutive_frames=1)
    image = np.zeros((100, 100, 3), np.uint8)
    boxes = np.array([[10., 10., 30., 70.], [60., 10., 80., 70.]])
    strong = Detections(boxes, np.array([PLAYER, REFEREE]), np.array([.9, .9]))
    for _ in range(3):
        before = tracker.update(strong, image)
    weak = Detections(boxes, strong.class_id, np.array([.15, .15]))
    after = tracker.update(weak, image)
    assert len(after) == 2
    np.testing.assert_array_equal(before.tracker_id, after.tracker_id)
    assert len(set(after.tracker_id)) == 2


def test_detector_preserves_low_scores_for_tracker():
    import torch
    from soccer_analytics.tracker import Detector
    detector = Detector.__new__(Detector)
    detector.imgsz, detector.device, detector.half = 640, "cpu", False
    detector.person_conf, detector.ball_conf, detector.track_low_conf = .3, .1, .1
    detector.remap_goalkeeper = True
    class FakeBoxes(SimpleNamespace):
        def __len__(self):
            return len(self.conf)
    boxes = FakeBoxes(xyxy=torch.tensor([[1., 1., 8., 20.]]),
                            cls=torch.tensor([float(PLAYER)]), conf=torch.tensor([.15]))
    detector.class_map = {PLAYER: PLAYER}
    detector.model = SimpleNamespace(predict=lambda *a, **k: [SimpleNamespace(boxes=boxes)])
    assert len(detector.detect(np.zeros((30, 30, 3), np.uint8))) == 1


def test_checkpoint_class_order_is_resolved_from_names():
    from soccer_analytics.tracker import canonical_class_map
    from soccer_analytics.core import BALL, GOALKEEPER
    assert canonical_class_map({0: "ball", 1: "player", 2: "goalkeeper", 3: "referee"}) == {
        0: BALL, 1: PLAYER, 2: GOALKEEPER, 3: REFEREE}
    assert canonical_class_map({0: "person", 32: "sports ball", 1: "bicycle"}) == {
        0: PLAYER, 32: BALL, 1: -1}


def test_uncalibrated_gps_does_not_invent_a_centre_pin():
    from soccer_analytics.sensors.gps_pitch import GPSPitchTransformer
    transformer = GPSPitchTransformer()
    assert np.isnan(transformer.gps_to_pitch(12.9, 77.5)).all()
    assert transformer.generate_density_grid([(12.9, 77.5)]).sum() == 0


def test_training_split_keeps_complete_sequences_together_and_is_deterministic():
    from soccer_analytics.data_split import split_sequences
    files = [f"SNMOT-{seq:03d}_{frame:06d}.jpg" for seq in range(4) for frame in range(10)]
    train, val, manifest = split_sequences(files)
    train2, val2, _ = split_sequences(reversed(files))
    assert train == train2 and val == val2
    assert not set(manifest["train_sequences"]).intersection(manifest["validation_sequences"])
    assert len(train) + len(val) == len(files)


def test_training_refuses_official_test_as_validation(tmp_path):
    from soccer_analytics.data_split import validate_training_data
    dataset = tmp_path / "data.yaml"
    dataset.write_text("train: images/train\nval: images/test\nnames: [ball, player]\n")
    with pytest.raises(ValueError, match="test data"):
        validate_training_data(dataset)


def test_team_qualified_jersey_bindings_keep_opponents_separate():
    from soccer_analytics.jersey_ocr import AutoBinder
    from soccer_analytics.sensors.sync import SensorVideoSync
    reader = SimpleNamespace(available=True, read=lambda *_: 7)
    binder = AutoBinder(reader, {(1, 7): 101, (2, 7): 202}, every=1)
    sync = SensorVideoSync()
    for frame in range(5):
        binder.step(frame, None, [(1, (0, 0, 10, 20), 1), (2, (20, 0, 30, 20), 2)], sync)
    assert sync.player_of(1) == 101 and sync.player_of(2) == 202
    sync.bind(9, 101)
    assert sync.player_of(1) is None and sync.player_of(9) == 101


def test_automatic_jersey_binding_cannot_steal_a_manual_binding():
    from soccer_analytics.jersey_ocr import AutoBinder
    from soccer_analytics.sensors.sync import SensorVideoSync
    reader = SimpleNamespace(available=True, read=lambda *_: 7)
    binder = AutoBinder(reader, {7: 101}, every=1)
    sync = SensorVideoSync()
    sync.bind(9, 101)
    for frame in range(6):
        binder.step(frame, None, [(1, (0, 0, 10, 20))], sync)
    assert sync.player_of(9) == 101 and sync.player_of(1) is None


def test_raw_invalid_optical_values_and_delayed_stale_imu_stay_unavailable():
    from soccer_analytics.sensors.hub_bridge import snapshot_to_sample
    state = {"device": {"connected": True},
             "vitals": {"rolling": {"valid": False, "bpm_raw": 155., "spo2_raw": 97.}},
             "imu": {"valid": True, "accel_body_mps2": [0, 0, 9.8],
                     "timestamp": {"host_unix_ns": 1_000_000_000, "transport_delay_ms": 6000.}},
             "server": {"host_unix_ns": 9_000_000_000}}
    sample = snapshot_to_sample(state, 7)
    assert sample.hr is None and sample.spo2 is None and sample.accel is None


def test_sensor_alignment_handles_out_of_order_delivery():
    from soccer_analytics.sensors.sync import SensorVideoSync
    from soccer_analytics.sensors.schema import SensorSample
    sync = SensorVideoSync(tolerance_s=.1)
    sync.ingest([SensorSample(player_id=7, t=10.), SensorSample(player_id=7, t=5.)])
    assert sync.sample_at(7, 10.).t == 10.


def test_recorded_video_without_time_anchor_cannot_inherit_live_wearables():
    from soccer_analytics.sensors.sync import SensorVideoSync
    from soccer_analytics.sensors.schema import SensorSample
    from soccer_analytics.sensors.fusion import FusionEngine
    sync = SensorVideoSync()
    sync.bind(1, 7)
    sync.ingest([SensorSample(player_id=7, t=10., hr=150., clock_synced=True)])
    fusion = FusionEngine(sync)
    fusion.step(.04, None, {1: ((2., 0.), (0., 0.))})
    assert not fusion.aligned
    assert fusion.alignment[1]["reason"] == "video_source_time_unavailable"
    fusion.step(.04, 10., {1: ((2., 0.), (0., 0.))})
    assert fusion.aligned[1].hr == 150.
    fusion.step(.04, 10.04, {})
    assert not fusion.aligned and not fusion.alignment


def test_touchline_jitter_does_not_create_repeated_restart_events():
    from soccer_analytics.metrics import MetricsEngine
    engine = MetricsEngine(fps=25.)
    for frame in range(100):
        engine.update(frame, {}, {}, (50., 70. if frame % 2 else 67.))
    assert not engine.throw_ins
    engine = MetricsEngine(fps=25.)
    for frame in range(5):
        engine.update(frame, {}, {}, (50., 65.))
    for frame in range(5, 35):
        engine.update(frame, {}, {}, (50., 70.))
    for frame in range(35, 40):
        engine.update(frame, {}, {}, (50., 67.))
    assert len(engine.throw_ins) == 1
    assert engine.throw_ins[0]["side"] == "near"


def test_rgb_calibration_and_torso_colours_assign_the_same_team(monkeypatch):
    import json
    from pathlib import Path
    from soccer_analytics.realtime import RealtimePipeline
    from soccer_analytics.team_assign import TeamAssigner
    data = {"calibrated": True, "team_1": {"primary_color_rgb": [255, 0, 0]},
            "team_2": {"primary_color_rgb": [0, 0, 255]}}
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: json.dumps(data))
    pipeline = RealtimePipeline.__new__(RealtimePipeline)
    pipeline.team = TeamAssigner()
    pipeline._load_team_seeds()
    rng = np.random.default_rng(44)
    for tid, bgr, expected in ((1, [0, 0, 250], 1), (2, [250, 0, 0], 2)):
        frame = (np.full((60, 30, 3), bgr) + rng.integers(0, 5, (60, 30, 3))).astype(np.uint8)
        for _ in range(3):
            team = pipeline.team.assign(tid, frame, [0, 0, 30, 60])
        assert team == expected


def test_saved_pixel_results_are_not_exposed_as_physical_metrics(tmp_path):
    import json
    from cpg44_api.store import ProductStore
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"metric": False, "players": {"1": {
        "team": 1, "distance_m": 500., "top_speed_ms": 50.,
        "metabolic_power_avg_wkg": 90., "player_load": 70.,
    }}}))
    snapshot = ProductStore(stats_path=stats, hub_url="http://127.0.0.1:9").live_frame_snapshot()
    player = snapshot["players"][0]
    assert snapshot["metric"] is False
    assert all(player[key] is None for key in ("distance_m", "top_speed_mps", "metabolic_power", "player_load"))


def test_camera_mapping_fails_closed_when_visual_support_disappears():
    tracker = PitchHomographyTracker(np.eye(3), (100, 100))
    blank = np.zeros((100, 100, 3), np.uint8)
    tracker.update(blank)
    tracker.update(blank)
    assert not tracker.quality()["valid"]
    assert np.isnan(tracker.transform([[10., 20.]])).all()


def test_camera_translation_preserves_a_known_pitch_location():
    import cv2
    rng = np.random.default_rng(44)
    image = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
    translation = np.array([[1., 0., 4.], [0., 1., 2.], [0., 0., 1.]])
    moved = cv2.warpPerspective(image, translation, (400, 300))
    tracker = PitchHomographyTracker(np.diag([.1, .1, 1.]), image.shape, scale=1.)
    tracker.update(image)
    tracker.update(moved)
    assert tracker.valid
    np.testing.assert_allclose(tracker.transform([[104., 102.]])[0], [10., 10.], atol=.1)


def test_pitch_evaluation_penalizes_misses_id_switches_and_reports_coverage():
    from soccer_analytics.evaluation import evaluate_frames
    truth = {i: {"players": [{"track_id": 7, "x": 10., "y": 20., "team": 1}]} for i in range(3)}
    predictions = {
        0: {"players": [{"track_id": 1, "x": 10.5, "y": 20., "team": 1}]},
        1: {"players": [{"track_id": 2, "x": 10.5, "y": 20., "team": 0}]},
    }
    score = evaluate_frames(truth, predictions)
    assert score["position_recall"] == pytest.approx(2 / 3, abs=1e-6)
    assert score["identity_switches_between_matches"] == 1
    assert score["team_coverage"] == pytest.approx(1 / 3, abs=1e-6)
    assert score["matched_position_mae_m"] == .5
