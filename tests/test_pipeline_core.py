"""Pure-NumPy unit tests for the geometry/tactics/metrics core.

These exercise everything that does NOT need the heavy CV stack
(cv2/ultralytics/sklearn/matplotlib), so they run with just numpy installed:

    python -m pytest tests/test_pipeline_core.py       # or: python tests/test_pipeline_core.py
"""

import numpy as np

from soccer_analytics.view import fit_homography, apply_homography, ViewTransformer
from soccer_analytics.config import PitchConfig
from soccer_analytics import tactics as T
from soccer_analytics.core import Detections, foot_points, BALL, PLAYER, REFEREE
from soccer_analytics.metrics import MetricsEngine
from soccer_analytics.tracker import interpolate_ball


def test_homography_round_trip():
    pc = PitchConfig()
    npnts = pc.named_points()
    names = ["tl_corner", "tr_corner", "br_corner", "bl_corner"]
    pitch = np.array([npnts[n] for n in names])
    Hcam = np.array([[8.0, 2.0, 300.0], [0.5, 10.0, 150.0], [0.0005, 0.001, 1.0]])
    img = apply_homography(Hcam, pitch)
    vt = ViewTransformer.from_points(img, pitch, pitch_length=pc.length, pitch_width=pc.width)
    assert np.abs(vt.transform(img) - pitch).max() < 0.01
    mid = vt.transform(apply_homography(Hcam, np.array([[52.5, 34.0]])))[0]
    assert abs(mid[0] - 52.5) < 0.05 and abs(mid[1] - 34) < 0.05


def test_out_of_pitch_is_nan():
    pc = PitchConfig()
    names = ["tl_corner", "tr_corner", "br_corner", "bl_corner"]
    pitch = np.array([pc.named_points()[n] for n in names])
    Hcam = np.array([[8.0, 2.0, 300.0], [0.5, 10.0, 150.0], [0.0005, 0.001, 1.0]])
    vt = ViewTransformer.from_points(apply_homography(Hcam, pitch), pitch)
    far = vt.transform(apply_homography(Hcam, np.array([[500.0, 500.0]])))[0]
    assert np.isnan(far[0])


def test_identity_view_passthrough():
    v = ViewTransformer()
    assert not v.is_metric
    assert np.allclose(v.transform([[3, 4]])[0], [3, 4])


def test_convex_hull_area():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    assert abs(T.convex_hull_area(sq) - 100) < 1e-6


def test_formation_433():
    xs = [2, 20, 20, 20, 20, 50, 50, 50, 80, 80, 80]
    ys = [34, 10, 26, 42, 58, 20, 34, 48, 16, 34, 52]
    assert T.estimate_formation(xs, ys, attack_dir=1) == "4-3-3"


def test_tactical_report():
    xs = [2, 20, 20, 20, 20, 50, 50, 50, 80, 80, 80]
    ys = [34, 10, 26, 42, 58, 20, 34, 48, 16, 34, 52]
    ta = T.TacticalAnalyzer()
    for _ in range(5):
        ta.update({1: {i: (xs[i], ys[i]) for i in range(11)}, 2: {}})
    rep = ta.report(1, 55.0)
    assert rep.formation == "4-3-3"
    assert rep.recommendations


def test_possession_and_speed():
    eng = MetricsEngine(fps=25.0, metric=True, possession_dist=2.0)
    for f in range(40):
        p1 = (10 + 0.2 * f, 34.0)
        eng.update(f, {1: p1, 2: (60.0, 34.0)}, {1: 1, 2: 2},
                   (10 + 0.2 * f + 0.5, 34.0))
    assert eng.possession_pct()[1] > 90
    sd = eng.speed_distance(window=5)
    assert abs(sd[1]["avg_speed"] - 5.0) < 0.6      # 0.2 m/frame * 25 fps
    assert sd[1]["distance"] > 5


def test_ball_interpolation():
    boxes = [np.array([10, 10, 20, 20.0]), None, None, np.array([40, 40, 50, 50.0]), None]
    fill = interpolate_ball(boxes)
    assert all(b is not None for b in fill)
    assert abs(fill[1][0] - 20) < 1e-6 and abs(fill[2][0] - 30) < 1e-6


def test_detections_filtering():
    d = Detections(np.array([[0, 0, 10, 20], [0, 0, 5, 5], [1, 1, 2, 2.0]]),
                   np.array([PLAYER, BALL, REFEREE]), np.array([.9, .5, .8]))
    assert len(d.of_class(BALL)) == 1
    assert np.allclose(foot_points(d.xyxy)[0], [5, 20])


def test_kalman_recovers_constant_velocity():
    from soccer_analytics.filters import TrackKalman
    kf = TrackKalman()
    # move along x at 5 m/s, dt=0.04, with small measurement noise
    rng = np.random.default_rng(0)
    x = 0.0
    for i in range(60):
        x += 5.0 * 0.04
        z = (x + rng.normal(0, 0.1), 34.0 + rng.normal(0, 0.1))
        kf.step(z, 0.04)
    assert abs(kf.speed - 5.0) < 0.6           # recovers speed
    assert abs(kf.acceleration[0]) < 1.5        # ~zero acceleration
    # occlusion: predict-only keeps moving forward, no crash
    p0 = kf.position[0]
    kf.step(None, 0.04)
    assert kf.position[0] > p0


def test_metabolic_energy_cost_flat():
    from soccer_analytics.catapult import energy_cost
    assert abs(energy_cost(0.0) - 3.6) < 1e-6   # flat, constant speed = 3.6 J/kg/m
    assert energy_cost(3.0) > energy_cost(0.0)   # accelerating costs more
    assert energy_cost(-3.0) < energy_cost(0.0)  # decelerating costs less


def test_loadengine_metabolic_power():
    from soccer_analytics.catapult import LoadEngine
    le = LoadEngine(mass_kg=75)
    # constant 6 m/s (HSR) for 5 s, no acceleration
    for i in range(125):
        le.update(1, 0.04, vel=(6.0, 0.0), acc=(0.0, 0.0))
    f = le.features(1)
    assert abs(f.total_distance - 30.0) < 1.0            # 6*5 = 30 m
    assert f.hsr_distance > 25                            # 6 > 5.5 m/s
    assert abs(f.metabolic_power_avg - 21.6) < 1.0        # 3.6 J/kg/m * 6 m/s
    assert f.energy_kcal > 0
    assert f.distance_by_zone.get("sprint", 0) == 0       # 6 < 7 m/s not sprint


def test_pitch_control_voronoi():
    from soccer_analytics.tactical_engine import pitch_control
    pc = PitchConfig()
    t1 = [(20, y) for y in range(10, 60, 10)]
    t2 = [(85, y) for y in range(10, 60, 10)]
    ctrl, grid = pitch_control({1: t1, 2: t2}, pc)
    assert 35 < ctrl[1] < 65 and 35 < ctrl[2] < 65      # split halves ~ balanced
    assert abs(ctrl[1] + ctrl[2] - 100) < 1e-6


def test_pressing_and_thirds():
    from soccer_analytics.tactical_engine import pressing_intensity, thirds_count
    assert pressing_intensity([(50, 34), (51, 35), (90, 10)], (50, 34), radius=8) == 2
    assert pressing_intensity([(50, 34)], None) == 0
    pc = PitchConfig()
    th = thirds_count([(10, 34), (52, 34), (95, 34)], pc, attack_dir=1)
    assert th == {"def": 1, "mid": 1, "att": 1}


def test_line_height_both_directions():
    # regression for the defensive-line-height sign bug: a deep line must read as
    # a small metres-from-own-goal for BOTH attack directions.
    from soccer_analytics.tactical_engine import TacticalEngine
    pc = PitchConfig()  # 105 x 68
    # team attacking +x (own goal x=0), deep back line near x=20 → line height ~20
    eng1 = TacticalEngine(pc)
    for _ in range(10):
        eng1.update({1: {i: (18 + i, 34) for i in range(6)}, 2: {}}, (50, 34), 1)
    r1 = eng1.report(1, 50.0)
    assert r1.avg_line_height < 40, r1.avg_line_height
    # team attacking -x (own goal x=105), deep line near x=85 → height = 105-85 ≈ 20
    eng2 = TacticalEngine(pc)
    for _ in range(10):
        eng2.update({1: {}, 2: {i: (82 + i, 34) for i in range(6)}}, (50, 34), 2)
    r2 = eng2.report(2, 50.0)
    assert r2.avg_line_height < 40, r2.avg_line_height   # was ~190 before the fix
    assert r2.avg_line_height <= pc.length               # physically possible


def test_tactical_engine_report():
    from soccer_analytics.tactical_engine import TacticalEngine
    pc = PitchConfig()
    eng = TacticalEngine(pc)
    xs = [2, 20, 20, 20, 20, 50, 50, 50, 80, 80, 80]
    ys = [34, 10, 26, 42, 58, 20, 34, 48, 16, 34, 52]
    for _ in range(20):
        eng.update({1: {i: (xs[i], ys[i]) for i in range(11)}, 2: {}}, (50, 34), 1)
    rep = eng.report(1, 55.0)
    assert rep.formation == "4-3-3"
    assert 0 <= rep.avg_control <= 100
    assert rep.recommendations


def test_xg_geometry_and_model():
    from soccer_analytics.shots import shot_geometry, XGModel
    # penalty spot: 11 m central to goal at x=0
    dist, ang = shot_geometry(11.0, 34.0, 0.0, 68.0)
    assert abs(dist - 11.0) < 0.1 and ang > 0
    m = XGModel()
    close = m.xg(6.0, 34.0, 0.0)      # 6 m central
    far = m.xg(25.0, 34.0, 0.0)       # 25 m central
    wide = m.xg(11.0, 55.0, 0.0)      # 11 m but wide angle
    central = m.xg(11.0, 34.0, 0.0)   # 11 m central
    assert close > far                # closer → higher xG
    assert central > wide             # tighter angle when wide → lower xG
    assert 0 < far < close < 1


def test_shot_detection():
    from soccer_analytics.shots import ShotDetector
    sd = ShotDetector()
    goals = {1: 105.0, 2: 0.0}
    # team 1 fires from (80,34) toward x=105 at 12 m/s → one shot
    sd.update(0, (80.0, 34.0), (12.0, 0.0), 1, goals, nearest_player=9)
    # ball still fast next frame — no double count (debounced by _in_shot)
    sd.update(1, (82.0, 34.0), (12.0, 0.0), 1, goals, nearest_player=9)
    # slow ball → nothing
    sd.update(2, (50.0, 34.0), (1.0, 0.0), 1, goals)
    s = sd.summary()
    assert s[1]["shots"] == 1 and s[1]["xg"] > 0
    assert s[2]["shots"] == 0


def test_jersey_vote_tracker():
    from soccer_analytics.jersey_ocr import JerseyVoteTracker
    v = JerseyVoteTracker(min_votes=4, min_fraction=0.5)
    for n in [7, 7, 3, 7, 7]:
        v.add(1, n)
    assert v.confident_number(1) == 7      # 4/5 majority
    v.add(2, 9); v.add(2, 4)
    assert v.confident_number(2) is None   # too few votes
    assert v.resolve() == {1: 7}


def test_jersey_tie_not_confident():
    from soccer_analytics.jersey_ocr import JerseyVoteTracker
    v = JerseyVoteTracker(min_votes=4, min_fraction=0.5)
    for n in [10, 16, 10, 16]:
        v.add(1, n)
    assert v.confident_number(1) is None   # 2–2 tie is NOT confident
    v.add(1, 10)
    assert v.confident_number(1) == 10     # now a clear 3–2 majority


def test_autobind_uniqueness_and_manual():
    from soccer_analytics.jersey_ocr import AutoBinder
    from soccer_analytics.sensors.sync import SensorVideoSync

    class StubReader:
        available = True
        def __init__(self, n): self.n = n
        def read(self, frame, bbox): return self.n

    sync = SensorVideoSync()
    sync.bind(99, 3)                        # manual binding — authoritative
    ab = AutoBinder(StubReader(10), {10: 7}, every=1)
    for fi in range(6):                     # track 5 reads #10 → player 7
        ab.step(fi, None, [(5, (0, 0, 10, 20)), (99, (0, 0, 10, 20))], sync)
    assert sync.player_of(5) == 7
    assert sync.player_of(99) == 3          # manual bind not clobbered
    for fi in range(6, 12):                 # same player, new track 22 (id switch)
        ab.step(fi, None, [(22, (0, 0, 10, 20))], sync)
    assert sync.player_of(22) == 7 and sync.player_of(5) is None   # rebound, not double


def test_shot_rebound_counted():
    from soccer_analytics.shots import ShotDetector
    sd = ShotDetector()
    goals = {1: 105.0, 2: 0.0}
    sd.update(0, (80.0, 34.0), (12.0, 0.0), 1, goals)   # shot 1
    sd.update(1, (85.0, 34.0), (12.0, 0.0), 1, goals)   # same shot (no double)
    sd.update(2, (90.0, 20.0), (0.0, 12.0), 1, goals)   # off-cone → clears flag
    sd.update(3, (92.0, 34.0), (12.0, 0.0), 1, goals)   # rebound toward goal → shot 2
    assert sd.summary()[1]["shots"] == 2                # ball never slowed below 6 m/s


def test_xg_geometry_and_monotonicity():
    from soccer_analytics.shots import shot_geometry, XGModel
    m = XGModel()
    # closer, central shot has higher xG than a far one
    near = m.xg(100, 34, 105)     # 5 m out, central
    far = m.xg(80, 34, 105)       # 25 m out, central
    assert near > far
    # central beats a tight angle at the same distance
    central = m.xg(95, 34, 105)   # 10 m central
    wide = m.xg(95, 60, 105)      # 10 m but out wide
    assert central > wide
    assert 0.0 < far < near < 1.0
    d, a = shot_geometry(105, 34, 105)   # on the goal line, central
    assert d == 0.0 and a > 0


def test_shot_detector_flags_shot():
    from soccer_analytics.shots import ShotDetector
    sd = ShotDetector()
    # ball at 90 m in team-1's attacking half moving fast toward goal at x=105
    sd.update(1, (90.0, 34.0), (10.0, 0.0), 1, {1: 105.0, 2: 0.0}, nearest_player=9)
    s = sd.summary()
    assert s[1]["shots"] == 1 and s[1]["xg"] > 0
    # slow ball → no shot
    sd2 = ShotDetector()
    sd2.update(1, (90.0, 34.0), (1.0, 0.0), 1, {1: 105.0, 2: 0.0})
    assert sd2.summary()[1]["shots"] == 0


def test_jersey_vote_tracker():
    from soccer_analytics.jersey_ocr import JerseyVoteTracker
    v = JerseyVoteTracker(min_votes=4, min_fraction=0.5)
    for _ in range(5):
        v.add(7, 10)
    v.add(7, 23)                       # one noisy read
    assert v.confident_number(7) == 10
    v.add(9, 5); v.add(9, 6)           # tie / too few → not confident
    assert v.confident_number(9) is None


def test_team_assign_shadow_invariant():
    # regression for the team-mixing bug: a white jersey in SHADOW (low Lab-L) must
    # be assigned to the same team as a bright white jersey — clustering on chroma
    # (a,b), not lightness. Colours are (L, a, b).
    from soccer_analytics.team_assign import TeamAssigner
    ta = TeamAssigner()
    green = [np.array([L, 100.0, 160.0]) for L in (150, 160, 170, 155)]   # green kit
    white = [np.array([L, 127.0, 127.0]) for L in (210, 230, 220, 205)]   # white kit
    ta.fit(green + white)
    bright_white = ta.predict_team(np.array([220.0, 127.0, 127.0]))
    shadow_white = ta.predict_team(np.array([160.0, 126.0, 126.0]))       # low L
    assert bright_white == shadow_white          # same team despite the shadow
    assert ta.predict_team(np.array([158.0, 101.0, 159.0])) != bright_white  # green ≠ white


def test_line_circle_constrained_homography():
    # synthesize a camera H, generate: 2 exact points, points on pitch lines
    # (touchline y=0, halfway x=52.5), and points ON the centre circle — the
    # constrained fit must recover the mapping (sideline-compression regression).
    from soccer_analytics.view import fit_homography_lines, apply_homography
    Hcam = np.array([[8.0, 2.0, 300.0], [0.5, 10.0, 150.0], [0.0005, 0.001, 1.0]])
    P = lambda pts: apply_homography(Hcam, np.asarray(pts, float))
    hard_dst = [[52.5, 34.0], [52.5, 0.0]]
    hard_src = P(hard_dst)
    cons = []
    for x in (10, 30, 70, 95):                      # far touchline y=0
        cons.append((P([[x, 0.0]])[0], "y", 0.0))
    for y in (10, 30, 50, 64):                      # halfway x=52.5
        cons.append((P([[52.5, y]])[0], "x", 52.5))
    for th in np.linspace(0, 2 * np.pi, 10, endpoint=False):   # centre circle
        px = [52.5 + 9.15 * np.cos(th), 34.0 + 9.15 * np.sin(th)]
        cons.append((P([px])[0], "circle", (52.5, 34.0, 9.15)))
    # biased init points (mimic clicked ellipse extremes ±2m error)
    init_dst = [[43.35, 34.0], [61.65, 34.0], [52.5, 24.85], [52.5, 43.15]]
    init_src = P([[43.35, 32.5], [61.65, 35.5], [51.0, 24.85], [54.0, 43.15]])
    Hinv = fit_homography_lines(hard_src, hard_dst, cons,
                                init_src=init_src, init_dst=init_dst)
    # check recovery at pitch extremes (the old failure: wings collapsed to centre)
    for probe in ([5.0, 60.0], [100.0, 60.0], [15.0, 10.0], [90.0, 50.0]):
        rec = apply_homography(Hinv, P([probe]))[0]
        assert abs(rec[0] - probe[0]) < 1.5 and abs(rec[1] - probe[1]) < 1.5, (probe, rec)


def test_reid_persists_across_gap():
    # a player leaves view (ByteTrack drops the id) and returns with a NEW id —
    # Re-ID must map it back to the SAME stable id.
    from soccer_analytics.reid import ReIDManager
    r = ReIDManager()
    green = np.array([150.0, 100.0, 160.0])
    for fi in range(5):
        m = r.update(fi, [{"bt_id": 1, "center": (100.0, 200.0), "team": 1, "color": green}])
    sid = m[1]
    for fi in range(5, 20):
        r.update(fi, [])                       # gone for 15 frames
    m2 = r.update(20, [{"bt_id": 9, "center": (110.0, 205.0), "team": 1, "color": green}])
    assert m2[9] == sid                        # re-identified as the same player
    assert r.reids >= 1
    # a different-team player nearby must NOT inherit the id
    white = np.array([210.0, 127.0, 127.0])
    m3 = r.update(21, [{"bt_id": 11, "center": (100.0, 200.0), "team": 2, "color": white}])
    assert m3[11] != sid


def test_pass_prediction():
    from soccer_analytics.pass_prediction import pass_options, best_receiver, _lane_clearness
    carrier = (40.0, 34.0)
    teammates = {7: (55.0, 20.0), 9: (50.0, 48.0)}   # both forward
    opponents = [(50.0, 34.0)]                         # marks nobody tightly
    opts = pass_options(carrier, teammates, opponents, attack_goal_x=105.0)
    assert opts and opts[0].score >= opts[-1].score    # sorted best-first
    # a defender directly in the lane lowers clearness
    clear = _lane_clearness(np.array([0.0, 0]), np.array([10.0, 0]), np.array([[5.0, 0.2]]))
    open_ = _lane_clearness(np.array([0.0, 0]), np.array([10.0, 0]), np.array([[5.0, 9.0]]))
    assert clear < open_
    br = best_receiver(carrier, teammates, opponents, 105.0)
    assert br is not None and 0 <= br.score <= 1


def test_trajectory_model_forward():
    import torch
    from soccer_analytics.trajectory import TrajectoryLSTM, K_IN, H_OUT
    m = TrajectoryLSTM(hidden=32)
    out = m(torch.randn(4, K_IN, 2))
    assert out.shape == (4, H_OUT, 2)
    # constant-velocity trajectory → predict_positions returns H future points
    xy = np.cumsum(np.tile([0.01, 0.0], (K_IN + 2, 1)), axis=0).astype(np.float32)
    fut = m.predict_positions(xy)
    assert fut.shape == (H_OUT, 2)


def test_substitution_advisor_ranks_fatigued_player():
    from soccer_analytics.substitution import SubstitutionAdvisor
    from soccer_analytics.loadtypes import WorkloadFeatures
    adv = SubstitutionAdvisor(min_minutes=2.0, min_snapshots=5)
    # player 7 fades badly (HSR/work rate collapse); player 9 maintains output
    for k in range(13):
        t = k * 30.0                                   # snapshot every 30 s
        fade = max(0.0, 1.0 - 0.12 * k)                # decaying rate
        f7 = WorkloadFeatures(player_id=7,
                              total_distance=sum(110 * max(0.0, 1 - 0.12 * i) for i in range(k + 1)),
                              hsr_distance=sum(28 * max(0.0, 1 - 0.15 * i) for i in range(k + 1)),
                              sprint_count=min(k, 6), energy_kcal=8.0 * (k + 1) * max(0.3, fade))
        f9 = WorkloadFeatures(player_id=9, total_distance=110.0 * (k + 1),
                              hsr_distance=28.0 * (k + 1), sprint_count=k,
                              energy_kcal=8.0 * (k + 1))
        adv.update(t, {7: f7, 9: f9})
    ranked = adv.advise({7: 0.4, 9: 0.1})
    assert ranked[0].player_id == 7                    # fatigued player first
    assert ranked[0].priority > ranked[-1].priority + 0.15
    assert any("rate" in r for r in ranked[0].reasons)
    fresh = [a for a in ranked if a.player_id == 9][0]
    assert fresh.fatigue < 0.15                        # maintained output ≈ no fatigue


def test_hardware_chain_end_to_end():
    """Wearable JSON → sync → fusion → Catapult load → injury → substitution.
    The exact chain campus hardware will drive (transport differs, logic identical)."""
    import math
    from soccer_analytics.sensors import FusionEngine, SensorVideoSync, HeuristicInjuryModel
    from soccer_analytics.sensors.schema import SensorSample
    from soccer_analytics.substitution import SubstitutionAdvisor
    sync = SensorVideoSync(); sync.bind(3, 7)          # vision track 3 = player 7
    fe = FusionEngine(sync)
    adv = SubstitutionAdvisor(min_minutes=0.5, min_snapshots=5)   # 60 s test clip
    inj = HeuristicInjuryModel()
    for i in range(1500):                              # 60 s @ 25 fps
        t = i * 0.04
        if i % 3 == 0:                                 # wearable ~8 Hz, worsening HR
            fe.ingest_sensors([SensorSample(7, t, hr=125 + i * 0.04, spo2=96.0,
                                            accel=(0.1, 0.2, 1.0 + math.sin(i) * 0.4))])
        speed = 7.5 if i < 700 else 3.0                # sprints early, fades late
        fe.step(0.04, t, {3: ((speed, 0.0), (0.0, 0.0))})
        if i % 250 == 0:
            adv.update(t, fe.load.all_features())
    f = fe.load.features(3)
    assert f.total_distance > 200 and f.avg_hr > 125   # fused vision + wearable
    assert not np.isnan(f.min_spo2)
    r = inj.predict(f)
    assert 0.0 <= r.risk <= 1.0
    ranked = adv.advise({3: r.risk})
    assert ranked and ranked[0].player_id == 3
    assert ranked[0].fatigue > 0.2                     # the fade is detected


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print("PASS", fn.__name__)
        except AssertionError as e:
            failed += 1; print("FAIL", fn.__name__, e)
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
