"""Regression tests for the measured tracking, ball, 3D and fusion components."""

import json

import numpy as np
import pytest

from soccer_analytics.ball3d import _ballistic, camera_from_homography, reconstruct
from soccer_analytics.ball_tracking import ball_path_metrics, select_ball_path
from soccer_analytics.coach_engine import CoachEngine, Sample, VestSeries
from soccer_analytics.imm_tracker import IMMTracker, expanded_iou
from soccer_analytics.mot_eval import evaluate
from soccer_analytics.sensors.motion_artifact import clean_heart_rate
from soccer_analytics.tracklet_link import TrackletLinker, tracklets_from_frames


def _gt_frames(n=60, k=4):
    out = {}
    for f in range(1, n + 1):
        ids = np.arange(1, k + 1)
        boxes = np.array([[100 + 150 * i + 2 * f, 200, 40, 90] for i in range(k)], float)
        out[f] = (ids, boxes)
    return out


def test_mot_metrics_perfect_and_switch():
    gt = _gt_frames()
    r = evaluate(gt, gt)
    assert r["HOTA"] == pytest.approx(1.0) and r["IDF1"] == pytest.approx(1.0) and r["IDSW"] == 0
    swapped = {f: (np.where(ids == 1, 2, np.where(ids == 2, 1, ids)) if f > 30 else ids, b)
               for f, (ids, b) in gt.items()}
    r2 = evaluate(gt, swapped)
    assert r2["IDSW"] == 2 and r2["HOTA"] < 1.0 and r2["DetA"] == pytest.approx(1.0)


def test_expanded_iou_grows_with_expansion():
    a = np.array([[100, 100, 20, 40.]])
    b = np.array([[125, 100, 20, 40.]])           # disjoint without expansion
    assert expanded_iou(a, np.array([0.0]), b, 0.0)[0, 0] == 0
    assert expanded_iou(a, np.array([0.6]), b, 0.3)[0, 0] > 0


def test_imm_tracker_keeps_identity_through_gap():
    tr = IMMTracker(use_appearance=False)
    ids = []
    for f in range(40):
        x = 100 + 6 * f
        dets = [] if 15 <= f < 20 else [[x, 200, x + 30, 280]]       # five missed frames
        out = tr.update(np.array(dets, float).reshape(-1, 4), np.full(len(dets), 0.9))
        ids += [tid for tid, _, _ in out]
    assert len(set(ids)) == 1


def test_linker_joins_fragment_and_interpolates():
    per_frame = {}
    for f in range(60):
        if f < 25:
            per_frame[f] = [(1, np.array([100 + 3 * f, 200, 130 + 3 * f, 280.]))]
        elif f >= 32:
            per_frame[f] = [(2, np.array([100 + 3 * f, 200, 130 + 3 * f, 280.]))]
        else:
            per_frame[f] = []
    emb = {1: {0: np.ones(8) / np.sqrt(8)}, 2: {0: np.ones(8) / np.sqrt(8)}}
    res = TrackletLinker(interpolate_gap=20).link(tracklets_from_frames(per_frame, emb=emb))
    assert len(set(res.identity_of.values())) == 1
    assert any(interp.any() for _, _, interp in res.frames.values())


def test_ball_path_rejects_isolated_false_positive():
    n = 40
    cands = {f: np.array([[100 + 10 * f, 300, 0.35]]) for f in range(n) if f % 7}
    cands[20] = np.vstack([cands.get(20, np.zeros((0, 3))), [[1500, 900, 0.6]]])  # far spurious box
    path = select_ball_path(cands, n)
    assert all(p.x < 1000 for p in path)
    gt = {f: (100 + 10 * f, 300) for f in range(n)}
    m = ball_path_metrics(path, gt, n)
    assert m["precision"] == 1.0 and m["recall"] > 0.9       # gaps inside the chain are filled


def _broadcast_camera():
    C, target, f = np.array([52.5, 100., 25.]), np.array([60., 34., 0.]), 1800
    z = (target - C) / np.linalg.norm(target - C)
    x = np.cross(z, [0, 0, 1.]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    K = np.array([[f, 0, 960], [0, f, 540], [0, 0, 1.]])
    return K @ np.column_stack((R, -R @ C))


def test_camera_recovered_from_homography():
    P = _broadcast_camera()
    cam = camera_from_homography(P[:, [0, 1, 3]], (1920, 1080))
    assert cam.K[0, 0] == pytest.approx(1800, rel=1e-3)
    assert np.allclose(cam.centre, [52.5, 100, 25], atol=0.05)


def test_ballistic_arc_reconstructed_in_3d():
    P = _broadcast_camera()
    rng = np.random.default_rng(3)
    v0 = np.array([15., -2., 8.])
    t = np.arange(0, (v0[2] + np.sqrt(v0[2] ** 2 + 2 * 9.81 * .1)) / 9.81, 1 / 25)
    X = _ballistic(np.array([60., 30., .1]), v0, t)
    h = np.column_stack((X, np.ones(len(X)))) @ P.T
    uv = h[:, :2] / h[:, 2:3] + rng.normal(0, 1.0, (len(X), 2))
    r = reconstruct(np.arange(len(X)), uv, np.linalg.inv(P[:, [0, 1, 3]]), (1920, 1080), 25)
    assert r.segments[0].kind == "air"
    assert abs(r.segments[0].peak_height_m - X[:, 2].max()) < 0.6
    assert np.nanmean(np.linalg.norm(r.xyz - X, axis=1)) < 1.0


def test_nlms_removes_cadence_from_heart_rate():
    fs, t, ta = 25, np.arange(0, 8, 1 / 25), np.arange(0, 8, 1 / 50)
    acc = np.column_stack([3 * np.sin(2 * np.pi * 2.8 * ta), 1.5 * np.sin(2 * np.pi * 2.8 * ta + .7), .8 * np.sin(2 * np.pi * 5.6 * ta)])
    ap = np.column_stack([np.interp(t, ta, acc[:, k]) for k in range(3)])
    ir = 80000 + 120 * np.sin(2 * np.pi * 2.5 * t) + 500 * ap[:, 0] + 300 * ap[:, 1]
    est = clean_heart_rate(ir, t, acc, ta, fs)
    assert abs(est["bpm"] - 150) < 4


def test_spo2_never_defaults_to_healthy_value():
    from soccer_analytics.sensors.hub import estimate_ppg_vitals
    fs, t = 25.0, np.arange(0, 8, 1 / 25)
    ir = 120000 + 4000 * np.sin(2 * np.pi * 1.2 * t)
    red = 90000 + 9000 * np.sin(2 * np.pi * 1.2 * t)      # R ~ 3: outside the curve
    out = estimate_ppg_vitals(ir, red, fs)
    assert out["spo2_estimate_pct"] is None and out["valid"]   # HR still reported


def test_coach_engine_flags_cardiac_drift_with_evidence():
    samples = []
    for k in range(3000):                                    # 10 minutes at 5 Hz
        t = k / 5
        sp = 3.0 + (k % 50) / 25.0
        samples.append(Sample(t, {7: (50, 30, 1, sp)}, None, 0, None))
    tt = np.arange(0, 600, 1.0)
    speed = np.interp(tt, [s.t for s in samples], [s.players[7][3] for s in samples])
    hr = 120 + 8 * speed + np.where(tt > 300, 15, 0)
    cards = CoachEngine().analyse(samples, {7: VestSeries(tt, hr, np.full(len(tt), .9))})
    drift = [c for c in cards if c.id == "drift-7"]
    assert drift and drift[0].kind == "fused" and drift[0].evidence


def test_analysis_review_roundtrip(tmp_path):
    from cpg44_api.analyses import AnalysisLibrary
    d = tmp_path / "m1"
    d.mkdir()
    (d / "analysis.json").write_text(json.dumps({"match": {"name": "x"}, "quality": {}, "provenance": {},
                                                 "events": [{"id": "e1", "status": "check"}]}))
    lib = AnalysisLibrary([tmp_path])
    assert lib.get("m1")["quality"]["checks_pending"] == 1
    lib.decide("m1", "e1", "dismiss")
    a = lib.get("m1")
    assert a["events"][0]["status"] == "dismissed" and a["quality"]["checks_pending"] == 0
    assert lib.get("../etc") is None


def test_line_refinement_pulls_drifted_calibration_back():
    import cv2
    from soccer_analytics.config import PitchConfig
    from soccer_analytics.line_refine import refine, template_points
    from soccer_analytics.view import apply_homography
    pc = PitchConfig()
    P = _broadcast_camera()
    Hp2i = P[:, [0, 1, 3]] / P[2, 3]
    img = np.zeros((1080, 1920, 3), np.uint8)
    img[:] = (60, 140, 70)                                   # grass
    for x, y in apply_homography(Hp2i, template_points(pc, 0.2)):
        if np.isfinite([x, y]).all() and 0 <= x < 1920 and 0 <= y < 1080:
            cv2.circle(img, (int(x), int(y)), 2, (235, 235, 235), -1)
    T = np.array([[1, 0, 14.0], [0, 1, -9.0], [0, 0, 1]])        # 17 px image drift
    H_drift = np.linalg.inv(T @ Hp2i)
    H_ref, info = refine(img, H_drift, pc)
    pts = template_points(pc, 5.0)
    truth = apply_homography(Hp2i, pts)
    err = lambda H: np.nanmedian(np.linalg.norm(apply_homography(np.linalg.inv(H), pts) - truth, axis=1))
    assert info["accepted"] and err(H_ref) < 0.4 * err(H_drift)


def test_decision_engine_prefers_open_forward_pass_and_shoots_close():
    from soccer_analytics.decision import decide, pass_options, xg
    assert 0.15 < xg(94, 34, 105, 68) < 0.35          # penalty spot, open play
    assert xg(80, 34, 105, 68) < 0.06                  # 25 m out
    mates = {7: (88.0, 30.0, 2.0, 0.0), 8: (60.0, 34.0, 0.0, 0.0)}
    opps = [(78.0, 40.0, 0.0, 0.0), (96.0, 50.0, 0.0, 0.0), (101.0, 34.0, 0.0, 0.0)]
    opts = pass_options((72.0, 34.0), mates, opps, 105.0, 68.0, 105.0)
    assert opts and opts[0].to == 7 and opts[0].value > 0
    blocked = pass_options((72.0, 34.0), {7: (88.0, 34.0, 0.0, 0.0)}, [(80.0, 34.0, 0.0, 0.0)], 105.0, 68.0, 105.0)
    assert blocked[0].p_arrive < opts[0].p_arrive      # a defender on the lane cuts the chance
    assert decide((99.0, 34.0), opts, 105.0, 68.0, 0)["call"] == "shoot"


def test_rolling_ball_agrees_with_a_non_physical_homography():
    """Clicked calibrations are not exact cameras. A rolling ball must land
    where the homography (which also places the players) puts it."""
    from soccer_analytics.view import apply_homography
    P = _broadcast_camera()
    G = P[:, [0, 1, 3]].copy()
    G[0, 1] *= 1.04                     # skew no pinhole camera can produce
    G[1, 0] += 0.02 * G[1, 1]
    Hi2p = np.linalg.inv(G)
    t = np.arange(0, 1.2, 1 / 25)
    xy = np.column_stack((40 + 9 * t - 1.5 * t * t, 30 + 3 * t))
    h = np.column_stack((xy, np.ones(len(t)))) @ G.T
    uv = h[:, :2] / h[:, 2:3]
    r = reconstruct(np.arange(len(t)), uv, Hi2p, (1920, 1080), 25)
    assert all(s.kind == "ground" for s in r.segments)
    on_h = apply_homography(Hi2p, uv)
    assert np.nanmax(np.linalg.norm(r.xyz[:, :2] - on_h, axis=1)) < 0.4


def test_keeper_gets_the_side_defending_his_goal():
    from soccer_analytics.core import GOALKEEPER, PLAYER
    from soccer_analytics.match_analysis import _keepers_by_side
    # side 2 attacks the right goal and pushes high; side 1 defends it deep
    pos = {}
    for f in range(40):
        pos[f] = {1: (95.0, 34.0), 2: (99.0, 30.0)}        # striker of side 2, last defender of side 1
        pos[f].update({10 + k: (70.0 + 4 * k, 20.0 + 5 * k) for k in range(4)})   # side 2 high line
        pos[f].update({20 + k: (80.0 + 4 * k, 22.0 + 5 * k) for k in range(4)})   # side 1 defending
        pos[f][99] = (103.0, 34.0)                          # keeper
    team_of = {1: 2, 2: 1, **{10 + k: 2 for k in range(4)}, **{20 + k: 1 for k in range(4)}}
    role_of = {i: PLAYER for i in team_of}
    role_of[99] = GOALKEEPER
    info = {}
    _keepers_by_side(pos, team_of, role_of, info, 105.0)
    assert team_of[99] == 1 and info["keepers_by_side"] == 1


def test_relay_clock_offset_uses_shortest_round_trips():
    from soccer_analytics.sensors.hub import RelayClockOffset
    c = RelayClockOffset()
    true_offset = 420_000_000                      # relay 420 ms ahead of this machine
    rng = np.random.default_rng(1)
    t = 10**18
    for k in range(20):
        up = int(5e6 + rng.exponential(60e6))      # queueing delays, asymmetric
        down = int(5e6 + rng.exponential(20e6))
        t0 = t + k * 10**9
        server = t0 + up + true_offset
        c.add(t0, server, t0 + up + down)
    est = c.estimate()
    assert abs(est["offset_ns"] - true_offset) < 25e6 and est["uncertainty_ns"] < 40e6
    assert abs(c.to_host_unix_ns(t + true_offset) - t) < 25e6


def _session(minutes=30, fade_id=7, rng_seed=0):
    from soccer_analytics.coach_engine import Sample
    rng = np.random.default_rng(rng_seed)
    out = []
    for k in range(int(minutes * 60 * 5)):
        t = k / 5
        players = {}
        team_dip = 0.55 if 20 * 60 <= t < 25 * 60 else 1.0          # whole team defending deep
        for pid in range(1, 21):
            tm = 1 if pid <= 10 else 2
            hot = 0.07 * team_dip
            if pid == fade_id and t > 15 * 60:
                hot *= 0.3                                            # this player fades
            sp = 7.6 if rng.random() < hot * 0.25 else 6.0 if rng.random() < hot else rng.uniform(0.5, 3.5)
            if pid == fade_id and t > 15 * 60 and sp > 7:
                sp = 6.6                                              # cannot reach earlier top speed
            players[pid] = (30 + pid * 2.0, 10 + (pid % 10) * 5.0, tm, sp)
        out.append(Sample(t, players, None))
    return out


def test_fatigue_flags_the_fading_player_not_the_team_dip():
    from soccer_analytics.fatigue import assess
    r = assess(_session(), attack_dir={1: 1, 2: -1})
    assert r["available"] and r["window_s"] == 300.0
    by = {p["id"]: p for p in r["players"]}
    assert by[7]["status"] == "tiring" and by[7] is r["players"][0]
    others = [p for p in r["players"] if p["id"] != 7]
    assert max(p["score"] for p in others) < 0.45                   # the team-wide dip is context
    assert any(s["out"] == 7 for s in r["subs"])
    assert by[7]["factors"]["sprint_speed"] > 0.3


def test_fatigue_refuses_short_clips():
    from soccer_analytics.fatigue import assess
    r = assess(_session(minutes=2))
    assert not r["available"] and "needs at least" in r["reason"]


def test_formation_change_is_reported_once_it_holds():
    from soccer_analytics.coach_engine import Sample
    from soccer_analytics.config import PipelineConfig
    from soccer_analytics.match_analysis import _game_plan
    four = [(20, 10), (20, 26), (20, 42), (20, 58), (40, 10), (40, 26), (40, 42), (40, 58), (60, 25), (60, 43)]
    three = [(20, 18), (20, 34), (20, 50), (40, 6), (40, 20), (40, 34), (40, 48), (40, 62), (60, 25), (60, 43)]
    rng = np.random.default_rng(0)
    samples = []
    for k in range(30 * 60):                                 # 30 min at 1 Hz, side 1 defending
        t = float(k)
        base = four if t < 15 * 60 else three
        players = {i + 1: (x + rng.normal(0, 1), y + rng.normal(0, 1), 1, 2.0) for i, (x, y) in enumerate(base)}
        players.update({20 + i: (70.0 + i, 5.0 + 6 * i, 2, 2.0) for i in range(10)})
        samples.append(Sample(t, players, (50.0, 34.0, 0.0), possession=2))
    plan = _game_plan(samples, list(range(len(samples))), {}, [], {1: 1, 2: -1}, PipelineConfig(), ("Green", "White"))
    tl = plan["teams"]["1"]["shape_timeline"]
    assert tl[0]["without_ball"] != tl[-1]["without_ball"]
    changes = [c for c in plan["cards"] if c["id"].startswith("plan-shape-1-without")]
    assert len(changes) == 1 and changes[0]["moments"][0] >= 15 * 60 - 300


def test_pass_needs_the_ball_to_travel():
    from soccer_analytics.metrics import MetricsEngine
    teams = {1: 1, 2: 1}

    def run(receiver_xy, ball_path):
        eng = MetricsEngine(fps=25, metric=True)
        players = {1: (50.0, 30.0), 2: receiver_xy}
        for f, b in enumerate(ball_path):
            eng.update(f, players, teams, b)
        return eng.passes

    # teammates 1.5 m apart over a loose ball: nearest player flips, no pass
    swap = [(50.4, 30.0)] * 10 + [(50.9, 30.5)] * 10
    assert run((50.0, 31.5), swap) == []
    # a 12 m pass along the ground
    path = [(50.5, 30.0)] * 10 + [(50.5 + k, 30.0) for k in range(1, 12)] + [(61.6, 30.0)] * 8
    out = run((62.0, 30.0), path)
    assert len(out) == 1 and out[0].from_id == 1 and out[0].to_id == 2
    assert abs(out[0].start[0] - 50.5) < 1.5 and abs(out[0].end[0] - 61.6) < 1.0


def test_vest_tag_moves_with_the_coach(tmp_path):
    from cpg44_api.analyses import AnalysisLibrary
    d = tmp_path / "m2"
    d.mkdir()
    (d / "analysis.json").write_text(json.dumps({"match": {"name": "x"}, "quality": {}, "provenance": {}, "events": [],
                                                 "players": [{"id": 3, "vest": 1}, {"id": 7}, {"id": 9}]}))
    lib = AnalysisLibrary([tmp_path])
    lib.tag_vest("m2", 7, 1)                       # the automatic match chose 3; the coach says 7
    by = {p["id"]: p for p in lib.get("m2")["players"]}
    assert "vest" not in by[3] and by[7]["vest"] == 1 and by[7]["vest_source"] == "coach"
    lib.tag_vest("m2", 9, 1)                       # one vest, one player
    by = {p["id"]: p for p in lib.get("m2")["players"]}
    assert "vest" not in by[7] and by[9]["vest"] == 1
    lib.tag_vest("m2", 9, None)
    assert "vest_source" not in {k for p in lib.get("m2")["players"] for k in p}
    with pytest.raises(KeyError):
        lib.tag_vest("m2", 42, 1)


def test_impossible_roll_becomes_a_flight_between_contacts():
    from soccer_analytics.match_analysis import _contact_flights
    fps = 25.0
    b = {}
    for f in range(0, 20):                       # at the kicker's feet
        b[f] = (30.0 + 0.1 * f, 30.0, 0.0)
    for f in range(20, 70):                      # the shadow of the airborne ball races out and back
        u = (f - 19) / 51.0
        b[f] = (32.0 + 30.0 * u + 80.0 * np.sin(np.pi * u), 30.0, 0.0)
    for f in range(70, 90):                      # lands 30 m on and is controlled
        b[f] = (62.0 + 0.2 * (f - 69), 30.0, 0.0)
    out = _contact_flights(b, fps)
    assert len(out) == 1
    a_f, e_f = out[0]
    T = (e_f - a_f) / fps
    mid = (a_f + e_f) // 2
    assert b[mid][2] > 0.9 * 9.81 * T * T / 8               # apex of a free flight of T seconds
    v = [np.hypot(b[f + 1][0] - b[f][0], b[f + 1][1] - b[f][1]) * fps for f in range(a_f, e_f)]
    assert max(v) < 36


def test_unexplainable_ball_positions_are_withheld():
    from soccer_analytics.match_analysis import _drop_impossible_ball
    b = {f: (50.0 + 0.2 * f, 30.0, 0.0) for f in range(30)}
    b[15] = (80.0, 30.0, 0.0)                    # one detection 27 m away
    assert _drop_impossible_ball(b, 25.0) == 1 and 15 not in b


def test_homography_snaps_are_smoothed_out():
    from soccer_analytics.match_analysis import _RawFrame, _smooth_homographies
    from soccer_analytics.view import apply_homography
    base = np.array([[0.06, 0.004, -10.0], [0.0005, 0.09, -20.0], [0.0, 0.00012, 1.0]])
    raw = []
    for f in range(100):
        H = base.copy()
        H[0, 2] += 0.02 * f                       # a slow pan
        if f % 10 == 5:
            H[0, 2] += 6.0                        # a line-snap jump of metres
        raw.append(_RawFrame([], np.zeros((0, 3)), H))
    info = _smooth_homographies(raw, 1920, 1080, 25.0)
    p = [apply_homography(r.H, [[960, 800]])[0][0] for r in raw]
    jumps = np.abs(np.diff(p))
    assert info["frames"] == 100 and jumps.max() < 0.5
