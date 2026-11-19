"""Recorded-match analysis: one evidence file for the review workspace.

``build_analysis`` runs the measured-best components end to end on a recorded
video and writes ``analysis.json`` (schema ``cpg44.match_analysis/1``):

  detect (all roles, low thresholds)
    -> IMM-EIoU tracker with camera-motion compensation and ReID appearance
    -> offline tracklet linking (+ short-gap interpolation, flagged)
    -> identity-level team separation (kit colour clustering, referee/keeper abstention)
    -> pitch mapping with the propagated calibration
    -> ball: temporal path selection + physics-based 3D arcs
    -> possession, passes, turnovers, shots (with 3D breakdown), sprints
    -> review queue ("check" items) for every uncertain decision
    -> coach decision cards (coach_engine), optionally fused with a vest log

Everything that is inferred rather than observed carries a flag, and metric
outputs exist only when the pitch is calibrated. See docs/TRACKING_BENCHMARK.md
for how each component was chosen.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .ball3d import reconstruct, shot_metrics
from .ball_tracking import select_ball_path
from .coach_engine import CoachEngine, Sample, VestSeries, HSR, SPRINT
from .config import PipelineConfig
from .core import BALL, GOALKEEPER, PLAYER, REFEREE, foot_points
from .device import resolve_device
from .imm_tracker import CameraMotion, IMMTracker, appearance_embedding
from .line_refine import refine as refine_lines
from .lens import undistort_frame
from .fatigue import assess as assess_fatigue
from .metrics import MetricsEngine
from .reid_model import ReIDEncoder
from .shots import XGModel
from .team_assign import TeamAssigner
from .tracker import Detector
from .tracklet_link import TrackletLinker, tracklets_from_frames
from .view import PitchHomographyTracker, ViewTransformer, apply_homography

SCHEMA = "cpg44.match_analysis/1"
REFEREE_TEAM = 3          # timeline code for match officials (drawn, never counted)
TEAM_COLOURS = {1: "#1f5f99", 2: "#c2552c", 0: "#5b6460"}


@dataclass
class _RawFrame:
    tracks: List[Tuple[int, np.ndarray, float]]
    balls: np.ndarray                       # (n,3) cx, cy, conf
    H: Optional[np.ndarray]                 # image -> pitch for this frame


def _progress(i, n):
    if i % 50 == 0:
        print(f"[analyze] {i}/{n or '?'} frames", flush=True)


STAGE1_VERSION = 1


def _stage1_cache_path(video, cfg, view, lens_k1, camera_mode, max_frames) -> Path:
    import hashlib
    h = hashlib.sha1()
    for part in (video, cfg.weights):
        try:
            st = Path(part).stat()
            h.update(f"{Path(part).resolve()}|{st.st_size}|{st.st_mtime_ns}".encode())
        except OSError:
            h.update(str(part).encode())
    H = np.asarray(view.H, float) if view.is_metric else np.zeros(1)
    h.update(np.round(H, 9).tobytes())
    h.update(f"{cfg.imgsz}|{cfg.pitch.length}|{cfg.pitch.width}|{lens_k1:.6f}|{camera_mode}|{max_frames}|{STAGE1_VERSION}".encode())
    root = Path(__file__).resolve().parent.parent / "data" / "cache" / "analysis"
    return root / f"{Path(video).stem}-{h.hexdigest()[:12]}.pkl"


def _load_stage1(path: Path):
    import pickle
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def _save_stage1(path: Path, payload) -> None:
    import pickle
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
    except OSError:
        pass


def build_analysis(video: str, cfg: PipelineConfig, view: ViewTransformer, out_dir: str,
                   max_frames: int = 0, render: bool = True, match_name: str = "",
                   vest_series: Optional[Dict[int, VestSeries]] = None,
                   team_names: Tuple[str, str] = ("Home", "Away"), lens_k1: float = 0.0,
                   camera_mode: str = "auto", vest_logs: Optional[List[str]] = None,
                   video_start_unix: Optional[float] = None) -> dict:
    """``camera_mode``: "fixed" (tripod/mast, the campus setup) keeps the
    calibrated homography and only re-snaps it gently to the painted lines;
    "panning"/"auto" propagate it with optical flow. ``lens_k1`` undistorts
    every frame first so the pinhole homography holds across the image."""
    t_start = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames) if total else max_frames
    dev = resolve_device(cfg.device)
    det = Detector(cfg.weights, imgsz=cfg.imgsz, device=dev.device, person_conf=0.1,
                   ball_conf=0.05, half=dev.half_ok, track_low_conf=0.1, remap_goalkeeper=False)
    reid = ReIDEncoder(device="cpu" if str(dev.device) == "cpu" else "")
    tracker = IMMTracker(use_imm=False, **_tracker_params(reid.available))
    cam = CameraMotion()
    fixed = camera_mode == "fixed"
    pitch = PitchHomographyTracker(view.H, (Hh, W), pitch_length=cfg.pitch.length,
                                   pitch_width=cfg.pitch.width) if (view.is_metric and not fixed) else None
    static_H = np.asarray(view.H, float).copy() if (view.is_metric and fixed) else None
    team = TeamAssigner()
    metric = view.is_metric

    # Detection, tracking and calibration propagation dominate the run time
    # (about 14 min on CPU for 30 s). Their output is cached per input so a
    # change to the later analysis stages rebuilds in seconds.
    cache_file = _stage1_cache_path(video, cfg, view, lens_k1, camera_mode, max_frames)
    cached = _load_stage1(cache_file)
    if cached is not None:
        (raw, emb_sum, emb_n, colours, roles, line_refinements, line_recoveries) = cached
        print(f"[analyze] reused detections and tracks from {cache_file.name}", flush=True)
    else:
        raw: List[_RawFrame] = []
        last_good = None
        line_refinements = line_recoveries = 0
        emb_sum: Dict[int, np.ndarray] = {}
        emb_n: Dict[int, int] = {}
        colours: Dict[int, List[np.ndarray]] = {}
        roles: Dict[int, List[int]] = {}
        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and fi >= max_frames):
                break
            frame = undistort_frame(frame, lens_k1)
            d = det.detect(frame)
            people = d.class_id != BALL
            P = d[people]
            warp = cam.update(frame, P.xyxy)
            emb = reid(frame, P.xyxy) if reid.available else appearance_embedding(frame, P.xyxy)
            res = tracker.update(P.xyxy, P.confidence, emb, warp)
            if pitch is not None:
                pitch.update(frame, P.xyxy)
                if pitch.valid:
                    if fi % 5 == 0:
                        Hn, info = refine_lines(frame, pitch.H, cfg.pitch)
                        if info.get("accepted"):
                            pitch.H = Hn
                            line_refinements += 1
                    last_good = (pitch.H.copy(), fi)
                elif last_good is not None and fi - last_good[1] <= 50:
                    # Flow lost support (fast pan, blur): try to re-register from the
                    # last good calibration against the painted lines.
                    Hn, info = refine_lines(frame, last_good[0], cfg.pitch)
                    if info.get("accepted") and info.get("inliers_after", 0) >= 0.4:
                        pitch.H, pitch.valid = Hn, True
                        line_recoveries += 1
            if static_H is not None and fi % 25 == 0:
                # Fixed camera: the mapping should not move; accept only small,
                # clearly better line fits (tripod knocks, wind, mast sway).
                Hn, info = refine_lines(frame, static_H, cfg.pitch, max_shift_px=15.0)
                if info.get("accepted"):
                    static_H = Hn
                    line_refinements += 1
            H = static_H.copy() if static_H is not None else (
                pitch.H.copy() if (pitch is not None and pitch.valid) else None)
            # attribute per-track evidence from the matching detection box
            for tid, box, score in res:
                k = int(np.argmin(np.abs(P.xyxy - box).sum(1))) if len(P) else -1
                if k >= 0 and np.abs(P.xyxy[k] - box).sum() < 1e-3:
                    v = emb[k]
                    if np.linalg.norm(v) > 0:
                        emb_sum[tid] = emb_sum.get(tid, 0) + v
                        emb_n[tid] = emb_n.get(tid, 0) + 1
                    roles.setdefault(tid, []).append(int(P.class_id[k]))
                if fi % 3 == 0:
                    c = team.shirt_color(frame, box)
                    if c is not None:
                        colours.setdefault(tid, []).append(c)
            B = d[~people]
            balls = np.column_stack(((B.xyxy[:, 0] + B.xyxy[:, 2]) / 2, (B.xyxy[:, 1] + B.xyxy[:, 3]) / 2,
                                     B.confidence)) if len(B) else np.zeros((0, 3))
            raw.append(_RawFrame([(t, b.copy(), s) for t, b, s in res], balls, H))
            fi += 1
            _progress(fi, total)
        _save_stage1(cache_file, (raw, emb_sum, emb_n, colours, roles, line_refinements, line_recoveries))
    cap.release()
    n = len(raw)
    print(f"[analyze] {n}/{n} frames; linking identities", flush=True)
    h_smoothing = _smooth_homographies(raw, W, Hh, fps) if metric and not fixed else {}

    # ---------------- identities ----------------
    per_frame = {f: [(t, b) for t, b, _ in r.tracks] for f, r in enumerate(raw)}
    embs = {tid: {0: emb_sum[tid] / max(emb_n[tid], 1)} for tid in emb_sum}
    tracklets = tracklets_from_frames(per_frame, emb=embs)
    linker = TrackletLinker(**_linker_params(reid.available))
    linked = linker.link(tracklets)
    ident = linked.identity_of

    # role per identity (majority of detector classes), team by kit colour clustering
    id_roles: Dict[int, List[int]] = {}
    id_cols: Dict[int, List[np.ndarray]] = {}
    for tid, fid in ident.items():
        id_roles.setdefault(fid, []).extend(roles.get(tid, []))
        id_cols.setdefault(fid, []).extend(colours.get(tid, []))
    role_of = {fid: (int(np.bincount(r, minlength=4).argmax()) if r else PLAYER) for fid, r in id_roles.items()}
    team_of, team_info = _assign_teams(id_cols, role_of, team)

    # ---------------- positions + kinematics ----------------
    # Offline, every foot point is projected first and judged against its own
    # neighbours in time (rolling median), instead of an online Kalman gate
    # that drops a player whenever a re-linked track or a calibration
    # correction makes the position jump.
    pos: Dict[int, Dict[int, Tuple[float, float, float, bool]]] = {}   # frame -> id -> (x,y,speed,interp)
    referees: Dict[int, Dict[int, Tuple[float, float]]] = {}
    boxes_out: Dict[int, List[Tuple[int, np.ndarray, bool]]] = {}
    funnel = {"boxes": 0, "referee": 0, "no_calibration": 0, "off_pitch": 0, "uncertain": 0,
              "outlier": 0, "placed": 0}
    uncertain_pins = 0
    tracks: Dict[int, List[Tuple[int, float, float, bool]]] = {}
    for f in range(n):
        ids, bx, interp = linked.frames.get(f, (np.zeros(0, int), np.zeros((0, 4)), np.zeros(0, bool)))
        boxes_out[f] = list(zip(ids.tolist(), bx, interp.tolist()))
        H = raw[f].H
        pos[f], referees[f] = {}, {}
        for fid, box, is_interp in zip(ids, bx, interp):
            funnel["boxes"] += 1
            if not metric or H is None:
                funnel["no_calibration"] += 1
                continue
            foot = foot_points(box)[0]
            p = apply_homography(H, [foot])[0]
            if not (np.isfinite(p).all() and -3 <= p[0] <= cfg.pitch.length + 3 and -3 <= p[1] <= cfg.pitch.width + 3):
                funnel["off_pitch"] += 1
                continue
            if _pin_sigma(H, foot, box) > 3.5:
                uncertain_pins += 1
                funnel["uncertain"] += 1
                continue
            if role_of.get(int(fid)) == REFEREE:
                funnel["referee"] += 1
                referees[f][int(fid)] = (float(p[0]), float(p[1]))
                continue
            tracks.setdefault(int(fid), []).append((f, float(p[0]), float(p[1]), bool(is_interp)))
    for fid, rows in tracks.items():
        a = np.asarray([(r[0], r[1], r[2]) for r in rows], float)
        keep = _robust_track_mask(a[:, 0], a[:, 1:3], fps, cfg.max_speed_mps)
        funnel["outlier"] += int((~keep).sum())
        for k in np.nonzero(keep)[0]:
            f, x, y, it = rows[k]
            pos[f][fid] = (x, y, float("nan"), it)
            funnel["placed"] += 1

    pos, velocity = _smooth_kinematics(pos, fps, cfg.max_speed_mps)
    if metric:
        _keepers_by_side(pos, team_of, role_of, team_info, cfg.pitch.length)

    # ---------------- ball ----------------
    path = select_ball_path({f: r.balls for f, r in enumerate(raw) if len(r.balls)}, n,
                            vmax_px=70.0 * W / 1920.0)
    ball_px = {p.frame: (p.x, p.y, p.interpolated, p.chain) for p in path}
    ball_xyz: Dict[int, Tuple[float, float, float]] = {}
    ball3d_segments = []
    if metric:
        chains: Dict[int, List] = {}
        for p in path:
            chains.setdefault(p.chain, []).append(p)
        for pts in chains.values():
            pts = [p for p in pts if raw[p.frame].H is not None]
            if not pts:
                continue
            frames_c = [p.frame for p in pts]
            # Each observation is fitted through its own frame's camera (the
            # camera pans and zooms), never warped into one reference frame.
            uv = np.asarray([[p.x, p.y] for p in pts], float)
            Hs = np.stack([np.asarray(raw[p.frame].H, float) for p in pts])
            ok = np.isfinite(uv).all(1)
            r3 = reconstruct(list(np.asarray(frames_c)[ok]), uv[ok], Hs[ok], (W, Hh), fps) if ok.sum() >= 5 else None
            if r3 is not None:
                fr_idx = {int(f): k for k, f in enumerate(r3.frames)}
                # A flight whose ends do not join the ball's path before and
                # after it is a misfit: fall back to the ground model there.
                ground_xy = {p.frame: apply_homography(raw[p.frame].H, [[p.x, p.y]])[0]
                             for p, keep in zip(pts, ok) if keep}
                for sg in r3.segments:
                    if sg.kind != "air":
                        continue
                    a_k, b_k = fr_idx.get(sg.start), fr_idx.get(sg.end)
                    bad = False
                    for k_in, k_out in ((a_k, (a_k or 0) - 1), (b_k, (b_k or 0) + 1)):
                        if k_in is None or not 0 <= k_out < len(r3.frames):
                            continue
                        gap = abs(r3.frames[k_out] - r3.frames[k_in]) / fps
                        jump = np.linalg.norm(r3.xyz[k_in] - r3.xyz[k_out])
                        if np.isfinite(jump) and jump > 1.0 + 40.0 * gap:
                            bad = True
                    if bad:
                        sg.kind = "ground"
                        sg.peak_height_m = 0.0
                        for f in range(sg.start, sg.end + 1):
                            k = fr_idx.get(f)
                            g = ground_xy.get(f)
                            if k is not None and g is not None and np.isfinite(g).all():
                                r3.xyz[k] = (g[0], g[1], 0.0)
                for f, xyz in zip(r3.frames, r3.xyz):
                    plausible = (np.isfinite(xyz).all() and -5 <= xyz[0] <= cfg.pitch.length + 5
                                 and -5 <= xyz[1] <= cfg.pitch.width + 5 and xyz[2] <= 25)
                    if plausible:
                        ball_xyz[int(f)] = (float(xyz[0]), float(xyz[1]), float(max(xyz[2], 0.0)))
                for s in r3.segments:
                    ball3d_segments.append({k: v for k, v in s.__dict__.items() if k != "params"})
            else:
                for p in pts:
                    g = apply_homography(raw[p.frame].H, [[p.x, p.y]])[0] if raw[p.frame].H is not None else None
                    if g is not None and np.isfinite(g).all():
                        ball_xyz[p.frame] = (float(g[0]), float(g[1]), 0.0)

    dropped = 0
    if metric and ball_xyz:
        _despike_ground_ball(ball_xyz)
        flights = _contact_flights(ball_xyz, fps)
        for a_f, b_f in flights:
            ball3d_segments.append({"start": a_f, "end": b_f, "kind": "air", "rmse_px": 0.0,
                                    "launch_speed_mps": None, "peak_height_m": round(9.81 * ((b_f - a_f) / fps) ** 2 / 8, 2),
                                    "method": "contact-to-contact"})
        ball3d_segments.sort(key=lambda sg: sg["start"])
        dropped = _drop_impossible_ball(ball_xyz, fps)

    # ---------------- events ----------------
    eng = MetricsEngine(fps=fps, metric=metric, possession_dist=2.0 if metric else 70.0,
                        max_speed_mps=cfg.max_speed_mps)
    eng.pitch_length, eng.pitch_width = cfg.pitch.length, cfg.pitch.width
    samples: List[Sample] = []
    sample_frames: List[int] = []
    footprints: List = []
    step = max(1, int(round(fps / 5)))
    possession_trace = []
    for f in range(n):
        players = {i: (x, y) for i, (x, y, _, _) in pos[f].items()}
        teams = {i: team_of.get(i, 0) for i in players}
        b = ball_xyz.get(f)
        observed_ball = (b[0], b[1]) if (b is not None and not ball_px.get(f, (0, 0, True))[2]
                                         and b[2] < 1.5) else None
        eng.update(f, players, teams, observed_ball)
        holder = eng._holder_id
        # One definition everywhere: the side in control (hysteresis state of
        # MetricsEngine), counted only while the ball is actually located.
        control = int(eng._cur_team) if b is not None else 0
        possession_trace.append(control)
        if f % step == 0:
            people = {i: (x, y, team_of.get(i, 0), sp) for i, (x, y, sp, _) in pos[f].items()}
            people.update({i: (x, y, REFEREE_TEAM, float("nan")) for i, (x, y) in referees[f].items()})
            samples.append(Sample(f / fps, people, b, control, holder))
            sample_frames.append(f)
            footprints.append(_camera_footprint(raw[f].H, W, Hh, cfg) if metric else None)

    coach = CoachEngine(cfg.pitch.length, cfg.pitch.width)
    coach.attack_dir = coach._infer_attack_dir(samples)
    decisions = _decisions(samples, sample_frames, velocity, coach.attack_dir, cfg)
    plan = _game_plan(samples, sample_frames, velocity, decisions, coach.attack_dir, cfg, team_names)
    events = _events(eng, samples, ball_xyz, pos, team_of, fps, cfg, ball_px, n, coach.attack_dir)
    checks = _checks(linked, tracklets, ident, role_of, team_of, ball_px, fps, n)
    events += checks

    vest_links: List[dict] = []
    if vest_logs:
        found, vest_links = _link_vests(vest_logs, pos, velocity, fps, video_start_unix, video)
        vest_series = {**(vest_series or {}), **found}
    players_out = _player_summaries(pos, team_of, role_of, fps, cfg)
    numbers = _read_numbers(video, boxes_out, n, lens_k1, cfg.device)
    # one shirt number per side at a time: when two identities on the same side
    # read the same number while both are on the pitch, keep the clearer reading
    seen_at = {}
    for f, row in pos.items():
        for i in row:
            seen_at.setdefault(i, set()).add(f)
    by_num: Dict[Tuple[int, int], List[int]] = {}
    for i, (num, share) in numbers.items():
        by_num.setdefault((team_of.get(i, 0), num), []).append(i)
    for (side, num), ids in by_num.items():
        ids.sort(key=lambda i: -numbers[i][1])
        for j in ids[1:]:
            if side in (1, 2) and any(len(seen_at.get(j, set()) & seen_at.get(k, set())) > fps for k in ids[:1]):
                numbers.pop(j, None)
    for pl in players_out:
        if pl["id"] in numbers:
            pl["number"], pl["number_share"] = numbers[pl["id"]]
    for link in vest_links:
        for pl in players_out:
            if link["confident"] and pl["id"] == link["track"]:
                pl["vest"] = link["vest"]
    quality = {
        "frames": n, "fps": fps, "metric": metric,
        "calibration_coverage": round(sum(r.H is not None for r in raw) / max(n, 1), 3) if metric else 0.0,
        "ball_seen": round(sum(1 for f in range(n) if f in ball_px and not ball_px[f][2]) / max(n, 1), 3),
        "ball_located_incl_interpolated": round(len(ball_px) / max(n, 1), 3),
        "identities": len(set(ident.values())),
        "raw_tracks": len(tracklets),
        "links": len(linked.links),
        "uncertain_links": sum(l.uncertain for l in linked.links),
        "appearance_model": "soccer_reid_r18" if reid.available else "colour_histogram",
        "pins_rejected_uncertain": int(uncertain_pins),
        "position_funnel": funnel,
        "calibration_line_refinements": int(line_refinements),
        "calibration_recoveries": int(line_recoveries),
        "calibration_smoothing": h_smoothing,
        "ball_positions_withheld": int(dropped),
        "camera_mode": "fixed" if fixed else "panning",
        "lens_k1": float(lens_k1),
        "team_separation": team_info,
        "checks_pending": sum(1 for e in events if e["status"] == "check"),
        "processing_s": round(time.time() - t_start, 1),
    }
    insights = [c.as_dict() for c in coach.analyse(samples, vest_series, quality)] if metric else []
    load = assess_fatigue(samples, vest_series, attack_dir=coach.attack_dir, width=cfg.pitch.width) if metric \
        else {"available": False, "reason": "needs a pitch calibration"}
    insights += _fatigue_cards(load, team_names)
    insights = sorted(insights + plan.pop("cards", []), key=lambda c: -c["priority"])
    analysis = {
        "schema": SCHEMA,
        "match": {"name": match_name or Path(video).stem, "video": Path(video).name,
                  "pitch_length_m": cfg.pitch.length, "pitch_width_m": cfg.pitch.width,
                  "pitch_kind": cfg.pitch.kind, "goal_width_m": cfg.pitch.goal_width,
                  "duration_s": round(n / fps, 2), "fps": fps, "width": W, "height": Hh},
        "provenance": {"generated_at": time.time(), "detector": Path(cfg.weights).as_posix(),
                       "tracker": "imm-eiou+cmc+" + quality["appearance_model"] + "+offline-link",
                       "ball": "dp-path + ballistic 3D (Grad et al. 2026)", "wearable": bool(vest_series),
                       "synthetic": False},
        "quality": quality,
        "teams": {str(k): {"name": team_names[k - 1], "colour": TEAM_COLOURS[k],
                           "attack_dir": coach.attack_dir.get(k, 1)} for k in (1, 2)},
        "players": players_out,
        "vests": vest_links,
        "timeline": _timeline(samples, ball_px, fps, step, footprints, sample_frames, velocity, decisions),
        # every frame, so 3D draws the real path and arcs, not 5 Hz chords
        "ball_track": [[round(f / fps, 3), round(x, 2), round(y, 2), round(z, 2), 0 if ball_px.get(f, (0, 0, True))[2] else 1]
                       for f, (x, y, z) in sorted(ball_xyz.items())],
        "plan": plan,
        "load": load,
        "possession": _possession_summary(possession_trace, fps),
        "events": sorted(events, key=lambda e: e["t"]),
        "ball3d": {"segments": ball3d_segments},
        "team_stats": _team_stats(eng, samples, players_out, team_of, fps),
        "insights": insights,
    }
    (out / "analysis.json").write_text(json.dumps(analysis, allow_nan=False, default=_json_default))
    if render:
        _render(video, out / "annotated.mp4", boxes_out, team_of, role_of, ball_px, fps, n, lens_k1)
    print(f"[analyze] wrote {out / 'analysis.json'} in {time.time() - t_start:.0f}s", flush=True)
    return analysis


# ---------------------------------------------------------------------------
def _tracker_params(reid: bool) -> dict:
    """Operating points selected on the 9 held-out validation clips."""
    if reid:
        return dict(use_appearance=True, app_veto=0.65, app_weight=0.5)
    return dict(use_appearance=True, app_veto=0.55, app_weight=0.25)


def _linker_params(reid: bool) -> dict:
    if reid:
        return dict(max_gap=750, motion_gate=1.2, app_gate=0.5, interpolate_gap=20)
    return dict(max_gap=60, motion_gate=0.6, app_gate=0.15, interpolate_gap=20)





def _despike_ground_ball(ball_xyz: Dict[int, Tuple[float, float, float]], k: int = 3) -> int:
    """Hampel filter on balls on the ground: a position more than 3 robust
    deviations (and 0.6 m) from the median of its +-3 frame neighbours is a
    detection flick or a gap fill overshoot, and takes the median instead."""
    frames = sorted(f for f, v in ball_xyz.items() if v[2] < 0.05)
    xy = np.asarray([ball_xyz[f][:2] for f in frames], float)
    fixed = 0
    for i, f in enumerate(frames):
        nb = [j for j in range(max(0, i - k), min(len(frames), i + k + 1)) if abs(frames[j] - f) <= k]
        if len(nb) < 5:
            continue
        med = np.median(xy[nb], axis=0)
        dev = np.hypot(*(xy[nb] - med).T)
        mad = 1.4826 * np.median(dev)
        if np.hypot(*(xy[i] - med)) > 3 * mad + 0.6:
            ball_xyz[f] = (float(med[0]), float(med[1]), 0.0)
            fixed += 1
    return fixed


def _drop_impossible_ball(ball_xyz: Dict[int, Tuple[float, float, float]], fps: float, vmax: float = 45.0) -> int:
    """Last guard: a ground position that would need the ball to travel faster
    than any ball does since the previous located frame is not shown."""
    kept_f, kept, dropped = None, None, 0
    for f in sorted(ball_xyz):
        x, y, z = ball_xyz[f]
        if kept is not None and f - kept_f <= 3 and z < 0.05:
            v = float(np.hypot(x - kept[0], y - kept[1])) * fps / (f - kept_f)
            if v > vmax:
                del ball_xyz[f]
                dropped += 1
                continue
        kept_f, kept = f, (x, y)
    return dropped

def _contact_flights(ball_xyz: Dict[int, Tuple[float, float, float]], fps: float,
                     roll_max: float = 30.0) -> List[Tuple[int, int]]:
    """Rebuild balls that 'roll' faster than any ball can as flights.

    The pitch mapping places an airborne ball where its line of sight meets
    the grass, far behind the real ball, so it seems to race across the pitch
    (up to 300 m/s on SNMOT-074). No kicked ball rolls faster than about
    30 m/s, so such a stretch is a flight. Between take-off and landing,
    where the ball is on the ground and well placed, gravity fixes the
    motion: horizontal position moves at constant speed and height follows
    z = g/2 (t - t0)(t1 - t), whose apex is g T^2 / 8 for a flight of T
    seconds. The unreliable positions in between are replaced; the ends are
    kept as measured."""
    frames = sorted(ball_xyz)
    if len(frames) < 6:
        return []
    sp = {}
    for f0, f1 in zip(frames[:-1], frames[1:]):
        if 0 < f1 - f0 <= 2:
            a, b = np.asarray(ball_xyz[f0]), np.asarray(ball_xyz[f1])
            sp[f1] = float(np.hypot(*(b[:2] - a[:2]))) * fps / (f1 - f0)
    fast = [f for f in frames if sp.get(f, 0.0) > roll_max]
    out, used = [], -1
    for f in fast:
        if f <= used:
            continue
        a_f = f
        while a_f - 1 in ball_xyz and sp.get(a_f, 99.0) > 15.0 and f - a_f < fps * 3:
            a_f -= 1                                     # back to the kick: slow, on the ground
        p0 = np.asarray(ball_xyz[a_f][:2])
        # Landing: the first stretch of 6 calm frames (0.24 s) from which a free
        # flight is possible. The shadow of a ball moving away and then back
        # towards the camera also slows to a stop at the top of the flight,
        # but only for a frame or three, so a short calm spell is not a landing.
        landing, b_f, calm = None, f, 0
        while b_f + 1 in ball_xyz and b_f - f < fps * 5:
            b_f += 1
            calm = calm + 1 if sp.get(b_f, 99.0) < 20.0 else 0
            if calm >= 6:
                cand = b_f - 5
                T = (cand - a_f) / fps
                if 0.24 <= T <= 5.0 and np.hypot(*(np.asarray(ball_xyz[cand][:2]) - p0)) / T <= 36.0:
                    landing = cand
                    break
        if landing is None:
            used = b_f
            continue
        T = (landing - a_f) / fps
        p1 = np.asarray(ball_xyz[landing][:2])
        for g in range(a_f, landing + 1):
            u = (g - a_f) / fps
            xy = p0 + (p1 - p0) * (u / T)
            ball_xyz[g] = (float(xy[0]), float(xy[1]), float(0.5 * 9.81 * u * (T - u)))
        out.append((a_f, landing))
        used = landing
    return out

def _smooth_homographies(raw, W: int, Hh: int, fps: float) -> dict:
    """Remove snap jumps from the per-frame pitch mapping of a moving camera.

    Line snapping corrects drift, but each correction can move points near the
    far goal by metres in one frame (measured: 4 to 9 m on SNMOT-074), which
    shakes every player and the ball. A camera pans smoothly, so a grid of
    image points is followed through each frame's mapping onto the pitch,
    each point's path gets a Hampel filter (spikes) and a Savitzky-Golay
    filter (1 s, quadratic: keeps real pans), and every frame's homography is
    refitted to the smoothed points.
    """
    from scipy.signal import savgol_filter
    idx = [f for f, r in enumerate(raw) if r.H is not None]
    if len(idx) < 15:
        return {}
    gx, gy = np.meshgrid(np.linspace(0.08, 0.92, 7) * W, np.linspace(0.35, 0.95, 5) * Hh)
    img = np.column_stack((gx.ravel(), gy.ravel())).astype(np.float64)
    P = np.full((len(raw), len(img), 2), np.nan)
    for f in idx:
        q = apply_homography(raw[f].H, img)
        if np.isfinite(q).all() and np.abs(q).max() < 1e4:
            P[f] = q
    valid = np.isfinite(P[:, 0, 0])
    k = max(3, int(round(fps * 0.24)))
    S = P.copy()
    for j in range(len(img)):
        for c in (0, 1):
            x = P[:, j, c]
            ok = np.isfinite(x)
            if ok.sum() < 15:
                continue
            xi = np.interp(np.arange(len(x)), np.nonzero(ok)[0], x[ok])
            med = np.array([np.median(xi[max(0, i - k):i + k + 1]) for i in range(len(xi))])
            mad = np.array([np.median(np.abs(xi[max(0, i - k):i + k + 1] - med[i])) for i in range(len(xi))])
            spike = np.abs(xi - med) > 3 * 1.4826 * mad + 0.3
            xi[spike] = med[spike]
            win = min(len(xi) - (1 - len(xi) % 2), max(5, int(fps) | 1))
            S[:, j, c] = np.where(ok, savgol_filter(xi, win, 2) if win >= 5 else xi, np.nan)
    moved = []
    for f in idx:
        if not valid[f]:
            continue
        Hs, _ = cv2.findHomography(img, S[f], 0)
        if Hs is not None and np.isfinite(Hs).all():
            moved.append(float(np.median(np.hypot(*(S[f] - P[f]).T))))
            raw[f] = _RawFrame(raw[f].tracks, raw[f].balls, Hs)
    return {"frames": len(moved), "median_correction_m": round(float(np.median(moved)), 3) if moved else 0.0,
            "p95_correction_m": round(float(np.percentile(moved, 95)), 3) if moved else 0.0}

def _robust_track_mask(frames: np.ndarray, xy: np.ndarray, fps: float, vmax: float,
                       half_window: int = 6) -> np.ndarray:
    """True for samples consistent with their temporal neighbours.

    A sample is an outlier when it lies further from the median of its +-6
    frame neighbourhood than a player could plausibly move in that time plus
    a 1.5 m calibration allowance (Hampel-style filter, robust to single-frame
    projection glitches and identity hand-overs).
    """
    keep = np.ones(len(frames), bool)
    if len(frames) < 5:
        return keep
    for i in range(len(frames)):
        lo, hi = max(0, i - half_window), min(len(frames), i + half_window + 1)
        nb = np.abs(frames[lo:hi] - frames[i]) <= half_window
        if nb.sum() < 3:
            continue
        med = np.median(xy[lo:hi][nb], axis=0)
        reach = 1.5 + vmax * half_window / fps * 0.5
        if np.hypot(*(xy[i] - med)) > reach:
            keep[i] = False
    return keep


def _pin_sigma(H: np.ndarray, foot: np.ndarray, box: np.ndarray) -> float:
    """Pitch-position standard deviation (m) of a foot pin.

    First-order propagation Sigma_pitch = J Sigma_u J^T of the image foot-point
    noise through the homography (enhancement guide D.3). Pixel noise grows
    with box height: the lowest box pixel jumps between feet during a stride
    (sigma_u = 6 % of the box height, at least 1.5 px).
    """
    h = float(box[3] - box[1])
    su = max(1.5, 0.06 * h)
    eps = 1.0
    base = apply_homography(H, [foot])[0]
    jx = (apply_homography(H, [foot + [eps, 0]])[0] - base) / eps
    jy = (apply_homography(H, [foot + [0, eps]])[0] - base) / eps
    J = np.column_stack((jx, jy))
    if not np.isfinite(J).all():
        return float("inf")
    cov = J @ np.diag([su * su * 0.25, su * su]) @ J.T
    return float(np.sqrt(np.trace(cov)))


def _smooth_kinematics(pos, fps, vmax, window_s: float = 0.6):
    """Offline, centred Savitzky-Golay smoothing of each identity's pitch track.

    Frame-level foot points jitter by tens of centimetres, which inflates
    instantaneous speed (a 0.3 m jitter at 25 fps is 7.5 m/s). Recorded footage
    allows a symmetric window, so position and speed are smoothed jointly on
    each contiguous run; runs shorter than the window keep a robust two-point
    estimate. Speeds above the physical cap are withheld, not clipped.
    """
    from scipy.signal import savgol_filter
    win = max(5, int(round(window_s * fps)) | 1)
    per: Dict[int, List[int]] = {}
    for f, row in pos.items():
        for i in row:
            per.setdefault(i, []).append(f)
    out = {f: dict(row) for f, row in pos.items()}
    velocity: Dict[int, Dict[int, Tuple[float, float]]] = {f: {} for f in pos}
    for i, frames in per.items():
        frames = sorted(frames)
        runs, cur = [], [frames[0]]
        for f in frames[1:]:
            if f == cur[-1] + 1:
                cur.append(f)
            else:
                runs.append(cur)
                cur = [f]
        runs.append(cur)
        for run in runs:
            xy = np.asarray([pos[f][i][:2] for f in run], float)
            if len(run) >= win:
                sm = savgol_filter(xy, win, 2, axis=0)
                vel = savgol_filter(xy, win, 2, deriv=1, delta=1.0 / fps, axis=0)
                speed = np.hypot(vel[:, 0], vel[:, 1])
            else:
                sm = xy
                vel = np.zeros_like(xy)
                speed = np.full(len(run), np.nan)
                if len(run) >= 3:
                    v = np.hypot(*(xy[-1] - xy[0])) / ((len(run) - 1) / fps)
                    speed[:] = v
            for k, f in enumerate(run):
                sp = float(speed[k]) if np.isfinite(speed[k]) and speed[k] <= vmax else float("nan")
                out[f][i] = (float(sm[k, 0]), float(sm[k, 1]), sp, pos[f][i][3])
                if np.isfinite(sp):
                    velocity[f][i] = (float(vel[k, 0]), float(vel[k, 1]))
    return out, velocity


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))





def _video_start_unix(video: str) -> Optional[float]:
    """Recording start from the container's creation_time, when the camera wrote one."""
    import subprocess
    from datetime import datetime
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time",
                              "-of", "default=nw=1:nk=1", video], capture_output=True, text=True, timeout=10).stdout.strip()
        return datetime.fromisoformat(out.replace("Z", "+00:00")).timestamp() if out else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None



def _read_numbers(video, boxes_out, n, lens_k1, device, every: int = 3, max_crops: int = 64):
    """Shirt number per identity from the trained reader, decided per track."""
    from .jersey_model import JerseyReader
    reader = JerseyReader(device="cpu" if str(device) == "cpu" else "")
    if not reader.available:
        return {}
    vote, seen = reader.vote(), {}
    cap = cv2.VideoCapture(video)
    f = 0
    while f < n:
        ok, frame = cap.read()
        if not ok:
            break
        if f % every == 0:
            rows = [(i, b) for i, b, interp in boxes_out.get(f, []) if not interp and seen.get(i, 0) < max_crops
                    and (b[3] - b[1]) >= 40]                    # too small to carry a readable number
            if rows:
                frame = undistort_frame(frame, lens_k1)
                out = reader.read_batch(frame, np.asarray([b for _, b in rows]))
                for k, (i, _) in enumerate(rows):
                    vote.add(int(i), out, k)
                    seen[i] = seen.get(i, 0) + 1
        f += 1
    cap.release()
    out = {}
    for i in seen:
        num, share = vote.decide(i)
        if num is not None:
            out[i] = (num, round(share, 2))
    return out


def _link_vests(paths, pos, velocity, fps, video_start_unix, video):
    """Find the wearer of each vest log and put its pulse on the match clock."""
    from .sensors import vest_log
    from .sensors.association import associate, imu_envelope, track_envelope
    start = video_start_unix if video_start_unix is not None else _video_start_unix(video)
    # a clock we set ourselves (live capture) is good to a second; container
    # metadata from a phone only to a few seconds
    max_lag = 2.0 if video_start_unix is not None else 10.0
    series: Dict[int, List[Tuple[float, float]]] = {}
    for f in sorted(pos):
        for i in pos[f]:
            v = velocity.get(f, {}).get(i)
            if v is not None:
                series.setdefault(i, []).append((f / fps, float(np.hypot(*v))))
    tracks = {i: track_envelope(np.asarray(r)[:, 0], np.asarray(r)[:, 1]) for i, r in series.items() if len(r) > fps * 10}
    links, out = [], {}
    for k, path in enumerate(paths):
        log = vest_log.load(path)
        vid = int(log.player_id or k + 1)
        if start is None or not len(log.imu_t):
            links.append({"vest": vid, "track": None, "confident": False, "reason": "no recording start time; pass --video-start"})
            continue
        m = associate({vid: imu_envelope(log.imu_t - start, log.accel)}, tracks, max_lag_s=max_lag)[vid]
        links.append({"vest": vid, "track": m.track, "score": m.score, "margin": m.margin, "lag_s": m.lag_s,
                      "overlap_s": m.overlap_s, "confident": m.confident, "candidates": m.candidates,
                      "calibration": log.calibration})
        if m.confident:
            t = log.hr_t - start - m.lag_s
            pl = np.interp(t, log.load_t - start - m.lag_s, log.player_load) if len(log.load_t) else None
            out[m.track] = VestSeries(t, log.hr, log.hr_quality, pl)
    return out, links

def _fatigue_cards(load: dict, team_names) -> List[dict]:
    """One card per suggested substitution, with the measured reasons."""
    cards = []
    if not load.get("available"):
        return cards
    by = {p["id"]: p for p in load["players"]}
    for s in load["subs"]:
        p = by[s["out"]]
        ev = [{"label": "Now", "value": f"{p['now']:.0f} m/min"}, {"label": "Own peak", "value": f"{p['peak']:.0f} m/min"}]
        if p.get("minutes_to_half_peak") is not None:
            ev.append({"label": "Half of peak in", "value": f"{p['minutes_to_half_peak']:.0f} min"})
        cards.append({
            "id": f"sub-{s['out']}", "kind": "physical", "team": s["team"],
            "title": f"Consider replacing {s['out']} {s['when']}",
            "finding": "; ".join(s["why"]).capitalize() + ". The drop is larger than the team's over the same minutes.",
            "action": f"Bring on a like-for-like {s['role']} player, or move {s['out']} to a role with fewer repeated runs.",
            "confidence": "high" if p["minutes_seen"] >= 20 else "medium" if p["minutes_seen"] >= 8 else "low",
            "evidence": ev, "moments": [], "players": [s["out"]], "objective": "Obj 3",
            "priority": 0.5 + 0.4 * s["score"]})
    return cards

def _keepers_by_side(pos, team_of, role_of, team_info, length):
    """Give each goalkeeper the side that defends his goal.

    Keepers wear a third kit, so colour clustering rightly abstains on them,
    but a keeper with no side never counts for ball control or shape. The
    offside law makes the defending side's last outfield player the one
    nearest its own goal almost all the time, so each frame votes for the side
    whose deepest outfield player is nearest the goal the keeper stands in.
    """
    kx: Dict[int, List[float]] = {}
    for frame in pos.values():
        for i, (x, *_r) in frame.items():
            if role_of.get(i) == GOALKEEPER and not team_of.get(i):
                kx.setdefault(i, []).append(x)
    done = 0
    for i, xs in kx.items():
        if len(xs) < 10:
            continue
        mx = float(np.median(xs))
        if min(mx, length - mx) > 25.0:          # not near a goal: no basis
            continue
        goal = 0.0 if mx < length / 2 else length
        votes = {1: 0, 2: 0}
        for frame in pos.values():
            if i not in frame:
                continue
            deep = {}
            for j, (x, *_r) in frame.items():
                t = team_of.get(j)
                if t in (1, 2) and role_of.get(j) == PLAYER:
                    deep[t] = min(deep.get(t, 1e9), abs(x - goal))
            if len(deep) == 2 and abs(deep[1] - deep[2]) > 0.5:
                votes[min(deep, key=deep.get)] += 1
        total = votes[1] + votes[2]
        if total >= 10 and max(votes.values()) >= 0.7 * total:
            team_of[i] = max(votes, key=votes.get)
            done += 1
    if done:
        team_info["keepers_by_side"] = done

def _assign_teams(id_cols, role_of, team: TeamAssigner):
    """Cluster identities (not frames) by median kit colour; abstain on outliers.

    Features are CIELAB with lightness down-weighted by half: chroma separates
    most kit pairs, but dark-vs-white kits differ mainly in lightness, which a
    chroma-only model misses. A trimmed 2-means drops referees, keepers and
    occlusion blends (far from both centres) before the final fit, so a bright
    third kit cannot capture a cluster. Calibrated kit seeds from the review
    desk, when present, decide which cluster is side 1.
    """
    ids = [i for i, c in id_cols.items() if len(c) >= 3 and role_of.get(i) == PLAYER]
    if len(ids) < 4:
        return {}, {"method": "insufficient", "assigned": 0}
    med = np.asarray([np.median(np.asarray(id_cols[i]), axis=0) for i in ids], float)
    feat = np.column_stack((0.5 * med[:, 0], med[:, 1], med[:, 2]))
    weight = np.asarray([len(id_cols[i]) for i in ids], float)

    # Trimmed 2-means (Cuesta-Albertos, Gordaliza & Matrán 1997): every pair
    # of identities seeds a run, the 20% of identities farthest from their
    # centre are ignored when re-estimating, and the lowest trimmed inertia wins.
    # Referees and keepers therefore cannot capture one of the two clusters.
    n = len(ids)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng = np.random.default_rng(0)
    if len(pairs) > 300:
        pairs = [pairs[k] for k in rng.choice(len(pairs), 300, replace=False)]
    best = None
    for i, j in pairs:
        C = feat[[i, j]].copy()
        for _ in range(15):
            dd = np.sqrt(((feat[:, None] - C[None]) ** 2).sum(2))
            near = dd.min(1)
            kept = near <= np.quantile(near, 0.8)
            lab = dd.argmin(1)
            newC = np.stack([np.average(feat[kept & (lab == k)], axis=0, weights=weight[kept & (lab == k)])
                             if (kept & (lab == k)).any() else C[k] for k in range(2)])
            if np.allclose(newC, C):
                break
            C = newC
        dd = np.sqrt(((feat[:, None] - C[None]) ** 2).sum(2))
        near = dd.min(1)
        kept = near <= np.quantile(near, 0.8)
        cost = float((weight[kept] * near[kept] ** 2).sum())
        balanced = min((dd.argmin(1)[kept] == 0).sum(), (dd.argmin(1)[kept] == 1).sum()) >= 2
        if balanced and (best is None or cost < best[0]):
            best = (cost, C)
    if best is None:
        return {}, {"method": "insufficient", "assigned": 0}
    C = best[1]
    d = np.sqrt(((feat[:, None] - C[None]) ** 2).sum(2))
    near = d.min(1)
    core = near <= np.quantile(near, 0.8)
    spread = np.median(near[core]) + 1e-6
    keep = near <= 3.0 * spread + 6
    lab = d.argmin(1)
    if team._seeds is not None:
        s1 = np.r_[0.5 * team._seeds[1][0], team._seeds[1][1:]]
        s2 = np.r_[0.5 * team._seeds[2][0], team._seeds[2][1:]]
        if np.linalg.norm(C[0] - s2) + np.linalg.norm(C[1] - s1) < np.linalg.norm(C[0] - s1) + np.linalg.norm(C[1] - s2):
            lab = 1 - lab
            C = C[::-1]
    margin = np.abs(d[:, 0] - d[:, 1])
    out = {}
    for k, i in enumerate(ids):
        out[i] = 0 if (not keep[k] or margin[k] < 3.0) else int(lab[k]) + 1
    sep = float(np.linalg.norm(C[0] - C[1]))
    return out, {"method": "identity-level trimmed 2-means on CIELAB (L/2, a, b)",
                 "assigned": int(sum(v > 0 for v in out.values())),
                 "abstained": int(sum(v == 0 for v in out.values())), "kit_separation": round(sep, 1)}


def _assign_teams_embedding(id_vec: Dict[int, np.ndarray], id_n: Dict[int, int], role_of,
                            min_obs: int = 10):
    """Split identities into two sides using their mean ReID embedding.

    Within one match the dominant direction of appearance variation between
    outfield players is the kit, so a weighted 2-means on L2-normalised
    identity embeddings separates the sides; identities far from both centres
    (referees, keepers in a third kit) abstain.
    """
    ids = [i for i, v in id_vec.items() if id_n.get(i, 0) >= min_obs and role_of.get(i) == PLAYER
           and np.isfinite(v).all() and np.linalg.norm(v) > 0]
    if len(ids) < 4:
        return {}, {"method": "insufficient", "assigned": 0}
    X = np.stack([id_vec[i] / np.linalg.norm(id_vec[i]) for i in ids])
    w = np.asarray([id_n[i] for i in ids], float)
    # deterministic farthest-pair initialisation, weighted Lloyd iterations
    mean = np.average(X, axis=0, weights=w)
    a = X[np.argmax(((X - mean) ** 2).sum(1))]
    b = X[np.argmax(((X - a) ** 2).sum(1))]
    C = np.stack([a, b])
    for _ in range(20):
        lab = np.argmax(X @ C.T, axis=1)
        newC = np.stack([np.average(X[lab == k], axis=0, weights=w[lab == k]) if (lab == k).any() else C[k]
                         for k in range(2)])
        newC /= np.linalg.norm(newC, axis=1, keepdims=True)
        if np.allclose(newC, C, atol=1e-6):
            break
        C = newC
    sim = X @ C.T
    lab = np.argmax(sim, axis=1)
    best = sim[np.arange(len(ids)), lab]
    margin = np.abs(sim[:, 0] - sim[:, 1])
    cut = np.median(best) - 2.5 * (np.median(np.abs(best - np.median(best))) + 1e-3)
    out = {}
    for k, i in enumerate(ids):
        out[i] = 0 if (best[k] < cut or margin[k] < 0.02) else int(lab[k]) + 1
    return out, {"method": "identity-level ReID embedding clustering", "assigned": int(sum(v > 0 for v in out.values())),
                 "abstained": int(sum(v == 0 for v in out.values())),
                 "kit_separation": round(float(1 - C[0] @ C[1]), 3)}


def _events(eng, samples, ball_xyz, pos, team_of, fps, cfg, ball_px, n, attack_dir):
    ev = []
    for k, p in enumerate(eng.passes):
        ev.append({"id": f"pass-{k}", "type": "pass", "t": round(p.frame / fps, 2), "frame": int(p.frame),
                   "team": int(p.team), "players": [int(p.from_id), int(p.to_id)],
                   "x": float(p.end[0]), "y": float(p.end[1]), "from": [float(p.start[0]), float(p.start[1])],
                   "status": "observed", "confidence": "medium"})
    # turnovers from the possession state of the samples
    prev = 0
    for s in samples:
        if s.possession in (1, 2):
            if prev in (1, 2) and s.possession != prev and s.ball is not None:
                ev.append({"id": f"turnover-{len(ev)}", "type": "turnover", "t": round(s.t, 2),
                           "frame": int(round(s.t * fps)), "team": int(s.possession), "lost_by": int(prev),
                           "players": [int(s.holder)] if s.holder is not None else [],
                           "x": float(s.ball[0]), "y": float(s.ball[1]), "status": "observed", "confidence": "medium"})
            prev = s.possession
    # shots: fast ball heading at a goal mouth, scored with xG and 3D breakdown
    xg = XGModel()
    frames = sorted(ball_xyz)
    last_shot = -1e9
    for a, b in zip(frames[:-1], frames[1:]):
        if b - a != 1 or b - last_shot < fps * 2:
            continue
        p0, p1 = np.asarray(ball_xyz[a]), np.asarray(ball_xyz[b])
        v = (p1 - p0) * fps
        speed = float(np.linalg.norm(v[:2]))
        if speed < 12.0:
            continue
        # A strike: the ball stays fast for four frames (not a one-frame jump
        # between two candidates) and leaves a player's feet.
        ahead = [ball_xyz.get(a + k) for k in range(5)]
        if any(q is None for q in ahead):
            continue
        sustained = np.linalg.norm(np.diff(np.asarray(ahead)[:, :2], axis=0), axis=1) * fps
        if np.median(sustained) < 12.0 or sustained.min() < 6.0:
            continue
        # 50 m/s (180 km/h) is beyond any recorded shot; faster means the 3D fit jumped
        if sustained.max() > 50.0:
            continue
        near = {i: np.hypot(x - p0[0], y - p0[1]) for i, (x, y, _, _) in pos.get(a, {}).items()}
        if not near or min(near.values()) > 2.5:
            continue
        kicker = min(near, key=near.get)
        # direction from the net movement over five frames, not one noisy step
        v = (np.asarray(ahead[-1]) - p0) * fps / (len(ahead) - 1)
        for goal_x in (0.0, cfg.pitch.length):
            dx = goal_x - p0[0]
            if dx == 0 or abs(v[0]) < 1e-6 or np.sign(dx) != np.sign(v[0]) or abs(dx) > 35:
                continue
            y_at = p0[1] + v[1] * dx / v[0]
            if abs(y_at - cfg.pitch.width / 2) > max(cfg.pitch.goal_width, 3.0):
                continue
            # the kicker must be on the side attacking this goal: rules out goal
            # kicks and clearances that happen to point at the far goal
            toward_side = 1 if attack_dir.get(1, 1) == (1 if goal_x > 0 else -1) else 2
            if team_of.get(kicker) in (1, 2) and team_of.get(kicker) != toward_side:
                continue
            seg = [ball_xyz[f] for f in range(a, min(a + 15, n)) if f in ball_xyz]
            sp = [np.linalg.norm(np.asarray(seg[i + 1]) - np.asarray(seg[i])) * fps for i in range(len(seg) - 1)]
            toward = 1 if goal_x > 0 else -1
            shooter_team = 1 if attack_dir.get(1, 1) == toward else 2
            defenders = [(x, y) for i, (x, y, _, _) in pos.get(a, {}).items() if team_of.get(i) not in (0, shooter_team)]
            m = shot_metrics(np.asarray(seg), np.r_[np.nan, sp], goal_x, cfg.pitch.width, np.asarray(defenders) if defenders else None)
            seen = np.mean([f in ball_px and not ball_px[f][2] for f in range(max(0, a - 10), min(n, a + 10))])
            ev.append({"id": f"shot-{len(ev)}", "type": "shot", "t": round(a / fps, 2), "frame": int(a),
                       "team": shooter_team, "players": [], "x": float(p0[0]), "y": float(p0[1]),
                       "xg": round(xg.xg(p0[0], p0[1], goal_x, cfg.pitch.width, cfg.pitch.goal_width), 3), "breakdown": m,
                       "status": "observed" if seen >= 0.7 else "check",
                       "confidence": "medium" if seen >= 0.7 else "low",
                       "reason": None if seen >= 0.7 else "ball partly hidden around the shot"})
            last_shot = b
            break
    # sprints (>= 1 s above SPRINT)
    for pid in {i for row in pos.values() for i in row}:
        run, start = 0, None
        for f in range(n):
            sp = pos[f].get(pid, (0, 0, 0, False))[2]
            if sp >= SPRINT:
                run += 1
                start = f if start is None else start
            else:
                if run >= fps:
                    x, y = pos[start][pid][:2]
                    ev.append({"id": f"sprint-{pid}-{start}", "type": "sprint", "t": round(start / fps, 2),
                               "frame": int(start), "team": int(team_of.get(pid, 0)), "players": [int(pid)],
                               "x": float(x), "y": float(y), "duration_s": round(run / fps, 1),
                               "status": "observed", "confidence": "medium"})
                run, start = 0, None
    return ev


def _checks(linked, tracklets, ident, role_of, team_of, ball_px, fps, n):
    out = []
    for k, l in enumerate(linked.links):
        if l.uncertain:
            fid = ident[l.to_id]
            f = tracklets[l.to_id].start
            out.append({"id": f"check-link-{k}", "type": "identity", "t": round(f / fps, 2), "frame": int(f),
                        "team": int(team_of.get(fid, 0)), "players": [int(fid)],
                        "detail": f"Re-joined after {l.gap_frames / fps:.1f}s out of view",
                        "gap_frames": int(l.gap_frames), "status": "check", "confidence": "low"})
    # missing ball spells
    gap_start = None
    for f in range(n + 1):
        seen = f < n and f in ball_px
        if not seen and gap_start is None:
            gap_start = f
        elif seen and gap_start is not None:
            if f - gap_start >= 3 * fps:
                out.append({"id": f"ball-gap-{gap_start}", "type": "ball_unseen", "t": round(gap_start / fps, 2),
                            "frame": int(gap_start), "team": 0, "players": [],
                            "detail": f"Ball not located for {(f - gap_start) / fps:.1f}s",
                            "status": "info", "confidence": "high"})
            gap_start = None
    unknown = [i for i, r in role_of.items() if r == PLAYER and team_of.get(i, 0) == 0]
    first_seen = {}
    for f in sorted(linked.frames):
        for fid in linked.frames[f][0]:
            first_seen.setdefault(int(fid), f)
    for i in unknown[:12]:
        f = first_seen.get(int(i), 0)
        out.append({"id": f"check-team-{i}", "type": "team", "t": round(f / fps, 2), "frame": int(f), "team": 0,
                    "players": [int(i)], "detail": "Kit colour did not match either team",
                    "status": "check", "confidence": "low"})
    return out


def _player_summaries(pos, team_of, role_of, fps, cfg, grid=(12, 8)):
    per: Dict[int, list] = {}
    for f, row in pos.items():
        for i, (x, y, sp, interp) in row.items():
            per.setdefault(i, []).append((f, x, y, sp, interp))
    out = []
    for i, rows in per.items():
        a = np.asarray(rows, float)
        obs = a[a[:, 4] == 0]
        if len(obs) < fps:                     # less than a second observed
            continue
        steps = np.hypot(np.diff(a[:, 1]), np.diff(a[:, 2]))
        gaps = np.diff(a[:, 0])
        dist = float(steps[(gaps == 1) & (steps < cfg.max_speed_mps / fps * 1.5)].sum())
        sp = a[:, 3]
        hsr = float((steps[(gaps == 1)] * (np.nan_to_num(sp[1:][gaps == 1]) >= HSR)).sum())
        heat, _, _ = np.histogram2d(obs[:, 1], obs[:, 2], bins=grid,
                                    range=[[0, cfg.pitch.length], [0, cfg.pitch.width]])
        heat = heat / max(heat.max(), 1)
        out.append({
            "id": int(i), "team": int(team_of.get(i, 0)), "role": {PLAYER: "player", GOALKEEPER: "goalkeeper"}.get(role_of.get(i), "player"),
            "label": f"{'AB'[team_of.get(i) - 1] if team_of.get(i) in (1, 2) else 'U'}{i}",
            "observed_s": round(len(obs) / fps, 1), "distance_m": round(dist, 1),
            "top_speed_kmh": _sustained_top_speed(a[:, 0], sp, fps),
            "hsr_m": round(hsr, 1),
            "avg_x": round(float(obs[:, 1].mean()), 1), "avg_y": round(float(obs[:, 2].mean()), 1),
            "heat": np.round(heat.T, 3).tolist(),
        })
    out.sort(key=lambda p: (p["team"] or 9, -p["observed_s"]))
    return out


def _sustained_top_speed(frames, speed, fps, window_s: float = 1.0):
    """Highest speed held for a full second (rolling median), in km/h.

    GPS vests report peak speed over ~1 s windows; a single-frame peak mostly
    measures calibration and foot-point jitter.
    """
    k = max(3, int(round(window_s * fps)))
    best = None
    f = np.asarray(frames, int)
    sp = np.asarray(speed, float)
    start = 0
    for i in range(1, len(f) + 1):
        if i == len(f) or f[i] != f[i - 1] + 1:
            run = sp[start:i]
            if len(run) >= k:
                med = np.asarray([np.nanmedian(run[j:j + k]) if np.isfinite(run[j:j + k]).sum() >= k // 2 else np.nan
                                  for j in range(len(run) - k + 1)])
                if np.isfinite(med).any():
                    v = float(np.nanmax(med))
                    best = v if best is None else max(best, v)
            start = i
    return None if best is None else round(best * 3.6, 1)


def _decisions(samples, sample_frames, velocity, attack_dir, cfg):
    """Pass options and the on-ball call for every moment with a ball carrier."""
    from .decision import decide, pass_options
    out = []
    for s, f in zip(samples, sample_frames):
        h = s.holder
        if h is None or h not in s.players or s.players[h][2] not in (1, 2):
            out.append(None)
            continue
        team = s.players[h][2]
        vel = velocity.get(f, {})
        goal_x = cfg.pitch.length if attack_dir.get(team, 1) > 0 else 0.0
        mates = {i: (x, y, *vel.get(i, (0.0, 0.0))) for i, (x, y, tm, _) in s.players.items() if tm == team and i != h}
        opps = [(x, y, *vel.get(i, (0.0, 0.0))) for i, (x, y, tm, _) in s.players.items() if tm in (1, 2) and tm != team]
        cx, cy = s.players[h][:2]
        opts = pass_options((cx, cy), mates, opps, goal_x, cfg.pitch.width, cfg.pitch.length, cfg.pitch.goal_width)
        pressure = sum(np.hypot(x - cx, y - cy) <= 5 for x, y, _, _ in opps)
        call = decide((cx, cy), opts, goal_x, cfg.pitch.width, pressure, cfg.pitch.goal_width)
        call.update(team=int(team), carrier=int(h), pressure=int(pressure))
        out.append({"options": [[o.to, o.x, o.y, o.p_arrive, o.value] for o in opts[:4]], "call": call})
    return out


def _game_plan(samples, sample_frames, velocity, decisions, attack_dir, cfg, team_names):
    """Formations, space conceded by zone, and progression routes, per side."""
    from .decision import pitch_control_grid
    from .tactics import estimate_formation
    L, Wd = cfg.pitch.length, cfg.pitch.width
    plan = {"teams": {}, "cards": []}
    nx, ny = 3, 3                          # thirds x channels
    for team in (1, 2):
        d = attack_dir.get(team, 1)
        names = team_names[team - 1]
        # formations from average positions in and out of possession
        phases = (("with_ball", lambda s: s.possession == team), ("without_ball", lambda s: s.possession not in (0, team)))

        def shape_of(rows, cond):
            acc: Dict[int, List[Tuple[float, float]]] = {}
            for s in rows:
                if cond(s):
                    for i, (x, y, tm, _) in s.players.items():
                        if tm == team:
                            acc.setdefault(i, []).append((x, y))
            keep = {i: np.mean(v, axis=0) for i, v in acc.items() if len(v) >= 10}
            visible = len(keep)
            return {"shape": estimate_formation([v[0] for v in keep.values()], [v[1] for v in keep.values()], d)
                    if visible >= 8 else None, "players_seen": visible,
                    "positions": [[int(i), round(float(v[0]), 1), round(float(v[1]), 1)] for i, v in keep.items()]}
        form = {phase: shape_of(samples, cond) for phase, cond in phases}
        # shape over time: 5 min windows for 25 min or more of play; a change
        # is reported only when the new shape holds for two windows running
        timeline = []
        if samples and samples[-1].t - samples[0].t >= 1500:
            t0 = samples[0].t
            while t0 + 300 <= samples[-1].t + 1e-6:
                rows = [s for s in samples if t0 <= s.t < t0 + 300]
                timeline.append({"t": round(t0, 1), **{ph: shape_of(rows, c)["shape"] for ph, c in phases}})
                t0 += 300
        for ph, label in (("without_ball", "out of possession"), ("with_ball", "in possession")):
            seq = [(w["t"], w[ph]) for w in timeline if w[ph]]
            for k in range(2, len(seq)):
                (_, a), (tb, b), (_, c) = seq[k - 2], seq[k - 1], seq[k]
                if b == c and b != a:
                    plan["cards"].append({"id": f"plan-shape-{team}-{ph}-{int(tb)}", "kind": "tactical", "team": team,
                        "title": f"{names} changed shape {label}: {a} to {b}",
                        "finding": f"From about {int(tb // 60)}:{int(tb % 60):02d} {names} lined up {b} {label}, after {a} earlier, and kept it.",
                        "action": "Check whether the opponent's attack changed at the same time and whether your plan still matches.",
                        "confidence": "medium", "evidence": [{"label": "Before", "value": a}, {"label": "After", "value": b}],
                        "moments": [float(tb)], "players": [], "objective": "Obj 3", "priority": 0.62})
        # space the opponent controls when this side defends, by zone (own-goal relative)
        space = np.zeros((ny, nx)); cnt = 0
        for k in range(0, len(samples), 5):
            s, f = samples[k], sample_frames[k]
            if s.possession in (0, team):
                continue
            vel = velocity.get(f, {})
            opp = [(x, y, *vel.get(i, (0, 0))) for i, (x, y, tm, _) in s.players.items() if tm not in (0, 3, team)]
            mine = [(x, y, *vel.get(i, (0, 0))) for i, (x, y, tm, _) in s.players.items() if tm == team]
            if len(opp) < 3 or len(mine) < 3:
                continue
            o, m = np.asarray(opp, float), np.asarray(mine, float)
            g = pitch_control_grid(o[:, :2], o[:, 2:], m[:, :2], m[:, 2:], L, Wd, 30, 18)
            g = g if d > 0 else g[:, ::-1]
            for j in range(ny):
                for i in range(nx):
                    space[j, i] += g[j * 6:(j + 1) * 6, i * 10:(i + 1) * 10].mean()
            cnt += 1
        zones = None
        if cnt >= 6:
            zones = (space / cnt).round(3).tolist()
            own_half = np.asarray(zones)[:, 0]          # defensive third
            j = int(np.argmax(own_half))
            if own_half[j] >= 0.55:
                side = ["far-side", "central", "near-side"][j]
                plan["cards"].append({"id": f"plan-space-{team}", "kind": "tactical", "team": team,
                    "title": f"Space conceded in the {side} defensive third",
                    "finding": f"When defending, the opponent controls {own_half[j]:.0%} of the {side} channel in front of goal.",
                    "action": "Narrow the back line on that side or drop a midfielder into it when possession is lost.",
                    "confidence": "medium" if cnt >= 30 else "low",
                    "evidence": [{"label": "Opponent control", "value": f"{own_half[j]:.0%}"}, {"label": "Moments", "value": str(cnt)}],
                    "moments": [], "players": [], "objective": "Obj 3", "priority": 0.62})
        # progression: where the best available pass pointed and how safe it was
        routes = np.zeros(3); safe = np.zeros(3)
        for dec in decisions:
            if dec and dec["call"].get("team") == team and dec["options"]:
                to, x, y, p_, v_ = dec["options"][0]
                c = 0 if y < Wd / 3 else 1 if y < 2 * Wd / 3 else 2
                routes[c] += 1; safe[c] += p_
        prog = None
        if routes.sum() >= 10:
            c = int(np.argmax(safe / np.maximum(routes, 1) * (routes > 2)))
            prog = {"channel": ["far side", "centre", "near side"][c], "share": round(float(routes[c] / routes.sum()), 2),
                    "mean_success": round(float(safe[c] / max(routes[c], 1)), 2)}
            plan["cards"].append({"id": f"plan-route-{team}", "kind": "tactical", "team": team,
                "title": f"Best route forward: the {prog['channel']}",
                "finding": f"The best pass available most often went to the {prog['channel']}, arriving {prog['mean_success']:.0%} of the time in the model.",
                "action": "Build attacks towards that channel and keep a runner there to receive.",
                "confidence": "medium" if routes.sum() >= 40 else "low",
                "evidence": [{"label": "Expected arrival", "value": f"{prog['mean_success']:.0%}"}, {"label": "Moments", "value": str(int(routes.sum()))}],
                "moments": [], "players": [], "objective": "Obj 3", "priority": 0.58})
        plan["teams"][str(team)] = {"name": names, "formation": form, "space_conceded": zones, "progression": prog,
                                    "shape_timeline": timeline}
    return plan


def _camera_footprint(H, W, Hh, cfg, per_edge: int = 12):
    """Pitch polygon seen by the camera: the image border projected through H,
    clipped to the pitch (with a small margin). None without calibration."""
    if H is None:
        return None
    t = np.linspace(0, 1, per_edge)
    border = np.vstack([np.column_stack((t * W, np.zeros_like(t))), np.column_stack((np.full_like(t, W), t * Hh)),
                        np.column_stack((W - t * W, np.full_like(t, Hh))), np.column_stack((np.zeros_like(t), Hh - t * Hh))])
    hom = np.column_stack((border, np.ones(len(border)))) @ np.asarray(H, float).T
    ref = np.asarray(H, float) @ np.array([W / 2, Hh, 1.0])     # bottom centre is always on the ground
    # Above the horizon the projective scale changes sign: those image points
    # never meet the ground. Keep only points on the camera's side of it.
    ok = np.sign(hom[:, 2]) == np.sign(ref[2])
    g = hom[ok, :2] / hom[ok, 2:3]
    g = g[np.isfinite(g).all(1)]
    if len(g) < 3:
        return None
    g[:, 0] = np.clip(g[:, 0], -4, cfg.pitch.length + 4)
    g[:, 1] = np.clip(g[:, 1], -4, cfg.pitch.width + 4)
    return [[round(float(x), 1), round(float(y), 1)] for x, y in g[::2]]


def _timeline(samples, ball_px, fps, step, footprints=None, sample_frames=None, velocity=None, decisions=None):
    rows = []
    for k, s in enumerate(samples):
        vel = velocity.get(sample_frames[k], {}) if (velocity is not None and sample_frames) else {}
        rows.append({
            "t": round(s.t, 2),
            "p": [[int(i), round(x, 2), round(y, 2), int(tm), round(sp, 2) if np.isfinite(sp) else None,
                   round(vel[i][0], 2) if i in vel else 0.0, round(vel[i][1], 2) if i in vel else 0.0]
                  for i, (x, y, tm, sp) in s.players.items()],
            "b": None if s.ball is None else [round(v, 2) for v in s.ball],
            "pos": int(s.possession), "h": None if s.holder is None else int(s.holder),
            "v": footprints[k] if footprints else None,
            **({"o": decisions[k]["options"], "d": decisions[k]["call"]} if decisions and decisions[k] else {}),
        })
    return {"rate_hz": round(fps / step, 2), "frames": rows}


def _possession_summary(trace, fps, bucket_s=5.0):
    k = max(1, int(bucket_s * fps))
    buckets = []
    for i in range(0, len(trace), k):
        seg = np.asarray(trace[i:i + k])
        known = seg[seg > 0]
        buckets.append({"t": round(i / fps, 1), "team1": float((known == 1).mean()) if len(known) else None,
                        "coverage": round(len(known) / max(len(seg), 1), 2)})
    arr = np.asarray(trace)
    known = arr[arr > 0]
    return {"team1_pct": round(100 * float((known == 1).mean()), 1) if len(known) else None,
            "team2_pct": round(100 * float((known == 2).mean()), 1) if len(known) else None,
            "coverage": round(len(known) / max(len(arr), 1), 3), "buckets": buckets}


def _team_stats(eng, samples, players, team_of, fps):
    out = {}
    for t in (1, 2):
        mine = [p for p in players if p["team"] == t]
        shape = []
        for s in samples:
            pts = np.asarray([(x, y) for (x, y, tm, _) in s.players.values() if tm == t])
            if len(pts) >= 5:
                shape.append((np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), pts[:, 0].mean()))
        a = np.asarray(shape) if shape else None
        out[str(t)] = {
            "players_seen": len(mine),
            "passes": sum(p.team == t for p in eng.passes),
            "distance_m": round(sum(p["distance_m"] for p in mine), 0),
            "hsr_m": round(sum(p["hsr_m"] for p in mine), 0),
            "length_m": None if a is None else round(float(a[:, 0].mean()), 1),
            "width_m": None if a is None else round(float(a[:, 1].mean()), 1),
            "centroid_x_m": None if a is None else round(float(a[:, 2].mean()), 1),
        }
    return out


def _render(video, path, boxes_out, team_of, role_of, ball_px, fps, n, lens_k1=0.0):
    cap = cv2.VideoCapture(video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = str(path) + ".tmp.mp4"
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    bgr = {1: (153, 95, 31), 2: (44, 85, 194), 0: (96, 100, 91)}
    trail: List[Tuple[int, int]] = []
    for f in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        frame = undistort_frame(frame, lens_k1)
        for fid, box, interp in boxes_out.get(f, []):
            x1, y1, x2, y2 = map(int, box)
            if role_of.get(fid) == REFEREE:
                col = (40, 40, 40)
                label = "ref"
            else:
                col = bgr.get(team_of.get(fid, 0), bgr[0])
                label = f"{'AB'[team_of[fid] - 1] if team_of.get(fid) in (1, 2) else 'U'}{fid}"
            cx = (x1 + x2) // 2
            # grounded ellipse under the feet + compact tag, not a box
            cv2.ellipse(frame, (cx, y2), (max(8, (x2 - x1) // 2), max(3, (x2 - x1) // 6)), 0, -40, 220, col, 2,
                        lineType=cv2.LINE_AA)
            if interp:
                cv2.ellipse(frame, (cx, y2), (max(8, (x2 - x1) // 2), max(3, (x2 - x1) // 6)), 0, -40, 220,
                            (255, 255, 255), 1, lineType=cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(frame, (cx - tw // 2 - 4, y1 - th - 10), (cx + tw // 2 + 4, y1 - 4), col, -1)
            cv2.putText(frame, label, (cx - tw // 2, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                        cv2.LINE_AA)
        b = ball_px.get(f)
        if b is not None:
            trail.append((int(b[0]), int(b[1])))
            trail = trail[-12:]
            for k in range(1, len(trail)):
                cv2.line(frame, trail[k - 1], trail[k], (40, 200, 255), 1 + k // 4, cv2.LINE_AA)
            cv2.circle(frame, (int(b[0]), int(b[1])), 7, (40, 200, 255) if not b[2] else (200, 200, 200), 2,
                       cv2.LINE_AA)
        else:
            trail = []
        wr.write(frame)
    cap.release()
    wr.release()
    # browsers need H.264; re-encode when ffmpeg is present, else keep mp4v
    import shutil
    import subprocess
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(path)], check=False)
        Path(tmp).unlink(missing_ok=True)
    else:
        Path(tmp).replace(path)


def write_legacy_stats(analysis: dict, path: Path) -> None:
    """Minimal stats.json for the existing recorded-result store."""
    q = analysis["quality"]
    stats = {
        "video": analysis["match"]["video"], "frames": q["frames"], "fps": q["fps"], "metric": q["metric"],
        "provenance": {"kind": "vision_analysis", "generated_at": analysis["provenance"]["generated_at"],
                       "detector_weights": analysis["provenance"]["detector"], "wearable_source": "none",
                       "synthetic_telemetry": False, "analysis_schema": analysis["schema"]},
        "possession_pct": {"1": analysis["possession"]["team1_pct"] or 0.0,
                           "2": analysis["possession"]["team2_pct"] or 0.0},
        "passes": {"team1": analysis["team_stats"]["1"]["passes"], "team2": analysis["team_stats"]["2"]["passes"]},
        "players": {str(p["id"]): {"team": p["team"], "distance_m": p["distance_m"],
                                   "top_speed_ms": (p["top_speed_kmh"] or 0) / 3.6, "hsr_m": p["hsr_m"]}
                    for p in analysis["players"]},
        "quality": {"status": "review" if q["checks_pending"] else "usable",
                    "warnings": [] if q["metric"] else ["pitch not calibrated; metric outputs withheld"]},
    }
    path.write_text(json.dumps(stats, indent=2, default=_json_default))


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Build the CPG44 recorded-match analysis (analysis.json).")
    p.add_argument("--video", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--calibration", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--name", default="")
    p.add_argument("--teams", default="Home,Away", help="comma-separated team names")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--device", default="", help="'' auto, '0' GPU, 'cpu'")
    p.add_argument("--camera", default="auto", choices=("auto", "fixed", "panning"),
                   help="fixed = tripod/mast camera (campus); overrides the calibration file")
    p.add_argument("--vest", action="append", default=[], help="vest log (wearable_raw.ndjson); repeat per vest")
    p.add_argument("--video-start", type=float, default=None,
                   help="unix time of the first frame; otherwise read from the file's creation_time")
    a = p.parse_args(argv)
    cfg = PipelineConfig(weights=a.weights, imgsz=a.imgsz, device=a.device)
    lens_k1, camera_mode = 0.0, a.camera
    if a.calibration:
        from .pitch import load_calibration
        from .config import PitchConfig
        from .pitch import calibration_meta
        meta = calibration_meta(a.calibration)
        cfg.pitch = PitchConfig.for_kind(meta["pitch_kind"], meta["pitch_length"], meta["pitch_width"])
        view = load_calibration(a.calibration, cfg.pitch)
        lens_k1, camera_mode = meta["k1"], (a.camera if a.camera != "auto" else meta["camera"])
    else:
        view = ViewTransformer()
    names = tuple((a.teams.split(",") + ["Away"])[:2])
    analysis = build_analysis(a.video, cfg, view, a.out, a.max_frames, not a.no_render, a.name, None, names,
                              lens_k1=lens_k1, camera_mode=camera_mode, vest_logs=a.vest or None,
                              video_start_unix=a.video_start)
    write_legacy_stats(analysis, Path(a.out) / "stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
