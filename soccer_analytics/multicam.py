"""Two or more fixed cameras on one pitch: one board.

Each camera is analysed on its own (``match_analysis``), in the same pitch
coordinates because each has its own calibration. This module then:

1. **Aligns the recordings in time** from their sound. Two phones by the same
   pitch hear the same whistle, shouts and ball strikes, so the cross
   correlation of their audio envelopes peaks at the offset between them.
2. **Matches players across cameras** at every moment by position on the pitch
   (same side, within 1.5 m), solved jointly so one player is never matched
   twice (Hungarian assignment).
3. **Links identities.** Camera B's player 12 is camera A's player 4 if they
   coincide for long enough; ids that never coincide stay separate.
4. **Fuses positions** as an inverse-variance mean. A foot point's error grows
   with distance from the camera (perspective: a fixed pixel error covers more
   ground further away), so the nearer camera counts more. Ground error is
   modelled as sigma = 0.25 m + 0.02 x distance.

The ball is taken from the camera nearest to it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

GATE_M = 1.5
LINK_S = 5.0


def audio_envelope(video: str, rate: int = 100) -> np.ndarray:
    """Loudness envelope at ``rate`` Hz from a video's sound track (ffmpeg)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
                         capture_output=True, timeout=600).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32)
    if len(x) < 8000:
        return np.zeros(0)
    hop = 8000 // rate
    n = len(x) // hop
    return np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1))


def audio_offset(env_a: np.ndarray, env_b: np.ndarray, rate: int = 100, max_s: float = 60.0) -> Tuple[float, float]:
    """Seconds to add to B's clock to land on A's, and the peak correlation.

    Envelopes are log-compressed and differenced so onsets (whistles, strikes)
    dominate over steady crowd noise."""
    def prep(e):
        d = np.diff(np.log1p(np.maximum(e, 0.0)))
        return (d - d.mean()) / (d.std() + 1e-9)
    a, b = prep(env_a), prep(env_b)
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    xc = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    lags = np.r_[np.arange(0, len(a)), np.arange(-len(b) + 1, 0)]
    vals = np.r_[xc[: len(a)], xc[n - len(b) + 1:]]
    ok = np.abs(lags) <= max_s * rate
    k = int(np.argmax(np.where(ok, vals, -np.inf)))
    overlap = min(len(a), len(b))
    return float(lags[k]) / rate, float(vals[k] / max(overlap, 1))


def _sigma(x: float, y: float, cam: Tuple[float, float, float]) -> float:
    return 0.25 + 0.02 * float(np.sqrt((x - cam[0]) ** 2 + (y - cam[1]) ** 2 + cam[2] ** 2))


def _frames(analysis: dict) -> List[dict]:
    return analysis["timeline"]["frames"]


def _at(frames: List[dict], t: float, tol: float = 0.11) -> Optional[dict]:
    lo, hi = 0, len(frames) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if frames[mid]["t"] < t:
            lo = mid + 1
        else:
            hi = mid
    best = min((frames[i] for i in (lo - 1, lo) if 0 <= i < len(frames)), key=lambda f: abs(f["t"] - t), default=None)
    return best if best is not None and abs(best["t"] - t) <= tol else None


def _match(pa: Sequence, pb: Sequence) -> List[Tuple[int, int]]:
    """Index pairs (i in pa, j in pb) of the same player: same side, within the gate."""
    if not len(pa) or not len(pb):
        return []
    C = np.full((len(pa), len(pb)), 1e6)
    for i, a in enumerate(pa):
        for j, b in enumerate(pb):
            if a[3] == b[3]:
                d = float(np.hypot(a[1] - b[1], a[2] - b[2]))
                if d <= GATE_M:
                    C[i, j] = d
    r, c = linear_sum_assignment(C)
    return [(int(i), int(j)) for i, j in zip(r, c) if C[i, j] < 1e6]


def fuse(analyses: Sequence[dict], cameras: Sequence[Tuple[float, float, float]],
         offsets: Sequence[float]) -> dict:
    """Merge per-camera analyses into one. ``cameras[k]`` = (x, y, height) in
    pitch metres; ``offsets[k]`` = camera k's time minus the reference camera's
    time for the same moment (0 for the reference, the first)."""
    ref = analyses[0]
    rate = ref["timeline"].get("rate_hz", 5)
    frames = [_frames(a) for a in analyses]
    t_end = min(f[-1]["t"] - off for f, off in zip(frames, offsets) if f)
    grid = np.arange(0.0, t_end, 1.0 / rate)

    # pass 1: how long each pair of ids coincides
    co: List[Dict[Tuple[int, int], int]] = [dict() for _ in analyses]
    for t in grid:
        fa = _at(frames[0], t)
        if fa is None:
            continue
        for k in range(1, len(analyses)):
            fb = _at(frames[k], t + offsets[k])
            if fb is None:
                continue
            for i, j in _match(fa["p"], fb["p"]):
                key = (fa["p"][i][0], fb["p"][j][0])
                co[k][key] = co[k].get(key, 0) + 1
    link: List[Dict[int, int]] = [{}]
    next_id = 1 + max((p["id"] for p in ref.get("players", [])), default=0)
    for k in range(1, len(analyses)):
        pairs = sorted(co[k].items(), key=lambda kv: -kv[1])
        used_a, m = set(), {}
        for (ia, ib), n in pairs:
            if n / rate >= LINK_S and ia not in used_a and ib not in m:
                m[ib] = ia
                used_a.add(ia)
        for p in analyses[k].get("players", []):
            if p["id"] not in m:
                m[p["id"]] = next_id
                next_id += 1
        link.append(m)

    # pass 2: fused positions per moment
    out_frames = []
    for t in grid:
        views = []
        for k in range(len(analyses)):
            f = _at(frames[k], t + offsets[k])
            if f is not None:
                views.append((k, f))
        if not views:
            continue
        acc: Dict[int, List] = {}
        for k, f in views:
            for row in f["p"]:
                gid = row[0] if k == 0 else link[k].get(row[0])
                if gid is None:
                    continue
                w = 1.0 / _sigma(row[1], row[2], cameras[k]) ** 2
                acc.setdefault(gid, []).append((w, row, k))
        players = []
        for gid, obs in acc.items():
            ws = np.asarray([o[0] for o in obs])
            xy = np.asarray([[o[1][1], o[1][2]] for o in obs])
            m = (ws[:, None] * xy).sum(0) / ws.sum()
            best = max(obs, key=lambda o: o[0])[1]
            players.append([int(gid), round(float(m[0]), 2), round(float(m[1]), 2), *best[3:]])
        balls = [(k, f["b"]) for k, f in views if f.get("b")]
        ball = min(balls, key=lambda kb: np.hypot(kb[1][0] - cameras[kb[0]][0], kb[1][1] - cameras[kb[0]][1]))[1] if balls else None
        base = dict(views[0][1])
        base.update({"t": round(float(t), 2), "p": players, "b": ball, "cams": [k for k, _ in views]})
        base.pop("v", None)
        out_frames.append(base)

    merged = json.loads(json.dumps(ref))
    merged["timeline"]["frames"] = out_frames
    merged.setdefault("quality", {})["cameras"] = len(analyses)
    merged["quality"]["camera_offsets_s"] = [round(float(o), 2) for o in offsets]
    ref_ids = {p["id"] for p in ref.get("players", [])}
    merged["quality"]["identity_links"] = [sum(1 for b in m.values() if b in ref_ids) for m in link[1:]]
    return merged


def camera_position(calibration_yaml: str) -> Tuple[float, float, float]:
    """Camera centre on the pitch (x, y, height) from a fixed-camera calibration."""
    from .ball3d import camera_from_homography
    from .pitch import calibration_meta, load_calibration
    meta = calibration_meta(calibration_yaml)
    view = load_calibration(calibration_yaml)
    w, h = (meta.get("image_size") or [1920, 1080])[:2]
    Hinv = np.linalg.inv(np.asarray(view.H, float))
    cam = camera_from_homography(Hinv / Hinv[2, 2], (int(w), int(h)))
    if cam is None:
        raise ValueError("camera position not recoverable from this calibration")
    c = cam.centre
    return float(c[0]), float(c[1]), float(c[2])


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Fuse analyses of the same match from several fixed cameras.")
    p.add_argument("--analysis", action="append", required=True, help="analysis folder, one per camera (first is the reference)")
    p.add_argument("--calibration", action="append", required=True, help="calibration YAML, same order")
    p.add_argument("--video", action="append", default=[], help="source videos, same order, for sound sync")
    p.add_argument("--offset", action="append", type=float, default=[], help="seconds per camera instead of sound sync")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if len(a.analysis) != len(a.calibration):
        raise SystemExit("one calibration per analysis")
    analyses = [json.loads((Path(d) / "analysis.json").read_text()) for d in a.analysis]
    cams = [camera_position(c) for c in a.calibration]
    if a.offset:
        offsets = [0.0] + list(a.offset[: len(analyses) - 1])
    elif len(a.video) == len(analyses):
        ref = audio_envelope(a.video[0])
        offsets = [0.0]
        for v in a.video[1:]:
            off, r = audio_offset(ref, audio_envelope(v))
            print(f"[multicam] {Path(v).name}: offset {off:+.2f} s (correlation {r:.2f})", flush=True)
            offsets.append(-off)
    else:
        offsets = [0.0] * len(analyses)
    merged = fuse(analyses, cams, offsets)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(merged))
    src = Path(a.analysis[0]) / "annotated.mp4"
    if src.is_file() and not (out / "annotated.mp4").exists():
        (out / "annotated.mp4").symlink_to(src.resolve())
    print(f"[multicam] wrote {out / 'analysis.json'}: {len(merged['timeline']['frames'])} moments, "
          f"{merged['quality']['identity_links']} identities linked", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
