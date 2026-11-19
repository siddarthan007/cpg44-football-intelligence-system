"""Within-match fatigue watch and substitution suggestions.

What it measures, and why these measures
----------------------------------------
* **Output against the player's own peak.** Bradley & Noakes (2013) showed
  that comparing the first and second half confounds fatigue with pacing and
  game state; the robust signal is the drop from a player's own most intense
  window ("peak 5 min") to the windows that follow. Rates are per *observed*
  minute, so leaving the camera view never looks like tiring.
* **Relative to the team.** When the whole side drops off because it is
  defending deep or the game is stopped, that is context, not fatigue
  (Paul, Bradley & Nassis 2015 on contextual factors). The player's decline is
  taken net of the team's median decline over the same windows.
* **Repeated-sprint speed.** Top speed reached in recent efforts against the
  best earlier efforts (Mohr, Krustrup & Bangsbo 2003, temporary fatigue).
* **Vest, when worn.** Heart rate per unit of running output rising
  (cardiac drift at the same work) and slower heart-rate recovery after hard
  bouts (Buchheit 2014). Pulse is used only where the hub's quality gate
  passed.

The result is a coaching cue with every factor shown, not an injury
probability. Load history across sessions (acute:chronic ratios) is a weak
injury predictor (Impellizzeri et al. 2020), so it is reported as context
only, elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HSR = 5.5          # m/s, 19.8 km/h
SPRINT = 7.0       # m/s, 25.2 km/h
MIN_OBS = 0.4      # a window counts when the player was seen for 40% of it


@dataclass
class _Win:
    t0: float
    obs_s: float
    hsr_m: float
    dist_m: float
    sprints: int
    top: float
    hr: float = float("nan")

    @property
    def hsr_rate(self) -> float:       # m per observed minute
        return self.hsr_m / self.obs_s * 60 if self.obs_s > 0 else float("nan")

    @property
    def work_rate(self) -> float:
        return self.dist_m / self.obs_s * 60 if self.obs_s > 0 else float("nan")


def window_length(duration_s: float) -> float:
    """5 min windows for a match; shorter (still >= 60 s) for short sessions."""
    return 300.0 if duration_s >= 25 * 60 else max(60.0, duration_s / 5)


def _windows(rows: np.ndarray, start: float, end: float, w: float, dt: float,
             hr: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> List[_Win]:
    """rows: (t, speed). Windows step by w/2 (50% overlap)."""
    out = []
    t0 = start
    while t0 + w <= end + 1e-6:
        m = (rows[:, 0] >= t0) & (rows[:, 0] < t0 + w)
        sp = rows[m, 1]
        obs = len(sp) * dt
        if obs >= MIN_OBS * w:
            hi = sp >= SPRINT
            sprints = int(np.sum(hi[1:] & ~hi[:-1]) + (hi[0] if len(hi) else 0))
            win = _Win(t0, obs, float(np.sum(sp[sp >= HSR]) * dt), float(np.sum(sp) * dt), sprints,
                       float(np.mean(np.sort(sp)[-5:])))           # fastest second in the window
            if hr is not None:
                ht, hv = hr
                k = (ht >= t0) & (ht < t0 + w) & np.isfinite(hv)
                if k.sum() >= 10:
                    win.hr = float(np.median(hv[k]))
            out.append(win)
        t0 += w / 2
    return out


def _hr_recovery(t: np.ndarray, speed: np.ndarray, ht: np.ndarray, hv: np.ndarray) -> List[Tuple[float, float]]:
    """(time, bpm drop 30 s after a high-intensity bout) for each bout."""
    out = []
    hot = speed >= HSR
    k = 0
    while k < len(hot):
        if hot[k]:
            e = k
            while e + 1 < len(hot) and (hot[e + 1] or t[e + 1] - t[e] < 1.0):
                e += 1
            if t[e] - t[k] >= 4.0:                       # a real effort, not a step
                te = t[e]
                at = np.isfinite(hv) & (np.abs(ht - te) <= 3)
                after = np.isfinite(hv) & (np.abs(ht - (te + 30)) <= 3)
                calm = (np.abs(t - (te + 15)) <= 15)
                if at.any() and after.any() and np.mean(speed[calm] < HSR) > 0.85:
                    out.append((float(te), float(np.median(hv[at]) - np.median(hv[after]))))
            k = e + 1
        else:
            k += 1
    return out


def assess(samples: Sequence, vests: Optional[Dict[int, object]] = None,
           roles: Optional[Dict[int, str]] = None, min_duration_s: float = 600.0,
           attack_dir: Optional[Dict[int, int]] = None, width: float = 68.0) -> dict:
    """Fatigue status per player and ranked substitution suggestions.

    ``samples`` are :class:`coach_engine.Sample` rows (t, players id -> (x, y,
    team, speed)). ``vests`` maps player id to a :class:`VestSeries`.
    """
    vests = vests or {}
    if len(samples) < 2:
        return {"available": False, "reason": "no tracked play", "min_duration_s": min_duration_s}
    t_all = np.asarray([s.t for s in samples])
    start, end = float(t_all[0]), float(t_all[-1])
    duration = end - start
    if duration < min_duration_s:
        return {"available": False, "duration_s": round(duration, 1), "min_duration_s": min_duration_s,
                "reason": f"needs at least {min_duration_s / 60:.0f} min of play to separate fatigue from game state"}
    dt = float(np.median(np.diff(t_all)))
    w = window_length(duration)

    per: Dict[int, List[Tuple[float, float]]] = {}
    team_of: Dict[int, List[int]] = {}
    xy: Dict[int, List[Tuple[float, float]]] = {}
    for s in samples:
        for pid, (x, y, tm, sp) in s.players.items():
            if tm in (1, 2) and np.isfinite(sp):
                per.setdefault(pid, []).append((s.t, sp))
                team_of.setdefault(pid, []).append(tm)
                xy.setdefault(pid, []).append((x, y))
    wins: Dict[int, List[_Win]] = {}
    side: Dict[int, int] = {}
    for pid, rows in per.items():
        a = np.asarray(rows)
        v = vests.get(pid)
        hr = (np.asarray(v.t), np.where(np.asarray(v.quality) >= 0.5, np.asarray(v.hr), np.nan)) if v is not None else None
        ws = _windows(a, start, end, w, dt, hr)
        if len(ws) >= 3:
            wins[pid] = ws
            side[pid] = int(np.bincount(team_of[pid]).argmax())

    # team trend: median HSR rate per window start, per side
    team_curve: Dict[int, Dict[float, float]] = {1: {}, 2: {}}
    for tm in (1, 2):
        starts = sorted({x.t0 for p, ws in wins.items() if side[p] == tm for x in ws})
        for t0 in starts:
            r = [x.hsr_rate for p, ws in wins.items() if side[p] == tm for x in ws if x.t0 == t0]
            if len(r) >= 3:
                team_curve[tm][t0] = float(np.median(r))

    roles = dict(roles or {})
    for tm in (1, 2):
        ids = [p for p in wins if side[p] == tm]
        d = (attack_dir or {}).get(tm, 1)
        depth = {p: d * float(np.mean(np.asarray(xy[p])[:, 0])) for p in ids}
        order = sorted(ids, key=depth.get)
        for k, p in enumerate(order):
            line = "back" if k < len(order) * 0.4 else "front" if k >= len(order) * 0.8 else "middle"
            y = float(np.mean(np.asarray(xy[p])[:, 1])) / width
            ch = "wide" if y < 0.25 or y > 0.75 else "central"
            roles.setdefault(p, f"{ch} {line}")

    players = []
    for pid, ws in wins.items():
        tm = side[pid]
        rates = np.asarray([x.hsr_rate for x in ws])
        peak = float(np.max(rates))
        cur = ws[-1]
        factors: Dict[str, float] = {}
        notes: List[str] = []
        if peak < 6.0:                        # role with little high-speed running: use work rate
            wr = np.asarray([x.work_rate for x in ws])
            peak, rates = float(wr.max()), wr
            metric = "work rate"
        else:
            metric = "high-speed running"
        drop = float(np.clip(1 - rates[-1] / peak, 0, 1)) if peak > 0 else 0.0
        factors["below_own_peak"] = drop
        tc = team_curve[tm]
        if len(tc) >= 3 and cur.t0 in tc:
            team_drop = float(np.clip(1 - tc[cur.t0] / max(tc.values()), 0, 1))
            factors["vs_team"] = float(np.clip(drop - team_drop, 0, 1))
        else:
            factors["vs_team"] = drop * 0.5
        tops = np.asarray([x.top for x in ws])
        best = float(np.max(tops[:-1])) if len(tops) > 1 else float(tops[0])
        if best >= SPRINT - 0.5:
            factors["sprint_speed"] = float(np.clip((best - tops[-1]) / best / 0.15, 0, 1))
        v = vests.get(pid)
        if v is not None:
            hrw = [(x.hr, x.work_rate) for x in ws if np.isfinite(x.hr) and x.work_rate > 5]
            if len(hrw) >= 3:
                eff = np.asarray([h / wr for h, wr in hrw])            # bpm per m/min
                factors["cardiac_drift"] = float(np.clip((eff[-1] / np.median(eff[:max(1, len(eff) // 2)]) - 1) / 0.15, 0, 1))
            rows = np.asarray(per[pid])
            rec = _hr_recovery(rows[:, 0], rows[:, 1], np.asarray(v.t), np.where(np.asarray(v.quality) >= 0.5, np.asarray(v.hr), np.nan))
            if len(rec) >= 4:
                half = len(rec) // 2
                early, late = np.median([r for _, r in rec[:half]]), np.median([r for _, r in rec[half:]])
                if early > 5:
                    factors["hr_recovery"] = float(np.clip((early - late) / early / 0.4, 0, 1))
                    notes.append(f"heart rate recovers {early:.0f} bpm in 30 s early, {late:.0f} now")
        weights = {"vs_team": 0.4, "below_own_peak": 0.2, "sprint_speed": 0.2, "cardiac_drift": 0.12, "hr_recovery": 0.08}
        used = {k: weights[k] for k in factors}
        score = float(sum(factors[k] * used[k] for k in used) / sum(used.values()))
        # trend over the last three windows, for a time-to-threshold projection
        proj = None
        if len(rates) >= 3 and peak > 0:
            tt = np.asarray([x.t0 for x in ws[-3:]]) / 60.0
            slope, icpt = np.polyfit(tt, rates[-3:], 1)
            if slope < 0 and rates[-1] > 0.5 * peak:
                proj = float(np.clip(((0.5 * peak) - (slope * tt[-1] + icpt)) / slope, 0, 120))
        status = "fresh" if score < 0.25 else "watch" if score < 0.45 else "tiring"
        if drop > 0.25:
            notes.insert(0, f"{metric} {rates[-1]:.0f} m/min now, peak {peak:.0f}")
        if factors.get("sprint_speed", 0) > 0.3:
            notes.append(f"top speed {tops[-1] * 3.6:.1f} km/h, best earlier {best * 3.6:.1f}")
        if factors.get("cardiac_drift", 0) > 0.3:
            notes.append("heart rate higher for the same running")
        players.append({
            "id": int(pid), "team": tm, "score": round(score, 2), "status": status, "metric": metric,
            "now": round(float(rates[-1]), 1), "peak": round(peak, 1), "minutes_seen": round(sum(x.obs_s for x in ws) / 120.0, 1),
            "factors": {k: round(v, 2) for k, v in factors.items()}, "notes": notes,
            "minutes_to_half_peak": None if proj is None or status == "fresh" else round(proj, 1),
            "role": roles.get(pid, ""),
            "curve": [[round(x.t0 + w / 2, 1), round(float(r), 1)] for x, r in zip(ws, rates)],
        })
    players.sort(key=lambda p: -p["score"])
    subs = []
    for tm in (1, 2):
        for p in [p for p in players if p["team"] == tm and p["status"] == "tiring"][:2]:
            subs.append({"team": tm, "out": p["id"], "role": p["role"], "score": p["score"],
                         "why": p["notes"][:2] or ["output well below own peak and team"],
                         "when": "at the next stoppage" if p["score"] >= 0.55 else "if it continues for another window"})
    return {"available": True, "window_s": w, "duration_s": round(duration, 1), "players": players, "subs": subs,
            "team_curve": {str(k): [[round(t0 + w / 2, 1), round(v, 1)] for t0, v in sorted(c.items())] for k, c in team_curve.items()}}

