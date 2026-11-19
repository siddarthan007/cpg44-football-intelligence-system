"""Coach decision engine: vision tactics + vision kinematics + vest physiology.

This is CPG44's answer to the report's Objective 3 ("machine-learning
performance trends with evidence-based tactical recommendations") under the
project's evidence rules. Every card it emits

* names the observation, the numbers behind it and the moments to review,
* states a confidence derived from sample size and data quality,
* separates *observed* facts from the *suggested* action, and
* abstains (emits nothing, or a data-quality card) when coverage is too low.

Rules are deliberately transparent and literature-anchored rather than an
opaque score. Thresholds are coaching heuristics to be tuned with coaches, not
validated medical or performance cut-offs:

- Team shape, compactness and line height: Fradua et al. 2013; Clemente et al.
  2013 (effective playing space); Low et al. 2020 review of tactical metrics.
- Turnover location and pressing response ("counter-press" within ~5 s):
  Vogelbein et al. 2014; Fernández-Navarro et al. 2016.
- High-speed running decline across a session: Mohr, Krustrup & Bangsbo 2003;
  Bradley & Noakes 2013 (transient fatigue after intense periods).
- Heart-rate drift at matched external load and impaired post-effort recovery:
  Buchheit 2014 (HR monitoring in team sports); Lamberts 2009.
- External/internal load coupling (PlayerLoad with HR): Impellizzeri, Marcora &
  Coutts 2019 training-load framework; Akubat et al. 2012.

The input is the match timeline produced by :mod:`soccer_analytics.match_analysis`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HSR = 5.5        # m/s, high-speed running threshold (19.8 km/h)
SPRINT = 7.0     # m/s, sprint threshold (25.2 km/h)


@dataclass
class Sample:
    t: float                                            # seconds from start
    players: Dict[int, Tuple[float, float, int, float]]  # id -> (x, y, team, speed)
    ball: Optional[Tuple[float, float, float]]           # x, y, z (z may be 0)
    possession: int = 0                                  # team in control (0 = loose/unknown)
    holder: Optional[int] = None


@dataclass
class VestSeries:
    """Aligned wearable observations for one tracked player."""
    t: np.ndarray            # seconds on the match clock
    hr: np.ndarray           # bpm (NaN where invalid)
    quality: np.ndarray      # 0..1
    player_load: Optional[np.ndarray] = None   # cumulative IMU PlayerLoad


@dataclass
class Insight:
    id: str
    kind: str                # tactical | physical | fused | data
    team: int
    title: str
    finding: str
    action: str
    confidence: str          # high | medium | low
    evidence: List[Dict[str, str]] = field(default_factory=list)
    moments: List[float] = field(default_factory=list)
    players: List[int] = field(default_factory=list)
    objective: str = "Obj 3"
    priority: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def _conf(n: float, high: float, medium: float) -> str:
    return "high" if n >= high else "medium" if n >= medium else "low"


def _fmt(v, unit="", d=0):
    return f"{v:.{d}f}{unit}"


class CoachEngine:
    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0,
                 attack_dir: Optional[Dict[int, int]] = None):
        self.L, self.W = pitch_length, pitch_width
        self.attack_dir = attack_dir or {}
        self.min_tactical_s = 60.0

    # ------------------------------------------------------------------ #
    def _infer_attack_dir(self, samples: Sequence[Sample]) -> Dict[int, int]:
        """Team whose mean x is smaller attacks towards +x (defends left goal)."""
        xs = {1: [], 2: []}
        for s in samples:
            for (x, y, team, _) in s.players.values():
                if team in xs:
                    xs[team].append(x)
        if not xs[1] or not xs[2]:
            return {1: 1, 2: -1}
        return {1: 1, 2: -1} if np.mean(xs[1]) < np.mean(xs[2]) else {1: -1, 2: 1}

    def _rel_x(self, x: float, team: int) -> float:
        """Distance from own goal line along the attack direction (0..L)."""
        return x if self.attack_dir.get(team, 1) > 0 else self.L - x

    # ------------------------------------------------------------------ #
    def analyse(self, samples: Sequence[Sample], vests: Optional[Dict[int, VestSeries]] = None,
                quality: Optional[dict] = None) -> List[Insight]:
        if not self.attack_dir:
            self.attack_dir = self._infer_attack_dir(samples)
        cards: List[Insight] = []
        cards += self._data_quality(samples, quality or {})
        span = samples[-1].t - samples[0].t if samples else 0.0
        if span < self.min_tactical_s:
            cards.append(Insight(
                "data-short", "data", 0, "Too little play for team-shape findings",
                f"This recording covers {span:.0f} s. Shape, turnover and channel findings need at least "
                f"{self.min_tactical_s:.0f} s, otherwise set pieces such as a kick-off dominate them.",
                "Analyse a longer stretch of play; individual running and the review queue are still valid.",
                "high", [{"label": "Footage analysed", "value": f"{span:.0f} s"}], objective="Obj 4", priority=0.2))
        elif len(samples) >= 50:
            for team in (1, 2):
                cards += self._shape(samples, team)
                cards += self._turnovers(samples, team)
                cards += self._channels(samples, team)
        if vests:
            cards += self._fused(samples, vests)
        # individual fatigue cards come from soccer_analytics.fatigue
        cards.sort(key=lambda c: -c.priority)
        return cards

    # ---------------- data quality ---------------- #
    def _data_quality(self, samples, q) -> List[Insight]:
        out = []
        seen = float(np.mean([s.ball is not None for s in samples])) if samples else 0.0
        if samples and seen < 0.6:
            out.append(Insight(
                "data-ball", "data", 0, "Ball visibility limits event detection",
                f"The ball was located in {seen:.0%} of sampled moments, so passes and "
                "possession are under-counted.", "Raise or widen the camera before the next session; "
                "confirm key moments in the review queue.", "high",
                [{"label": "Ball located", "value": f"{seen:.0%}"}], objective="Obj 4", priority=0.3))
        pending = int(q.get("checks_pending", 0))
        if pending:
            out.append(Insight(
                "data-checks", "data", 0, f"{pending} moments need a human decision",
                "Uncertain identity links, possible shots and long ball gaps were held back "
                "instead of guessed.", "Resolve them in the review queue; statistics update after each decision.",
                "high", [{"label": "Pending", "value": str(pending)}], objective="Obj 4", priority=0.25))
        return out

    # ---------------- team shape ---------------- #
    def _shape(self, samples, team) -> List[Insight]:
        rows_in, rows_out, lines = [], [], []
        for s in samples:
            pts = np.asarray([(x, y) for (x, y, tm, _) in s.players.values() if tm == team])
            if len(pts) < 5:
                continue
            length = float(np.ptp(pts[:, 0]))
            width = float(np.ptp(pts[:, 1]))
            rel = np.asarray([self._rel_x(x, team) for x in pts[:, 0]])
            line = float(np.sort(rel)[1]) if len(rel) > 2 else float(rel.min())   # skip keeper
            row = (s.t, length, width, line)
            if s.possession == team:
                rows_in.append(row)
            elif s.possession and s.possession != team:
                rows_out.append(row)
            lines.append(line)
        out = []
        if len(rows_out) >= 30:
            a = np.asarray(rows_out)
            length, width, line = a[:, 1].mean(), a[:, 2].mean(), a[:, 3].mean()
            worst = a[np.argsort(-a[:, 1])[:3], 0].tolist()
            if length > 38:
                out.append(Insight(
                    f"shape-stretched-{team}", "tactical", team, "Out of possession the block is stretched",
                    f"Without the ball the team spans {length:.0f} m from deepest to highest outfield "
                    f"player, leaving space between the lines.",
                    "Ask the front line to drop 5-8 m when the ball is lost, or push the back line up "
                    "together so the distance between lines shrinks.",
                    _conf(len(a), 750, 300),
                    [{"label": "Team length (no ball)", "value": _fmt(length, " m")},
                     {"label": "Team width (no ball)", "value": _fmt(width, " m")},
                     {"label": "Moments measured", "value": str(len(a))}],
                    moments=worst, priority=0.7 + min(0.2, (length - 38) / 50)))
            elif length < 28 and line < 25:
                out.append(Insight(
                    f"shape-deep-{team}", "tactical", team, "Compact but very deep block",
                    f"The team is compact ({length:.0f} m) but the back line averages only {line:.0f} m "
                    "from its own goal, inviting pressure and long shots.",
                    "Step the line up on backward passes so the block defends further from goal.",
                    _conf(len(a), 750, 300),
                    [{"label": "Team length (no ball)", "value": _fmt(length, " m")},
                     {"label": "Back line from goal", "value": _fmt(line, " m")}],
                    moments=a[np.argsort(a[:, 3])[:3], 0].tolist(), priority=0.55))
        if len(rows_in) >= 30:
            a = np.asarray(rows_in)
            width = a[:, 2].mean()
            if width < 0.45 * self.W:
                out.append(Insight(
                    f"shape-narrow-{team}", "tactical", team, "Attacks stay narrow",
                    f"In possession the team uses {width:.0f} m of a {self.W:.0f} m wide pitch; "
                    "opponents can defend the middle without being pulled apart.",
                    "Hold one wide player on each touchline in build-up to stretch the defence.",
                    _conf(len(a), 750, 300),
                    [{"label": "Width in possession", "value": _fmt(width, " m")},
                     {"label": "Share of pitch width", "value": f"{width / self.W:.0%}"}],
                    moments=a[np.argsort(a[:, 2])[:3], 0].tolist(), priority=0.5))
        return out

    # ---------------- turnovers and counter-press ---------------- #
    def _turnovers(self, samples, team) -> List[Insight]:
        losses = []          # (t, rel_x, y)
        regains = []         # seconds to regain after each loss
        prev = 0
        loss_t = None
        for s in samples:
            if s.possession in (1, 2):
                if prev == team and s.possession != team and s.ball is not None:
                    losses.append((s.t, self._rel_x(s.ball[0], team), s.ball[1]))
                    loss_t = s.t
                if loss_t is not None and s.possession == team and prev != team:
                    regains.append(s.t - loss_t)
                    loss_t = None
                prev = s.possession
        out = []
        if len(losses) >= 4:
            a = np.asarray(losses)
            own_half = a[:, 1] < self.L / 2
            share = float(own_half.mean())
            if share >= 0.5:
                out.append(Insight(
                    f"turnover-own-half-{team}", "tactical", team, "Losing the ball in our own half",
                    f"{int(own_half.sum())} of {len(a)} lost possessions happened in the defensive half, "
                    "where a loss is most dangerous.",
                    "Offer a safer option in build-up (a dropping midfielder or a switch) and review the "
                    "marked moments for the recurring pattern.",
                    _conf(len(a), 12, 6),
                    [{"label": "Losses in own half", "value": f"{int(own_half.sum())}/{len(a)}"}],
                    moments=a[own_half, 0].tolist()[:6], priority=0.75))
        if len(regains) >= 3:
            quick = float(np.mean(np.asarray(regains) <= 5.0))
            out.append(Insight(
                f"counterpress-{team}", "tactical", team,
                "Quick regains after losing the ball" if quick >= 0.4 else "Slow reaction after losing the ball",
                f"The ball was won back within 5 s in {quick:.0%} of {len(regains)} measured losses "
                f"(median {np.median(regains):.1f} s).",
                "Keep the nearest two players pressing immediately after a loss." if quick < 0.4 else
                "The immediate press works; protect it by keeping the rest shape compact behind it.",
                _conf(len(regains), 10, 5),
                [{"label": "Regained within 5 s", "value": f"{quick:.0%}"},
                 {"label": "Median time to regain", "value": _fmt(float(np.median(regains)), " s", 1)}],
                priority=0.45 if quick >= 0.4 else 0.6))
        return out

    # ---------------- channels / overloads ---------------- #
    def _channels(self, samples, team) -> List[Insight]:
        lanes = np.zeros(3)
        for s in samples:
            if s.possession == team and s.ball is not None:
                y = s.ball[1]
                lanes[0 if y < self.W / 3 else 1 if y < 2 * self.W / 3 else 2] += 1
        total = lanes.sum()
        if total < 60:
            return []
        share = lanes / total
        k = int(np.argmax(share))
        names = ["the far-side channel", "the centre", "the near-side channel"]
        if share[k] >= 0.5:
            return [Insight(
                f"channel-{team}", "tactical", team, f"Play funnels through {names[k]}",
                f"{share[k]:.0%} of the time in possession the ball was in {names[k]}.",
                "Opponents can shift to that side; practise switching play when they overload it.",
                _conf(total, 900, 450),
                [{"label": n.split(' ')[-2] if ' ' in n else n, "value": f"{v:.0%}"}
                 for n, v in zip(["far channel", "centre", "near channel"], share)],
                priority=0.4)]
        return []

    # ---------------- vision-only physical ---------------- #
    # ---------------- wearable + vision fusion ---------------- #
    def _fused(self, samples, vests: Dict[int, VestSeries]) -> List[Insight]:
        out = []
        speed_of: Dict[int, List[Tuple[float, float]]] = {}
        for s in samples:
            for pid, (x, y, team, sp) in s.players.items():
                if pid in vests and np.isfinite(sp):
                    speed_of.setdefault(pid, []).append((s.t, sp))
        for pid, vs in vests.items():
            ok = np.isfinite(vs.hr) & (vs.quality >= 0.5)
            if ok.sum() < 60 or pid not in speed_of:
                if ok.sum() < 60:
                    out.append(Insight(
                        f"vest-coverage-{pid}", "data", 0, f"Player {pid}: too little valid pulse",
                        f"Only {int(ok.sum())} quality-checked heart-rate samples were usable.",
                        "Tighten the vest so the optical sensor stays on skin; check the pod placement.",
                        "high", [{"label": "Valid pulse samples", "value": str(int(ok.sum()))}],
                        players=[pid], objective="Obj 2", priority=0.2))
                continue
            sp = np.asarray(speed_of[pid])
            spd = np.interp(vs.t[ok], sp[:, 0], sp[:, 1], left=np.nan, right=np.nan)
            hr = vs.hr[ok]
            t = vs.t[ok]
            m = np.isfinite(spd)
            if m.sum() < 60:
                continue
            spd, hr, t = spd[m], hr[m], t[m]
            # 1) cardiac drift at matched speed (1 m/s bands)
            mid = t[0] + (t[-1] - t[0]) / 2
            diffs = []
            for lo in np.arange(0.5, 6.0, 1.0):
                band = (spd >= lo) & (spd < lo + 1)
                e, l = band & (t < mid), band & (t >= mid)
                if e.sum() >= 10 and l.sum() >= 10:
                    diffs.append(np.median(hr[l]) - np.median(hr[e]))
            if len(diffs) >= 2:
                drift = float(np.median(diffs))
                sp_e = float(np.mean(spd[t < mid]))
                sp_l = float(np.mean(spd[t >= mid]))
                if drift >= 8 and sp_l <= sp_e * 1.05:
                    out.append(Insight(
                        f"drift-{pid}", "fused", 0, f"Player {pid}: working harder for the same running",
                        f"At matched running speeds heart rate is {drift:.0f} bpm higher than earlier, while "
                        f"average speed did not rise ({sp_e:.1f} -> {sp_l:.1f} m/s). Camera and vest agree.",
                        "Prioritise this player for a check-in or a lower-intensity role; review hydration "
                        "and recovery between bouts. This is a workload cue, not a diagnosis.",
                        _conf(len(diffs), 4, 2),
                        [{"label": "HR rise at same speed", "value": _fmt(drift, " bpm")},
                         {"label": "Speed bands compared", "value": str(len(diffs))},
                         {"label": "Mean speed early/late", "value": f"{sp_e:.1f} / {sp_l:.1f} m/s"}],
                        players=[pid], objective="Obj 1 + Obj 3", priority=0.9))
            # 2) recovery after high-intensity bursts
            bursts = np.nonzero((spd[1:] < 2.0) & (spd[:-1] >= HSR))[0]
            recov = []
            for b in bursts:
                t0 = t[b + 1]
                win = (t >= t0) & (t <= t0 + 30)
                low = win & (spd < 2.5)
                if low.sum() >= 5:
                    recov.append(hr[b] - np.min(hr[low]))
            if len(recov) >= 3:
                r = float(np.median(recov))
                if r < 8:
                    out.append(Insight(
                        f"recovery-{pid}", "fused", 0, f"Player {pid}: heart rate is not coming down between efforts",
                        f"After high-speed efforts, heart rate fell a median of only {r:.0f} bpm within 30 s "
                        "of easing off.",
                        "Give the player a longer reset after sprints (rotate pressing duties).",
                        _conf(len(recov), 6, 3),
                        [{"label": "HR drop in 30 s", "value": _fmt(r, " bpm")},
                         {"label": "Efforts measured", "value": str(len(recov))}],
                        players=[pid], objective="Obj 1 + Obj 3", priority=0.8))
            # 3) internal/external coupling sanity (binding integrity)
            if vs.player_load is not None and len(vs.player_load) == len(vs.t):
                pl = np.diff(vs.player_load[ok][m], prepend=vs.player_load[ok][m][0])
                if np.std(pl) > 0 and np.std(spd) > 0:
                    r = float(np.corrcoef(pl, spd)[0, 1])
                    if r < 0.1:
                        out.append(Insight(
                            f"binding-{pid}", "data", 0, f"Vest {pid} may be linked to the wrong track",
                            f"Vest movement and camera speed for this track barely correlate (r = {r:.2f}).",
                            "Confirm the player-vest link before trusting this player's fused numbers.",
                            "medium", [{"label": "IMU-camera correlation", "value": f"{r:.2f}"}],
                            players=[pid], objective="Obj 1", priority=0.85))
        return out
