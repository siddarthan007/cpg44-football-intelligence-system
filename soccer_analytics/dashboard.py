"""Multi-window graphical dashboard for live / near-real-time analysis.

Windows (separate concerns):
    "Match"      annotated broadcast frame (players tagged, ball, possession)
    "Radar"      top-down minimap + Voronoi pitch-control shading
    "Heatmap"    live-accumulating team occupancy
    "Analytics"  rich panel: possession, team performance, player performance
                 table, load-review score, tactical recommendations, FPS

All drawn with NumPy/OpenCV so it stays fast enough for live use. Under WSL the
windows need WSLg (built into Windows 11); pass ``windows=False`` for headless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from .annotate import TEAM_BGR

PANEL_W = 680
PANEL_H = 940
_BG = (22, 22, 26)
_HDR = (255, 255, 255)
RISK_BGR = {"high": (60, 60, 235), "moderate": (40, 180, 235), "low": (90, 190, 90)}


@dataclass
class DashboardState:
    fps: float = 0.0
    possession: Dict[int, float] = field(default_factory=lambda: {1: 0.0, 2: 0.0})
    team_stats: Dict[int, dict] = field(default_factory=dict)   # per team tactical metrics
    players: List[dict] = field(default_factory=list)           # perf rows (sorted)
    recommendations: List[str] = field(default_factory=list)
    tagged_count: int = 0
    metric: bool = True
    wearable_vitals: Dict[int, dict] = field(default_factory=dict)


def _text(img, s, xy, sc=0.5, col=(225, 225, 225), th=1):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)


def _wrap(s: str, width: int) -> List[str]:
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def _possession(img, y, poss):
    _text(img, "POSSESSION", (18, y), 0.66, _HDR); y += 16
    bar_w = PANEL_W - 36
    p1 = poss.get(1, 0.0)
    split = int(bar_w * p1 / 100.0)
    cv2.rectangle(img, (18, y), (18 + split, y + 30), TEAM_BGR[1], -1)
    cv2.rectangle(img, (18 + split, y), (18 + bar_w, y + 30), TEAM_BGR[2], -1)
    _text(img, f"{p1:.0f}%", (24, y + 21), 0.62, (0, 0, 0), 2)
    _text(img, f"{poss.get(2,0):.0f}%", (18 + bar_w - 56, y + 21), 0.62, (0, 0, 0), 2)
    return y + 52


def _team_perf(img, y, team_stats):
    _text(img, "TEAM PERFORMANCE", (18, y), 0.66, _HDR); y += 26
    rows = [("Control %", "control", "{:.0f}"), ("Formation", "formation", "{}"),
            ("Width m", "width", "{:.0f}"), ("Line m", "line_height", "{:.0f}"),
            ("Pressing", "pressing", "{:.1f}"), ("Att-third %", "att_third", "{:.0f}"),
            ("Shots", "shots", "{}"), ("xG", "xg", "{:.2f}"),
            ("Phase", "phase", "{}")]
    c1x, c2x = 250, 460
    cv2.circle(img, (c1x, y - 5), 7, TEAM_BGR[1], -1)
    cv2.circle(img, (c2x, y - 5), 7, TEAM_BGR[2], -1)
    _text(img, "T1", (c1x + 14, y), 0.56, _HDR); _text(img, "T2", (c2x + 14, y), 0.56, _HDR)
    y += 24
    t1, t2 = team_stats.get(1, {}), team_stats.get(2, {})
    for label, key, fmt in rows:
        _text(img, label, (22, y), 0.54, (175, 175, 182))
        for tx, ts in ((c1x, t1), (c2x, t2)):
            v = ts.get(key)
            s = fmt.format(v) if v is not None and v != "" else "-"
            _text(img, str(s)[:16], (tx, y), 0.54)
        y += 24
    return y + 12


def _player_table(img, y, players, metric):
    _text(img, "PLAYER PERFORMANCE", (18, y), 0.66, _HDR)
    _text(img, "cyan=wearable", (PANEL_W - 168, y), 0.5, (255, 255, 0)); y += 24
    hdr = "ID     dist   v.max  metP  review" if metric else "ID       dist(px)   v.max"
    _text(img, hdr, (44, y), 0.5, (150, 150, 158)); y += 22
    for p in players[:9]:
        cv2.circle(img, (24, y - 6), 7, TEAM_BGR.get(p.get("team", 0), (180, 180, 180)), -1)
        if p.get("wearable"):
            cv2.circle(img, (24, y - 6), 8, (255, 255, 0), 2)
        tag = str(p.get("tag", "?"))
        if metric:
            line = "{:<5} {:5.0f}m {:4.1f} {:5.0f}  {:.2f}".format(
                tag, p.get("distance", 0), p.get("top_speed", 0),
                p.get("metabolic", 0), p.get("risk", 0.0))
            _text(img, line, (44, y), 0.54)
            lvl = p.get("level", "low")
            cv2.circle(img, (PANEL_W - 22, y - 6), 7, RISK_BGR.get(lvl, (150, 150, 150)), -1)
        else:
            _text(img, "{:<5} {:8.0f}   {:5.1f}".format(
                tag, p.get("distance", 0), p.get("top_speed", 0)), (44, y), 0.54)
        y += 26
    return y + 12


def _wearable_strip(img, y, vitals: Dict[int, dict]):
    _text(img, "WEARABLE (ESP32 HUB)", (18, y), 0.66, _HDR); y += 24
    if not vitals:
        _text(img, "waiting for hub  http://127.0.0.1:8081/", (22, y), 0.5, (150, 150, 158))
        return y + 28
    for pid, v in list(vitals.items())[:4]:
        hr = v.get("hr")
        spo2 = v.get("spo2")
        hr_s = "--" if hr is None else f"{hr:.0f}"
        spo2_s = "--" if spo2 is None else f"{spo2:.0f}"
        _text(img, f"P{pid}  HR {hr_s} bpm   SpO2 {spo2_s}%", (22, y), 0.56, (255, 255, 0))
        y += 22
    return y + 10


def _recs(img, y, recs):
    _text(img, "TACTICAL RECOMMENDATIONS", (18, y), 0.66, _HDR); y += 26
    for msg in recs[:6]:
        for i, line in enumerate(_wrap(msg, 58)):
            _text(img, ("- " if i == 0 else "  ") + line, (18, y), 0.5, (210, 210, 165))
            y += 20
        y += 5
    return y


def render_dashboard_panel(state: DashboardState, height: int = PANEL_H) -> np.ndarray:
    img = np.full((max(height, PANEL_H), PANEL_W, 3), _BG, np.uint8)
    _text(img, "MATCH ANALYTICS", (18, 40), 0.95, _HDR, 2)
    _text(img, f"{state.fps:4.1f} FPS", (PANEL_W - 150, 38), 0.72, (120, 220, 120), 2)
    if not state.metric:
        _text(img, "PIXEL MODE - calibrate for metres/load/injury",
              (18, 62), 0.5, (80, 160, 235))
    y = 82
    y = _possession(img, y, state.possession)
    if state.metric:
        y = _team_perf(img, y, state.team_stats)
    y = _player_table(img, y, state.players, state.metric)
    y = _wearable_strip(img, y, state.wearable_vitals)
    if state.metric:
        y = _recs(img, y, state.recommendations)
    _text(img, f"wearables bound: {state.tagged_count}", (18, img.shape[0] - 18),
          0.5, (255, 255, 0))
    return img


def compose_dashboard(match, panel, radar=None, heatmap=None,
                      total_width: int = 1540) -> np.ndarray:
    """Composite dashboard sized to FIT a laptop screen at native text size:

        ┌───────────────┬─────────┐
        │  match video  │  panel  │   left column = total_width - PANEL_W
        ├───────┬───────┤ (native │   panel is NEVER squashed → text readable
        │ radar │ heat  │  680px) │
        └───────┴───────┴─────────┘
    """
    left_w = max(total_width - PANEL_W, 640)
    mh, mw = match.shape[:2]
    m = cv2.resize(match, (left_w, int(mh * left_w / mw)), interpolation=cv2.INTER_AREA)

    # radar + heatmap row under the match, sharing the column width
    tiles = [x for x in (radar, heatmap) if x is not None]
    if tiles:
        tw = (left_w - 8 * (len(tiles) - 1)) // len(tiles)
        row_h = max(int(t.shape[0] * tw / t.shape[1]) for t in tiles)
        row = np.full((row_h, left_w, 3), _BG, np.uint8)
        x0 = 0
        for t in tiles:
            r = cv2.resize(t, (tw, int(t.shape[0] * tw / t.shape[1])),
                           interpolation=cv2.INTER_AREA)
            row[:r.shape[0], x0:x0 + tw] = r
            x0 += tw + 8
        left = np.vstack([m, np.full((6, left_w, 3), _BG, np.uint8), row])
    else:
        left = m

    # right column: panel at NATIVE size (text never scaled down)
    lh = left.shape[0]
    if panel.shape[0] < lh:                       # pad panel to column height
        pad = np.full((lh - panel.shape[0], panel.shape[1], 3), _BG, np.uint8)
        p = np.vstack([panel, pad])
    else:                                          # pad the left column instead
        pad = np.full((panel.shape[0] - lh, left_w, 3), _BG, np.uint8)
        left = np.vstack([left, pad])
        p = panel
    return np.hstack([left, p])


WINDOW = "Soccer Analytics — live  (q quit, space pause)"


def display_available() -> bool:
    """True if an X display is actually reachable. A dead display makes Qt ABORT
    the whole process (uncatchable core dump), so this must be checked BEFORE any
    cv2 window call. Under WSL, a dead WSLg is revived by `wsl --shutdown`."""
    import os
    import socket
    disp = os.environ.get("DISPLAY", "")
    if not disp:
        return False
    num = disp.rsplit(":", 1)[-1].split(".")[0] or "0"
    sock_path = f"/tmp/.X11-unix/X{num}"
    if not os.path.exists(sock_path):
        return False
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(0.5)
        s.connect(sock_path)
        s.close()
        return True
    except OSError:
        return False


class Dashboard:
    """Shows the composited dashboard live in ONE window, auto-fitted to the
    screen (``display_width``). The recording keeps full resolution — only the
    on-screen image is scaled, so text stays crisp in the mp4 and readable live.
    ``windows=False`` for headless runs (still records if an out path is set)."""

    def __init__(self, windows: bool = True, display_width: int = 1540):
        if windows and not display_available():
            print("[dashboard] no reachable display — running headless (recording "
                  "continues). Under WSL: run `wsl --shutdown` from PowerShell and "
                  "reopen the terminal to revive WSLg, then rerun for the live window.")
            windows = False
        self.windows = windows
        self.display_width = display_width
        self._created = False

    def show_composite(self, composite) -> int:
        """Display the full composite dashboard; returns the pressed key (-1 if
        headless). ``q`` to quit, ``space`` to pause."""
        if not self.windows or composite is None:
            return -1
        try:
            ch, cw = composite.shape[:2]
            if cw > self.display_width:          # fit the screen horizontally
                s = self.display_width / cw
                composite = cv2.resize(composite, (self.display_width, int(ch * s)),
                                       interpolation=cv2.INTER_AREA)
            if not self._created:
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW, composite.shape[1], composite.shape[0])
                self._created = True
            cv2.imshow(WINDOW, composite)
            return cv2.waitKey(1) & 0xFF
        except cv2.error:
            # no display available (e.g. WSLg not running) → go headless, keep running
            print("[dashboard] no display — continuing headless (still recording if --out)")
            self.windows = False
            return -1

    def wait(self) -> int:                     # for the paused state
        return cv2.waitKey(30) & 0xFF if self.windows else -1

    def close(self):
        if self.windows:
            cv2.destroyAllWindows()
