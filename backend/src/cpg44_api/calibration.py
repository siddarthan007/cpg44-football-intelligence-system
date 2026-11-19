"""Interactive pitch calibration for the web Pitch setup flow.

A session holds one still frame (taken from an uploaded clip or photo). The
client clicks landmarks on the frame; ``solve`` returns the homography, the
lens curvature (fitted jointly when enough well-spread landmarks exist), the
pitch model projected back onto the frame (curved when the lens is curved),
and the reprojection error of every click. ``snap`` refines the fit against
the painted lines. ``save`` writes a YAML the pipeline loads directly.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml

from soccer_analytics.config import PitchConfig
from soccer_analytics.lens import loo_errors, fit_k1, undistort_frame, undistort_points
from soccer_analytics.line_refine import refine as refine_lines
from soccer_analytics.pitch import view_from_named_points

SAFE = re.compile(r"^[a-z0-9-]{1,64}$")


def _distort(pts: np.ndarray, k1: float, w: int, h: int) -> np.ndarray:
    """Forward lens model: pinhole pixels -> pixels as the camera records them."""
    if not k1:
        return pts
    c = np.array([w / 2.0, h / 2.0])
    xn = (pts - c) / float(w)
    r2 = (xn ** 2).sum(1, keepdims=True)
    return xn * (1 + k1 * r2) * w + c


def _pitch_polylines(pc: PitchConfig) -> List[np.ndarray]:
    L, W, cy = pc.length, pc.width, pc.width / 2
    segs = [[(0, 0), (L, 0)], [(L, 0), (L, W)], [(L, W), (0, W)], [(0, W), (0, 0)], [(L / 2, 0), (L / 2, W)]]
    for x0, d in ((0.0, 1.0), (L, -1.0)):
        if pc.kind == "futsal":
            # two 6 m quarter circles around the posts joined by a straight line
            r, half = pc.penalty_box_length, pc.penalty_box_width / 2
            a = np.linspace(0, np.pi / 2, 24)
            top = [(x0 + d * r * np.sin(t), cy - half - r * np.cos(t)) for t in a]
            bot = [(x0 + d * r * np.sin(t), cy + half + r * np.cos(t)) for t in a[::-1]]
            segs.append(top + bot)
            continue
        for depth, half in ((pc.penalty_box_length, pc.penalty_box_width / 2), (pc.goal_box_length, pc.goal_box_width / 2)):
            segs.append([(x0, cy - half), (x0 + d * depth, cy - half), (x0 + d * depth, cy + half), (x0, cy + half)])
    out = []
    for poly in segs:
        pts = []
        for a, b in zip(poly[:-1], poly[1:]):
            n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / 1.0))
            pts += list(np.linspace(a, b, n))
        out.append(np.asarray(pts))
    t = np.linspace(0, 2 * np.pi, 72)
    out.append(np.column_stack((L / 2 + pc.centre_circle_radius * np.cos(t), cy + pc.centre_circle_radius * np.sin(t))))
    return out


class CalibrationStudio:
    def __init__(self, work: Path, saved: Path):
        self.work, self.saved = Path(work), Path(saved)
        self.work.mkdir(parents=True, exist_ok=True)
        self.saved.mkdir(parents=True, exist_ok=True)

    # ---------------- sessions ----------------
    def _dir(self, sid: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{12}", sid or ""):
            raise KeyError("unknown calibration session")
        d = self.work / sid
        if not (d / "frame.jpg").is_file():
            raise KeyError("unknown calibration session")
        return d

    def create(self, data: bytes, filename: str, at_s: float = 2.0) -> dict:
        sid = uuid.uuid4().hex[:12]
        d = self.work / sid
        d.mkdir(parents=True)
        suffix = Path(filename or "x.mp4").suffix.lower()
        src = d / f"source{suffix}"
        src.write_bytes(data)
        frame = None
        if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            frame = cv2.imread(str(src))
        else:
            cap = cv2.VideoCapture(str(src))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(0, int(at_s * fps)), max(0, n - 1)))
            ok, frame = cap.read()
            cap.release()
        if frame is None:
            raise ValueError("could not read a frame from this file")
        cv2.imwrite(str(d / "frame.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        h, w = frame.shape[:2]
        duration = None
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            cap = cv2.VideoCapture(str(src))
            fps, n = cap.get(cv2.CAP_PROP_FPS) or 25, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = round(n / fps, 2) if fps else None
            cap.release()
        meta = {"id": sid, "width": w, "height": h, "created": time.time(), "source": Path(filename or "").name,
                "at_s": at_s, "duration_s": duration, "version": 0}
        (d / "meta.json").write_text(json.dumps(meta))
        return meta

    def regrab(self, sid: str, at_s: float) -> dict:
        """Take a different moment from the same uploaded clip."""
        d = self._dir(sid)
        src = next((p for p in d.glob("source.*")), None)
        if src is None or src.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            raise ValueError("this calibration was started from a photo")
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(0, int(at_s * fps)), max(0, n - 1)))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise ValueError("no frame at that time")
        cv2.imwrite(str(d / "frame.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        meta = json.loads((d / "meta.json").read_text())
        meta.update(at_s=at_s, duration_s=round(n / fps, 2) if fps else None, version=meta.get("version", 0) + 1)
        (d / "meta.json").write_text(json.dumps(meta))
        return meta

    def frame_path(self, sid: str) -> Path:
        return self._dir(sid) / "frame.jpg"

    def landmarks(self, length: float, width: float, kind: str = "football") -> dict:
        pc = PitchConfig.for_kind(kind, length, width)
        return {"points": {k: [round(v[0], 2), round(v[1], 2)] for k, v in pc.named_points().items()},
                "lines": [p.round(2).tolist() for p in _pitch_polylines(pc)]}

    # ---------------- fitting ----------------
    def solve(self, sid: str, points: Dict[str, List[float]], length: float, width: float,
              k1: Optional[float] = None, homography: Optional[list] = None, kind: str = "football") -> dict:
        d = self._dir(sid)
        meta = json.loads((d / "meta.json").read_text())
        w, h = meta["width"], meta["height"]
        pc = PitchConfig.for_kind(kind, float(length), float(width))
        tpl = pc.named_points()
        pts = {k: v for k, v in points.items() if k in tpl}
        if len(pts) < 4 and homography is None:
            return {"ok": False, "reason": f"{4 - len(pts)} more landmarks needed", "needed": 4 - len(pts)}
        auto = k1 is None
        lens_info = {}
        if auto:
            k1, lens_info = fit_k1([pts[k] for k in pts], [tpl[k] for k in pts], w, h)
        und = {k: undistort_points([v], k1, w, h)[0].tolist() for k, v in pts.items()}
        if homography is not None:
            H = np.asarray(homography, float)
        else:
            try:
                H = view_from_named_points(und, pc).H
            except Exception as exc:                     # degenerate clicks
                return {"ok": False, "reason": str(exc)}
        rep = self._report(H, pts, tpl, pc, k1, w, h, lens_info, auto)
        if len(und) >= 5:
            errs = loo_errors(np.asarray([und[k] for k in und], float), np.asarray([tpl[k] for k in und], float))
            loo = float(np.median(errs[np.isfinite(errs)])) if np.isfinite(errs).any() else float("inf")
            rep["check_px"] = round(float(loo), 2) if np.isfinite(loo) else None
            if rep["check_px"] is not None:
                rep["grade"] = "good" if loo <= 0.005 * w else "fair" if loo <= 0.012 * w else "poor"
        return rep

    def _report(self, H, pts, tpl, pc, k1, w, h, lens_info, auto):
        Hinv = np.linalg.inv(H)
        errors = {}
        for k, v in pts.items():
            p = cv2.perspectiveTransform(np.asarray([[tpl[k]]], float), Hinv).reshape(2)
            p = _distort(p[None], k1, w, h)[0]
            errors[k] = round(float(np.hypot(*(p - np.asarray(v)))), 1)
        lines = []
        for poly in _pitch_polylines(pc):
            hom = np.column_stack((poly, np.ones(len(poly)))) @ Hinv.T
            front = hom[:, 2] > 0 if (Hinv @ np.array([pc.length / 2, pc.width / 2, 1.0]))[2] > 0 else hom[:, 2] < 0
            px = hom[:, :2] / hom[:, 2:3]
            px = _distort(px, k1, w, h)
            seg = [[round(float(x), 1), round(float(y), 1)] if (f and abs(x) < 5 * w and abs(y) < 5 * h) else None
                   for (x, y), f in zip(px, front)]
            lines.append(seg)
        rms = float(np.sqrt(np.mean(np.square(list(errors.values()))))) if errors else None
        grade = "good" if rms is not None and rms <= 4 else "fair" if rms is not None and rms <= 10 else "poor"
        return {"ok": True, "homography": (H / H[2, 2]).tolist(), "k1": k1, "k1_auto": auto, "lens": lens_info,
                "errors_px": errors, "rms_px": None if rms is None else round(rms, 2), "grade": grade,
                "overlay": lines}

    def snap(self, sid: str, points, length, width, k1, homography, kind: str = "football") -> dict:
        """Refine the homography against painted lines on the (undistorted) frame."""
        d = self._dir(sid)
        meta = json.loads((d / "meta.json").read_text())
        w, h = meta["width"], meta["height"]
        frame = undistort_frame(cv2.imread(str(d / "frame.jpg")), float(k1 or 0))
        pc = PitchConfig.for_kind(kind, float(length), float(width))
        H = np.asarray(homography, float)
        Hn, info = refine_lines(frame, H, pc, max_shift_px=60.0)
        rep = self._report(Hn, {k: v for k, v in points.items() if k in pc.named_points()},
                           pc.named_points(), pc, float(k1 or 0), w, h, {}, False)
        rep["snap"] = info
        return rep

    # ---------------- persistence ----------------
    def save(self, sid: str, name: str, points, length, width, k1, homography, camera: str,
             kind: str = "football") -> dict:
        d = self._dir(sid)
        meta = json.loads((d / "meta.json").read_text())
        slug = re.sub(r"[^a-z0-9]+", "-", (name or "campus").lower()).strip("-")[:48] or "campus"
        if not SAFE.match(slug):
            raise ValueError("invalid name")
        if kind not in {"football", "futsal"}:
            raise ValueError("pitch kind must be football or futsal")
        if camera not in {"fixed", "panning", "auto"}:
            raise ValueError("camera must be fixed, panning or auto")
        doc = {
            "name": name, "created": time.strftime("%Y-%m-%d %H:%M"), "source": meta.get("source"),
            "image_size": [meta["width"], meta["height"]],
            "pitch": {"length": float(length), "width": float(width), "kind": kind},
            "lens": {"k1": float(k1 or 0.0)}, "camera": camera,
            "points": {k: [float(v[0]), float(v[1])] for k, v in points.items()},
            "homography": np.asarray(homography, float).tolist(),
        }
        path = self.saved / f"{slug}.yaml"
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        thumb = cv2.imread(str(d / "frame.jpg"))
        cv2.imwrite(str(self.saved / f"{slug}.jpg"), cv2.resize(thumb, (480, int(480 * thumb.shape[0] / thumb.shape[1]))))
        return {"id": slug, "path": str(path)}

    def list(self) -> List[dict]:
        out = []
        for p in sorted(self.saved.glob("*.yaml")):
            try:
                doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            out.append({"id": p.stem, "name": doc.get("name", p.stem), "created": doc.get("created"),
                        "pitch": doc.get("pitch"), "camera": doc.get("camera", "auto"),
                        "k1": (doc.get("lens") or {}).get("k1", 0.0), "has_thumb": (p.with_suffix(".jpg")).is_file()})
        return out

    def saved_path(self, cid: str) -> Optional[Path]:
        if not SAFE.match(cid or ""):
            return None
        p = self.saved / f"{cid}.yaml"
        return p if p.is_file() else None
