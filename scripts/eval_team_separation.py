"""Team separation accuracy against SoccerNet ground-truth sides.

For each validation clip: tracks come from the cached detections (frozen
tracker + linker settings), kit colours are read from the frames, and two
methods assign sides:

* ``frame_vote`` - previous behaviour: fit 2-means on warm-up colours, then a
  rolling per-track vote (team_assign.TeamAssigner.assign_from_color);
* ``identity_cluster`` - new: cluster *identities* by median kit colour with
  outlier abstention (match_analysis._assign_teams).

Predicted identities are matched to ground-truth tracklets by majority IoU;
accuracy is computed up to the left/right label swap over players and
goalkeepers. Coverage = share of matched identities that received a side.

    PYTHONPATH=. python scripts/eval_team_separation.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_tracking as bt  # noqa: E402
from soccer_analytics.core import PLAYER  # noqa: E402
from soccer_analytics.match_analysis import _assign_teams  # noqa: E402
from soccer_analytics.mot_eval import box_iou_xywh  # noqa: E402
from soccer_analytics.team_assign import TeamAssigner  # noqa: E402
from soccer_analytics.tracklet_link import TrackletLinker, tracklets_from_frames  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def gt_sides(seq):
    import configparser
    info = configparser.ConfigParser()
    info.read(seq / "gameinfo.ini")
    side = {}
    for k, v in info["Sequence"].items():
        if k.startswith("trackletid_"):
            v = v.strip()
            if "team left" in v:
                side[int(k.split("_")[1])] = "L"
            elif "team right" in v:
                side[int(k.split("_")[1])] = "R"
    return side


def run(seq):
    dets, emb, warps, n = bt.load_cache(seq, "soccernet_1280")
    reid = bt.load_cache(seq, "soccernet_1280", "reid")[1]
    pf, _ = bt.run_imm(dets, n, reid, warps, use_imm=False, app_veto=0.65, app_weight=0.5)
    linked, res = bt.link(pf, dets, reid, max_gap=750, motion_gate=1.2, app_gate=0.5, interpolate_gap=20)
    ident = res.identity_of
    images = sorted((seq / "img1").glob("*.jpg"))
    ta = TeamAssigner()
    colours, warm = {}, []
    for f in range(0, n, 3):
        frame = cv2.imread(str(images[f]))
        for tid, box in pf.get(f, []):
            c = ta.shirt_color(frame, box)
            if c is not None:
                colours.setdefault(ident[tid], []).append((f, tid, c))
                if f < 120:
                    warm.append(c)
    # new: identity-level clustering
    id_cols = {i: [c for _, _, c in rows] for i, rows in colours.items()}
    new_team, _ = _assign_teams(id_cols, {i: PLAYER for i in id_cols}, TeamAssigner())
    # old: frame-level vote keyed by raw track id, then majority per identity
    old = TeamAssigner()
    old.fit(warm)
    votes = {}
    for i, rows in colours.items():
        for f, tid, c in sorted(rows, key=lambda r: r[0]):
            t = old.assign_from_color(tid, c)
            if t:
                votes.setdefault(i, []).append(t)
    old_team = {i: Counter(v).most_common(1)[0][0] for i, v in votes.items()}
    # ReID-embedding clustering of identities
    from soccer_analytics.match_analysis import _assign_teams_embedding
    id_emb = {}
    emb_map = bt.det_embeddings(pf, dets, reid)
    for tid, per in emb_map.items():
        v = np.mean(list(per.values()), axis=0)
        id_emb.setdefault(ident[tid], []).append((len(per), v))
    id_vec = {i: np.average([v for _, v in rows], axis=0, weights=[w for w, _ in rows]) for i, rows in id_emb.items()}
    id_n = {i: sum(w for w, _ in rows) for i, rows in id_emb.items()}
    reid_team, _ = _assign_teams_embedding(id_vec, id_n, {i: PLAYER for i in id_vec})

    gt = bt.gt_for(seq)[0]
    sides = gt_sides(seq)
    overlap = {}
    for f, items in linked.items():
        if f + 1 not in gt:
            continue
        gids, gb = gt[f + 1]
        pb = bt.to_xywh([b for _, b in items])
        iou = box_iou_xywh(gb, pb)
        for j, (pid, _) in enumerate(items):
            k = int(np.argmax(iou[:, j])) if len(gids) else -1
            if k >= 0 and iou[k, j] >= 0.5:
                overlap.setdefault(int(pid), Counter())[int(gids[k])] += 1
    # linked ids are already final identities
    matched = {i: c.most_common(1)[0][0] for i, c in overlap.items()}
    out = {}
    for name, team in (("frame_vote", old_team), ("identity_cluster", new_team), ("reid_cluster", reid_team)):
        pairs = [(team.get(i, 0), sides.get(g)) for i, g in matched.items() if sides.get(g)]
        assigned = [(t, s) for t, s in pairs if t in (1, 2)]
        a = sum(1 for t, s in assigned if (t == 1) == (s == "L"))
        acc = max(a, len(assigned) - a) / max(1, len(assigned))
        out[name] = {"accuracy": acc, "coverage": len(assigned) / max(1, len(pairs)), "identities": len(pairs)}
    return out


def main():
    rows = {}
    for seq in bt.sequences("val", 0):
        rows[seq.name] = run(seq)
        print(seq.name, rows[seq.name], flush=True)
    summary = {m: {k: float(np.mean([r[m][k] for r in rows.values()])) for k in ("accuracy", "coverage")}
               for m in ("frame_vote", "identity_cluster", "reid_cluster")}
    summary["per_clip"] = rows
    out = ROOT / "data" / "benchmarks" / "team_separation_val.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_clip"}, indent=2))


if __name__ == "__main__":
    main()
