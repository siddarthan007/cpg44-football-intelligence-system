"""Two fixed cameras on a futsal court fused into one board."""

import numpy as np

from soccer_analytics.multicam import audio_offset, fuse

CAMS = [(-1.0, -1.0, 4.0), (41.0, 21.0, 4.0)]          # diagonally opposite corners


def _truth(n=10, secs=60, rate=5, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, secs, 1 / rate)
    pos = np.zeros((len(t), n, 2))
    pos[0] = rng.uniform([2, 2], [38, 18], (n, 2))
    for k in range(1, len(t)):
        pos[k] = np.clip(pos[k - 1] + rng.normal(0, 0.25, (n, 2)), [0.5, 0.5], [39.5, 19.5])
    return t, pos


def _camera_view(t, pos, cam, id_base, offset, seed):
    rng = np.random.default_rng(seed)
    frames = []
    for k, tt in enumerate(t):
        rows = []
        for i, (x, y) in enumerate(pos[k]):
            d = np.sqrt((x - cam[0]) ** 2 + (y - cam[1]) ** 2 + cam[2] ** 2)
            sig = 0.25 + 0.02 * d
            if d > 34 and rng.random() < 0.5:                 # far players often missed
                continue
            rows.append([id_base + i, x + rng.normal(0, sig), y + rng.normal(0, sig), 1 + i % 2, 3.0, 0, 0])
        frames.append({"t": round(float(tt + offset), 2), "p": rows, "b": None, "pos": 0, "h": None})
    players = [{"id": id_base + i, "team": 1 + i % 2} for i in range(pos.shape[1])]
    return {"timeline": {"rate_hz": 5, "frames": frames}, "players": players, "quality": {}}


def test_fused_board_links_identities_and_is_more_accurate():
    t, pos = _truth()
    a = _camera_view(t, pos, CAMS[0], 1, 0.0, 1)
    b = _camera_view(t, pos, CAMS[1], 101, 2.0, 2)        # camera B started 2 s earlier
    m = fuse([a, b], CAMS, [0.0, 2.0])
    assert m["quality"]["identity_links"] == [10]
    err_f, err_a = [], []
    for k, f in enumerate(m["timeline"]["frames"][5:200], start=5):
        truth = pos[k]
        for row in f["p"]:
            i = row[0] - 1
            if 0 <= i < 10:
                err_f.append(np.hypot(row[1] - truth[i, 0], row[2] - truth[i, 1]))
        for row in a["timeline"]["frames"][k]["p"]:
            i = row[0] - 1
            err_a.append(np.hypot(row[1] - truth[i, 0], row[2] - truth[i, 1]))
    assert np.mean(err_f) < 0.8 * np.mean(err_a)
    seen = np.mean([len([r for r in f["p"] if r[0] <= 10]) for f in m["timeline"]["frames"]])
    assert seen > 9.8                                     # far players missed by one camera are kept by the other


def test_sound_sync_recovers_the_offset():
    rng = np.random.default_rng(4)
    rate = 100
    base = rng.gamma(2.0, 1.0, 9000) * 10
    for k in rng.choice(9000, 40, replace=False):          # whistles and ball strikes
        base[k:k + 15] += 400
    a = base[300:] + rng.normal(0, 2, 8700)                # A started 3 s later than B
    b = base + rng.normal(0, 2, 9000)
    off, r = audio_offset(a, b, rate)
    assert abs(off - (-3.0)) < 0.02 and r > 0.3
