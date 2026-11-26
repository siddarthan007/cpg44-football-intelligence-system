"""Product API contract tests (no GPU, no Node)."""

import os
import sys
import asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))
os.environ.setdefault("CPG44_HUB_URL", "http://127.0.0.1:9")  # force hub miss

import httpx
from cpg44_api.main import STORE, create_app
from cpg44_api.store import ProductStore


def test_product_store_only_accepts_cpg44_relay_origin():
    store = ProductStore(hostinger_url="https://cpg44.nivaspms.com/")
    assert store.hostinger_url == "https://cpg44.nivaspms.com"

    try:
        ProductStore(hostinger_url="https://203.0.113.12")
    except ValueError as exc:
        assert "cpg44.nivaspms.com" in str(exc)
    else:
        raise AssertionError("direct-IP relay origin was accepted")


def test_health_and_wearable_join_key():
    async def check():
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            h = await client.get("/api/v1/health")
            assert h.status_code == 200
            assert h.json()["status"] == "ok"

            body = {
                "match_id": "MATCH_001",
                "global_player_id": "PLAYER_17",
                "timestamp": 1723984321.123,
                "source": "wearable",
                "metrics": {"gps_speed": 7.1, "acceleration": 1.9},
            }
            posted = await client.post("/api/v1/observations/wearable", json=body)
            assert posted.status_code == 200
            row = posted.json()
            assert row["match_id"] == "MATCH_001"
            assert row["global_player_id"] == "PLAYER_17"
            assert row["source"] == "wearable"

            listed = await client.get(
                "/api/v1/observations", params={"source": "wearable"}
            )
            assert listed.status_code == 200
            assert any(r["global_player_id"] == "PLAYER_17" for r in listed.json())

            matches = await client.get("/api/v1/matches")
            assert matches.status_code == 200
            assert matches.json()[0]["match_id"] == "live"

            live_body = {
                "match_id": "live",
                "timestamp": 1723984321.2,
                "source_kind": "recorded_file",
                "metric": True,
                "players": [{"global_player_id": "TRACK_7", "track_id": 7, "x": 41.2, "y": 22.4}],
                "ball": {"x": 52.5, "y": 34.0},
                "data_quality": {"status": "usable", "warnings": []},
            }
            assert STORE.ingest_live_frame(live_body)["ok"] is True

            jpeg = b"\xff\xd8CPG44-live-frame\xff\xd9"
            posted_frame = await client.post(
                "/api/v1/live/frame",
                content=jpeg,
                headers={"Content-Type": "image/jpeg"},
            )
            assert posted_frame.status_code == 200
            image, version = STORE.live_jpeg_snapshot()
            assert image == jpeg
            assert version is not None

            live = await client.get("/api/v1/live")
            assert live.status_code == 200
            assert live.json()["provenance"]["live"] is True
            assert live.json()["provenance"]["input_kind"] == "recorded_file"
            assert live.json()["players"][0]["track_id"] == 7

            frame = await client.get("/api/v1/live/frame")
            assert frame.status_code == 200
            assert frame.headers["content-type"] == "image/jpeg"
            assert frame.content == jpeg

            registered = await client.post(
                "/api/v1/cameras/register",
                json={"id": "demo_cam", "name": "Demo", "type": "file", "source": "demo/sample_match.mp4"},
            )
            assert registered.status_code == 200
            assert registered.json()["id"] == "demo_cam"

            vision = await client.get("/api/v1/vision/status")
            assert vision.status_code == 200
            assert vision.json()["running"] is False

            old_hub = STORE.hub_url
            STORE.hub_url = "http://127.0.0.1:9999"
            try:
                wearable_status = await client.get("/api/v1/wearables/status")
                assert wearable_status.status_code == 200
                assert wearable_status.json()["connected"] is False
            finally:
                STORE.hub_url = old_hub

            route_paths = {route.path for route in create_app().routes}
            assert "/api/v1/live/mjpeg" in route_paths
            assert "/api/v1/cameras/{camera_id}/mjpeg" in route_paths
    asyncio.run(check())


if __name__ == "__main__":
    test_health_and_wearable_join_key()
    print("web api tests passed")


def test_squad_round_trip_and_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("CPG44_SQUAD_FILE", str(tmp_path / "squad.json"))

    async def check():
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/api/v1/squad")).json()["players"] == []
            bad = await client.put("/api/v1/squad", json={"players": [{"name": "A", "number": 7}, {"name": "B", "number": 7}]})
            assert bad.status_code == 422 and "used twice" in bad.json()["detail"]
            bad = await client.put("/api/v1/squad", json={"players": [{"name": "A", "vest": 1}, {"name": "B", "vest": 1}]})
            assert bad.status_code == 422
            ok = await client.put("/api/v1/squad", json={"team_name": "Thapar FC", "players": [
                {"name": " Rohan Gill ", "number": "7", "vest": 1, "position": "MF"}, {"name": "Aarav", "number": None}]})
            assert ok.status_code == 200
            got = (await client.get("/api/v1/squad")).json()
            assert got["players"][0] == {"name": "Rohan Gill", "number": 7, "vest": 1, "position": "MF"}
    asyncio.run(check())


def test_live_frame_carries_pitch_geometry():
    store = ProductStore()
    store.ingest_live_frame({"players": [], "pitch": {"kind": "futsal", "length_m": 40.0, "width_m": 20.0}})
    m = store.live_frame_snapshot()["match"]
    assert (m["pitch_length_m"], m["pitch_width_m"], m["pitch_kind"]) == (40.0, 20.0, "futsal")
    # implausible sizes are ignored, the last good geometry stays
    store.ingest_live_frame({"players": [], "pitch": {"kind": "football", "length_m": 4000, "width_m": "x"}})
    m = store.live_frame_snapshot()["match"]
    assert m["pitch_length_m"] == 40.0


def test_upload_passes_name_and_vest_logs(tmp_path, monkeypatch):
    """The coach's match name and chosen vest recordings reach the analysis command."""
    import subprocess
    from soccer_analytics import modes

    seen = {}

    class FakeProc:
        stdout = iter(["[analyze] 1/1 frames\n"])
        pid = 1
        def wait(self):
            return 1
        def poll(self):
            return 1

    def fake_popen(cmd, **kw):
        seen["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    eng = modes.EngineManager(tmp_path)
    eng.jobs["m1"] = modes.ProcessingProgress(match_id="m1", mode=modes.MatchMode.UPLOAD, status="queued", started_at=0.0)
    eng._run_upload_pipeline("m1", tmp_path / "v.mp4", tmp_path / "w.pt", None, None, "Saturday friendly", ["/x/vest.ndjson"])
    cmd = seen["cmd"]
    assert cmd[cmd.index("--name") + 1] == "Saturday friendly"
    assert cmd[cmd.index("--vest") + 1] == "/x/vest.ndjson"


def test_vest_sessions_listing(tmp_path, monkeypatch):
    import json
    from cpg44_api import main
    monkeypatch.setattr(main, "ROOT", tmp_path)
    d = tmp_path / "data" / "live_sessions" / "20260821T131001Z"
    d.mkdir(parents=True)
    (d / "wearable_raw.ndjson").write_text(json.dumps({"player_id": 3, "t": "imu"}) + "\n")
    (tmp_path / "data" / "live_sessions" / "20260821T120000Z").mkdir()        # empty: not listed
    rows = main._vest_sessions()
    assert [r["id"] for r in rows] == ["20260821T131001Z"]
    assert rows[0]["vests"] == [3] and abs(rows[0]["start_unix"] - 1787317801.0) < 1
    assert main._vest_session_log("../../etc") is None


def test_relay_tls_trusts_public_roots_and_the_saved_file(tmp_path, monkeypatch):
    """A saved chain without a self-signed root must not break relay calls."""
    import ssl
    from cpg44_api import main
    monkeypatch.setenv("CPG44_RELAY_CA_FILE", str(Path(__file__).resolve().parents[1] / "configs" / "relay_ca.pem"))
    ctx = main._relay_tls()
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
    assert ctx.cert_store_stats()["x509_ca"] >= 100          # the public roots, not only the file
    monkeypatch.setenv("CPG44_RELAY_CA_FILE", str(tmp_path / "missing.pem"))
    assert main._relay_tls().cert_store_stats()["x509_ca"] >= 100


def test_camera_sources_persist_delete_and_phone_needs_video(tmp_path):
    saved = tmp_path / "cameras.json"
    store = ProductStore()
    store.load_cameras(saved)
    store.register_camera({"id": "cam_a", "name": "Stand", "type": "rtsp", "source": "rtsp://10.0.0.2/s"})
    again = ProductStore()
    again.load_cameras(saved)
    assert again.get_camera("cam_a")["name"] == "Stand"
    assert again.remove_camera("cam_a") is not None and ProductStore().get_camera("cam_a") is None

    async def check():
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # a phone row never exposes its room key, and cannot track before video arrives
            STORE.register_camera({"id": "phone_0badc0de", "name": "Phone camera", "type": "phone",
                                   "source": "phone://0badc0de?key=secret&relay=https://cpg44.nivaspms.com&exp=1"})
            rows = (await client.get("/api/v1/cameras")).json()
            row = next(r for r in rows if r["id"] == "phone_0badc0de")
            assert "source" not in row and row["status"] == "waiting"
            assert (await client.post("/api/v1/cameras/phone_0badc0de/start", json={})).status_code == 409
            pairing = (await client.get("/api/v1/cameras/phone_0badc0de/pairing")).json()
            assert pairing["url"].endswith("/cam#r=0badc0de&k=secret&e=1") and "<svg" in pairing["qr_svg"]
            assert (await client.delete("/api/v1/cameras/phone_0badc0de")).status_code == 200
            assert (await client.delete("/api/v1/cameras/phone_0badc0de")).status_code == 404
            assert (await client.get("/api/v1/cameras/nope/pairing")).status_code == 404
            bad = await client.post("/api/v1/cameras/register", json={"type": "file", "source": "no/such/clip.mp4"})
            assert bad.status_code == 422
    asyncio.run(check())


def test_pitch_tracker_reset_returns_to_calibration():
    import numpy as np
    from soccer_analytics.view import PitchHomographyTracker
    H0 = np.eye(3)
    t = PitchHomographyTracker(H0, (1080, 1920))
    t.H = np.diag([2.0, 2.0, 1.0]); t.prev_gray = np.zeros((4, 4), np.uint8)
    t.reset()
    assert np.allclose(t.H, H0) and t.prev_gray is None
