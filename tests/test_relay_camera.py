"""Phone camera pairing on the relay: auth, signalling and newest-frame forwarding."""

import json

import pytest
from fastapi.testclient import TestClient

import cpg44_api.hostinger_sensor_relay as relay_module
from cpg44_api import relay_camera

TOKEN = "k" * 40


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(relay_module, "RELAY_TOKEN", TOKEN)
    relay_camera.ROOMS.rooms.clear()
    return TestClient(relay_module.app)


def _room(client):
    r = client.post("/api/v1/cam/rooms", headers={"X-Auth": TOKEN})
    assert r.status_code == 200
    return r.json()


def test_room_needs_master_token_and_key_is_not_the_token(client):
    assert client.post("/api/v1/cam/rooms", headers={"X-Auth": "wrong"}).status_code == 401
    room = _room(client)
    assert room["key"] != TOKEN and room["phone_path"].startswith("/cam#r=")
    with pytest.raises(Exception):
        with client.websocket_connect(f"/ws/cam/{room['room']}?role=phone&key=bad") as ws:
            ws.receive_text()


def test_signalling_is_forwarded_between_phone_and_base(client):
    room = _room(client)
    base_url = f"/ws/cam/{room['room']}?role=base&key={room['key']}"
    phone_url = f"/ws/cam/{room['room']}?role=phone&key={room['key']}"
    with client.websocket_connect(base_url) as base:
        assert json.loads(base.receive_text()) == {"type": "peer", "present": False}
        with client.websocket_connect(phone_url) as phone:
            assert json.loads(phone.receive_text())["present"] is True
            assert json.loads(base.receive_text())["present"] is True
            phone.send_text(json.dumps({"type": "offer", "sdp": "v=0"}))
            assert json.loads(base.receive_text()) == {"type": "offer", "sdp": "v=0"}
            base.send_text(json.dumps({"type": "answer", "sdp": "v=0 a"}))
            assert json.loads(phone.receive_text())["type"] == "answer"
            phone.send_text(json.dumps({"type": "evil", "x": 1}))       # unknown types are dropped
            phone.send_text(json.dumps({"type": "stats", "mode": "direct", "fps": 30}))
            assert json.loads(base.receive_text())["type"] == "stats"


def test_fallback_frames_go_phone_to_base_only(client):
    room = _room(client)
    with client.websocket_connect(f"/ws/cam/{room['room']}?role=base&key={room['key']}") as base:
        base.receive_text()
        with client.websocket_connect(f"/ws/cam/{room['room']}?role=phone&key={room['key']}") as phone:
            phone.receive_text(); base.receive_text()
            jpeg = b"\xff\xd8" + b"x" * 100 + b"\xff\xd9"
            phone.send_bytes(jpeg)
            assert base.receive_bytes() == jpeg
            phone.send_bytes(b"not a jpeg")                         # rejected, nothing forwarded
            phone.send_text(json.dumps({"type": "stats", "mode": "relay"}))
            assert json.loads(base.receive_text())["mode"] == "relay"
    status = client.get(f"/api/v1/cam/rooms/{room['room']}", headers={"X-Auth": TOKEN}).json()
    assert status["fallback_frames"]["in"] == 1 and status["fallback_frames"]["out"] == 1


def test_phone_page_is_public_and_keeps_key_in_fragment(client):
    html = client.get("/cam").text
    assert "location.hash" in html and "getUserMedia" in html


def _ws(room, role, key=None, exp=True):
    url = f"/ws/cam/{room['room']}?role={role}&key={key or room['key']}"
    return url + (f"&exp={room['expires_at']}" if exp else "")


def _closed_with(client, url):
    from starlette.websockets import WebSocketDisconnect
    with client.websocket_connect(url) as ws:
        with pytest.raises(WebSocketDisconnect) as err:
            ws.receive_text()
    return err.value.code


def test_signed_code_survives_a_relay_restart(client):
    room = _room(client)
    assert room["phone_path"].endswith(f"&e={room['expires_at']}")
    relay_camera.ROOMS.rooms.clear()                       # what a redeploy does
    with client.websocket_connect(_ws(room, "base")) as base:
        assert json.loads(base.receive_text()) == {"type": "peer", "present": False}
    # without the expiry, or with a forged key, the room is not rebuilt
    relay_camera.ROOMS.rooms.clear()
    assert _closed_with(client, _ws(room, "phone", exp=False)) == relay_camera.CLOSE_ENDED
    assert _closed_with(client, _ws(room, "phone", key="0" * 32)) == relay_camera.CLOSE_ENDED


def test_expired_and_ended_codes_close_with_4404(client):
    room = _room(client)
    relay_camera.ROOMS.rooms.clear()
    stale = dict(room, expires_at=room["expires_at"] - 10 * 24 * 3600)
    assert _closed_with(client, _ws(stale, "phone")) == relay_camera.CLOSE_ENDED
    with client.websocket_connect(_ws(room, "phone")) as phone:
        phone.receive_text()
        assert client.delete(f"/api/v1/cam/rooms/{room['room']}", headers={"X-Auth": "bad"}).status_code == 401
        assert client.delete(f"/api/v1/cam/rooms/{room['room']}", headers={"X-Auth": TOKEN}).json()["was_open"]
        from starlette.websockets import WebSocketDisconnect
        with pytest.raises(WebSocketDisconnect) as err:
            phone.receive_text()
        assert err.value.code == relay_camera.CLOSE_ENDED
    assert _closed_with(client, _ws(room, "phone")) == relay_camera.CLOSE_ENDED   # stays ended


def test_second_phone_replaces_the_first_with_4409(client):
    room = _room(client)
    from starlette.websockets import WebSocketDisconnect
    with client.websocket_connect(_ws(room, "phone")) as first:
        first.receive_text()
        with client.websocket_connect(_ws(room, "phone")) as second:
            second.receive_text()
            with pytest.raises(WebSocketDisconnect) as err:
                while True:
                    first.receive_text()
            assert err.value.code == relay_camera.CLOSE_REPLACED


def test_answer_keeps_the_offer_id_and_ping_gets_pong(client):
    room = _room(client)
    with client.websocket_connect(_ws(room, "base")) as base:
        base.receive_text()
        base.send_text(json.dumps({"type": "ping", "t": 1}))
        assert json.loads(base.receive_text())["type"] == "pong"
        with client.websocket_connect(_ws(room, "phone")) as phone:
            phone.receive_text(); base.receive_text()
            base.send_text(json.dumps({"type": "control", "offer": True}))
            assert json.loads(phone.receive_text()) == {"type": "control", "offer": True}
            base.send_text(json.dumps({"type": "answer", "id": 3, "sdp": "v=0"}))
            assert json.loads(phone.receive_text())["id"] == 3
