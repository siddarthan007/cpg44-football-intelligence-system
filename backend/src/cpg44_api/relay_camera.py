"""Phone camera pairing through the relay.

Why this shape
--------------
A phone browser only opens its camera on an HTTPS page, so the capture page is
served from the relay's public HTTPS name. The video itself should not travel
through the relay: WebRTC sends hardware-encoded H.264 straight from the phone
to the field laptop (same Wi-Fi: host candidates; different networks: STUN),
typically 100-200 ms glass to frame. The relay only carries the few kilobytes
of session description that set the link up (signalling).

When no direct path exists (symmetric NAT, strict campus Wi-Fi), the phone
falls back to sending JPEG frames over the same WebSocket. The relay keeps one
frame per room and forwards the newest one when the laptop is ready, so a slow
link lowers the frame rate instead of building up delay. Nothing is written to
disk and nothing outlives the room.

Pairing never exposes the relay's master token: the laptop asks for a room with
the token and receives a random per-room key with a short lifetime, which is
what the QR code carries (in the URL fragment, so it is not sent in requests or
logged by proxies).
"""

from __future__ import annotations

import asyncio
import hmac
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

ROOM_TTL_S = 4 * 3600
MAX_ROOMS = 8
MAX_TEXT = 64 * 1024
MAX_FRAME = 600 * 1024
MAX_FALLBACK_FPS = 15.0


@dataclass
class Room:
    room: str
    key: str
    created: float
    expires: float
    peers: Dict[str, WebSocket] = field(default_factory=dict)
    frame: Optional[bytes] = None
    frame_event: asyncio.Event = field(default_factory=asyncio.Event)
    frames_in: int = 0
    frames_out: int = 0
    frames_dropped: int = 0
    last_frame_at: float = 0.0
    phone_stats: dict = field(default_factory=dict)


class CameraRooms:
    def __init__(self):
        self.rooms: Dict[str, Room] = {}

    def prune(self, now: Optional[float] = None) -> None:
        now = now or time.time()
        for rid in [r for r, v in self.rooms.items() if v.expires < now and not v.peers]:
            del self.rooms[rid]

    def create(self, ttl_s: float = ROOM_TTL_S) -> Room:
        self.prune()
        if len(self.rooms) >= MAX_ROOMS:
            oldest = min((r for r in self.rooms.values() if not r.peers), key=lambda r: r.created, default=None)
            if oldest is None:
                raise HTTPException(status_code=429, detail="too many active camera rooms")
            del self.rooms[oldest.room]
        now = time.time()
        room = Room(secrets.token_hex(4), secrets.token_urlsafe(18), now, now + max(60.0, min(ttl_s, ROOM_TTL_S)))
        self.rooms[room.room] = room
        return room

    def check(self, room_id: str, key: Optional[str]) -> Optional[Room]:
        room = self.rooms.get(room_id)
        if room is None or not key or room.expires < time.time():
            return None
        return room if hmac.compare_digest(key, room.key) else None


ROOMS = CameraRooms()


def make_router(authorized: Callable[[Optional[str]], bool]) -> APIRouter:
    router = APIRouter()

    @router.post("/api/v1/cam/rooms")
    async def create_room(ttl_s: float = Query(ROOM_TTL_S, ge=60, le=ROOM_TTL_S),
                          x_auth: Optional[str] = Header(None, alias="X-Auth")):
        if not authorized(x_auth):
            raise HTTPException(status_code=401, detail="invalid relay token")
        room = ROOMS.create(ttl_s)
        return {"room": room.room, "key": room.key, "expires_at": room.expires,
                "phone_path": f"/cam#r={room.room}&k={room.key}"}

    @router.get("/api/v1/cam/rooms/{room_id}")
    async def room_status(room_id: str, x_auth: Optional[str] = Header(None, alias="X-Auth")):
        if not authorized(x_auth):
            raise HTTPException(status_code=401, detail="invalid relay token")
        room = ROOMS.rooms.get(room_id)
        if room is None:
            raise HTTPException(status_code=404, detail="no such room")
        return {"room": room.room, "peers": sorted(room.peers), "expires_at": room.expires,
                "fallback_frames": {"in": room.frames_in, "out": room.frames_out, "dropped": room.frames_dropped},
                "phone": room.phone_stats}

    @router.websocket("/ws/cam/{room_id}")
    async def cam_socket(ws: WebSocket, room_id: str, role: str = Query(...), key: Optional[str] = Query(None)):
        room = ROOMS.check(room_id, key)
        if room is None or role not in ("phone", "base"):
            await ws.close(code=1008, reason="invalid room or key")
            return
        await ws.accept()
        old = room.peers.get(role)
        if old is not None:                       # a reconnect replaces the stale socket
            try:
                await old.close(code=1012, reason="replaced")
            except RuntimeError:
                pass
        room.peers[role] = ws
        other = "base" if role == "phone" else "phone"
        await _notify_presence(room)
        sender = asyncio.create_task(_frame_pump(room, ws)) if role == "base" else None
        min_gap = 1.0 / MAX_FALLBACK_FPS
        try:
            while True:
                msg = await asyncio.wait_for(ws.receive(), timeout=45.0)
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    data = msg["bytes"]
                    if role != "phone" or len(data) > MAX_FRAME or not data.startswith(b"\xff\xd8"):
                        continue
                    now = time.time()
                    if now - room.last_frame_at < min_gap:
                        room.frames_dropped += 1
                        continue
                    if room.frame is not None:
                        room.frames_dropped += 1   # never delivered: a newer frame wins
                    room.frame, room.last_frame_at = data, now
                    room.frames_in += 1
                    room.frame_event.set()
                    continue
                text = msg.get("text")
                if not text or len(text) > MAX_TEXT:
                    continue
                try:
                    body = json.loads(text)
                except ValueError:
                    continue
                if not isinstance(body, dict) or body.get("type") not in ("offer", "answer", "ice", "stats", "ping", "control"):
                    continue
                if body["type"] == "ping":
                    await ws.send_text(json.dumps({"type": "pong", "t": body.get("t"), "relay_t": time.time()}))
                    continue
                if body["type"] == "stats" and role == "phone":
                    room.phone_stats = {k: body[k] for k in ("mode", "width", "height", "fps", "kbps", "rtt_ms", "battery") if k in body}
                peer = room.peers.get(other)
                if peer is not None:
                    await peer.send_text(text)
        except (WebSocketDisconnect, asyncio.TimeoutError, RuntimeError):
            pass
        finally:
            if room.peers.get(role) is ws:
                del room.peers[role]
            if sender is not None:
                sender.cancel()
            await _notify_presence(room)

    @router.get("/cam", response_class=HTMLResponse, include_in_schema=False)
    async def phone_page():
        return PHONE_PAGE

    return router


async def _notify_presence(room: Room) -> None:
    for role, sock in list(room.peers.items()):
        other = "base" if role == "phone" else "phone"
        try:
            await sock.send_text(json.dumps({"type": "peer", "present": other in room.peers}))
        except RuntimeError:
            pass


async def _frame_pump(room: Room, ws: WebSocket) -> None:
    """Forward the newest fallback frame to the laptop, one at a time."""
    try:
        while True:
            await room.frame_event.wait()
            room.frame_event.clear()
            frame, room.frame = room.frame, None
            if frame is None:
                continue
            await asyncio.wait_for(ws.send_bytes(frame), timeout=2.0)
            room.frames_out += 1
    except (asyncio.CancelledError, asyncio.TimeoutError, RuntimeError):
        return


PHONE_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#14201b"><title>CPG44 pitch camera</title>
<style>
:root{--ink:#14201b;--paper:#f4f1ea;--ok:#2f7d4f;--warn:#c98a12;--bad:#b4442d}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--ink);color:var(--paper);font:15px/1.4 system-ui,-apple-system,sans-serif}
main{position:fixed;inset:0;display:grid;grid-template-rows:auto 1fr auto;padding:env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left)}
header,footer{display:flex;align-items:center;gap:10px;padding:12px 16px}
header b{font-weight:650;letter-spacing:.01em}.pill{margin-left:auto;padding:4px 10px;border-radius:999px;background:#ffffff1a;font-size:13px;white-space:nowrap}
.pill.ok{background:var(--ok)}.pill.warn{background:var(--warn);color:#1b1305}.pill.bad{background:var(--bad)}
video{width:100%;height:100%;object-fit:contain;background:#000}
footer{justify-content:space-between;font:13px ui-monospace,SFMono-Regular,Menlo,monospace;color:#cfd6cf}
button{font:600 15px system-ui;border:0;border-radius:10px;padding:12px 18px;background:var(--paper);color:var(--ink)}
button.ghost{background:#ffffff1f;color:var(--paper)}
.hint{position:absolute;left:16px;right:16px;bottom:84px;text-align:center;font-size:14px;color:#e6e9e4;text-shadow:0 1px 2px #000}
@media (orientation:landscape){.hint{display:none}}
</style></head><body><main>
<header><b>CPG44 pitch camera</b><span id="state" class="pill">Starting</span></header>
<video id="v" autoplay muted playsinline></video>
<p class="hint">Turn the phone sideways and fix it high on a tripod, whole pitch in view.</p>
<footer><span id="stats">&nbsp;</span><span><button id="flip" class="ghost" type="button">Flip</button> <button id="go" type="button">Start</button></span></footer>
</main><script>
const q=new URLSearchParams(location.hash.slice(1)),room=q.get('r'),key=q.get('k');
const $=id=>document.getElementById(id),v=$('v'),stateEl=$('state'),statsEl=$('stats');
let ws,pc,stream,facing='environment',mode='',lock,fallbackTimer,lastBytes=0,lastT=0,running=false,sending=false;
const ICE=[{urls:'stun:stun.cloudflare.com:3478'},{urls:'stun:stun.l.google.com:19302'}];
function state(t,c){stateEl.textContent=t;stateEl.className='pill '+(c||'')}
if(!room||!key){state('Open this page from the QR code','bad')}
async function camera(){
  if(stream)stream.getTracks().forEach(t=>t.stop());
  stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:facing},width:{ideal:1920},height:{ideal:1080},frameRate:{ideal:30,max:30}}});
  const tr=stream.getVideoTracks()[0];if('contentHint' in tr)tr.contentHint='motion';
  v.srcObject=stream;
  if(pc){const s=pc.getSenders().find(s=>s.track&&s.track.kind==='video');if(s)await s.replaceTrack(tr)}
}
function send(o){if(ws&&ws.readyState===1)ws.send(JSON.stringify(o))}
function connect(){
  const u=new URL('/ws/cam/'+encodeURIComponent(room),location.href);u.protocol=u.protocol.replace('http','ws');
  u.searchParams.set('role','phone');u.searchParams.set('key',key);
  ws=new WebSocket(u);ws.binaryType='arraybuffer';
  ws.onopen=()=>state('Waiting for the laptop','warn');
  ws.onclose=()=>{state('Reconnecting','warn');stopFallback();setTimeout(()=>running&&connect(),1500)};
  ws.onmessage=async e=>{let m;try{m=JSON.parse(e.data)}catch{return}
    if(m.type==='peer'){if(m.present)offer();else{state('Waiting for the laptop','warn');stopFallback()}}
    if(m.type==='answer'&&pc){await pc.setRemoteDescription({type:'answer',sdp:m.sdp})}
    if(m.type==='control'&&m.flip){facing=facing==='environment'?'user':'environment';camera()}};
}
async function offer(){
  if(pc)pc.close();stopFallback();mode='';
  pc=new RTCPeerConnection({iceServers:ICE});
  const tr=stream.getVideoTracks()[0];
  const tx=pc.addTransceiver(tr,{direction:'sendonly',streams:[stream]});
  const caps=RTCRtpSender.getCapabilities&&RTCRtpSender.getCapabilities('video');
  if(caps&&tx.setCodecPreferences){const h=caps.codecs.filter(c=>/H264/i.test(c.mimeType)),o=caps.codecs.filter(c=>!/H264/i.test(c.mimeType));if(h.length)tx.setCodecPreferences([...h,...o])}
  const p=tx.sender.getParameters();p.encodings=p.encodings&&p.encodings.length?p.encodings:[{}];
  p.encodings[0].maxBitrate=6e6;p.degradationPreference='maintain-resolution';try{await tx.sender.setParameters(p)}catch{}
  pc.onconnectionstatechange=()=>{const s=pc.connectionState;
    if(s==='connected'){mode='direct';state('Direct link','ok');stopFallback()}
    if(s==='failed'||s==='disconnected'){startFallback()}};
  await pc.setLocalDescription(await pc.createOffer());
  await new Promise(r=>{if(pc.iceGatheringState==='complete')return r();const t=setTimeout(r,2500);pc.onicegatheringstatechange=()=>{if(pc.iceGatheringState==='complete'){clearTimeout(t);r()}}});
  send({type:'offer',sdp:pc.localDescription.sdp});
  clearTimeout(fallbackTimer);fallbackTimer=setTimeout(()=>{if(mode!=='direct')startFallback()},8000);
}
const cv=document.createElement('canvas');
function startFallback(){if(mode==='relay')return;mode='relay';state('Relayed, lower rate','warn');tick()}
function stopFallback(){clearTimeout(fallbackTimer);if(mode==='relay')mode=''}
function tick(){
  if(mode!=='relay'||!ws||ws.readyState!==1)return;
  if(!sending&&ws.bufferedAmount<200000&&v.videoWidth){sending=true;
    const w=Math.min(960,v.videoWidth),h=Math.round(w*v.videoHeight/v.videoWidth);cv.width=w;cv.height=h;
    cv.getContext('2d').drawImage(v,0,0,w,h);
    cv.toBlob(b=>{if(b&&ws.readyState===1)b.arrayBuffer().then(a=>ws.send(a));sending=false},'image/jpeg',.72)}
  setTimeout(tick,80);
}
async function report(){
  if(!running)return;let fps=0,kbps=0,rtt=null,w=v.videoWidth,h=v.videoHeight;
  if(pc&&mode==='direct'){const r=await pc.getStats();r.forEach(s=>{
    if(s.type==='outbound-rtp'&&s.kind==='video'){fps=s.framesPerSecond||0;w=s.frameWidth||w;h=s.frameHeight||h;
      const now=s.timestamp;if(lastT)kbps=Math.round(8*(s.bytesSent-lastBytes)/(now-lastT));lastBytes=s.bytesSent;lastT=now}
    if(s.type==='candidate-pair'&&s.nominated&&s.currentRoundTripTime!=null)rtt=Math.round(1000*s.currentRoundTripTime)})}
  statsEl.textContent=mode?`${w}x${h}  ${fps?fps.toFixed(0)+' fps  ':''}${kbps?kbps+' kbps':''}${rtt!=null?'  '+rtt+' ms':''}`:'';
  send({type:'stats',mode,width:w,height:h,fps,kbps,rtt_ms:rtt});
  setTimeout(report,2000);
}
$('go').onclick=async()=>{if(running)return;try{await camera();running=true;$('go').textContent='Live';
  if('wakeLock' in navigator){try{lock=await navigator.wakeLock.request('screen')}catch{}}
  connect();report()}catch(e){state('Camera blocked: '+e.message,'bad')}};
$('flip').onclick=()=>{facing=facing==='environment'?'user':'environment';if(stream)camera()};
document.addEventListener('visibilitychange',async()=>{if(document.visibilityState==='visible'&&running&&'wakeLock' in navigator){try{lock=await navigator.wakeLock.request('screen')}catch{}}});
</script></body></html>"""
