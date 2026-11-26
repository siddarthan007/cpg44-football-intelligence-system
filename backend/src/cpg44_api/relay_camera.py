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
the token and receives a per-room key with a fixed lifetime, which is what the
QR code carries (in the URL fragment, so it is not sent in requests or logged by
proxies). The key is an HMAC of the room and its expiry made with the token, so
a code stays valid when the relay restarts: the room is rebuilt the first time
either side reconnects. Without a token the key is random and lives in memory.

Close codes the two ends act on:

* 4404: the code is unknown, expired or was ended by the laptop. Retrying
  cannot help; the phone asks for a new code.
* 4409: the same role connected again elsewhere (a second tab, a second
  laptop process). The older one stops instead of fighting for the room.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

ROOM_TTL_S = 12 * 3600               # a full matchday on one code
IDLE_TIMEOUT_S = 75.0                # both ends send something at least every 15 s
CLOSE_ENDED, CLOSE_REPLACED = 4404, 4409
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


def _sign(secret: str, room: str, exp: int) -> str:
    return hmac.new(secret.encode(), f"cam:{room}:{exp}".encode(), hashlib.sha256).hexdigest()[:32]


class CameraRooms:
    def __init__(self, secret: Callable[[], str] = lambda: ""):
        self.rooms: Dict[str, Room] = {}
        self.ended: Dict[str, float] = {}   # rooms the laptop closed, until they expire
        self.secret = secret

    def prune(self, now: Optional[float] = None) -> None:
        now = now or time.time()
        for rid in [r for r, v in self.rooms.items() if v.expires < now and not v.peers]:
            del self.rooms[rid]
        for rid in [r for r, exp in self.ended.items() if exp < now]:
            del self.ended[rid]

    def _make_space(self) -> None:
        if len(self.rooms) >= MAX_ROOMS:
            oldest = min((r for r in self.rooms.values() if not r.peers), key=lambda r: r.created, default=None)
            if oldest is None:
                raise HTTPException(status_code=429, detail="too many active camera rooms")
            del self.rooms[oldest.room]

    def create(self, ttl_s: float = ROOM_TTL_S) -> Room:
        self.prune()
        self._make_space()
        now = time.time()
        room_id = secrets.token_hex(4)
        exp = int(now + max(60.0, min(ttl_s, ROOM_TTL_S)))
        secret = self.secret()
        key = _sign(secret, room_id, exp) if secret else secrets.token_urlsafe(18)
        room = Room(room_id, key, now, float(exp))
        self.rooms[room.room] = room
        return room

    def check(self, room_id: str, key: Optional[str], exp: Optional[int] = None) -> Optional[Room]:
        now = time.time()
        if not key or room_id in self.ended:
            return None
        room = self.rooms.get(room_id)
        if room is not None:
            return room if room.expires >= now and hmac.compare_digest(key, room.key) else None
        # Not in memory: the relay restarted since the code was made. A signed key
        # with a future expiry rebuilds the room.
        secret = self.secret()
        if not secret or exp is None or exp < now or not re.fullmatch(r"[0-9a-f]{8}", room_id):
            return None
        if not hmac.compare_digest(key, _sign(secret, room_id, int(exp))):
            return None
        self.prune()
        try:
            self._make_space()
        except HTTPException:
            return None
        room = Room(room_id, key, now, float(exp))
        self.rooms[room_id] = room
        return room

    def end(self, room_id: str) -> Optional[Room]:
        room = self.rooms.pop(room_id, None)
        self.ended[room_id] = room.expires if room else time.time() + ROOM_TTL_S
        return room


ROOMS = CameraRooms()


def make_router(authorized: Callable[[Optional[str]], bool], secret: Callable[[], str] = lambda: "") -> APIRouter:
    router = APIRouter()
    ROOMS.secret = secret

    @router.post("/api/v1/cam/rooms")
    async def create_room(ttl_s: float = Query(ROOM_TTL_S, ge=60, le=ROOM_TTL_S),
                          x_auth: Optional[str] = Header(None, alias="X-Auth")):
        if not authorized(x_auth):
            raise HTTPException(status_code=401, detail="invalid relay token")
        room = ROOMS.create(ttl_s)
        exp = int(room.expires)
        return {"room": room.room, "key": room.key, "expires_at": exp,
                "phone_path": f"/cam#r={room.room}&k={room.key}&e={exp}"}

    @router.delete("/api/v1/cam/rooms/{room_id}")
    async def end_room(room_id: str, x_auth: Optional[str] = Header(None, alias="X-Auth")):
        """The laptop forgot this camera: tell the phone, and refuse the code from now on."""
        if not authorized(x_auth):
            raise HTTPException(status_code=401, detail="invalid relay token")
        room = ROOMS.end(room_id)
        for sock in list(room.peers.values()) if room else []:
            try:
                await sock.close(code=CLOSE_ENDED, reason="ended by the laptop")
            except RuntimeError:
                pass
        return {"ok": True, "was_open": room is not None}

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
    async def cam_socket(ws: WebSocket, room_id: str, role: str = Query(...), key: Optional[str] = Query(None),
                         exp: Optional[int] = Query(None)):
        room = ROOMS.check(room_id, key, exp) if role in ("phone", "base") else None
        await ws.accept()      # accept first so the browser sees the close code, not a failed handshake
        if room is None:
            await ws.close(code=CLOSE_ENDED, reason="this code has ended")
            return
        old = room.peers.get(role)
        if old is not None:                       # the newest connection wins
            try:
                await old.close(code=CLOSE_REPLACED, reason="opened elsewhere")
            except RuntimeError:
                pass
        room.peers[role] = ws
        other = "base" if role == "phone" else "phone"
        await _notify_presence(room)
        sender = asyncio.create_task(_frame_pump(room, ws)) if role == "base" else None
        min_gap = 1.0 / MAX_FALLBACK_FPS
        try:
            while True:
                msg = await asyncio.wait_for(ws.receive(), timeout=IDLE_TIMEOUT_S)
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
                    try:                          # a dying peer must not end this side's session
                        await peer.send_text(text)
                    except Exception:
                        pass
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
        except Exception:                         # closed under us; its own handler cleans up
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
    except (asyncio.CancelledError, asyncio.TimeoutError, RuntimeError, WebSocketDisconnect):
        return


PHONE_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#14201b"><title>Pitch camera</title>
<style>
:root{--ink:#14201b;--paper:#f4f1ea;--ok:#2f7d4f;--warn:#c98a12;--bad:#b4442d}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--ink);color:var(--paper);font:15px/1.4 system-ui,-apple-system,sans-serif}
main{position:fixed;inset:0;display:grid;grid-template-rows:auto 1fr auto;padding:env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left)}
header,footer{display:flex;align-items:center;gap:10px;padding:12px 16px;min-width:0}
header b{font-weight:650;white-space:nowrap}.pill{margin-left:auto;padding:4px 10px;border-radius:999px;background:#ffffff1a;font-size:13px;max-width:70%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pill.ok{background:var(--ok)}.pill.warn{background:var(--warn);color:#1b1305}.pill.bad{background:var(--bad)}
.stage{position:relative;min-height:0}video{width:100%;height:100%;object-fit:contain;background:#000;display:block}
.note{position:absolute;inset:auto 16px 16px;margin:0;padding:12px 14px;border-radius:12px;background:#000a;text-align:center;font-size:14px}
.note:empty{display:none}
footer{justify-content:space-between;font:13px ui-monospace,SFMono-Regular,Menlo,monospace;color:#cfd6cf}
#stats{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.btns{display:flex;gap:8px;flex:none}
button{font:600 15px system-ui;border:0;border-radius:10px;padding:12px 18px;background:var(--paper);color:var(--ink);min-height:44px}
button.ghost{background:#ffffff1f;color:var(--paper)}button.stop{background:var(--bad);color:#fff}
@media (orientation:landscape){.note.tip{display:none}}
</style></head><body><main>
<header><b>Pitch camera</b><span id="state" class="pill" role="status">Ready</span></header>
<div class="stage"><video id="v" autoplay muted playsinline></video>
<p id="note" class="note tip">Turn the phone sideways and fix it high on a tripod with the whole pitch in view, then press Start.</p></div>
<footer><span id="stats"></span><span class="btns"><button id="flip" class="ghost" type="button">Flip</button><button id="go" type="button">Start</button></span></footer>
</main><script>
const q=new URLSearchParams(location.hash.slice(1)),room=q.get('r'),key=q.get('k'),exp=q.get('e');
const $=id=>document.getElementById(id),v=$('v'),stateEl=$('state'),statsEl=$('stats'),noteEl=$('note'),go=$('go');
const ICE=[{urls:'stun:stun.cloudflare.com:3478'},{urls:'stun:stun.l.google.com:19302'}];
let ws,pc,stream,facing='environment',mode='',lock,running=false,sending=false,retry=0,retryTimer,reportTimer,fallbackTimer,offerId=0,offerAt=0,lastBytes=0,lastT=0;
function state(t,c){stateEl.textContent=t;stateEl.className='pill '+(c||'')}
function note(t,tip){noteEl.textContent=t||'';noteEl.className='note'+(tip?' tip':'')}
if(!room||!key){state('Code missing','bad');note('Open this page by scanning the code on the laptop.');go.disabled=true}
async function camera(){
  if(stream)stream.getTracks().forEach(t=>t.stop());
  stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:facing},width:{ideal:1920},height:{ideal:1080},frameRate:{ideal:30,max:30}}});
  const tr=stream.getVideoTracks()[0];if('contentHint' in tr)tr.contentHint='motion';
  tr.onended=()=>{if(running)camera().catch(e=>stop('Camera stopped: '+e.message,'bad'))};
  v.srcObject=stream;
  if(pc){const s=pc.getSenders().find(s=>s.track&&s.track.kind==='video');if(s)await s.replaceTrack(tr)}
}
function send(o){if(ws&&ws.readyState===1)ws.send(JSON.stringify(o))}
function closePeer(){clearTimeout(fallbackTimer);if(pc){pc.onconnectionstatechange=null;pc.close();pc=null}mode=''}
function stop(msg,cls,why){
  running=false;clearTimeout(retryTimer);clearTimeout(reportTimer);closePeer();
  if(ws){ws.onclose=null;try{ws.close()}catch{}ws=null}
  if(stream){stream.getTracks().forEach(t=>t.stop());stream=null}v.srcObject=null;
  if(lock){lock.release().catch(()=>{});lock=null}
  go.textContent='Start';go.className='';statsEl.textContent='';state(msg||'Stopped',cls);note(why||'')
}
function connect(){
  clearTimeout(retryTimer);
  const u=new URL('/ws/cam/'+encodeURIComponent(room),location.href);u.protocol=u.protocol.replace('http','ws');
  u.searchParams.set('role','phone');u.searchParams.set('key',key);if(exp)u.searchParams.set('exp',exp);
  const sock=new WebSocket(u);ws=sock;sock.binaryType='arraybuffer';
  sock.onopen=()=>{retry=0;if(mode!=='direct')state('Waiting for the laptop','warn')};
  sock.onclose=e=>{if(ws!==sock||!running)return;ws=null;
    if(e.code===4404){stop('Code ended','bad','This code has ended. Show a new code on the laptop and scan it again.');return}
    if(e.code===4409){stop('Opened elsewhere','warn','This code was opened on another screen. Press Start to use this phone instead.');return}
    if(mode==='relay')mode='';
    retry++;const wait=Math.min(10000,800*2**Math.min(retry,4));
    if(mode!=='direct')state(navigator.onLine?'Reconnecting':'No network','warn');
    retryTimer=setTimeout(()=>running&&connect(),wait)};
  sock.onmessage=async e=>{if(typeof e.data!=='string')return;let m;try{m=JSON.parse(e.data)}catch{return}
    if(m.type==='peer'){if(m.present){if(!pc||pc.connectionState!=='connected')offer()}else if(mode!=='direct')state('Waiting for the laptop','warn')}
    if(m.type==='control'&&m.offer&&!(pc&&pc.connectionState==='connecting'&&Date.now()-offerAt<15000))offer(true);
    if(m.type==='answer'&&pc&&m.id===offerId&&pc.signalingState==='have-local-offer'){try{await pc.setRemoteDescription({type:'answer',sdp:m.sdp})}catch{startFallback()}}};
}
async function offer(force){
  if(!stream||(!force&&Date.now()-offerAt<5000))return;
  offerAt=Date.now();closePeer();const id=++offerId;
  const p=new RTCPeerConnection({iceServers:ICE});pc=p;
  const tr=stream.getVideoTracks()[0];
  const tx=p.addTransceiver(tr,{direction:'sendonly',streams:[stream]});
  const caps=RTCRtpSender.getCapabilities&&RTCRtpSender.getCapabilities('video');
  if(caps&&tx.setCodecPreferences){const h=caps.codecs.filter(c=>/H264/i.test(c.mimeType)),o=caps.codecs.filter(c=>!/H264/i.test(c.mimeType));if(h.length)tx.setCodecPreferences([...h,...o])}
  const prm=tx.sender.getParameters();prm.encodings=prm.encodings&&prm.encodings.length?prm.encodings:[{}];
  prm.encodings[0].maxBitrate=6e6;prm.degradationPreference='maintain-resolution';try{await tx.sender.setParameters(prm)}catch{}
  p.onconnectionstatechange=()=>{if(pc!==p)return;const s=p.connectionState;
    if(s==='connected'){mode='direct';sending=false;state('Live, direct','ok');note('');clearTimeout(fallbackTimer)}
    if(s==='disconnected'){clearTimeout(fallbackTimer);fallbackTimer=setTimeout(()=>{if(pc===p&&p.connectionState!=='connected')startFallback()},4000)}
    if(s==='failed')startFallback()};
  await p.setLocalDescription(await p.createOffer());
  await new Promise(r=>{if(p.iceGatheringState==='complete')return r();const t=setTimeout(r,2500);p.onicegatheringstatechange=()=>{if(p.iceGatheringState==='complete'){clearTimeout(t);r()}}});
  if(pc!==p)return;
  send({type:'offer',id,sdp:p.localDescription.sdp});state('Connecting video','warn');
  clearTimeout(fallbackTimer);fallbackTimer=setTimeout(()=>{if(pc===p&&mode!=='direct')startFallback()},10000);
}
const cv=document.createElement('canvas');
function startFallback(){if(mode==='relay'||!running)return;mode='relay';state('Live, through the relay','warn');note('No direct link on this network, so video goes through the relay at a lower rate.');tick()}
function tick(){
  if(mode!=='relay'||!running)return;
  if(ws&&ws.readyState===1&&!sending&&ws.bufferedAmount<200000&&v.videoWidth){sending=true;
    const w=Math.min(960,v.videoWidth),h=Math.round(w*v.videoHeight/v.videoWidth);cv.width=w;cv.height=h;
    cv.getContext('2d').drawImage(v,0,0,w,h);
    cv.toBlob(b=>{const s=ws;if(b&&s&&s.readyState===1)b.arrayBuffer().then(a=>{s.send(a);sending=false});else sending=false},'image/jpeg',.72)}
  setTimeout(tick,80);
}
async function report(){
  if(!running)return;let fps=0,kbps=0,rtt=null,w=v.videoWidth,h=v.videoHeight;
  if(pc&&mode==='direct'){try{const r=await pc.getStats();r.forEach(s=>{
    if(s.type==='outbound-rtp'&&s.kind==='video'){fps=s.framesPerSecond||0;w=s.frameWidth||w;h=s.frameHeight||h;
      const now=s.timestamp;if(lastT&&now>lastT)kbps=Math.round(8*(s.bytesSent-lastBytes)/(now-lastT));lastBytes=s.bytesSent;lastT=now}
    if(s.type==='candidate-pair'&&s.nominated&&s.currentRoundTripTime!=null)rtt=Math.round(1000*s.currentRoundTripTime)})}catch{}}
  statsEl.textContent=mode?`${w}x${h}${fps?'  '+fps.toFixed(0)+' fps':''}${kbps?'  '+(kbps/1000).toFixed(1)+' Mbps':''}`:'';
  send({type:'stats',mode,width:w,height:h,fps,kbps,rtt_ms:rtt});
  reportTimer=setTimeout(report,2000);
}
async function wake(){if('wakeLock' in navigator&&!lock){try{lock=await navigator.wakeLock.request('screen');lock.onrelease=()=>{lock=null}}catch{}}}
go.onclick=async()=>{
  if(running){stop('Stopped','', 'Press Start to send video again.');return}
  try{await camera()}catch(e){state('Camera blocked','bad');note('Allow camera access for this page in the browser settings, then press Start. ('+e.message+')');return}
  running=true;retry=0;go.textContent='Stop';go.className='stop';note('Turn the phone sideways and keep the whole pitch in view.',true);
  wake();connect();report()};
$('flip').onclick=()=>{facing=facing==='environment'?'user':'environment';if(stream)camera().catch(()=>{})};
document.addEventListener('visibilitychange',()=>{if(document.visibilityState!=='visible'||!running)return;wake();
  const tr=stream&&stream.getVideoTracks()[0];if(!tr||tr.readyState==='ended')camera().catch(()=>{});
  if(!ws){connect()}});
window.addEventListener('online',()=>{if(running&&!ws)connect()});
</script></body></html>"""
