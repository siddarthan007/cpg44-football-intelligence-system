import { useCallback, useEffect, useState } from 'react';
import { Glyph } from '../components/Glyph';
import { api, apiUrl } from '../lib/api';

type Camera = { id: string; name: string; type: string; source: string; status: string; frame_age_s?: number | null; calibration_path?: string };
type Vision = { running?: boolean; status?: string; camera_id?: string | null; error?: string | null };
type Calib = { id: string; name: string; has_thumb?: boolean };
type Pairing = { camera_id: string; room: string; url: string; qr_svg: string; expires_at: number };
type RoomState = { peers: string[]; phone?: { mode?: string; width?: number; height?: number; fps?: number; kbps?: number; rtt_ms?: number | null } };

const KIND: Record<string, { label: string; g: 'camera' | 'live' | 'upload' | 'sources' }> = {
  phone: { label: 'Phone', g: 'live' }, rtsp: { label: 'Network camera', g: 'camera' }, usb: { label: 'USB camera', g: 'camera' },
  file: { label: 'Video file', g: 'upload' }, browser_jpeg: { label: 'Browser', g: 'sources' },
};

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(apiUrl(path), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((data as { detail?: string }).detail ?? `HTTP ${r.status}`);
  return data as T;
}

function PhoneLink({ state }: { state?: RoomState }) {
  const p = state?.phone;
  const here = state?.peers.includes('phone');
  if (!state) return <span className="muted">Checking the relay</span>;
  if (!here) return <span className="muted">Waiting for the phone to open the link</span>;
  if (!p?.mode) return <span>Phone connected, starting video</span>;
  return <span className="num" style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
    <b style={{ color: p.mode === 'direct' ? 'var(--ok)' : 'var(--warn)' }}>{p.mode === 'direct' ? 'Direct link' : 'Through the relay'}</b>
    {p.width ? <span>{p.width}x{p.height}</span> : null}{p.fps ? <span>{Math.round(p.fps)} fps</span> : null}
    {p.rtt_ms != null ? <span>{p.rtt_ms} ms</span> : null}{p.kbps ? <span>{(p.kbps / 1000).toFixed(1)} Mbps</span> : null}
  </span>;
}

export const CameraNetworkPage = () => {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [vision, setVision] = useState<Vision>({});
  const [calibs, setCalibs] = useState<Calib[]>([]);
  const [calib, setCalib] = useState('');
  const [pairing, setPairing] = useState<Pairing>();
  const [room, setRoom] = useState<RoomState>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const [form, setForm] = useState({ type: 'rtsp', source: '', name: 'Pitch camera' });

  const refresh = useCallback(() => Promise.all([api.cameras(), api.visionStatus()]).then(([c, v]) => {
    setCameras(c as unknown as Camera[]); setVision(v as Vision);
  }).catch(() => undefined), []);
  useEffect(() => {
    void refresh();
    fetch(apiUrl('/api/v1/calibrations')).then(r => r.json()).then((rows: Calib[]) => { setCalibs(rows); if (rows[0]) setCalib(rows[0].id); }).catch(() => setCalibs([]));
    const timer = window.setInterval(() => void refresh(), 2000);
    return () => window.clearInterval(timer);
  }, [refresh]);
  useEffect(() => {
    if (!pairing) return;
    const poll = () => fetch(apiUrl(`/api/v1/phone-camera/${pairing.room}`)).then(r => r.ok ? r.json() : undefined).then(setRoom).catch(() => undefined);
    void poll();
    const timer = window.setInterval(poll, 2000);
    return () => window.clearInterval(timer);
  }, [pairing]);

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true); setError(undefined);
    try { await fn(); await refresh(); } catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); }
  };
  const pair = () => act(async () => { setRoom(undefined); setPairing(await postJson<Pairing>('/api/v1/phone-camera/pair', {})); });
  const start = (id: string) => act(() => api.startCamera(id, calib ? { calibration_id: calib } : undefined));
  const stop = () => act(() => api.stopVision());
  const register = () => act(async () => { await api.registerCamera({ ...form, calibration_path: '' }); setForm({ ...form, source: '' }); });
  const phoneLive = room?.phone?.mode;

  return <div className="page">
    <header className="page-header"><h1>Cameras</h1></header>
    {error && <div className="notice bad">{error}</div>}

    <div className="setup">
      <section className="panel" aria-label="Phone camera">
        <div className="panel-head"><h2>Use a phone as the pitch camera</h2></div>
        <div className="phone-pair">
          <div className="qr">{pairing ? <div dangerouslySetInnerHTML={{ __html: pairing.qr_svg }} aria-label="QR code for the phone" />
            : <button className="btn" disabled={busy} onClick={() => void pair()}><Glyph g="live" size={16} />Show pairing code</button>}</div>
          <ol className="steps-plain">
            <li>Scan the code with the phone and press Start.</li>
            <li>Fix the phone high on a tripod, sideways, whole pitch in view.</li>
            <li>Choose the calibration for this ground, then run tracking.</li>
          </ol>
        </div>
        {pairing && <div className="phone-state">
          <PhoneLink state={room} />
          <span style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
            <button className="btn ghost small" disabled={busy} onClick={() => void pair()}>New code</button>
            {vision.running && vision.camera_id === pairing.camera_id
              ? <button className="btn small" disabled={busy} onClick={() => void stop()}><Glyph g="pause" size={14} />Stop</button>
              : <button className="btn small" disabled={busy || !phoneLive || Boolean(vision.running)} onClick={() => void start(pairing.camera_id)}><Glyph g="play" size={14} />Run tracking</button>}
          </span>
        </div>}
      </section>

      <aside className="panel" aria-label="Ground calibration">
        <div className="panel-head"><h2>Ground</h2></div>
        {!calibs.length ? <p className="muted">No saved calibration yet. Pitch setup makes one from a single frame.</p>
          : <div className="calib-pick">{calibs.map(c => <button key={c.id} aria-pressed={calib === c.id} onClick={() => setCalib(c.id)}>
            {c.has_thumb ? <img src={apiUrl(`/api/v1/calibrations/${c.id}/thumb`)} alt="" /> : <Glyph g="pitch" size={22} />}<span>{c.name}</span></button>)}</div>}
        <p className="muted" style={{ fontSize: 12.5, marginTop: 10 }}>Used when tracking starts.</p>
      </aside>
    </div>

    <section className="panel" aria-label="Sources">
      <div className="panel-head"><h2>Sources</h2>{vision.running && <span className="num muted">tracking {vision.camera_id}, {vision.status}</span>}</div>
      {!cameras.length ? <div className="empty">No sources yet</div> : <div className="source-list">
        {cameras.map(c => { const k = KIND[c.type] ?? KIND.file; const on = vision.running && vision.camera_id === c.id;
          return <div key={c.id} className="source-row">
            <Glyph g={k.g} size={18} />
            <div><strong>{c.name}</strong><small>{k.label}{c.calibration_path ? ', calibrated' : ', no calibration'}</small></div>
            <span className={`state ${c.status}`}>{on ? 'tracking' : c.status}</span>
            {on ? <button className="btn ghost small" disabled={busy} onClick={() => void stop()}>Stop</button>
              : <button className="btn ghost small" disabled={busy || Boolean(vision.running)} onClick={() => void start(c.id)}>Run</button>}
          </div>; })}
      </div>}
      <details className="add-source">
        <summary>Add a network camera, USB camera or file</summary>
        <div className="add-source-form">
          <div className="seg" role="group" aria-label="Kind">{(['rtsp', 'usb', 'file'] as const).map(t => <button key={t} aria-pressed={form.type === t} onClick={() => setForm({ ...form, type: t })}>{KIND[t].label}</button>)}</div>
          <input aria-label="Name" value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} />
          <input aria-label="Source" value={form.source} onChange={e => setForm({ ...form, source: e.target.value })}
            placeholder={form.type === 'usb' ? '0' : form.type === 'file' ? 'demo/clips/SNMOT-060_750f.mp4' : 'rtsp://192.168.1.20/stream'} />
          <button className="btn small" disabled={busy || !form.source} onClick={() => void register()}>Add</button>
        </div>
      </details>
    </section>
  </div>;
};
