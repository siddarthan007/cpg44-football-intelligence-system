import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Glyph } from '../components/Glyph';
import { api, apiUrl } from '../lib/api';

type Link_ = { relay?: boolean; phone?: boolean; mode?: string; fps?: number; width?: number | null; height?: number | null; age_s?: number | null; ended?: boolean; error?: string | null };
type Camera = { id: string; name: string; type: string; status: string; tracking?: boolean; calibration_path?: string; link?: Link_ | null };
type Vision = { running?: boolean; status?: string; camera_id?: string | null; error?: string | null };
type Calib = { id: string; name: string; has_thumb?: boolean };
type Pairing = { camera_id: string; url: string; qr_svg: string; expires_at: number | null };

const KIND: Record<string, string> = { rtsp: 'Network camera', usb: 'USB camera', file: 'Video file', browser_jpeg: 'Browser camera' };

async function send<T>(path: string, method: string, body?: unknown): Promise<T> {
  const r = await fetch(apiUrl(path), { method, headers: { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((data as { detail?: string }).detail ?? `HTTP ${r.status}`);
  return data as T;
}

/** What the phone link is doing, in one short line. */
function linkState(c: Camera): { text: string; tone: 'ok' | 'warn' | 'bad' | 'off' } {
  const l = c.link ?? {};
  if (l.ended) return { text: 'Code ended', tone: 'bad' };
  if (c.status === 'online') return { text: l.mode === 'relay' ? 'Live through the relay' : 'Live, direct link', tone: l.mode === 'relay' ? 'warn' : 'ok' };
  if (l.phone) return { text: 'Phone connected, starting video', tone: 'warn' };
  if (l.relay) return { text: 'Waiting for the phone', tone: 'off' };
  return { text: 'Reaching the relay', tone: 'off' };
}

function PhoneCard({ cam, pairing, ground, busy, onRun, onStop, onNew, onForget, onShowCode }: {
  cam: Camera; pairing?: Pairing; ground?: string; busy: boolean; onRun: () => void; onStop: () => void; onNew: () => void; onForget: () => void; onShowCode: () => void;
}) {
  const live = cam.status === 'online';
  const st = linkState(cam);
  const [tick, setTick] = useState(0);
  useEffect(() => {                                  // a light preview: one small still a second
    if (!live) return;
    const t = window.setInterval(() => setTick(n => n + 1), 1000);
    return () => window.clearInterval(t);
  }, [live]);
  const showQr = !live && pairing && !cam.link?.ended;
  return <section className="panel phone-card" aria-label="Phone camera">
    <div className="panel-head">
      <h2>Phone camera</h2>
      <span className={`pill ${st.tone}`}><i />{st.text}</span>
    </div>
    <div className="phone-body">
      <div className="phone-view">
        {live ? <img src={apiUrl(`/api/v1/cameras/${cam.id}/frame?w=640&t=${tick}`)} alt="What the phone sees" />
          : showQr ? <div className="qr" dangerouslySetInnerHTML={{ __html: pairing.qr_svg }} role="img" aria-label="Pairing code for the phone" />
          : <div className="phone-wait">
            <Glyph g={cam.link?.ended ? 'warn' : 'camera'} size={28} />
            <span>{cam.link?.ended ? 'This code has ended. Make a new one and scan it again.' : 'No video yet'}</span>
            {!cam.link?.ended && <button className="btn ghost small" onClick={onShowCode}>Show the code</button>}
          </div>}
      </div>
      <div className="phone-side">
        {live ? <dl className="facts">
          <div><dt>Picture</dt><dd className="num">{cam.link?.width}x{cam.link?.height}</dd></div>
          <div><dt>Frames</dt><dd className="num">{Math.round(cam.link?.fps ?? 0)}<small> fps</small></dd></div>
          <div><dt>Ground</dt><dd>{ground ?? (cam.calibration_path ? 'calibrated' : <span className="state-warn">not set</span>)}</dd></div>
        </dl> : <ol className="steps-plain">
          <li>Scan the code with the phone and press Start.</li>
          <li>Fix it high on a tripod, sideways, with the whole pitch in view.</li>
          <li>When the picture shows here, run tracking.</li>
        </ol>}
        <div className="phone-actions">
          {cam.tracking
            ? <button className="btn" disabled={busy} onClick={onStop}><Glyph g="pause" size={14} />Stop tracking</button>
            : <button className="btn" disabled={busy || !live} onClick={onRun}><Glyph g="play" size={14} />Run tracking</button>}
          {live && <Link className="btn ghost" to={`/prepare/pitch?camera=${cam.id}`}><Glyph g="pitch" size={14} />Calibrate from this view</Link>}
        </div>
        <div className="phone-actions quiet">
          <button className="link-btn" disabled={busy} onClick={onNew}>New code</button>
          <button className="link-btn danger" disabled={busy} onClick={onForget}>Forget this phone</button>
        </div>
      </div>
    </div>
  </section>;
}

export const CameraNetworkPage = () => {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [vision, setVision] = useState<Vision>({});
  const [calibs, setCalibs] = useState<Calib[]>([]);
  const [calib, setCalib] = useState('');
  const [pairing, setPairing] = useState<Pairing>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const [form, setForm] = useState({ type: 'rtsp', source: '', name: '' });

  const refresh = useCallback(() => Promise.all([api.cameras(), api.visionStatus()]).then(([c, v]) => {
    setCameras(c as unknown as Camera[]); setVision(v as Vision);
  }).catch(() => undefined), []);
  useEffect(() => {
    void refresh();
    fetch(apiUrl('/api/v1/calibrations')).then(r => r.json()).then((rows: Calib[]) => { setCalibs(rows); if (rows[0]) setCalib(c => c || rows[0].id); }).catch(() => setCalibs([]));
    const timer = window.setInterval(() => void refresh(), 2000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const phone = cameras.find(c => c.type === 'phone');
  const others = cameras.filter(c => c.type !== 'phone');
  // after a reload the code is fetched again from the saved pairing
  useEffect(() => {
    if (phone && pairing?.camera_id !== phone.id && phone.status !== 'online')
      send<Pairing>(`/api/v1/cameras/${phone.id}/pairing`, 'GET').then(setPairing).catch(() => undefined);
  }, [phone?.id, phone?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true); setError(undefined);
    try { await fn(); await refresh(); } catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); }
  };
  const pair = () => act(async () => setPairing(await send<Pairing>('/api/v1/phone-camera/pair', 'POST', { replace: phone?.id, calibration_id: calib || undefined })));
  const start = (id: string) => act(() => api.startCamera(id, calib ? { calibration_id: calib } : undefined));
  const stop = () => act(() => api.stopVision());
  const forget = (c: Camera) => {
    if (!window.confirm(`Remove ${c.name}?${c.tracking ? ' Tracking stops.' : ''}`)) return;
    void act(async () => { await send(`/api/v1/cameras/${c.id}`, 'DELETE'); if (c.id === pairing?.camera_id) setPairing(undefined); });
  };
  const register = () => act(async () => { await api.registerCamera({ ...form, name: form.name || KIND[form.type] }); setForm({ ...form, source: '', name: '' }); });
  const tracked = cameras.find(c => c.tracking);

  return <div className="page">
    <header className="page-header top">
      <h1>Cameras</h1>
      {tracked && <span className="pill ok"><i />Tracking {tracked.name}</span>}
    </header>
    {error && <div className="notice bad" role="alert">{error}</div>}
    {vision.error && !vision.running && <div className="notice bad">Tracking stopped: {vision.error}</div>}

    <div className="setup">
      {phone ? <PhoneCard cam={phone} pairing={pairing?.camera_id === phone.id ? pairing : undefined} busy={busy}
        ground={calibs.find(c => c.id === calib)?.name}
        onRun={() => void start(phone.id)} onStop={() => void stop()} onNew={() => void pair()} onForget={() => forget(phone)}
        onShowCode={() => void send<Pairing>(`/api/v1/cameras/${phone.id}/pairing`, 'GET').then(setPairing).catch(e => setError(String(e)))} />
        : <section className="panel phone-empty" aria-label="Phone camera">
          <Glyph g="camera" size={30} />
          <div>
            <h2>Use a phone as the pitch camera</h2>
            <p className="muted">Video goes straight from the phone to this laptop. Any phone with a browser works.</p>
          </div>
          <button className="btn" disabled={busy} onClick={() => void pair()}><Glyph g="live" size={15} />Pair a phone</button>
        </section>}

      <aside className="panel" aria-label="Ground calibration">
        <div className="panel-head"><h2>Ground</h2><Link className="link-btn" to="/prepare/pitch">New</Link></div>
        {!calibs.length ? <p className="muted">No ground saved yet. Pitch setup makes one from a single frame.</p>
          : <div className="calib-pick">{calibs.map(c => <button key={c.id} aria-pressed={calib === c.id} onClick={() => setCalib(c.id)}>
            {c.has_thumb ? <img src={apiUrl(`/api/v1/calibrations/${c.id}/thumb`)} alt="" /> : <Glyph g="pitch" size={22} />}<span>{c.name}</span></button>)}</div>}
        <p className="muted small-note">Applied when tracking starts.</p>
      </aside>
    </div>

    <section className="panel" aria-label="Other sources">
      <div className="panel-head"><h2>Other sources</h2></div>
      {others.length > 0 && <ul className="source-list">
        {others.map(c => <li key={c.id} className="source-row">
          <Glyph g={c.type === 'file' ? 'upload' : 'camera'} size={18} />
          <div className="source-name"><strong>{c.name}</strong><small>{KIND[c.type] ?? c.type}</small></div>
          <span className={`state ${c.tracking ? 'online' : ''}`}>{c.tracking ? 'tracking' : c.status === 'online' ? 'online' : ''}</span>
          {c.tracking ? <button className="btn ghost small" disabled={busy} onClick={() => void stop()}>Stop</button>
            : <button className="btn ghost small" disabled={busy || Boolean(vision.running)} onClick={() => void start(c.id)}>Run</button>}
          <button className="icon-btn" aria-label={`Remove ${c.name}`} title="Remove" disabled={busy} onClick={() => forget(c)}><Glyph g="x" size={14} /></button>
        </li>)}
      </ul>}
      <div className="add-source-form">
        <div className="seg" role="group" aria-label="Kind">{(['rtsp', 'usb', 'file'] as const).map(t =>
          <button key={t} aria-pressed={form.type === t} onClick={() => setForm({ ...form, type: t })}>{KIND[t]}</button>)}</div>
        <input aria-label="Source" value={form.source} onChange={e => setForm({ ...form, source: e.target.value })}
          placeholder={form.type === 'usb' ? 'Camera number, such as 0' : form.type === 'file' ? 'demo/clips/SNMOT-060_750f.mp4' : 'rtsp://192.168.1.20/stream'} />
        <input aria-label="Name" value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} placeholder="Name (optional)" />
        <button className="btn small" disabled={busy || !form.source.trim()} onClick={() => void register()}>Add</button>
      </div>
    </section>
  </div>;
};
