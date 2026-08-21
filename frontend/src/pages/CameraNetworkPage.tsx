import { useEffect, useState } from 'react';
import { EmptyState, ErrorBanner, PageHeader, StatusBadge } from '../components/common';
import { api, apiUrl } from '../lib/api';

type Camera = { id: string; name: string; type: string; source: string; status: string; frame_age_s?: number | null; calibrated?: boolean };

export const CameraNetworkPage = () => {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [name, setName] = useState('Pitch camera');
  const [source, setSource] = useState('');
  const [type, setType] = useState('rtsp');
  const [nonce, setNonce] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const refresh = () => api.cameras().then((body) => { setCameras(body as unknown as Camera[]); setError(null); }).catch((reason: unknown) => setError(reason instanceof Error ? reason.message : 'Camera API unavailable'));
  useEffect(() => { void refresh(); const timer = window.setInterval(() => { void refresh(); setNonce(Date.now()); }, 1000); return () => window.clearInterval(timer); }, []);
  const register = async () => {
    try { await api.registerCamera({ name, source, type }); setSource(''); await refresh(); }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Camera was not registered'); }
  };
  const phoneUrl = import.meta.env.DEV
    ? `${window.location.protocol}//${window.location.hostname}:8000/camera?id=phone_1`
    : `${window.location.origin}/camera?id=phone_1`;
  return (
    <div className="page">
      <PageHeader eyebrow="VIDEO SOURCES" title="Camera gateway" description="Register real sources and inspect their measured health. Browser phones can send preview frames; RTSP sources remain the preferred path for full-rate inference." />
      <ErrorBanner message={error} />
      <div className="two-column">
        <section className="card-clean">
          <div className="section-heading"><div><span className="eyebrow">REGISTRY</span><h2>Camera sources</h2></div><StatusBadge status={cameras.some((camera) => camera.status === 'online') ? 'online' : 'offline'} /></div>
          {!cameras.length ? <EmptyState title="No camera has been registered." /> : <div className="camera-list">{cameras.map((camera) => <article key={camera.id}><div><strong>{camera.name}</strong><span>{camera.type} · {camera.source || 'source not supplied'}</span></div><div><StatusBadge status={camera.status} /><span>{camera.frame_age_s == null ? 'no frames' : `${camera.frame_age_s}s old`}</span></div>{camera.type === 'browser_jpeg' && <img src={`${apiUrl(`/api/v1/cameras/${camera.id}/frame`)}?t=${nonce}`} alt={`${camera.name} latest frame`} />}</article>)}</div>}
        </section>
        <section className="card-clean">
          <div className="section-heading"><div><span className="eyebrow">ADD SOURCE</span><h2>Register camera</h2></div></div>
          <div className="form-grid">
            <label className="field wide"><span>Name</span><input value={name} onChange={(event) => setName(event.target.value)} /></label>
            <label className="field"><span>Transport</span><select value={type} onChange={(event) => setType(event.target.value)}><option value="rtsp">RTSP</option><option value="file">Video file</option></select></label>
            <label className="field wide"><span>Source URL or path</span><input value={source} onChange={(event) => setSource(event.target.value)} placeholder="rtsp://… or /path/video.mp4" /></label>
          </div>
          <button className="btn-solid" disabled={!name || !source} onClick={register}>Register source</button>
          <div className="quality-callout neutral camera-phone"><strong>Phone preview ingest</strong><p>Open this URL on a phone connected to the same network:</p><code>{phoneUrl}</code><p>It sends JPEG previews at up to 5 fps. It is not labelled as a full-rate inference stream.</p></div>
        </section>
      </div>
    </div>
  );
};
