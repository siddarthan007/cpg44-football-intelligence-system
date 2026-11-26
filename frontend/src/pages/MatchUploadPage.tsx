import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Glyph } from '../components/Glyph';
import { api, apiUrl } from '../lib/api';

type Job = { match_id: string; status: string; progress_pct?: number | null; current_frame?: number; total_frames?: number; fps?: number | null; error_message?: string | null };
type VestSession = { id: string; start_unix: number | null; duration_s: number | null; active_s: number | null; vests: number[] };
const stamp = (s: number | null) => s ? new Date(s * 1000).toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }) : 'Unknown time';
type Calib = { id: string; name: string; pitch?: { length: number; width: number }; camera: string; k1: number; has_thumb: boolean };

export const MatchUploadPage = () => {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState('Campus match');
  const [calibs, setCalibs] = useState<Calib[]>([]);
  const [calib, setCalib] = useState('');
  const [job, setJob] = useState<Job | null>(null);
  const [sessions, setSessions] = useState<VestSession[]>([]);
  const [picked, setPicked] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { fetch(apiUrl('/api/v1/vest-sessions')).then(r => r.ok ? r.json() : []).then(setSessions).catch(() => setSessions([])); }, []);
  useEffect(() => { fetch(apiUrl('/api/v1/calibrations')).then(r => r.json()).then((rows: Calib[]) => { setCalibs(rows); if (rows[0]) setCalib(rows[0].id); }).catch(() => setCalibs([])); }, []);
  useEffect(() => {
    if (!job || ['completed', 'failed'].includes(job.status)) return;
    const timer = window.setInterval(() => api.progress(job.match_id).then(b => setJob(b as unknown as Job)).catch(() => undefined), 1500);
    return () => window.clearInterval(timer);
  }, [job?.match_id, job?.status]);

  const upload = async () => {
    if (!file) return;
    setBusy(true); setError(null); setJob(null);
    const form = new FormData();
    form.append('file', file); form.append('match_name', name);
    if (calib) form.append('calibration_id', calib);
    if (picked.length) form.append('vest_sessions', picked.join(','));
    try { const r = await api.uploadMatch(form); setJob({ match_id: r.match_id, status: r.status, progress_pct: 0 }); }
    catch (e) { setError(e instanceof Error ? e.message : 'Upload failed'); } finally { setBusy(false); }
  };
  const running = Boolean(job && !['completed', 'failed'].includes(job.status));

  return <div className="page">
    <div className="top"><h1>Add footage</h1></div>
    {(error || job?.error_message) && <p className="notice bad">{error ?? job?.error_message}</p>}
    <div className="setup">
      <section className="panel">
        <label className="drop">
          <input type="file" accept="video/mp4,video/quicktime,video/x-matroska,video/webm" onChange={e => setFile(e.target.files?.[0] ?? null)} />
          <Glyph g="upload" size={34} />
          <span><strong>{file?.name ?? 'Drop the match recording'}</strong>{file ? `${(file.size / 1024 / 1024).toFixed(0)} MB` : 'MP4, MOV, MKV or WebM'}</span>
        </label>
        {job && <div style={{ marginTop: 14 }}>
          <div style={{ height: 8, borderRadius: 4, background: 'var(--paper-2)', overflow: 'hidden' }}><div style={{ width: `${job.progress_pct ?? 0}%`, height: '100%', background: 'var(--ink)', transition: 'width .5s' }} /></div>
          <div className="board-legend"><span className="num">{(job.progress_pct ?? 0).toFixed(0)}%</span><span className="num">{job.current_frame ?? 0} / {job.total_frames || '?'} frames</span>{job.fps ? <span className="num">{job.fps.toFixed(1)} fps</span> : null}
            {job.status === 'completed' && <Link className="btn small" to={`/review/${encodeURIComponent(job.match_id)}`}>Open match</Link>}</div>
        </div>}
      </section>
      <aside className="panel">
        <div className="kv" style={{ marginBottom: 14 }}><label htmlFor="mn">Name</label>
          <input id="mn" value={name} onChange={e => setName(e.target.value)} style={{ width: 170, padding: '6px 8px', border: '1px solid var(--rule-strong)', borderRadius: 7 }} /></div>
        <div className="panel-head"><h2>Pitch calibration</h2><Link className="btn ghost small" to="/prepare/pitch">New</Link></div>
        {!calibs.length ? <p className="muted">None saved. Without one, players are tracked in pixels only.</p>
          : <div style={{ display: 'grid', gap: 8 }}>{calibs.map(c => <button key={c.id} onClick={() => setCalib(c.id)} className="panel" aria-pressed={calib === c.id}
            style={{ display: 'flex', gap: 10, alignItems: 'center', textAlign: 'left', cursor: 'pointer', padding: 8, borderColor: calib === c.id ? 'var(--ink)' : undefined }}>
            {c.has_thumb ? <img src={apiUrl(`/api/v1/calibrations/${c.id}/thumb`)} alt="" style={{ width: 72, borderRadius: 6 }} /> : <Glyph g="pitch" size={28} />}
            <span><strong style={{ display: 'block' }}>{c.name}</strong><span className="muted num" style={{ fontSize: 12 }}>{c.pitch ? `${c.pitch.length} x ${c.pitch.width} m` : ''} {c.camera}</span></span>
          </button>)}
            <button className="btn ghost small" onClick={() => setCalib('')} aria-pressed={!calib}>No calibration</button></div>}
        {sessions.length > 0 && <>
          <div className="panel-head" style={{ marginTop: 18 }}><h2>Vest recordings</h2>{picked.length > 0 && <span className="muted num">{picked.length} added</span>}</div>
          <div className="vest-pick">{sessions.slice(0, 8).map(v => <label key={v.id}>
            <input type="checkbox" checked={picked.includes(v.id)} onChange={e => setPicked(p => e.target.checked ? [...p, v.id] : p.filter(x => x !== v.id))} />
            <span><strong>{stamp(v.start_unix)}</strong><small className="num">{v.vests.length ? `vest ${v.vests.join(', ')}` : 'vest'}{v.active_s ? `, ${v.active_s >= 3600 ? `${(v.active_s / 3600).toFixed(1)} h` : `${Math.max(1, Math.round(v.active_s / 60))} min`}` : ''}</small></span>
          </label>)}</div>
        </>}
        <button className="btn" style={{ marginTop: 16, width: '100%', justifyContent: 'center' }} disabled={!file || busy || running} onClick={upload}>
          <Glyph g="play" size={14} />{busy ? 'Uploading' : running ? 'Analysing' : 'Analyse'}</button>
      </aside>
    </div>
  </div>;
};
