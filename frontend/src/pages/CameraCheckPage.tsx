import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Glyph } from '../components/Glyph';
import { apiUrl } from '../lib/api';

interface CheckItem { key: string; label: string; value: string | number; grade: 'good' | 'check' | 'fix'; advice: string | null }
interface CheckResult { verdict: 'good' | 'check' | 'fix'; items: CheckItem[]; overlay?: string | null }

/* What a good campus camera position looks like, drawn instead of explained. */
const Placement = () => <svg viewBox="0 0 320 170" width="100%" role="img" aria-label="Camera high on the halfway line, side of the pitch">
  <rect x="40" y="20" width="240" height="110" fill="var(--board)" stroke="var(--board-line)" />
  <line x1="160" y1="20" x2="160" y2="130" stroke="var(--board-line)" /><circle cx="160" cy="75" r="16" fill="none" stroke="var(--board-line)" />
  <rect x="40" y="48" width="30" height="54" fill="none" stroke="var(--board-line)" /><rect x="250" y="48" width="30" height="54" fill="none" stroke="var(--board-line)" />
  <path d="M160 152 L44 24 M160 152 L276 24" stroke="var(--ink)" strokeDasharray="4 4" fill="none" />
  <rect x="150" y="144" width="20" height="14" rx="3" fill="var(--ink)" /><circle cx="160" cy="151" r="3.5" fill="var(--ball)" />
  <text x="178" y="160" fontSize="11" fill="var(--ink-2)">4 to 6 m high, fixed</text>
</svg>;

export const CameraCheckPage = () => {
  const [file, setFile] = useState<File>();
  const [preview, setPreview] = useState<string>();
  const [result, setResult] = useState<CheckResult>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();

  const pick = async (f: File) => {
    setResult(undefined); setError(undefined);
    if (f.type.startsWith('video/')) {
      const url = URL.createObjectURL(f);
      const v = document.createElement('video'); v.src = url; v.muted = true;
      await new Promise<void>(r => { v.onloadeddata = () => r(); });
      v.currentTime = Math.min(2, v.duration / 2);
      await new Promise<void>(r => { v.onseeked = () => r(); });
      const c = document.createElement('canvas'); c.width = v.videoWidth; c.height = v.videoHeight;
      c.getContext('2d')!.drawImage(v, 0, 0);
      const blob = await new Promise<Blob>(r => c.toBlob(b => r(b!), 'image/jpeg', 0.92));
      URL.revokeObjectURL(url); f = new File([blob], 'frame.jpg', { type: 'image/jpeg' });
    }
    setFile(f); setPreview(URL.createObjectURL(f));
    setBusy(true);
    try {
      const form = new FormData(); form.append('image', f);
      const r = await fetch(apiUrl('/api/v1/camera-check'), { method: 'POST', body: form });
      const j = await r.json(); if (!r.ok) throw new Error(j.detail ?? r.statusText);
      setResult(j);
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };

  const verdict = result?.verdict;
  return <div className="page">
    <div className="top"><h1>Camera check</h1>
      {verdict && <span className={verdict === 'good' ? 'state-ok' : verdict === 'check' ? 'state-warn' : 'state-bad'} style={{ display: 'flex', gap: 6, alignItems: 'center', fontWeight: 700 }}>
        <Glyph g={verdict === 'fix' ? 'warn' : 'check'} size={20} />{verdict === 'good' ? 'Ready to record' : verdict === 'check' ? 'Usable' : 'Fix before kick-off'}</span>}
    </div>
    {error && <p className="notice bad">{error}</p>}
    <div className="setup">
      <section className="panel">
        <label className="drop" style={{ minHeight: preview ? 0 : 280, padding: preview ? 0 : 18 }}>
          <input type="file" accept="image/*,video/*" onChange={e => e.target.files?.[0] && void pick(e.target.files[0])} />
          {result?.overlay ? <img src={result.overlay} alt="Lines and players the system can see" style={{ width: '100%', display: 'block' }} />
            : preview ? <img src={preview} alt="Frame being checked" style={{ width: '100%', display: 'block', opacity: busy ? 0.5 : 1 }} />
            : <><Glyph g="camera" size={34} /><span><strong>Drop a photo or clip from the camera position</strong></span></>}
        </label>
        {result?.overlay && <div className="board-legend"><span><i className="dot" style={{ background: 'var(--ball)' }} />lines found</span>
          <span><i className="dot" style={{ background: 'var(--ok)' }} />players large enough</span><span><i className="dot" style={{ background: 'var(--bad)' }} />too small</span></div>}
      </section>
      <aside className="panel">
        {!result ? <><Placement /><ul className="check-list" style={{ marginTop: 10 }}>
          <li><Glyph g="eye" size={16} /><span>Whole pitch in frame</span></li>
          <li><Glyph g="camera" size={16} /><span>1080p, 25 to 60 fps, exposure locked</span></li>
          <li><Glyph g="sources" size={16} /><span>Tripod or clamp, no zoom during play</span></li>
        </ul></> : <ul className="check-list">{result.items.map(i => <li key={i.key}>
          <span className={i.grade === 'good' ? 'state-ok' : i.grade === 'check' ? 'state-warn' : 'state-bad'}><Glyph g={i.grade === 'good' ? 'check' : 'warn'} size={16} /></span>
          <span>{i.label}</span><span className="num muted">{String(i.value)}</span>
          {i.advice && <p>{i.advice}</p>}
        </li>)}</ul>}
        {result && <Link className="btn" style={{ marginTop: 14 }} to="/prepare/pitch">Next: pitch setup</Link>}
        {file && !result && busy && <p className="loading-line">Checking</p>}
      </aside>
    </div>
  </div>;
};
