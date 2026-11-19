import { useCallback, useEffect, useRef, useState } from 'react';
import { Glyph } from '../components/Glyph';
import { apiUrl } from '../lib/api';

type Cls = 'player' | 'goalkeeper' | 'referee' | 'ball';
interface Box { cls: Cls; box: [number, number, number, number]; source?: string; score?: number }
interface Frame { id: string; reviewed: boolean; boxes: number }
interface Status { state: string; error?: string; results?: Record<string, { map50_95: number; ball_recall: number | null }> & { recommend: string } }
const CLS: Cls[] = ['player', 'goalkeeper', 'referee', 'ball'];
const COLOUR: Record<Cls, string> = { player: '#2251a3', goalkeeper: '#e2c53a', referee: '#1b2320', ball: '#f2b705' };
const PROJECT = 'campus';

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(apiUrl(path), init); const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail ?? r.statusText); return j as T;
}

function Annotator({ frame, onSaved }: { frame: Frame; onSaved: () => void }) {
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [size, setSize] = useState<[number, number]>([1920, 1080]);
  const [sel, setSel] = useState<number | null>(null);
  const [cls, setCls] = useState<Cls>('player');
  const [draft, setDraft] = useState<[number, number, number, number] | null>(null);
  const svg = useRef<SVGSVGElement>(null);

  useEffect(() => { call<{ boxes: Box[]; width: number; height: number }>(`/api/v1/adapt/${PROJECT}/labels/${frame.id}`)
    .then(r => { setBoxes(r.boxes); setSize([r.width, r.height]); setSel(null); }).catch(() => setBoxes([])); }, [frame.id]);

  const pt = (e: React.PointerEvent) => { const r = svg.current!.getBoundingClientRect(); return [((e.clientX - r.left) / r.width) * size[0], ((e.clientY - r.top) / r.height) * size[1]] as [number, number]; };
  const save = useCallback(async () => { await call(`/api/v1/adapt/${PROJECT}/labels/${frame.id}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ boxes }) }); onSaved(); }, [boxes, frame.id, onSaved]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest('input')) return;
      const n = Number(e.key);
      if (n >= 1 && n <= 4) { const c = CLS[n - 1]; setCls(c); if (sel != null) setBoxes(b => b.map((x, i) => i === sel ? { ...x, cls: c } : x)); }
      if ((e.key === 'Delete' || e.key === 'Backspace') && sel != null) { setBoxes(b => b.filter((_, i) => i !== sel)); setSel(null); }
      if (e.key.toLowerCase() === 'b') setCls('ball');
      if (e.key === 'Enter') void save();
    };
    window.addEventListener('keydown', onKey); return () => window.removeEventListener('keydown', onKey);
  }, [sel, save]);

  return <div>
    <div style={{ position: 'relative' }}>
      <img src={apiUrl(`/api/v1/adapt/${PROJECT}/frames/${frame.id}.jpg`)} alt="Frame to label" style={{ width: '100%', display: 'block', borderRadius: 8 }} draggable={false} />
      <svg ref={svg} viewBox={`0 0 ${size[0]} ${size[1]}`} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', cursor: 'crosshair' }}
        onPointerDown={e => { const [x, y] = pt(e); if (cls === 'ball') { setBoxes(b => [...b, { cls: 'ball', box: [x - 7, y - 7, x + 7, y + 7], source: 'human' }]); return; } setDraft([x, y, x, y]); setSel(null); }}
        onPointerMove={e => { if (draft) { const [x, y] = pt(e); setDraft([draft[0], draft[1], x, y]); } }}
        onPointerUp={() => { if (draft) { const [a, b, c, d] = draft; if (Math.abs(c - a) > 6 && Math.abs(d - b) > 10) setBoxes(bs => [...bs, { cls, box: [Math.min(a, c), Math.min(b, d), Math.max(a, c), Math.max(b, d)], source: 'human' }]); setDraft(null); } }}>
        {boxes.map((b, i) => <rect key={i} x={b.box[0]} y={b.box[1]} width={b.box[2] - b.box[0]} height={b.box[3] - b.box[1]} fill={i === sel ? 'rgba(242,183,5,.18)' : 'transparent'}
          stroke={COLOUR[b.cls]} strokeWidth={i === sel ? 4 : 2.5} strokeDasharray={b.source === 'model' ? '8 5' : undefined}
          onPointerDown={e => { e.stopPropagation(); setSel(i); }} style={{ cursor: 'pointer' }} />)}
        {draft && <rect x={Math.min(draft[0], draft[2])} y={Math.min(draft[1], draft[3])} width={Math.abs(draft[2] - draft[0])} height={Math.abs(draft[3] - draft[1])} fill="none" stroke={COLOUR[cls]} strokeWidth={3} />}
      </svg>
    </div>
    <div className="panel-head" style={{ marginTop: 10 }}>
      <div className="seg" role="group" aria-label="Class">{CLS.map((c, i) => <button key={c} aria-pressed={cls === c} onClick={() => { setCls(c); if (sel != null) setBoxes(b => b.map((x, k) => k === sel ? { ...x, cls: c } : x)); }}>
        <i className="dot" style={{ background: COLOUR[c] }} />{i + 1} {c}</button>)}</div>
      <div style={{ display: 'flex', gap: 8 }}>
        <button className="btn ghost small" disabled={sel == null} onClick={() => { setBoxes(b => b.filter((_, i) => i !== sel)); setSel(null); }}><Glyph g="x" size={14} />Delete</button>
        <button className="btn small" onClick={() => void save()}><Glyph g="check" size={14} />Save, next</button>
      </div>
    </div>
    <p className="muted" style={{ fontSize: 12.5, margin: '6px 0 0' }}>Dashed boxes are model guesses. Drag to add, click to select, keys 1 to 4 set the class, B adds a ball with one click, Enter saves.</p>
  </div>;
}

export const AdaptPage = () => {
  const [frames, setFrames] = useState<Frame[]>([]);
  const [current, setCurrent] = useState(0);
  const [status, setStatus] = useState<Status>({ state: 'idle' });
  const [busy, setBusy] = useState<string>();
  const [error, setError] = useState<string>();

  const refresh = useCallback(() => {
    call<Frame[]>(`/api/v1/adapt/${PROJECT}/frames`).then(setFrames).catch(() => setFrames([]));
    call<Status>(`/api/v1/adapt/${PROJECT}/status`).then(setStatus).catch(() => undefined);
  }, []);
  useEffect(refresh, [refresh]);
  useEffect(() => { if (status.state !== 'training') return; const id = setInterval(refresh, 5000); return () => clearInterval(id); }, [status.state, refresh]);

  const run = async (label: string, fn: () => Promise<unknown>) => { setBusy(label); setError(undefined); try { await fn(); refresh(); } catch (e) { setError((e as Error).message); } finally { setBusy(undefined); } };
  const upload = (f: File) => run('Sampling frames', async () => { const form = new FormData(); form.append('file', f); form.append('frames', '60'); await call(`/api/v1/adapt/${PROJECT}/videos`, { method: 'POST', body: form }); });
  const reviewed = frames.filter(f => f.reviewed).length;
  const res = status.results;
  const trained = status.state === 'done';
  const stages = [
    { label: 'Clips', done: frames.length > 0, current: !frames.length },
    { label: frames.length ? `Review ${reviewed}/${frames.length}` : 'Review', done: reviewed >= 10, current: frames.length > 0 && reviewed < 10 },
    { label: 'Train', done: trained, current: reviewed >= 10 && !trained },
    { label: 'Compare', done: false, current: trained },
  ];

  return <div className="page">
    <div className="top"><h1>Adapt to your ground</h1>
      <ol className="steps" aria-label="Progress">
        {stages.map((st, i) => <li key={i}>{i > 0 && <span className="sep" aria-hidden="true" />}
          <span className={`step ${st.done ? 'done' : ''}`} aria-current={st.current ? 'step' : undefined}><b>{st.done ? <Glyph g="check" size={13} /> : i + 1}</b>{st.label}</span></li>)}
      </ol>
    </div>
    {error && <p className="notice bad">{error}</p>}
    <div className="setup">
      <section className="panel">
        {!frames.length ? <label className="drop"><input type="file" accept="video/*" onChange={e => e.target.files?.[0] && void upload(e.target.files[0])} />
          <Glyph g="upload" size={34} /><span><strong>{busy ?? 'Drop a clip from your ground'}</strong>About 60 varied frames are taken from it</span></label>
          : <Annotator frame={frames[current]} onSaved={() => { refresh(); setCurrent(c => Math.min(frames.length - 1, c + 1)); }} />}
      </section>
      <aside className="panel">
        {!frames.length && <ol className="adapt-how">
          <li><Glyph g="upload" size={18} /><span><strong>Add a clip</strong>a minute of play from your camera</span></li>
          <li><Glyph g="eye" size={18} /><span><strong>Check the boxes</strong>at least 10 frames</span></li>
          <li><Glyph g="model" size={18} /><span><strong>Train</strong>runs on this computer</span></li>
          <li><Glyph g="check" size={18} /><span><strong>Keep the better model</strong>scored on frames it never saw</span></li>
        </ol>}
        {frames.length > 0 && <>
          <div className="panel-head"><h2>Frames</h2><span className="num muted">{reviewed} reviewed</span></div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>{frames.map((f, i) => <button key={f.id} onClick={() => setCurrent(i)} title={f.id}
            style={{ width: 22, height: 22, borderRadius: 5, border: i === current ? '2px solid var(--ink)' : '1px solid var(--rule-strong)', background: f.reviewed ? 'var(--ok)' : f.boxes ? 'var(--paper-2)' : 'var(--sheet)', cursor: 'pointer' }} />)}</div>
          <div style={{ display: 'grid', gap: 8, marginTop: 14 }}>
            <label className="btn ghost small" style={{ justifyContent: 'center' }}><input type="file" accept="video/*" hidden onChange={e => e.target.files?.[0] && void upload(e.target.files[0])} /><Glyph g="upload" size={14} />Add another clip</label>
            <button className="btn ghost small" disabled={!!busy} onClick={() => run('Pre-labelling', () => call(`/api/v1/adapt/${PROJECT}/prelabel`, { method: 'POST' }))}><Glyph g="model" size={14} />{busy === 'Pre-labelling' ? 'Labelling' : 'Let the model guess'}</button>
            <button className="btn" disabled={reviewed < 10 || status.state === 'training' || !!busy} onClick={() => run('Training', () => call(`/api/v1/adapt/${PROJECT}/train`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ epochs: 20 }) }))}>
              <Glyph g="play" size={14} />{status.state === 'training' ? 'Training' : 'Train on these frames'}</button>
          </div></>}
        {status.state === 'failed' && <p className="notice bad" style={{ marginTop: 10 }}>{status.error}</p>}
        {res && <div style={{ marginTop: 16 }}>
          <div className="panel-head"><h2>On your held-out frames</h2></div>
          {(['base', 'adapted'] as const).map(k => <div key={k} style={{ marginBottom: 10 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}><span>{k === 'base' ? 'Current model' : 'Adapted'}</span><b className="num">{(res[k].map50_95 * 100).toFixed(1)}</b></div>
            <span className="bar" style={{ width: `${res[k].map50_95 * 100}%`, background: k === 'adapted' ? 'var(--ink)' : '#b9b3a5', height: 8 }} />
            {res[k].ball_recall != null && <small className="muted">ball found {(res[k].ball_recall! * 100).toFixed(0)}%</small>}
          </div>)}
          {res.recommend === 'adapted' ? <button className="btn" onClick={() => run('Promoting', () => call(`/api/v1/adapt/${PROJECT}/promote`, { method: 'POST' }))}><Glyph g="check" size={14} />Use the adapted model</button>
            : <p className="muted">The current model is still better here. Review more frames and train again.</p>}
        </div>}
      </aside>
    </div>
  </div>;
};
