import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { Glyph } from '../components/Glyph';
import { apiUrl } from '../lib/api';

type Pt = [number, number];
interface Session { id: string; width: number; height: number; duration_s: number | null; at_s?: number; version?: number }
interface Solve {
  ok: boolean; reason?: string; homography?: number[][]; k1?: number; k1_auto?: boolean; lens?: Record<string, unknown>;
  errors_px?: Record<string, number>; rms_px?: number; check_px?: number | null; grade?: 'good' | 'fair' | 'poor'; overlay?: (Pt | null)[][];
  snap?: { accepted: boolean; median_px_before?: number; median_px_after?: number };
}

type Kind = 'football' | 'futsal';
const PRESETS: { label: string; l: number; w: number; kind: Kind }[] = [
  { label: 'Full size', l: 105, w: 68, kind: 'football' }, { label: 'Campus', l: 100, w: 64, kind: 'football' },
  { label: '9 a side', l: 75, w: 50, kind: 'football' }, { label: '7 a side', l: 60, w: 40, kind: 'football' },
  { label: 'Futsal', l: 40, w: 20, kind: 'futsal' },
];
// Points that are unbiased under perspective (circle extremes are not).
const LANDMARKS = ['tl_corner', 'tr_corner', 'bl_corner', 'br_corner', 'half_top', 'half_bottom', 'centre',
  'l_pen_top', 'l_pen_bottom', 'l_pen_top_line', 'l_pen_bottom_line', 'l_goal_top', 'l_goal_bottom', 'l_pen_spot',
  'r_pen_top', 'r_pen_bottom', 'r_pen_top_line', 'r_pen_bottom_line', 'r_goal_top', 'r_goal_bottom', 'r_pen_spot',
  'l_arc_top_line', 'l_arc_bottom_line', 'r_arc_top_line', 'r_arc_bottom_line'];
const STEPS = ['Frame', 'Pitch', 'Landmarks', 'Save'] as const;

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(apiUrl(path), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail ?? r.statusText);
  return j as T;
}

export const PitchSetupPage = () => {
  const [step, setStep] = useState(0);
  const [session, setSession] = useState<Session>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const [size, setSize] = useState<{ l: number; w: number; kind: Kind }>({ l: 105, w: 68, kind: 'football' });
  const [camera, setCamera] = useState<'fixed' | 'panning'>('fixed');
  const [tpl, setTpl] = useState<{ points: Record<string, Pt>; lines: Pt[][] }>();
  const [points, setPoints] = useState<Record<string, Pt>>({});
  const [active, setActive] = useState<string>('tl_corner');
  const [solve, setSolve] = useState<Solve>();
  const [lensManual, setLensManual] = useState<number | null>(null);
  const [name, setName] = useState('Campus main pitch');
  const [saved, setSaved] = useState<string>();
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    fetch(apiUrl(`/api/v1/calibration/landmarks?length=${size.l}&width=${size.w}&kind=${size.kind}`)).then(r => r.json()).then(setTpl).catch(() => setTpl(undefined));
  }, [size.l, size.w, size.kind]);

  const runSolve = useCallback(async (pts: Record<string, Pt>, k1: number | null) => {
    if (!session || Object.keys(pts).length < 4) { setSolve(undefined); return; }
    try {
      setSolve(await post<Solve>(`/api/v1/calibration/sessions/${session.id}/solve`, { points: pts, length: size.l, width: size.w, kind: size.kind, k1 }));
      setError(undefined);
    } catch (e) { setError((e as Error).message); }
  }, [session, size.l, size.w, size.kind]);
  useEffect(() => { void runSolve(points, lensManual); }, [points, lensManual, runSolve]);

  const upload = async (file: File) => {
    setBusy(true); setError(undefined);
    try {
      const form = new FormData(); form.append('file', file); form.append('at_s', '2');
      const r = await fetch(apiUrl('/api/v1/calibration/sessions'), { method: 'POST', body: form });
      const j = await r.json(); if (!r.ok) throw new Error(j.detail ?? r.statusText);
      setSession(j); setPoints({}); setSolve(undefined); setStep(1);
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };
  const regrab = async (at: number) => {
    if (!session) return;
    try { setSession({ ...session, ...(await post<Session>(`/api/v1/calibration/sessions/${session.id}/frame`, { at_s: at })) }); }
    catch (e) { setError((e as Error).message); }
  };

  const nextUnset = (pts: Record<string, Pt>) => LANDMARKS.find(k => !pts[k] && tpl?.points[k]) ?? active;
  const clickFrame = (e: React.MouseEvent) => {
    if (!session || step !== 2) return;
    const r = imgRef.current!.getBoundingClientRect();
    const p: Pt = [((e.clientX - r.left) / r.width) * session.width, ((e.clientY - r.top) / r.height) * session.height];
    const next = { ...points, [active]: p };
    setPoints(next); setActive(nextUnset(next));
  };
  const snap = async () => {
    if (!session || !solve?.homography) return;
    setBusy(true);
    try { setSolve(await post<Solve>(`/api/v1/calibration/sessions/${session.id}/snap`, { points, length: size.l, width: size.w, kind: size.kind, k1: solve.k1 ?? 0, homography: solve.homography })); }
    catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };
  const save = async () => {
    if (!session || !solve?.homography) return;
    setBusy(true);
    try {
      const r = await post<{ id: string }>(`/api/v1/calibration/sessions/${session.id}/save`,
        { name, points, length: size.l, width: size.w, kind: size.kind, k1: solve.k1 ?? 0, homography: solve.homography, camera });
      setSaved(r.id);
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };

  const count = Object.keys(points).length;
  const done = [Boolean(session), Boolean(session) && step > 1, Boolean(solve?.ok), Boolean(saved)];
  const frameUrl = session ? apiUrl(`/api/v1/calibration/sessions/${session.id}/frame?v=${session.version ?? 0}`) : '';
  const L = size.l, W = size.w;
  const overlayPaths = useMemo(() => (solve?.overlay ?? []).map(line => {
    let d = '', pen = false;
    for (const p of line) { if (!p) { pen = false; continue; } d += `${pen ? 'L' : 'M'}${p[0]},${p[1]} `; pen = true; }
    return d;
  }), [solve]);

  return <div className="page setup-page">
    <div className="top">
      <h1>Pitch setup</h1>
      <div className="steps" role="list" aria-label="Setup steps">
        {STEPS.map((s, i) => <span key={s} style={{ display: 'contents' }}>
          {i > 0 && <span className="sep" aria-hidden="true" />}
          <button role="listitem" className={done[i] && step !== i ? 'done' : ''} aria-current={step === i ? 'step' : undefined}
            disabled={i > 0 && !session} onClick={() => setStep(i)}><b>{done[i] && step !== i ? <Glyph g="check" size={13} /> : i + 1}</b>{s}</button>
        </span>)}
      </div>
    </div>
    {error && <p className="notice bad">{error}</p>}

    <div className="setup">
      <section className="panel" aria-label="Camera frame">
        {!session ? <label className="drop">
          <input type="file" accept="video/*,image/*" onChange={e => e.target.files?.[0] && void upload(e.target.files[0])} />
          <Glyph g="camera" size={34} />
          <span><strong>{busy ? 'Reading the clip' : 'Drop a clip or photo from the match camera'}</strong>The same position and zoom you will record with</span>
        </label> : <>
          <div className={`frame-box ${step === 2 ? 'picking' : ''}`} onClick={clickFrame}>
            <img ref={imgRef} src={frameUrl} alt="Frame used for calibration" draggable={false} />
            <svg viewBox={`0 0 ${session.width} ${session.height}`} preserveAspectRatio="none">
              {step >= 2 && overlayPaths.map((d, i) => <path key={i} d={d} className="ov" />)}
              {Object.entries(points).map(([k, p], i) => {
                const err = solve?.errors_px?.[k];
                const col = err == null ? '#f2b705' : err <= 4 ? '#2e7048' : err <= 10 ? '#a8650a' : '#b23a2e';
                return <g key={k} className="pin" transform={`translate(${p[0]},${p[1]})`}>
                  <circle r={9} fill={col} /><text x={14} y={6}>{i + 1}</text>
                </g>;
              })}
            </svg>
          </div>
          {session.duration_s ? <div className="slider" style={{ marginTop: 10 }}>
            <input type="range" min={0} max={session.duration_s} step={0.5} defaultValue={session.at_s ?? 2}
              onMouseUp={e => void regrab(Number((e.target as HTMLInputElement).value))}
              onKeyUp={e => void regrab(Number((e.target as HTMLInputElement).value))} aria-label="Moment in the clip" />
            <span className="muted">Pick a moment with the lines clearly visible</span>
          </div> : null}
        </>}
      </section>

      <aside className="panel" aria-label="Setup controls">
        {step <= 1 && <>
          <div className="panel-head"><h2>Pitch size</h2></div>
          <div className="seg seg-wrap" role="group" aria-label="Pitch size preset" style={{ marginBottom: 10 }}>
            {PRESETS.map(p => <button key={p.label} aria-pressed={size.l === p.l && size.w === p.w && size.kind === p.kind} onClick={() => setSize({ l: p.l, w: p.w, kind: p.kind })}>{p.label}</button>)}
          </div>
          <div className="kv">
            <label htmlFor="pl">Length (m)</label><input id="pl" type="number" min={20} max={130} value={size.l} onChange={e => setSize({ ...size, l: Number(e.target.value) })} />
            <label htmlFor="pw">Width (m)</label><input id="pw" type="number" min={15} max={100} value={size.w} onChange={e => setSize({ ...size, w: Number(e.target.value) })} />
          </div>
          <div className="panel-head" style={{ marginTop: 16 }}><h2>Camera</h2></div>
          <div className="seg" role="group" aria-label="Camera mounting">
            <button aria-pressed={camera === 'fixed'} onClick={() => setCamera('fixed')}><Glyph g="camera" size={15} />Fixed on a tripod</button>
            <button aria-pressed={camera === 'panning'} onClick={() => setCamera('panning')}><Glyph g="eye" size={15} />Follows play</button>
          </div>
          <button className="btn" style={{ marginTop: 16 }} disabled={!session} onClick={() => setStep(2)}>Mark landmarks</button>
        </>}

        {step === 2 && tpl && <>
          <div className="panel-head"><h2>Tap a point, then find it on the frame</h2><span className="muted num">{count} of 4 to 8</span></div>
          <svg className="mini-pitch" viewBox={`-3 -3 ${L + 6} ${W + 6}`} role="group" aria-label="Pitch landmarks">
            <rect x={-3} y={-3} width={L + 6} height={W + 6} fill="var(--board)" rx={2} />
            {tpl.lines.map((ln, i) => <polyline key={i} points={ln.map(p => p.join(',')).join(' ')} fill="none" stroke="var(--board-line)" strokeWidth={0.35} />)}
            {LANDMARKS.filter(k => tpl.points[k]).map(k => {
              const [x, y] = tpl.points[k];
              return <g key={k} className={`lm ${points[k] ? 'set' : ''} ${active === k ? 'active' : ''}`} onClick={() => setActive(k)}
                role="button" aria-label={k.replace(/_/g, ' ')} aria-pressed={active === k}>
                <circle cx={x} cy={y} r={2.1} /><circle cx={x} cy={y} r={4} fill="transparent" stroke="none" />
              </g>;
            })}
          </svg>
          <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap' }}>
            <button className="btn ghost small" disabled={!points[active]} onClick={() => { const n = { ...points }; delete n[active]; setPoints(n); }}><Glyph g="x" size={14} />Remove point</button>
            <button className="btn ghost small" disabled={!count} onClick={() => { setPoints({}); setSolve(undefined); }}><Glyph g="undo" size={14} />Start over</button>
          </div>
          {solve?.ok && <>
            <div className="grade" style={{ marginTop: 16 }}>
              <span className={solve.grade === 'good' ? 'state-ok' : solve.grade === 'fair' ? 'state-warn' : 'state-bad'}><Glyph g={solve.grade === 'poor' ? 'warn' : 'check'} size={22} /></span>
              <div><b>{(solve.check_px ?? solve.rms_px)?.toFixed(1)} px</b><div className="muted">{solve.check_px != null ? 'error predicting each point from the others' : 'average click error, add a 5th point to check'}</div></div>
            </div>
            {solve.grade === 'poor' && <p className="notice" style={{ marginTop: 10 }}>Add points far apart, on both halves and both touchlines if visible</p>}
            <div className="slider" style={{ marginTop: 14 }}>
              <label htmlFor="k1" style={{ display: 'flex', gap: 8, alignItems: 'center' }}><Glyph g="lens" size={16} />Lens curve <span className="num muted">{(solve.k1 ?? 0).toFixed(2)}{solve.k1_auto ? ' auto' : ''}</span></label>
              <input id="k1" type="range" min={-0.5} max={0.2} step={0.01} value={lensManual ?? solve.k1 ?? 0} onChange={e => setLensManual(Number(e.target.value))} />
              {lensManual !== null && <button className="btn ghost small" onClick={() => setLensManual(null)}>Back to auto</button>}
            </div>
            <button className="btn" style={{ marginTop: 14 }} onClick={() => setStep(3)}>Looks right</button>
          </>}
          {!solve?.ok && count >= 4 && solve?.reason && <p className="notice">{solve.reason}</p>}
        </>}

        {step === 3 && <>
          <div className="panel-head"><h2>Fine tune and save</h2></div>
          <button className="btn ghost" disabled={busy || !solve?.homography} onClick={snap}><Glyph g="snap" size={16} />Snap to painted lines</button>
          {solve?.snap && <p className="muted" style={{ marginTop: 6 }}>{solve.snap.accepted ? `Moved onto the lines (${solve.snap.median_px_before} to ${solve.snap.median_px_after} px)` : 'Already on the lines, kept as is'}</p>}
          <div className="kv" style={{ marginTop: 16 }}>
            <label htmlFor="cn">Name</label><input id="cn" value={name} onChange={e => setName(e.target.value)} style={{ width: 170, padding: '6px 8px', border: '1px solid var(--rule-strong)', borderRadius: 7 }} />
          </div>
          <button className="btn" style={{ marginTop: 14 }} disabled={busy || !solve?.homography} onClick={save}><Glyph g="save" size={16} />Save calibration</button>
          {saved && <p style={{ marginTop: 12 }}><Glyph g="check" size={15} /> Saved. <Link to="/ingest">Analyse footage with it</Link></p>}
        </>}
      </aside>
    </div>
  </div>;
};
