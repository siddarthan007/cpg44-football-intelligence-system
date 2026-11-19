import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { BoardLegend, BoardPitch, type BoardMode } from '../components/BoardPitch';
import { Glyph, type GlyphName } from '../components/Glyph';
import { MatchStrip } from '../components/MatchStrip';
import { PlayerCard } from '../components/PlayerCard';
import { LoadView } from '../components/LoadView';
import { PlanView } from '../components/PlanView';
import { apiUrl } from '../lib/api';
import { applySquad, getSquad } from '../lib/squad';
import type { Analysis, AnalysisEvent, AnalysisSummary, Insight } from '../lib/analysis';
import { clock, decide, eventLabel, getAnalysis, listAnalyses, TEAM_HEX } from '../lib/analysis';

const Replay3D = lazy(() => import('../components/Replay3D').then(m => ({ default: m.Replay3D })));

type BoardTab = BoardMode | '3d';
const BOARD: { key: BoardTab; label: string; g: GlyphName }[] = [
  { key: 'moment', label: 'Moment', g: 'moment' }, { key: 'options', label: 'Options', g: 'tactics' }, { key: 'shape', label: 'Shape', g: 'shape' },
  { key: 'heat', label: 'Heat', g: 'heat' }, { key: 'links', label: 'Passes', g: 'links' }, { key: '3d', label: '3D', g: 'cube' },
];
type LowerTab = 'findings' | 'plan' | 'load' | 'check' | 'players' | 'teams' | 'shots';
const KIND_GLYPH: Record<Insight['kind'], GlyphName> = { tactical: 'tactics', physical: 'run', fused: 'pulse', data: 'eye' };

const Conf = ({ c }: { c: Insight['confidence'] }) => <span className={`conf ${c}`} title={`${c} confidence`}><i /><i /><i /></span>;

export const ReviewPage = () => {
  const { id } = useParams();
  const nav = useNavigate();
  const [params, setParams] = useSearchParams();
  const [list, setList] = useState<AnalysisSummary[]>();
  const [a, setA] = useState<Analysis>();
  const [error, setError] = useState<string>();
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const [team, setTeam] = useState<0 | 1 | 2>(0);
  const [selected, setSelected] = useState<number>();
  const [focus, setFocus] = useState<string>();
  const video = useRef<HTMLVideoElement>(null);
  const board = (params.get('board') as BoardTab) || 'moment';
  const lower = (params.get('tab') as LowerTab) || 'findings';
  const wide = params.get('view') === 'wide';
  const boardPanel = useRef<HTMLElement>(null);
  const [full, setFull] = useState(false);
  useEffect(() => { const f = () => setFull(document.fullscreenElement === boardPanel.current); document.addEventListener('fullscreenchange', f); return () => document.removeEventListener('fullscreenchange', f); }, []);
  const toggleFull = () => { if (document.fullscreenElement) void document.exitFullscreen(); else void boardPanel.current?.requestFullscreen?.(); };
  const setParam = (k: string, v: string) => { const n = new URLSearchParams(params); n.set(k, v); setParams(n, { replace: true }); };

  useEffect(() => { listAnalyses().then(setList).catch(e => setError(String(e.message ?? e))); }, []);
  useEffect(() => { if (!id && list?.length) nav(`/review/${list[0].id}`, { replace: true }); }, [id, list, nav]);
  const load = useCallback(() => {
    if (!id) return;
    Promise.all([getAnalysis(id), getSquad()]).then(([r, sq]) => { setA(applySquad(r, sq)); setError(undefined); }).catch(e => setError(String(e.message ?? e)));
  }, [id]);
  useEffect(load, [load]);

  useEffect(() => {          // local clock when there is no video
    if (!a || a.video_url || !playing) return;
    let raf = 0, last = performance.now();
    const step = (now: number) => {
      setT(v => { const n = v + ((now - last) / 1000) * rate; if (n >= a.match.duration_s) { setPlaying(false); return a.match.duration_s; } return n; });
      last = now; raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [a, playing, rate]);
  useEffect(() => { if (video.current) video.current.playbackRate = rate; }, [rate]);

  const seek = (s: number) => { setT(s); if (video.current) video.current.currentTime = s; };
  const toggle = () => { const v = video.current; if (v) { if (v.paused) void v.play(); else v.pause(); } else setPlaying(p => !p); };
  const replay = (e: AnalysisEvent) => {          // half-speed 3D replay from just before the strike
    setParam('board', '3d'); setRate(0.5); seek(Math.max(0, (e.t ?? 0) - 1.5));
    const v = video.current; if (v) void v.play(); else setPlaying(true);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };
  const decideOn = async (e: AnalysisEvent, d: 'confirm' | 'dismiss' | 'reset') => { if (a) { await decide(a.id, e.id, d); load(); } };

  useEffect(() => {          // keyboard: space play, arrows seek
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest('input, select, textarea, [role=slider]')) return;
      if (e.code === 'Space') { e.preventDefault(); toggle(); }
      if (e.key === 'ArrowRight') seek(Math.min((a?.match.duration_s ?? 0), t + 2));
      if (e.key === 'ArrowLeft') seek(Math.max(0, t - 2));
      if (e.key === 'f' && !e.metaKey && !e.ctrlKey) toggleFull();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  const counts = useMemo(() => {
    if (!a) return null;
    const live = a.events.filter(e => e.status !== 'dismissed');
    const n = (type: string, tm: number) => live.filter(e => e.type === type && e.team === tm).length;
    return { passes: [n('pass', 1), n('pass', 2)], shots: [n('shot', 1), n('shot', 2)], won: [n('turnover', 1), n('turnover', 2)], sprints: [n('sprint', 1), n('sprint', 2)] };
  }, [a]);

  if (error && !a) return <div className="page"><p className="notice bad">{error}</p></div>;
  if (list && !list.length) return <div className="page"><div className="empty"><Glyph g="review" size={40} /><h1>No analysed match yet</h1>
    <p>Calibrate the pitch, then add footage.</p><div style={{ display: 'flex', gap: 8 }}><Link className="btn ghost" to="/prepare/pitch">Pitch setup</Link><Link className="btn" to="/ingest">Add footage</Link></div></div></div>;
  if (!a) return <div className="page"><p className="loading-line">Opening the match</p></div>;

  const q = a.quality;
  const queue = a.events.filter(e => ['check', 'confirmed', 'dismissed'].includes(e.status) && e.type !== 'ball_unseen');
  const open = queue.filter(e => e.status === 'check').length;
  const shots = a.events.filter(e => e.type === 'shot' && e.status !== 'dismissed');
  const player = a.players.find(p => p.id === selected);
  const name = (k: number) => a.teams[String(k) as '1' | '2']?.name ?? 'Unassigned';
  const p1 = a.possession.team1_pct, p2 = a.possession.team2_pct;
  const maxDist = Math.max(1, ...a.players.map(p => p.distance_m));
  const clip = (s: number | null) => s == null || !a.video_url ? null : apiUrl(`/api/v1/analyses/${a.id}/clip?t=${s.toFixed(1)}`);

  return <div className="page review">
    <div className="top">
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
        <h1>{a.match.name}</h1>
        {list && list.length > 1 && <select aria-label="Choose match" value={a.id} onChange={e => nav(`/review/${e.target.value}`)} className="btn ghost small">
          {list.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}</select>}
      </div>
      <div className="health" aria-label="Analysis quality">
        <span title={q.metric ? `Pitch calibrated in ${Math.round(q.calibration_coverage * 100)}% of frames` : 'Pitch not calibrated: no metres'}
          className={q.metric ? 'state-ok' : 'state-warn'}><Glyph g="pitch" size={16} />{q.metric ? `${Math.round(q.calibration_coverage * 100)}%` : 'off'}</span>
        <span title={`Ball located in ${Math.round(q.ball_seen * 100)}% of frames`} className={q.ball_seen >= 0.7 ? 'state-ok' : 'state-warn'}><Glyph g="ball" size={16} />{Math.round(q.ball_seen * 100)}%</span>
        <span title={`${q.identities} players followed`}><Glyph g="squad" size={16} />{q.identities}</span>
        <details className="info-pop"><summary className="icon-btn" aria-label="How this was measured"><Glyph g="info" size={16} /></summary><div>
          <ul>
            <li>Players: {a.provenance.tracker}, {q.raw_tracks} pieces joined into {q.identities}.</li>
            <li>Sides: {q.team_separation.method}.</li>
            <li>Ball: {a.provenance.ball}.</li>
            <li>Speeds smoothed over 0.6 s; top speed held for 1 s.</li>
            {q.position_funnel && <li>Placed {q.position_funnel.placed} of {q.position_funnel.boxes} player boxes; {q.position_funnel.off_pitch} off the pitch, {q.position_funnel.outlier} outliers removed.</li>}
            <li>Findings are coaching cues, not medical advice.</li>
          </ul></div></details>
      </div>
    </div>

    <div className="scoreline" aria-label="Ball control">
      <span className="side"><i className="dot a" />{name(1)}</span>
      {p1 != null ? <><span className="ctrl-bar" aria-hidden="true"><i style={{ width: `${p1}%` }} /><i style={{ width: `${p2}%` }} /></span>
        <span className="ctrl-val">{Math.round(p1)} : {Math.round(p2 ?? 0)}</span></> : <span className="muted">control not measured</span>}
      <span className="side">{name(2)}<i className="dot b" /></span>
    </div>

    <div className={`stage ${wide ? 'wide' : ''}`}>
      <section className="panel" aria-label="Footage" style={{ padding: 10 }}>
        <div className="film">
          {a.video_url ? <video ref={video} src={apiUrl(a.video_url)} playsInline muted preload="auto" onClick={toggle}
            onTimeUpdate={e => setT(e.currentTarget.currentTime)} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} />
            : <div className="empty" style={{ color: '#b8bdb4' }}><Glyph g="camera" size={32} /></div>}
        </div>
        <div className="transport">
          <button className="icon-btn" onClick={() => seek(Math.max(0, t - 5))} aria-label="Back 5 seconds"><Glyph g="back" size={16} /></button>
          <button className="btn small" onClick={toggle} aria-label={playing ? 'Pause' : 'Play'}><Glyph g={playing ? 'pause' : 'play'} size={15} /></button>
          <button className="icon-btn" onClick={() => seek(Math.min(a.match.duration_s, t + 5))} aria-label="Forward 5 seconds"><Glyph g="fwd" size={16} /></button>
          <div className="seg" role="group" aria-label="Speed">{[0.25, 0.5, 1, 2].map(r => <button key={r} aria-pressed={rate === r} onClick={() => setRate(r)}>{r}x</button>)}</div>
          <time>{clock(t)} / {clock(a.match.duration_s)}</time>
        </div>
      </section>

      <section className="panel board-panel" aria-label="Board" style={{ padding: 10 }} ref={boardPanel}>
        <div className="panel-head" style={{ marginBottom: 8 }}>
          <div className="seg board-tabs" role="tablist" aria-label="Board view">
            {BOARD.map(b => <button key={b.key} role="tab" aria-selected={board === b.key} disabled={!q.metric} aria-label={b.label} title={b.label} onClick={() => setParam('board', b.key)}><Glyph g={b.g} size={15} /><span className="lbl">{b.label}</span></button>)}
          </div>
          <div className="seg" aria-label="Show side">
            {([0, 1, 2] as const).map(k => <button key={k} aria-pressed={team === k} onClick={() => setTeam(k)} title={k ? name(k) : 'Both sides'}>
              {k ? <i className={`dot ${k === 1 ? 'a' : 'b'}`} /> : 'Both'}</button>)}
          </div>
          <div className="board-size">
            <button className="icon-btn wide-toggle" onClick={() => setParam('view', wide ? 'split' : 'wide')} aria-pressed={wide} title={wide ? 'Equal split' : 'Larger board'}><Glyph g={wide ? 'split' : 'widen'} size={16} /></button>
            <button className="icon-btn" onClick={toggleFull} title={full ? 'Exit full screen (F)' : 'Full screen (F)'}><Glyph g={full ? 'shrink' : 'expand'} size={16} /></button>
          </div>
        </div>
        {!q.metric ? <div className="empty"><Glyph g="pitch" size={36} /><span>Calibrate the pitch to place players</span><Link className="btn ghost small" to="/prepare/pitch">Pitch setup</Link></div>
          : board === '3d' ? <Suspense fallback={<div className="empty">Loading 3D</div>}><Replay3D analysis={a} t={t} selected={selected} /></Suspense>
          : <><div className="board-wrap"><BoardPitch analysis={a} t={t} mode={board} team={team} selected={selected} onSelect={id => { setSelected(id); setParam('tab', 'players'); }} /></div><BoardLegend mode={board} analysis={a} t={t} /></>}
      </section>
    </div>

    <section className="panel" aria-label="Match strip">
      <div className="panel-head" style={{ marginBottom: 4 }}>
        <p className="legend"><span><i className="lg-pass" />pass</span><span><i className="lg-won" />ball won</span><span><i className="lg-shot" />shot</span><span><i className="lg-sprint" />sprint</span><span><i className="lg-check" />to check</span></p>
      </div>
      <MatchStrip analysis={a} t={t} onSeek={seek} focus={focus} onEvent={e => setFocus(e.id)} />
    </section>

    <section className="panel">
      <div className="tabs" role="tablist" aria-label="Details">
        {([['findings', 'Findings', a.insights.length], ['plan', 'Plan', 0], ['load', 'Load', a.load?.subs?.length ?? 0], ['check', 'To check', open], ['players', 'Players', 0], ['teams', 'Teams', 0], ['shots', 'Shots', shots.length]] as [LowerTab, string, number][])
          .map(([k, label, n]) => <button key={k} role="tab" aria-selected={lower === k} onClick={() => setParam('tab', k)}>{label}{n ? <span className="count">{n}</span> : null}</button>)}
      </div>

      {lower === 'findings' && (!a.insights.length ? <div className="empty"><Glyph g="tactics" size={32} />Nothing crossed the evidence bar yet</div>
        : <div>{a.insights.map(c => <div key={c.id} className={`finding ${c.kind}`}>
          <span className="kind" title={c.kind}><Glyph g={KIND_GLYPH[c.kind]} size={16} /></span>
          <div>
            <h3>{c.title}{c.team ? <i className="dot" style={{ background: TEAM_HEX[c.team], marginLeft: 8 }} /> : null}</h3>
            <p>{c.finding}</p>
            <p className="act"><Glyph g="fwd" size={14} />{c.action}</p>
            {c.evidence.length > 0 && <div className="nums">{c.evidence.slice(0, 3).map(ev => <div key={ev.label}><b>{ev.value}</b><small>{ev.label}</small></div>)}</div>}
          </div>
          <div className="jumps"><Conf c={c.confidence} />{c.moments.slice(0, 3).map(m => <button key={m} className="btn ghost small" onClick={() => seek(m)}>{clock(m)}</button>)}</div>
        </div>)}</div>)}

      {lower === 'plan' && <PlanView a={a} />}

      {lower === 'load' && <LoadView a={a} />}

      {lower === 'check' && (a.squad_issues ?? []).map(t => <div key={t} className="notice" style={{ marginBottom: 8 }}>{t}</div>)}
      {lower === 'check' && (!queue.length ? <div className="empty"><Glyph g="check" size={32} />Nothing to check</div>
        : <div>{queue.map(e => <div key={e.id} className={`queue-item ${e.status} ${focus === e.id ? 'focus' : ''}`}>
          <button className="queue-time" onClick={() => { setFocus(e.id); if (e.t != null) seek(e.t); }}>{clock(e.t)}</button>
          <div><strong>{eventLabel[e.type] ?? e.type}{e.players.length ? ` ${e.players.join(' and ')}` : ''}</strong><span>{e.detail ?? e.reason ?? ''}</span></div>
          {e.status === 'check' ? <div className="queue-actions">
            <button className="btn small" onClick={() => decideOn(e, 'confirm')}><Glyph g="check" size={14} />Yes</button>
            <button className="btn ghost small" onClick={() => decideOn(e, 'dismiss')}><Glyph g="x" size={14} />No</button>
          </div> : <div className="queue-actions"><span className={e.status === 'confirmed' ? 'state-ok' : 'muted'}><Glyph g={e.status === 'confirmed' ? 'check' : 'x'} size={15} /></span>
            <button className="icon-btn" aria-label="Undo" onClick={() => decideOn(e, 'reset')}><Glyph g="undo" size={14} /></button></div>}
        </div>)}</div>)}

      {lower === 'players' && <div className="lower">
        <div style={{ overflowX: 'auto' }}><table className="players">
          <thead><tr><th>Player</th><th className="num">Seen</th><th>Distance</th><th className="num">Top km/h</th><th className="num">High speed</th></tr></thead>
          <tbody>{a.players.map(p => <tr key={p.id} aria-selected={selected === p.id} onClick={() => setSelected(p.id)}>
            <td><i className="dot" style={{ background: TEAM_HEX[p.team], marginRight: 8 }} />{p.label}{p.role === 'goalkeeper' ? ' GK' : ''}</td>
            <td className="num">{Math.round(p.observed_s)} s</td>
            <td><span className="num">{p.distance_m.toFixed(0)} m</span><span className="bar" style={{ width: `${(p.distance_m / maxDist) * 100}%`, background: TEAM_HEX[p.team] }} /></td>
            <td className="num">{p.top_speed_kmh?.toFixed(1) ?? 'n/a'}</td><td className="num">{p.hsr_m.toFixed(0)} m</td></tr>)}</tbody>
        </table></div>
        {player ? <PlayerCard key={player.id} analysis={a} player={player} onChange={load} /> : <div className="empty"><Glyph g="run" size={30} />Pick a player</div>}
      </div>}

      {lower === 'teams' && counts && <div className="versus" style={{ maxWidth: 620 }}>
        {([['Passes', counts.passes], ['Shots', counts.shots], ['Balls won', counts.won], ['Sprints', counts.sprints],
          ['Distance (m)', [a.team_stats['1'].distance_m, a.team_stats['2'].distance_m]],
          ['Team length (m)', [a.team_stats['1'].length_m ?? 0, a.team_stats['2'].length_m ?? 0]],
          ['Team width (m)', [a.team_stats['1'].width_m ?? 0, a.team_stats['2'].width_m ?? 0]]] as [string, number[]][]).map(([k, [x, y]]) => {
          const tot = Math.max(x + y, 1e-9);
          return <div key={k} className="versus-row"><span className="k">{k}</span><span className="v a">{Math.round(x)}</span>
            <div className="versus-bar"><i className="a" style={{ width: `${(x / tot) * 100}%` }} /><i className="b" style={{ width: `${(y / tot) * 100}%` }} /></div>
            <span className="v b">{Math.round(y)}</span></div>;
        })}
      </div>}

      {lower === 'shots' && (!shots.length ? <div className="empty"><Glyph g="target" size={32} />No shots in this footage</div>
        : <div className="lower">{shots.map(s => s.breakdown && <div key={s.id} className="panel shot-card">
          <div className="panel-head"><h2><i className="dot" style={{ background: TEAM_HEX[s.team], marginRight: 8 }} />{clock(s.t)} {name(s.team)}</h2>
            <div style={{ display: 'flex', gap: 6 }}>
              <button className="btn ghost small" onClick={() => replay(s)}><Glyph g="cube" size={14} />Replay</button>
              {clip(s.t) && <a className="btn ghost small" href={clip(s.t)!} download><Glyph g="save" size={14} />Clip</a>}
            </div></div>
          <dl><div><dt>Speed</dt><dd>{s.breakdown.shot_speed_kmh ?? 'n/a'}<small> km/h</small></dd></div><div><dt>To goal</dt><dd>{s.breakdown.distance_to_goal_m}<small> m</small></dd></div>
            <div><dt>Angle</dt><dd>{s.breakdown.goal_angle_deg}°</dd></div><div><dt>In the way</dt><dd>{s.breakdown.players_in_the_way ?? 'n/a'}</dd></div><div><dt>xG</dt><dd>{s.xg?.toFixed(2)}</dd></div></dl>
        </div>)}</div>)}
    </section>
  </div>;
};
