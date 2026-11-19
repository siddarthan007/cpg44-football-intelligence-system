import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { Glyph, type GlyphName } from '../components/Glyph';
import { api, apiUrl } from '../lib/api';
import type { Analysis, AnalysisSummary } from '../lib/analysis';
import { clock, getAnalysis, listAnalyses, TEAM_HEX } from '../lib/analysis';
import { applySquad, getSquad, type Squad } from '../lib/squad';

type State = 'done' | 'warn' | 'todo';
interface Step { to: string; g: GlyphName; title: string; status: string; state: State }
type Vest = { connected?: boolean; clock_synced?: boolean; battery?: { pct?: number } | null };
type Board = 'speed' | 'rate' | 'hsr';

const BOARD: Record<Board, { label: string; unit: string }> = {
  speed: { label: 'Top speed', unit: 'km/h' }, rate: { label: 'Work rate', unit: 'm/min' }, hsr: { label: 'High speed', unit: 'm' },
};
const when = (s?: number) => (s ? new Date(s * 1000).toLocaleDateString(undefined, { day: 'numeric', month: 'short' }) : '');

/** Compact two-sided figure: home value, a split bar, away value. */
function Versus({ label, a, b, fmt = (v: number) => `${Math.round(v)}` }: { label: string; a: number; b: number; fmt?: (v: number) => string }) {
  const total = a + b;
  return <div className="vs-row">
    <span className="vs-a num">{fmt(a)}</span>
    <span className="vs-mid"><small>{label}</small><span className="vs-bar" aria-hidden="true">
      <i style={{ width: total ? `${(a / total) * 100}%` : '50%', background: TEAM_HEX[1] }} /><i style={{ width: total ? `${(b / total) * 100}%` : '50%', background: TEAM_HEX[2] }} /></span></span>
    <span className="vs-b num">{fmt(b)}</span>
  </div>;
}

function Leaders({ a }: { a: Analysis }) {
  const [board, setBoard] = useState<Board>('speed');
  const rows = useMemo(() => {
    // only players followed for a fair share of the clip, so a 3 s cameo never tops the table
    const minSeen = Math.min(60, a.match.duration_s * 0.3);
    const pool = a.players.filter(p => (p.team === 1 || p.team === 2) && p.observed_s >= minSeen);
    const value = (p: (typeof pool)[number]) => board === 'speed' ? p.top_speed_kmh ?? 0 : board === 'rate' ? (p.distance_m / p.observed_s) * 60 : p.hsr_m;
    return pool.map(p => ({ p, v: value(p) })).filter(r => r.v > 0).sort((x, y) => y.v - x.v).slice(0, 6);
  }, [a, board]);
  const max = rows[0]?.v || 1;
  return <section className="panel leaders" aria-label="Leaders">
    <div className="panel-head"><h2>Leaders</h2>
      <div className="seg" role="group" aria-label="Measure">{(Object.keys(BOARD) as Board[]).map(k => <button key={k} aria-pressed={board === k} onClick={() => setBoard(k)}>{BOARD[k].label}</button>)}</div>
    </div>
    {!rows.length ? <div className="empty"><Glyph g="run" size={28} />Needs a calibrated pitch</div>
      : <ol className="leader-list">{rows.map((r, i) => <li key={r.p.id}>
        <span className="rank num">{i + 1}</span>
        <Link to={`/review/${a.id}?tab=players`} className="who"><i className="dot" style={{ background: TEAM_HEX[r.p.team] }} />{r.p.label}</Link>
        <span className="lead-bar" aria-hidden="true"><i style={{ width: `${(r.v / max) * 100}%`, background: TEAM_HEX[r.p.team] }} /></span>
        <b className="num">{r.v.toFixed(board === 'speed' ? 1 : 0)}<small>{BOARD[board].unit}</small></b>
      </li>)}</ol>}
  </section>;
}

function Latest({ s, a }: { s: AnalysisSummary; a?: Analysis }) {
  const name = (k: 1 | 2) => a?.teams[String(k) as '1' | '2']?.name ?? `Team ${k}`;
  const n = (type: string, tm: number) => a?.events.filter(e => e.type === type && e.team === tm && e.status !== 'dismissed').length ?? 0;
  const p1 = a?.possession.team1_pct, p2 = a?.possession.team2_pct;
  const top = a?.insights.filter(c => c.kind !== 'data').slice(0, 2) ?? [];
  return <section className="panel latest" aria-label="Latest match">
    <Link to={`/review/${s.id}`} className="latest-film">
      {s.has_video ? <img src={apiUrl(`/api/v1/analyses/${s.id}/thumb`)} alt="" /> : <span className="latest-film-empty"><Glyph g="camera" size={30} /></span>}
      <span className="latest-play"><Glyph g="play" size={18} /></span>
    </Link>
    <div className="latest-body">
      <div className="latest-title">
        <div><small className="muted">{when(s.generated_at)}{s.duration_s ? `, ${clock(s.duration_s)}` : ''}</small><h2>{s.name}</h2></div>
        {s.checks_pending > 0 && <Link className="chip-warn" to={`/review/${s.id}?tab=check`}>{s.checks_pending} to check</Link>}
      </div>
      {!a ? <progress className="spinner" aria-label="Loading the match" />
        : !a.quality.metric ? <p className="muted">Pitch not calibrated for this footage, so there are no distances or positions.</p>
        : <div className="vs">
          <div className="vs-head"><span><i className="dot a" />{name(1)}</span><span>{name(2)}<i className="dot b" /></span></div>
          {p1 != null && p2 != null && <Versus label="Ball control" a={p1} b={p2} fmt={v => `${Math.round(v)}%`} />}
          <Versus label="Passes" a={n('pass', 1)} b={n('pass', 2)} />
          <Versus label="Shots" a={n('shot', 1)} b={n('shot', 2)} />
          <Versus label="Distance" a={a.team_stats['1'].distance_m} b={a.team_stats['2'].distance_m} fmt={v => `${(v / 1000).toFixed(1)} km`} />
        </div>}
      {top.length > 0 && <ul className="latest-cues">{top.map(c => <li key={c.id}><Glyph g="fwd" size={14} /><span><strong>{c.title}</strong>{c.action}</span></li>)}</ul>}
      <Link className="btn" to={`/review/${s.id}`}><Glyph g="review" size={15} />Open match</Link>
    </div>
  </section>;
}

export const SessionPage = () => {
  const [calibs, setCalibs] = useState<Array<{ id: string }>>();
  const [cams, setCams] = useState<unknown[]>();
  const [vest, setVest] = useState<Vest>();
  const [teams, setTeams] = useState<{ calibrated?: boolean }>();
  const [squad, setSquad] = useState<Squad | null>();
  const [recent, setRecent] = useState<AnalysisSummary[]>();
  const [latest, setLatest] = useState<Analysis>();
  const [backend, setBackend] = useState<boolean>();

  useEffect(() => {
    api.health().then(() => setBackend(true)).catch(() => setBackend(false));
    fetch(apiUrl('/api/v1/calibrations')).then(r => r.json()).then(setCalibs).catch(() => setCalibs([]));
    api.cameras().then(setCams).catch(() => setCams([]));
    api.wearableStatus().then(v => setVest(v as Vest)).catch(() => setVest({ connected: false }));
    api.teamProfiles().then(t => setTeams(t as { calibrated?: boolean })).catch(() => setTeams({}));
    getSquad().then(setSquad);
    listAnalyses().then(setRecent).catch(() => setRecent([]));
  }, []);
  useEffect(() => {
    if (!recent?.length || squad === undefined) return;
    getAnalysis(recent[0].id).then(a => setLatest(applySquad(a, squad))).catch(() => setLatest(undefined));
  }, [recent, squad]);

  const players = squad?.players.length ?? 0, linked = squad?.players.filter(p => p.vest != null).length ?? 0;
  const steps: Step[] = [
    { to: '/workshop/cameras', g: 'camera', title: 'Camera', status: cams == null ? '' : cams.length ? `${cams.length} connected` : 'none connected', state: cams?.length ? 'done' : 'todo' },
    { to: '/prepare/pitch', g: 'pitch', title: 'Pitch', status: calibs == null ? '' : calibs.length ? `${calibs.length} saved` : 'not calibrated', state: calibs?.length ? 'done' : 'todo' },
    { to: '/prepare/kits', g: 'kits', title: 'Kits', status: teams?.calibrated ? 'colours set' : 'automatic', state: teams?.calibrated ? 'done' : 'warn' },
    { to: '/squad', g: 'squad', title: 'Squad', status: squad === undefined ? '' : players ? `${players} players, ${linked} vests` : 'no names yet', state: players ? 'done' : 'warn' },
    { to: '/vest', g: 'vest', title: 'Vests', status: vest == null ? '' : vest.connected ? (vest.clock_synced ? 'streaming' : 'syncing clock') : 'none streaming', state: vest?.connected ? (vest.clock_synced ? 'done' : 'warn') : 'todo' },
  ];

  return <div className="page home">
    <div className="top"><h1>Matchday</h1>
      <div className="top-actions"><Link className="btn ghost" to="/live"><Glyph g="live" size={16} />Live</Link><Link className="btn" to="/ingest"><Glyph g="upload" size={16} />Add footage</Link></div>
    </div>
    {backend === false && <p className="notice bad">The analysis service is not running. Start it with scripts/demo.sh</p>}

    <nav className="flow" aria-label="Before kick-off">
      {steps.map(s => <Link key={s.title} to={s.to} className={s.state}>
        <span className="ico"><Glyph g={s.g} size={20} /></span>
        <strong>{s.title}</strong><small>{s.status || ' '}</small>
        {s.state === 'done' && <span className="tick state-ok"><Glyph g="check" size={16} /></span>}
      </Link>)}
    </nav>

    {recent && !recent.length ? <section className="panel empty home-empty"><Glyph g="review" size={40} /><h2>No matches yet</h2>
      <p className="muted">Calibrate the pitch, then add footage or go live.</p>
      <div className="top-actions"><Link className="btn ghost" to="/prepare/pitch">Pitch setup</Link><Link className="btn" to="/ingest">Add footage</Link></div></section>
      : recent && <>
        <div className="home-grid">
          <Latest s={recent[0]} a={latest} />
          {latest?.quality.metric ? <Leaders a={latest} /> : <section className="panel leaders"><div className="panel-head"><h2>Leaders</h2></div>
            <div className="empty"><Glyph g="run" size={28} />{latest ? 'Needs a calibrated pitch' : <progress className="spinner" aria-label="Loading" />}</div></section>}
        </div>
        {recent.length > 1 && <section aria-label="Earlier matches">
          <div className="panel-head"><h2>Earlier matches</h2><Link className="btn ghost small" to="/review">All matches</Link></div>
          <div className="matches">{recent.slice(1, 7).map(r => <Link key={r.id} to={`/review/${r.id}`} className="match-card">
            {r.has_video ? <img src={apiUrl(`/api/v1/analyses/${r.id}/thumb`)} alt="" loading="lazy" /> : <span className="match-card-empty"><Glyph g="camera" size={24} /></span>}
            <div><strong>{r.name}</strong><span>{clock(r.duration_s)}{r.checks_pending ? <b className="state-warn" title="to check"> {r.checks_pending}</b> : null}</span></div>
          </Link>)}</div>
        </section>}
      </>}
  </div>;
};
