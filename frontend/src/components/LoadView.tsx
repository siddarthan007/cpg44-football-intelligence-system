import type { Analysis, LoadPlayer } from '../lib/analysis';
import { TEAM_HEX } from '../lib/analysis';
import { Glyph } from './Glyph';

const STATUS: Record<LoadPlayer['status'], { label: string; g: 'check' | 'eye' | 'warn' }> = {
  fresh: { label: 'Fresh', g: 'check' }, watch: { label: 'Watch', g: 'eye' }, tiring: { label: 'Tiring', g: 'warn' },
};

function Spark({ p, team }: { p: LoadPlayer; team?: [number, number][] }) {
  const W = 160, H = 36;
  const all = [...p.curve, ...(team ?? [])];
  const t0 = Math.min(...all.map(r => r[0])), t1 = Math.max(...all.map(r => r[0]));
  const vmax = Math.max(1, ...all.map(r => r[1]));
  const path = (rows: [number, number][]) => rows.map(([t, v], i) => `${i ? 'L' : 'M'}${(((t - t0) / Math.max(t1 - t0, 1)) * (W - 4) + 2).toFixed(1)},${(H - 3 - (v / vmax) * (H - 8)).toFixed(1)}`).join(' ');
  return <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${p.metric} per minute over the session, now ${p.now}, peak ${p.peak}`}>
    {team && team.length > 1 && <path d={path(team)} fill="none" stroke="var(--ink-3)" strokeWidth={1.2} strokeDasharray="3 3" />}
    <path d={path(p.curve)} fill="none" stroke="var(--ink)" strokeWidth={2} strokeLinejoin="round" />
  </svg>;
}

export function LoadView({ a }: { a: Analysis }) {
  const load = a.load;
  const name = (k: number) => a.teams[String(k) as '1' | '2']?.name ?? `Side ${k}`;
  if (!load) return <div className="empty"><Glyph g="run" size={32} />Re-run the analysis to measure load</div>;
  if (!load.available) return <div className="empty"><Glyph g="run" size={32} />
    <span>Load and fatigue need {Math.round((load.min_duration_s ?? 600) / 60)} minutes of play{load.duration_s != null ? `; this footage is ${Math.round(load.duration_s)} s` : ''}.</span>
    <small className="muted">Short clips cannot tell a tired player from a quiet spell of the game.</small></div>;
  const flagged = load.players!.filter(p => p.status !== 'fresh');
  const fresh = load.players!.filter(p => p.status === 'fresh');
  const vestOf = new Map(a.players.filter(p => p.vest != null).map(p => [p.id, p.vest!]));
  return <div>
    {(load.subs ?? []).length > 0 && <div className="sub-calls">{load.subs!.map(s => <div key={s.out} className="sub-call">
      <i className="dot" style={{ background: TEAM_HEX[s.team] }} />
      <div><strong>{name(s.team)}: replace {s.out} {s.when}</strong><span>{s.why.join(', ')}. Like for like: {s.role}.</span></div>
    </div>)}</div>}
    {flagged.length > 0 && <div className="load-head"><span>Player</span><span>Now against own peak, per minute</span><span>Status</span></div>}
    {flagged.map(p => <div key={p.id} className={`load-row ${p.status}`}>
      <div className="who"><i className="dot" style={{ background: TEAM_HEX[p.team] }} /><b className="num">{p.id}</b>
        {vestOf.has(p.id) && <span className="vest-tag" title="Wearing a vest">vest {vestOf.get(p.id)}</span>}<small>{p.role}</small></div>
      <div className="trend"><Spark p={p} team={load.team_curve?.[String(p.team) as '1' | '2']} />
        <span className="num">{p.now.toFixed(0)}<small> / {p.peak.toFixed(0)} m</small></span>
        <small className="muted">{p.notes.filter(n => !n.startsWith(p.metric)).join(', ')}</small></div>
      <div className="state"><Glyph g={STATUS[p.status].g} size={15} />{STATUS[p.status].label}
        {p.minutes_to_half_peak != null && <small className="muted">half of peak in {Math.round(p.minutes_to_half_peak)} min</small>}</div>
    </div>)}
    {fresh.length > 0 && <div className="fresh-row"><span className="state-ok"><Glyph g="check" size={15} />Holding their level</span>
      <div className="chips">{fresh.map(p => <span key={p.id} className="chip-id" title={`${p.role}: ${p.now.toFixed(0)} of ${p.peak.toFixed(0)} m/min peak`}>
        <i className="dot" style={{ background: TEAM_HEX[p.team] }} />{p.id}{vestOf.has(p.id) && <span className="vest-tag">vest {vestOf.get(p.id)}</span>}</span>)}</div></div>}
    <p className="muted" style={{ fontSize: 12, marginTop: 10 }}>Solid line: this player. Dashed: the side's median. A drop shared by the whole side is game state, not fatigue. Coaching cue, not a medical assessment.</p>
  </div>;
}
