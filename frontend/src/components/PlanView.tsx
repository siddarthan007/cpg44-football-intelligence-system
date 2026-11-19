import type { Analysis } from '../lib/analysis';
import { TEAM_HEX } from '../lib/analysis';
import { PitchLines } from './BoardPitch';

type TeamPlan = NonNullable<Analysis['plan']>['teams']['1'];

function Shape({ a, team, plan, phase }: { a: Analysis; team: 1 | 2; plan: TeamPlan; phase: 'with_ball' | 'without_ball' }) {
  const L = a.match.pitch_length_m, W = a.match.pitch_width_m;
  const f = plan.formation[phase];
  const empty = !f.shape && f.positions.length === 0;
  return <figure style={{ margin: 0 }}>
    <svg viewBox={`-2 -2 ${L + 4} ${W + 4}`} width="100%" role="img" aria-label={`${plan.name} ${phase.replace('_', ' ')}: ${f.shape ?? 'shape unclear'}`}>
      <rect x={-2} y={-2} width={L + 4} height={W + 4} fill="var(--board)" />
      <PitchLines L={L} W={W} kind={a.match.pitch_kind} />
      {empty && <g transform={`translate(${L / 2},${W / 2})`}>
        <rect x={-21} y={-4.2} width={42} height={8.4} rx={2} fill="var(--sheet)" stroke="var(--rule-strong)" strokeWidth={0.2} />
        <text y={1.3} textAnchor="middle" style={{ font: '500 3.6px var(--text)', fill: 'var(--ink-2)' }}>
          {phase === 'with_ball' ? 'Not seen on the ball' : 'Not seen defending'}</text></g>}
      {f.positions.map(([id, x, y]) => <g key={id} transform={`translate(${x},${y})`}>
        <circle r={2} fill={TEAM_HEX[team]} stroke="#fff" strokeWidth={0.35} /><text y={0.7} textAnchor="middle" style={{ font: '700 1.8px var(--mono)', fill: '#fff' }}>{id}</text></g>)}
    </svg>
    <figcaption style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12.5, marginTop: 4 }}>
      <span className="muted">{phase === 'with_ball' ? 'with the ball' : 'without the ball'}</span>
      <b className="num">{f.shape ?? (empty ? '' : f.players_seen < 8 ? `${f.players_seen} seen` : 'unclear')}</b></figcaption>
  </figure>;
}

function Space({ zones, team }: { zones: number[][]; team: 1 | 2 }) {
  // zones[channel][third], third 0 = own defensive third
  const cell = 34;
  return <svg viewBox={`0 0 ${cell * 3 + 2} ${cell * 3 + 18}`} width={140} role="img" aria-label="Opponent control by zone when defending">
    {zones.map((row, j) => row.map((v, i) => <g key={`${i}-${j}`}>
      <rect x={1 + i * cell} y={1 + j * cell} width={cell - 2} height={cell - 2} rx={4} fill={TEAM_HEX[team === 1 ? 2 : 1]} opacity={Math.max(0.08, (v - 0.3) * 1.6)} />
      <text x={1 + i * cell + cell / 2 - 1} y={1 + j * cell + cell / 2 + 4} textAnchor="middle" style={{ font: '600 11px var(--mono)', fill: 'var(--ink)' }}>{Math.round(v * 100)}</text>
    </g>))}
    <text x={1} y={cell * 3 + 14} style={{ font: '500 10px var(--text)', fill: 'var(--ink-3)' }}>own goal</text>
    <text x={cell * 3} y={cell * 3 + 14} textAnchor="end" style={{ font: '500 10px var(--text)', fill: 'var(--ink-3)' }}>attack</text>
  </svg>;
}

export function PlanView({ a }: { a: Analysis }) {
  if (!a.plan) return <div className="empty">Re-run the analysis to build the plan</div>;
  return <div className="lower" style={{ gridTemplateColumns: 'repeat(2, minmax(0, 1fr))' }}>
    {([1, 2] as const).map(team => {
      const p = a.plan!.teams[String(team) as '1' | '2'];
      return <div key={team} className="panel">
        <div className="panel-head"><h2><i className="dot" style={{ background: TEAM_HEX[team], marginRight: 8 }} />{p.name}</h2></div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          <Shape a={a} team={team} plan={p} phase="with_ball" /><Shape a={a} team={team} plan={p} phase="without_ball" />
        </div>
        <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start', marginTop: 12, flexWrap: 'wrap' }}>
          {p.space_conceded ? <div><div className="muted" style={{ fontSize: 12.5, marginBottom: 4 }}>Opponent control when defending, %</div><Space zones={p.space_conceded} team={team} /></div>
            : <span className="muted">Not enough defending moments for a space map</span>}
          {p.progression && <div><div className="muted" style={{ fontSize: 12.5 }}>Best route forward</div>
            <div style={{ font: '700 18px var(--display)', marginTop: 4 }}>{p.progression.channel}</div>
            <div className="num muted">{Math.round(p.progression.mean_success * 100)}% arrive, {Math.round(p.progression.share * 100)}% of options</div></div>}
        </div>
      </div>;
    })}
  </div>;
}
