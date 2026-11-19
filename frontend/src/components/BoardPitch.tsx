import { useMemo } from 'react';
import type { Analysis, TimelineSample } from '../lib/analysis';
import { ballAt, interpolateAt, pitchControl, TEAM_HEX, trackWindow } from '../lib/analysis';

export type BoardMode = 'moment' | 'options' | 'shape' | 'heat' | 'links';
const REF = 3;
const GHOST_S = 4;            // how long a player who left the camera view stays as a ghost
const M = 4;                  // margin around the pitch (m)

interface Props { analysis: Analysis; t: number; mode: BoardMode; team: 0 | 1 | 2; selected?: number; onSelect?: (id: number) => void }

function hull(points: [number, number][]) {
  const p = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cross = (o: number[], a: number[], b: number[]) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo: [number, number][] = [], up: [number, number][] = [];
  for (const q of p) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
  for (const q of [...p].reverse()) { while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}

/** Sample at or before t (binary search). */
function at(frames: TimelineSample[], t: number) {
  let lo = 0, hi = frames.length - 1;
  if (hi < 0) return -1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (frames[mid].t <= t) lo = mid; else hi = mid - 1; }
  return lo;
}

export function PitchLines({ L, W, kind = 'football' }: { L: number; W: number; kind?: 'football' | 'futsal' }) {
  const cy = W / 2;
  if (kind === 'futsal') {
    // FIFA futsal Law 1: 3 m goal, 3 m centre circle, 6 m arcs around the posts joined by 3.16 m
    const area = (x0: number, d: number) => `M${x0},${cy - 7.58} A6,6 0 0 ${d > 0 ? 1 : 0} ${x0 + d * 6},${cy - 1.58} L${x0 + d * 6},${cy + 1.58} A6,6 0 0 ${d > 0 ? 1 : 0} ${x0},${cy + 7.58}`;
    return <g className="board-lines" fill="none">
      <rect x={0} y={0} width={L} height={W} />
      <line x1={L / 2} y1={0} x2={L / 2} y2={W} />
      <circle cx={L / 2} cy={cy} r={3} /><circle cx={L / 2} cy={cy} r={0.15} className="spot" />
      <path d={area(0, 1)} /><path d={area(L, -1)} />
      {[6, 10, L - 6, L - 10].map(x => <circle key={x} cx={x} cy={cy} r={0.15} className="spot" />)}
      <path d={`M0,${cy - 1.5} L-0.8,${cy - 1.5} L-0.8,${cy + 1.5} L0,${cy + 1.5}`} className="goal" />
      <path d={`M${L},${cy - 1.5} L${L + 0.8},${cy - 1.5} L${L + 0.8},${cy + 1.5} L${L},${cy + 1.5}`} className="goal" />
    </g>;
  }
  const box = (x0: number, dir: number, d: number, w: number) =>
    `M${x0},${cy - w / 2} L${x0 + dir * d},${cy - w / 2} L${x0 + dir * d},${cy + w / 2} L${x0},${cy + w / 2}`;
  return <g className="board-lines" fill="none">
    <rect x={0} y={0} width={L} height={W} />
    <line x1={L / 2} y1={0} x2={L / 2} y2={W} />
    <circle cx={L / 2} cy={cy} r={9.15} /><circle cx={L / 2} cy={cy} r={0.35} className="spot" />
    <path d={box(0, 1, 16.5, 40.32)} /><path d={box(L, -1, 16.5, 40.32)} />
    <path d={box(0, 1, 5.5, 18.32)} /><path d={box(L, -1, 5.5, 18.32)} />
    <circle cx={11} cy={cy} r={0.35} className="spot" /><circle cx={L - 11} cy={cy} r={0.35} className="spot" />
    <path d={`M16.5,${cy - 7.3} A9.15,9.15 0 0 1 16.5,${cy + 7.3}`} />
    <path d={`M${L - 16.5},${cy - 7.3} A9.15,9.15 0 0 0 ${L - 16.5},${cy + 7.3}`} />
    <path d={`M0,${cy - 3.66} L-1.6,${cy - 3.66} L-1.6,${cy + 3.66} L0,${cy + 3.66}`} className="goal" />
    <path d={`M${L},${cy - 3.66} L${L + 1.6},${cy - 3.66} L${L + 1.6},${cy + 3.66} L${L},${cy + 3.66}`} className="goal" />
  </g>;
}

export function BoardPitch({ analysis, t, mode, team, selected, onSelect }: Props) {
  const L = analysis.match.pitch_length_m, W = analysis.match.pitch_width_m;
  const frames = analysis.timeline.frames;
  const idx = at(frames, t);
  const now = idx >= 0 ? frames[idx] : undefined;

  // players visible now, plus ghosts: seen in the last few seconds but outside the camera now
  const { live, ghosts, refs } = useMemo(() => {
    const live = (now?.p ?? []).filter(p => p[3] !== REF && (!team || p[3] === team));
    const refs = (now?.p ?? []).filter(p => p[3] === REF);
    const seen = new Set(live.map(p => p[0]));
    const ghosts: { id: number; x: number; y: number; tm: number; age: number }[] = [];
    if (mode === 'moment' && idx > 0) {
      for (let k = idx - 1; k >= 0 && frames[k].t > t - GHOST_S; k--) {
        for (const p of frames[k].p) {
          if (p[3] === REF || seen.has(p[0]) || (team && p[3] !== team)) continue;
          seen.add(p[0]);
          ghosts.push({ id: p[0], x: p[1], y: p[2], tm: p[3], age: t - frames[k].t });
        }
      }
    }
    return { live, ghosts, refs };
  }, [now, idx, frames, t, team, mode]);

  const trails = useMemo(() => {
    const out = new Map<number, [number, number][]>();
    if (mode !== 'moment') return out;
    for (let k = Math.max(0, idx - 10); k <= idx; k++) for (const [id, x, y, tm] of frames[k]?.p ?? []) {
      if (tm === REF) continue;
      if (!out.has(id)) out.set(id, []);
      out.get(id)!.push([x, y]);
    }
    return out;
  }, [frames, idx, mode]);

  // Same interpolation as the 3D replay: players between the two samples around t,
  // the ball from the full-rate track, so both views agree at every moment.
  const track = analysis.ball_track;
  const lerp = useMemo(() => new Map((interpolateAt(frames, t)?.players ?? []).map(p => [p.id, [p.x, p.y] as [number, number]])), [frames, t]);
  const ballNow = track?.length ? ballAt(track, t) : now?.b ?? null;
  const ballTrail = useMemo(() => track?.length ? trackWindow(track, t - 1.2, t).map(r => [r[1], r[2], r[3]] as [number, number, number])
    : frames.slice(Math.max(0, idx - 6), idx + 1).filter(f => f.b).map(f => f.b!), [track, frames, idx, t]);

  const heat = useMemo(() => {
    if (mode !== 'heat') return null;
    const rows = analysis.players.filter(p => (selected ? p.id === selected : !team || p.team === team));
    if (!rows.length) return null;
    const gy = rows[0].heat.length, gx = rows[0].heat[0]?.length ?? 0;
    const grid = Array.from({ length: gy }, () => new Array(gx).fill(0));
    for (const p of rows) p.heat.forEach((r, j) => r.forEach((v, i) => { grid[j][i] += v * p.observed_s; }));
    const max = Math.max(...grid.flat(), 1e-9);
    return { grid: grid.map(r => r.map(v => v / max)), gx, gy };
  }, [analysis, mode, team, selected]);

  const links = useMemo(() => {
    if (mode !== 'links') return [];
    const count = new Map<string, number>();
    for (const e of analysis.events) {
      if (e.type !== 'pass' || e.status === 'dismissed' || e.players.length < 2 || (team && e.team !== team)) continue;
      const k = `${e.players[0]}-${e.players[1]}`;
      count.set(k, (count.get(k) ?? 0) + 1);
    }
    const byId = new Map(analysis.players.map(p => [p.id, p]));
    return [...count.entries()].map(([k, n]) => { const [a, b] = k.split('-').map(Number); return { a: byId.get(a), b: byId.get(b), n }; })
      .filter(l => l.a && l.b);
  }, [analysis, mode, team]);

  const shirt = useMemo(() => new Map(analysis.players.filter(p => p.number != null).map(p => [p.id, p.number!])), [analysis]);
  const names = useMemo(() => new Map(analysis.players.map(p => [p.id, p.label])), [analysis]);
  const vestIds = useMemo(() => new Map(analysis.players.filter(p => p.vest != null).map(p => [p.id, p.vest!])), [analysis]);
  // markers are drawn in metres; on small pitches (futsal, 7 a side) shrink them
  const k = Math.max(0.5, Math.min(1, L / 105));
  const view = now?.v;
  // Options: the side on the ball, its control surface, and the passes the model rates.
  const onBall = now?.d?.team ?? (now?.pos || 0);
  const control = useMemo(() => (mode === 'options' && now && onBall ? pitchControl(now, onBall, L, W) : null), [mode, now, onBall, L, W]);
  const carrier = now?.d ? now.p.find(p => p[0] === now.d!.carrier) : undefined;
  // Control is only drawn where the camera sees: players outside the frame are
  // unknown, so shading there would be invented.
  const seen = view && view.length > 2 ? view.map(p => p.join(',')).join(' ') : null;
  return <svg className="board-pitch" viewBox={`${-M} ${-M} ${L + 2 * M} ${W + 2 * M}`} role="img"
    aria-label={`Pitch at ${t.toFixed(1)} s: ${live.length} players in camera view`}>
    <defs>
      <marker id="arrow" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="4" markerHeight="4" orient="auto-start-reverse"><path d="M0,0 L6,3 L0,6 z" fill="var(--ink)" /></marker>
      <pattern id="mow" width={L / 12} height={W} patternUnits="userSpaceOnUse">
        <rect width={L / 24} height={W} className="mow-a" /><rect x={L / 24} width={L / 24} height={W} className="mow-b" />
      </pattern>
    </defs>
    <rect x={-M} y={-M} width={L + 2 * M} height={W + 2 * M} className="board-surround" />
    <rect x={0} y={0} width={L} height={W} fill="url(#mow)" />
    {mode === 'moment' && view && view.length > 2 && <polygon className="cam-view" points={view.map(p => p.join(',')).join(' ')} />}
    {control && seen && <clipPath id="seen-area"><polygon points={seen} /></clipPath>}
    {control && <g clipPath={seen ? 'url(#seen-area)' : undefined}>{control.map((row, j) => row.map((v, i) => <rect key={`c${i}-${j}`} x={(i * L) / row.length} y={(j * W) / control.length}
      width={L / row.length + 0.05} height={W / control.length + 0.05} fill={v >= 0.5 ? TEAM_HEX[onBall] : TEAM_HEX[onBall === 1 ? 2 : 1]}
      opacity={Math.abs(v - 0.5) * 0.7} />))}</g>}
    {control && seen && <polygon points={seen} className="seen-edge" />}
    {heat && heat.grid.map((row, j) => row.map((v, i) => v > 0.04 && <rect key={`${i}-${j}`}
      x={(i * L) / heat.gx} y={(j * W) / heat.gy} width={L / heat.gx} height={W / heat.gy}
      className="heat-cell" style={{ opacity: Math.min(0.85, 0.12 + v * 0.8) }} />))}
    <PitchLines L={L} W={W} kind={analysis.match.pitch_kind} />

    {mode === 'shape' && now && [1, 2].filter(k => !team || team === k).map(k => {
      const pts = now.p.filter(p => p[3] === k).map(p => lerp.get(p[0]) ?? [p[1], p[2]] as [number, number]);
      if (pts.length < 3) return null;
      const dir = analysis.teams[String(k) as '1' | '2'].attack_dir;
      const xs = pts.map(p => p[0]).sort((a, b) => dir > 0 ? a - b : b - a);
      const line = xs[Math.min(1, xs.length - 1)];
      return <g key={k} className="shape">
        <polygon points={hull(pts).map(p => p.join(',')).join(' ')} style={{ fill: TEAM_HEX[k], stroke: TEAM_HEX[k] }} />
        <line x1={line} x2={line} y1={0} y2={W} style={{ stroke: TEAM_HEX[k] }} className="last-line" />
      </g>;
    })}

    {mode === 'links' && links.map(({ a, b, n }, i) => <line key={i} x1={a!.avg_x} y1={a!.avg_y} x2={b!.avg_x} y2={b!.avg_y}
      className="pass-link" style={{ stroke: TEAM_HEX[a!.team], strokeWidth: 0.25 + n * 0.35 }} />)}
    {mode === 'links' && analysis.players.filter(p => p.team && (!team || p.team === team)).map(p =>
      <g key={p.id} transform={`translate(${p.avg_x},${p.avg_y}) scale(${k})`} className="node" onClick={() => onSelect?.(p.id)}>
        <circle r={1.6} style={{ fill: TEAM_HEX[p.team] }} /><text y={0.55} textAnchor="middle">{p.id}</text>
      </g>)}

    {mode === 'moment' && [...trails.entries()].map(([id, pts]) => pts.length > 1 &&
      <polyline key={id} points={pts.map(p => p.join(',')).join(' ')} className="trail" />)}
    {mode === 'moment' && ghosts.map(g => <g key={`g${g.id}`} transform={`translate(${g.x},${g.y}) scale(${k})`} className="chip ghost"
      onClick={() => onSelect?.(g.id)} style={{ opacity: Math.max(0.35, 1 - g.age / GHOST_S) }}>
      <circle r={1.45} className="body" style={{ fill: TEAM_HEX[g.tm], stroke: TEAM_HEX[g.tm] }} /><text y={0.5} textAnchor="middle">{g.id}</text>
      <title>{`Player ${g.id}, last seen ${g.age.toFixed(1)} s ago`}</title>
    </g>)}
    {mode === 'options' && carrier && now?.o?.map(([to, x, y, p]) => {
      const best = now.d?.call !== 'shoot' && now.d?.to === to;
      return <g key={`o${to}`} className="option">
        <line x1={carrier[1]} y1={carrier[2]} x2={x} y2={y} stroke={best ? 'var(--ink)' : 'var(--ink-3)'} strokeWidth={best ? 0.6 : 0.3}
          strokeDasharray={best ? undefined : '1 .7'} markerEnd="url(#arrow)" opacity={0.35 + 0.65 * p} />
        <g transform={`translate(${(carrier[1] + x) / 2},${(carrier[2] + y) / 2 - 1.2 * k}) scale(${k})`}>
          <rect x={-2.6} y={-1.15} width={5.2} height={2.1} rx={0.6} fill={best ? 'var(--ink)' : 'var(--sheet)'} stroke="var(--ink)" strokeWidth={0.12} />
          <text textAnchor="middle" y={0.45} style={{ font: '700 1.35px var(--mono)', fill: best ? 'var(--paper)' : 'var(--ink)' }}>{Math.round(p * 100)}%</text>
        </g>
      </g>;
    })}
    {mode === 'options' && carrier && now?.d?.call === 'shoot' && (() => {
      const gx = analysis.teams[String(now.d!.team) as '1' | '2'].attack_dir > 0 ? L : 0;
      return <g><polygon points={`${carrier[1]},${carrier[2]} ${gx},${W / 2 - 3.66} ${gx},${W / 2 + 3.66}`} fill="var(--ball)" opacity={0.35} />
        <text x={carrier[1]} y={carrier[2] - 3} textAnchor="middle" style={{ font: '700 1.6px var(--mono)', fill: 'var(--ink)' }}>xG {now.d!.xg.toFixed(2)}</text></g>;
    })()}
    {(mode === 'moment' || mode === 'shape' || mode === 'options') && refs.map(([id, x, y]) => <g key={`r${id}`} className="ref" transform={`translate(${x},${y}) scale(${k})`}>
      <circle r={0.8} /><title>Match official</title></g>)}
    {(mode === 'moment' || mode === 'shape' || mode === 'options') && live.map(([id, x0, y0, tm, sp]) => { const [x, y] = lerp.get(id) ?? [x0, y0]; return <g key={id} transform={`translate(${x},${y}) scale(${k})`}
      className={`chip ${selected === id ? 'selected' : ''}`} onClick={() => onSelect?.(id)} role="button"
      aria-label={`Player ${id}${sp != null ? `, ${(sp * 3.6).toFixed(0)} km/h` : ''}`}>
      {now?.h === id && <circle r={2.6} className="holder-ring" />}
      {vestIds.has(id) && <circle r={2.05} className="vest-ring"><title>Wearing vest {vestIds.get(id)}</title></circle>}
      <circle r={1.45} className="body" style={{ fill: TEAM_HEX[tm] }} /><text y={0.5} textAnchor="middle">{shirt.get(id) ?? id}</text>
      <title>{names.get(id) ?? `Player ${id}`}</title>
    </g>; })}
    {mode === 'moment' && ballTrail.length > 1 && <polyline className="ball-trail" points={ballTrail.map(b => `${b[0]},${b[1]}`).join(' ')} />}
    {(mode === 'moment' || mode === 'shape' || mode === 'options') && ballNow && <g transform={`translate(${ballNow[0]},${ballNow[1]}) scale(${k})`}>
      <ellipse rx={0.9} ry={0.5} className="ball-shadow" />
      <circle cy={-Math.min(ballNow[2], 8) * 0.35} r={0.75 + Math.min(ballNow[2], 6) * 0.06} className="ball" />
    </g>}
    <g transform={`translate(${L / 2 + (analysis.teams['1'].attack_dir > 0 ? 6 : -6) * k},${-1.6 * k}) scale(${k})`}>
      <path d={analysis.teams['1'].attack_dir > 0 ? 'M-3,0 H3 M1.5,-1 L3,0 L1.5,1' : 'M3,0 H-3 M-1.5,-1 L-3,0 L-1.5,1'}
        stroke={TEAM_HEX[1]} strokeWidth={0.4} fill="none" />
    </g>
  </svg>;
}

const CALL: Record<string, string> = { shoot: 'Shoot', pass: 'Pass forward', recycle: 'Keep it, play safe', carry: 'Carry the ball' };
export function BoardLegend({ mode, analysis, t }: { mode: BoardMode; analysis?: Analysis; t?: number }) {
  if (mode === 'options') {
    const f = analysis && t != null ? analysis.timeline.frames[Math.max(0, at(analysis.timeline.frames, t))] : undefined;
    return <div className="board-legend">
      {f?.d ? <><strong style={{ color: 'var(--ink)' }}>{CALL[f.d.call]}{f.d.to != null && f.d.call !== 'shoot' ? ` to ${f.d.to}` : ''}</strong>
        <span>xG here {f.d.xg.toFixed(2)}</span>{f.d.pressure ? <span>{f.d.pressure} pressing</span> : null}</>
        : <span>No player on the ball at this moment</span>}
      <span style={{ marginLeft: 'auto' }}>shading: who reaches each spot first, inside the camera view</span>
    </div>;
  }
  if (mode !== 'moment') return null;
  return <div className="board-legend">
    <span><i className="swatch-view" />camera sees</span>
    <span><i className="swatch-ghost" />left the frame</span>
    <span><i className="dot" style={{ background: 'var(--ink)', width: 7, height: 7 }} />official</span>
    <span><i className="dot" style={{ background: 'var(--ball)' }} />ball</span>
  </div>;
}
