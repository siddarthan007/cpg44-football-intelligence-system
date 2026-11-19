import { useMemo } from 'react';
import { PitchLines } from './BoardPitch';
import { TEAM_HEX } from '../lib/analysis';
import type { HeatmapGrid, LivePlayer } from '../lib/types';

const M = 4;

interface Props {
  L: number; W: number; kind?: 'football' | 'futsal';
  players: LivePlayer[]; ball: { x: number; y: number } | null; heat?: HeatmapGrid | null;
  mode: 'positions' | 'heat'; selected?: number; onSelect?: (id: number) => void;
}

const placed = (p: LivePlayer) => p.x != null && p.y != null && Number.isFinite(p.x) && Number.isFinite(p.y);

/** Top-down pitch for the live feed: same lines and markers as the match board. */
export function LiveBoard({ L, W, kind, players, ball, heat, mode, selected, onSelect }: Props) {
  const k = Math.max(0.5, Math.min(1, L / 105));
  const shown = players.filter(placed);
  const cells = useMemo(() => {
    if (mode !== 'heat' || !heat?.cells?.length) return null;
    const max = Math.max(...heat.cells, 1e-9);
    return heat.cells.map((v, n) => ({ i: n % heat.cols, j: Math.floor(n / heat.cols), v: v / max })).filter(c => c.v > 0.04);
  }, [heat, mode]);
  return <svg className="board-pitch" viewBox={`${-M} ${-M} ${L + 2 * M} ${W + 2 * M}`} role="img"
    aria-label={`Live pitch: ${shown.length} players placed`}>
    <defs>
      <pattern id="mow-live" width={L / 12} height={W} patternUnits="userSpaceOnUse">
        <rect width={L / 24} height={W} className="mow-a" /><rect x={L / 24} width={L / 24} height={W} className="mow-b" />
      </pattern>
    </defs>
    <rect x={-M} y={-M} width={L + 2 * M} height={W + 2 * M} className="board-surround" />
    <rect x={0} y={0} width={L} height={W} fill="url(#mow-live)" />
    {cells && heat && cells.map(c => <rect key={`${c.i}-${c.j}`} x={(c.i * L) / heat.cols} y={(c.j * W) / heat.rows}
      width={L / heat.cols} height={W / heat.rows} className="heat-cell" style={{ opacity: Math.min(0.85, 0.12 + c.v * 0.8) }} />)}
    <PitchLines L={L} W={W} kind={kind} />
    {mode === 'positions' && shown.map(p => <g key={p.track_id} transform={`translate(${p.x},${p.y}) scale(${k})`}
      className={`chip ${selected === p.track_id ? 'selected' : ''}`} onClick={() => onSelect?.(p.track_id)} role="button"
      aria-label={`${p.name || `Player ${p.track_id}`}${p.speed_mps != null ? `, ${(p.speed_mps * 3.6).toFixed(0)} km/h` : ''}`}>
      {p.wearable && <circle r={2.05} className="vest-ring" />}
      <circle r={1.45} className="body" style={{ fill: TEAM_HEX[p.team] ?? TEAM_HEX[0] }} />
      <text y={0.5} textAnchor="middle">{p.jersey ?? p.track_id}</text>
    </g>)}
    {mode === 'positions' && ball && <g transform={`translate(${ball.x},${ball.y}) scale(${k})`}>
      <ellipse rx={0.9} ry={0.5} className="ball-shadow" /><circle r={0.75} className="ball" />
    </g>}
  </svg>;
}
