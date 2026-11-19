import { useState } from 'react';
import type { Analysis, AnalysisPlayer } from '../lib/analysis';
import { TEAM_HEX } from '../lib/analysis';
import { apiUrl } from '../lib/api';

export function PlayerCard({ analysis, player, onChange }: { analysis: Analysis; player: AnalysisPlayer; onChange?: () => void }) {
  const sprints = analysis.events.filter(e => e.type === 'sprint' && e.players.includes(player.id)).length;
  const L = analysis.match.pitch_length_m, W = analysis.match.pitch_width_m;
  const gy = player.heat.length, gx = player.heat[0]?.length ?? 1;
  const [vest, setVest] = useState(player.vest != null ? String(player.vest) : '');
  const [busy, setBusy] = useState(false);
  const save = async (value: string) => {
    setBusy(true);
    await fetch(apiUrl(`/api/v1/analyses/${analysis.id}/vest`), { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ player: player.id, vest: value === '' ? null : Number(value) }) }).catch(() => undefined);
    setBusy(false);
    onChange?.();
  };
  return <div className="panel player-card">
    <div className="player-card-head"><i className="dot" style={{ background: TEAM_HEX[player.team], width: 12, height: 12 }} /><strong>{player.label}</strong></div>
    <dl>
      <div><dt>Distance</dt><dd>{player.distance_m.toFixed(0)}<small> m</small></dd></div>
      <div><dt>Top speed</dt><dd>{player.top_speed_kmh?.toFixed(1) ?? 'n/a'}<small> km/h</small></dd></div>
      <div><dt>High speed</dt><dd>{player.hsr_m.toFixed(0)}<small> m</small></dd></div>
      <div><dt>Sprints</dt><dd>{sprints}</dd></div>
    </dl>
    <svg className="mini-heat" viewBox={`0 0 ${L} ${W}`} role="img" aria-label={`Where ${player.label} played`}>
      <rect width={L} height={W} fill="var(--board)" />
      {player.heat.map((row, j) => row.map((v, i) => v > 0.03 && <rect key={`${i}-${j}`} x={(i * L) / gx} y={(j * W) / gy} width={L / gx} height={W / gy}
        fill="var(--team-b)" opacity={Math.min(0.85, 0.15 + v * 0.75)} />))}
      <rect width={L} height={W} fill="none" stroke="var(--board-line)" strokeWidth={0.4} />
      <line x1={L / 2} x2={L / 2} y1={0} y2={W} stroke="var(--board-line)" strokeWidth={0.4} />
    </svg>
    <form className="vest-tag-form" onSubmit={e => { e.preventDefault(); void save(vest); }}>
      <label htmlFor={`vest-${player.id}`}>Vest worn</label>
      <div>
        <input id={`vest-${player.id}`} inputMode="numeric" placeholder="none" value={vest} onChange={e => setVest(e.target.value.replace(/\D/g, ''))} />
        <button className="btn small" type="submit" disabled={busy || vest === (player.vest != null ? String(player.vest) : '')}>Save</button>
      </div>
      <small>{player.vest == null ? 'Tag the player wearing a vest; vest data then appears on this player.'
        : (player as AnalysisPlayer & { vest_source?: string }).vest_source === 'coach' ? 'Tagged by hand.' : 'Matched from the vest\'s own movement.'}</small>
    </form>
  </div>;
}
