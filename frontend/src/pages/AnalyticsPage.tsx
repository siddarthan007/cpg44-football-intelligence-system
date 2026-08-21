import { TacticalPitch } from '../components/TacticalPitch';
import { EmptyState, ErrorBanner, PageHeader, ProvenanceBanner, StatusBadge, Value } from '../components/common';
import { useLive } from '../lib/useLive';

export const AnalyticsPage = () => {
  const { data, error } = useLive('live');
  const positioned = (data?.players ?? []).filter((player) => typeof player.x === 'number' && typeof player.y === 'number');
  const shot1 = data?.shots_xg?.team1 ?? data?.shots_xg?.['1'];
  const shot2 = data?.shots_xg?.team2 ?? data?.shots_xg?.['2'];
  const tactics = Object.entries(data?.tactics ?? {});
  return (
    <div className="page">
      <PageHeader eyebrow="MATCH ANALYSIS" title="Tactical and workload review" description="Every figure remains tied to the analysis artifact that produced it; unavailable spatial products stay unavailable." actions={<StatusBadge status={data?.data_quality?.status ?? 'unavailable'} />} />
      <ErrorBanner message={error} />
      <ProvenanceBanner snapshot={data} />
      <div className="metric-strip">
        <div><span>Team 1 passes</span><strong>{data?.passes?.team1 ?? data?.passes?.['1'] ?? '-'}</strong></div>
        <div><span>Team 2 passes</span><strong>{data?.passes?.team2 ?? data?.passes?.['2'] ?? '-'}</strong></div>
        <div><span>Team 1 xG</span><strong><Value value={shot1?.xg} digits={2} /></strong></div>
        <div><span>Team 2 xG</span><strong><Value value={shot2?.xg} digits={2} /></strong></div>
      </div>
      <div className="two-column analytics-grid">
        <section className="card-clean">
          <div className="section-heading"><div><span className="eyebrow">LATEST FRAME</span><h2>Measured pitch positions</h2></div><span className="muted">{positioned.length} players</span></div>
          {positioned.length ? <TacticalPitch players={positioned.map((player) => ({ global_player_id: player.global_player_id, jersey: player.jersey ?? undefined, team: player.team, x: player.x as number, y: player.y as number, speed_mps: player.speed_mps ?? undefined, wearable: player.wearable }))} ball={data?.ball ?? undefined} showVoronoi={false} showTrails={false} /> : <EmptyState title="No calibrated per-frame coordinates were persisted in this artifact." />}
        </section>
        <section className="card-clean">
          <div className="section-heading"><div><span className="eyebrow">TACTICAL STATE</span><h2>Team reports</h2></div></div>
          {!tactics.length ? <EmptyState title="No tactical report is available." /> : <div className="tactic-list">{tactics.map(([team, report]) => <article key={team}><strong>{team.replace('team', 'Team ')}</strong><dl>{Object.entries(report).slice(0, 8).map(([key, value]) => <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd>{typeof value === 'number' ? value.toFixed(2) : String(value ?? '-')}</dd></div>)}</dl></article>)}</div>}
        </section>
      </div>
      <section className="card-clean">
        <div className="section-heading"><div><span className="eyebrow">PLAYER LOAD</span><h2>Evidence table</h2></div><span className="muted">Heuristic load indicator, not medical prediction</span></div>
        {!data?.players.length ? <EmptyState title="No player aggregates are available." /> : <div className="table-scroll"><table><thead><tr><th>Track</th><th>Team</th><th>Distance (m)</th><th>Top speed</th><th>HSR (m)</th><th>Sprints</th><th>Load review</th></tr></thead><tbody>{data.players.map((player) => <tr key={player.global_player_id}><td><strong>{player.global_player_id}</strong></td><td>{player.team || '-'}</td><td><Value value={player.distance_m} /></td><td><Value value={player.top_speed_mps} digits={2} /></td><td><Value value={player.hsr_m ?? player.load?.hsr_m} /></td><td>{player.sprints ?? player.load?.sprints ?? '-'}</td><td>{player.load_indicator?.score == null ? '-' : `${Math.round(player.load_indicator.score * 100)} / 100 ${player.load_indicator.severity ?? ''}`}</td></tr>)}</tbody></table></div>}
      </section>
    </div>
  );
};
