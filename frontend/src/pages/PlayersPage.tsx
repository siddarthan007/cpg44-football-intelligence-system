import { useLive } from '../lib/useLive';
import { ErrorBanner, EmptyState, Panel } from '../components/common';

export function PlayersPage({ matchId }: { matchId: string }) {
  const { data, error } = useLive(matchId);
  const players = [...(data?.analytics.players ?? [])].sort((a, b) => b.distance_m - a.distance_m);
  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Players</h1>
          <p className="muted small">Load from the last vision run (`demo_out/stats.json` or a new pipeline out dir).</p>
        </div>
      </div>
      <ErrorBanner message={error} />
      <Panel title="Workload table">
        {players.length === 0 ? (
          <EmptyState title="No stats yet — run bash scripts/demo.sh" />
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Player</th>
                  <th>Team</th>
                  <th className="num">Dist m</th>
                  <th className="num">Vmax</th>
                  <th className="num">HSR</th>
                  <th className="num">Sprints</th>
                  <th className="num">W/kg</th>
                  <th>Vest</th>
                </tr>
              </thead>
              <tbody>
                {players.map((p) => (
                  <tr key={p.global_player_id}>
                    <td className="mono">{p.global_player_id}</td>
                    <td>{p.team_id}</td>
                    <td className="num">{p.distance_m.toFixed(1)}</td>
                    <td className="num">{p.top_speed_mps.toFixed(2)}</td>
                    <td className="num">{p.hsr_m.toFixed(1)}</td>
                    <td className="num">{p.sprints}</td>
                    <td className="num">{p.metabolic_wkg.toFixed(1)}</td>
                    <td>{p.wearable ? 'yes' : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}
