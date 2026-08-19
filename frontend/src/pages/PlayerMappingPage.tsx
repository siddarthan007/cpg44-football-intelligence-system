import React, { useEffect, useState } from "react";

export const PlayerMappingPage: React.FC = () => {
  const [players, setPlayers] = useState<any[]>([]);

  useEffect(() => {
    fetch(`http://${window.location.hostname}:8000/api/v1/matches/live/players`)
      .then((r) => r.json())
      .then((d) => setPlayers(d))
      .catch(() => {});
  }, []);

  return (
    <div style={{ padding: "1.5rem", maxWidth: "1100px", margin: "0 auto" }}>
      <div style={{ marginBottom: "1.5rem" }}>
        <h2 style={{ fontSize: "1.3rem", fontWeight: 700, color: "var(--text-main)" }}>
          Squad Roster &amp; Identity Tagging
        </h2>
        <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginTop: "0.25rem" }}>
          Assign jersey numbers, team kit colors, and bind ESP32 wearable IDs to vision tracklets.
        </p>
      </div>

      <div className="card-clean">
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border-light)", textAlign: "left", color: "var(--text-muted)" }}>
              <th style={{ padding: "0.65rem" }}>JERSEY</th>
              <th style={{ padding: "0.65rem" }}>PLAYER NAME</th>
              <th style={{ padding: "0.65rem" }}>TEAM</th>
              <th style={{ padding: "0.65rem" }}>POSITION</th>
              <th style={{ padding: "0.65rem" }}>WEARABLE SENSOR</th>
            </tr>
          </thead>
          <tbody>
            {players.map((p) => (
              <tr key={p.global_player_id} style={{ borderBottom: "1px solid var(--border-light)" }}>
                <td style={{ padding: "0.65rem", fontWeight: 700 }}>#{p.jersey}</td>
                <td style={{ padding: "0.65rem" }}>{p.name}</td>
                <td style={{ padding: "0.65rem", color: p.team === 1 ? "var(--team-blue)" : "var(--team-red)", fontWeight: 600 }}>
                  Team {p.team}
                </td>
                <td style={{ padding: "0.65rem" }}>{p.position}</td>
                <td style={{ padding: "0.65rem" }}>
                  {p.wearable ? (
                    <span style={{ padding: "0.2rem 0.5rem", borderRadius: "4px", background: "var(--accent-blue-light)", color: "var(--accent-blue)", fontWeight: 600, fontSize: "0.75rem" }}>
                      ESP32 #{p.wearable_id}
                    </span>
                  ) : (
                    <span style={{ color: "var(--text-light)" }}>None</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
