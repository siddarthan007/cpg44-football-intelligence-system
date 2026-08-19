import React, { useEffect, useState } from "react";

export const PlayerMappingPage: React.FC = () => {
  const [players, setPlayers] = useState<any[]>([]);

  useEffect(() => {
    fetch(`http://${window.location.hostname}:8000/api/v1/matches/live/players`)
      .then((res) => res.json())
      .then((data) => setPlayers(data))
      .catch(() => {});
  }, []);

  return (
    <div>
      <div className="card-header" style={{ marginBottom: "1.25rem" }}>
        <div>
          <h2 style={{ fontSize: "1.4rem", fontWeight: 700 }}>
            👥 Squad Roster &amp; Track-to-Player Identity Mapping
          </h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem", marginTop: "0.25rem" }}>
            Map YOLO track IDs to squad roster names, EasyOCR jersey numbers, and ESP32 wearable IDs.
          </p>
        </div>
      </div>

      <div className="card">
        <table className="data-table">
          <thead>
            <tr>
              <th>Global ID</th>
              <th>Jersey #</th>
              <th>Player Name</th>
              <th>Team</th>
              <th>Position</th>
              <th>Wearable Sensor</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {players.map((p) => (
              <tr key={p.global_player_id}>
                <td><strong>{p.global_player_id}</strong></td>
                <td>#{p.jersey}</td>
                <td>{p.name}</td>
                <td>
                  <span style={{ color: p.team === 1 ? "var(--team-blue)" : "var(--team-red)" }}>
                    Team {p.team}
                  </span>
                </td>
                <td>{p.position}</td>
                <td>
                  {p.wearable ? (
                    <span className="badge-pill badge-wearable">ESP32 ID #{p.wearable_id}</span>
                  ) : (
                    <span style={{ color: "var(--text-muted)" }}>None</span>
                  )}
                </td>
                <td>
                  <button className="btn btn-secondary" style={{ padding: "0.25rem 0.5rem", fontSize: "0.75rem" }}>
                    Edit Binding
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
