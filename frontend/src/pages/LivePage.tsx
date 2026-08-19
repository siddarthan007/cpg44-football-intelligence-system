import React, { useEffect, useState } from "react";
import { TacticalPitch } from "../components/TacticalPitch";

export const LivePage: React.FC = () => {
  const [mode, setMode] = useState<"demo" | "live" | "upload">("demo");
  const [matchData, setMatchData] = useState<any>(null);
  const [wsConnected, setWsConnected] = useState(false);

  useEffect(() => {
    // Connect to live WebSocket
    const wsUrl = `ws://${window.location.hostname}:8000/ws/live`;
    let ws: WebSocket;
    let pollInterval: any;

    try {
      ws = new WebSocket(wsUrl);
      ws.onopen = () => setWsConnected(true);
      ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);
          setMatchData(data);
        } catch (e) {}
      };
      ws.onclose = () => setWsConnected(false);
    } catch (e) {
      setWsConnected(false);
    }

    // Fallback polling if WS disconnects
    pollInterval = setInterval(() => {
      fetch(`http://${window.location.hostname}:8000/api/v1/matches/live/analytics`)
        .then((res) => res.json())
        .then((data) => setMatchData(data))
        .catch(() => {});
    }, 400);

    return () => {
      if (ws) ws.close();
      clearInterval(pollInterval);
    };
  }, []);

  const players = matchData?.players || [];
  const ball = matchData?.ball;
  const possession = matchData?.possession_pct || { "1": 55, "2": 45 };

  return (
    <div>
      {/* Scoreboard Banner */}
      <div className="scoreboard-banner">
        <div className="team-score">
          <span className="team-name blue">Blue Knights (Home)</span>
          <span className="score-digits">2 - 1</span>
          <span className="team-name red">Red Hawks (Away)</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <span className="match-clock">⏱ 34:12 (1st Half)</span>
          <div className="mode-tabs">
            <button
              className={`mode-tab ${mode === "demo" ? "active" : ""}`}
              onClick={() => setMode("demo")}
            >
              Demo Showcase
            </button>
            <button
              className={`mode-tab ${mode === "live" ? "active" : ""}`}
              onClick={() => setMode("live")}
            >
              Live Camera
            </button>
            <button
              className={`mode-tab ${mode === "upload" ? "active" : ""}`}
              onClick={() => setMode("upload")}
            >
              Uploaded Video
            </button>
          </div>
        </div>
      </div>

      {/* Main Analysis Grid */}
      <div className="grid-2">
        {/* Left: Video Feed / Detection Canvas */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">
              🎥 Video Feed &amp; YOLOv8 Tracking Overlay
            </h3>
            <span className="badge-pill badge-low">
              {wsConnected ? "● 25.4 FPS (RTX 5060)" : "● Replaying Stream"}
            </span>
          </div>

          <div
            style={{
              position: "relative",
              background: "#050811",
              borderRadius: "8px",
              minHeight: "340px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              overflow: "hidden",
              border: "1px solid var(--border-subtle)",
            }}
          >
            {/* Simulated Live Stream Canvas */}
            <div style={{ textAlign: "center", padding: "2rem" }}>
              <div style={{ fontSize: "3rem", marginBottom: "0.5rem" }}>⚽</div>
              <div style={{ fontWeight: 600, color: "var(--accent-cyan)" }}>
                YOLOv8 + ByteTrack MOT Active
              </div>
              <div style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
                Tracking 22 Players + Ball · Kalman Occlusion Coasting · CIELAB Kit Tagging
              </div>
            </div>

            {/* Overlaid Live Stats Box */}
            <div
              style={{
                position: "absolute",
                bottom: "12px",
                left: "12px",
                right: "12px",
                background: "rgba(15, 23, 42, 0.85)",
                backdropFilter: "blur(6px)",
                padding: "0.75rem 1rem",
                borderRadius: "6px",
                border: "1px solid var(--border-subtle)",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem" }}>
                <span>Possession: Blue {possession["1"]}% / Red {possession["2"]}%</span>
                <span>Ball Speed: {ball?.speed_mps || 0} m/s</span>
                <span>Active Wearables: 2</span>
              </div>
              <div className="possession-bar">
                <div className="bar-blue" style={{ width: `${possession["1"]}%` }}></div>
                <div className="bar-red" style={{ width: `${possession["2"]}%` }}></div>
              </div>
            </div>
          </div>
        </div>

        {/* Right: 2D Top-Down Pitch Radar */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">
              🗺️ 2D Top-Down Tactical Pitch (Radar)
            </h3>
            <span className="badge-pill badge-wearable">Homography: 105m × 68m</span>
          </div>

          <TacticalPitch players={players} ball={ball} />

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(3, 1fr)",
              gap: "0.75rem",
              marginTop: "1rem",
            }}
          >
            <div className="stat-box">
              <div className="stat-label">Team 1 Formation</div>
              <div className="stat-val" style={{ fontSize: "1.2rem", color: "var(--team-blue)" }}>
                4-3-3
              </div>
            </div>
            <div className="stat-box">
              <div className="stat-label">Team 2 Formation</div>
              <div className="stat-val" style={{ fontSize: "1.2rem", color: "var(--team-red)" }}>
                4-4-2
              </div>
            </div>
            <div className="stat-box">
              <div className="stat-label">Pressing Index</div>
              <div className="stat-val" style={{ fontSize: "1.2rem", color: "var(--accent-amber)" }}>
                0.76 <span className="stat-unit">high</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Players Live Tracking Table */}
      <div className="card" style={{ marginTop: "1.25rem" }}>
        <div className="card-header">
          <h3 className="card-title">🏃 Live Player Workload &amp; Health</h3>
          <span className="badge-pill badge-low">Fusing Vision + ESP32 Telemetry</span>
        </div>

        <table className="data-table">
          <thead>
            <tr>
              <th>Jersey / Player</th>
              <th>Team</th>
              <th>Live Speed</th>
              <th>Distance</th>
              <th>HSR (&gt;5.5 m/s)</th>
              <th>Sprints</th>
              <th>Heart Rate</th>
              <th>SpO2</th>
              <th>PlayerLoad</th>
              <th>Injury Risk</th>
            </tr>
          </thead>
          <tbody>
            {players.map((p: any) => (
              <tr key={p.global_player_id}>
                <td>
                  <strong>#{p.jersey}</strong> {p.global_player_id}
                  {p.wearable && (
                    <span className="badge-pill badge-wearable" style={{ marginLeft: "6px" }}>
                      VEST
                    </span>
                  )}
                </td>
                <td>
                  <span style={{ color: p.team === 1 ? "var(--team-blue)" : "var(--team-red)" }}>
                    Team {p.team}
                  </span>
                </td>
                <td>{p.speed_mps || 0} m/s</td>
                <td>{p.distance_m || 0} m</td>
                <td>{p.hsr_m || 0} m</td>
                <td>{p.sprints || 0}</td>
                <td>{p.hr ? `${p.hr} BPM` : "—"}</td>
                <td>{p.spo2 ? `${p.spo2}%` : "—"}</td>
                <td>{p.player_load || 0}</td>
                <td>
                  <span
                    className={`badge-pill ${
                      p.injury_risk === "high"
                        ? "badge-high"
                        : p.injury_risk === "medium"
                        ? "badge-med"
                        : "badge-low"
                    }`}
                  >
                    {p.injury_risk || "low"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
