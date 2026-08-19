import React, { useEffect, useState } from "react";

export const PlayerTaggingPage: React.FC = () => {
  const [teams, setTeams] = useState<any>(null);
  const [events, setEvents] = useState<any[]>([]);
  const [newEventType, setNewEventType] = useState("shot");
  const [newEventPlayer, setNewEventPlayer] = useState(17);
  const [newEventTeam, setNewEventTeam] = useState(1);
  const [newEventDesc, setNewEventDesc] = useState("Shot on Target");

  const fetchTeams = () => {
    fetch(`http://${window.location.hostname}:8000/api/v1/tagging/teams`)
      .then((r) => r.json())
      .then((d) => setTeams(d))
      .catch(() => {});
  };

  const fetchEvents = () => {
    fetch(`http://${window.location.hostname}:8000/api/v1/tagging/events`)
      .then((r) => r.json())
      .then((d) => setEvents(d))
      .catch(() => {});
  };

  useEffect(() => {
    fetchTeams();
    fetchEvents();
  }, []);

  const handleAddEvent = async () => {
    await fetch(`http://${window.location.hostname}:8000/api/v1/tagging/events`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        type: newEventType,
        team: newEventTeam,
        player_jersey: newEventPlayer,
        time_str: "41:32",
        timestamp_s: 2492.0,
        description: newEventDesc,
      }),
    });
    fetchEvents();
  };

  return (
    <div style={{ padding: "1.5rem", maxWidth: "1150px", margin: "0 auto" }}>
      <div style={{ marginBottom: "1.5rem" }}>
        <h2 style={{ fontSize: "1.3rem", fontWeight: 700, color: "var(--text-main)" }}>
          Match Event Tagging &amp; Team Kit Profiling
        </h2>
        <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginTop: "0.25rem" }}>
          Configure CIELAB jersey color clusters, manage live tactical tags, and review gameplay events.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1.1fr 1fr", gap: "1.5rem" }}>
        {/* Left: Team Kit Color Calibration */}
        <div className="card-clean">
          <h3 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "1rem" }}>
            Team Kit CIELAB Calibration
          </h3>

          <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
            {teams && (
              <>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0.75rem", background: "var(--bg-subtle)", borderRadius: "6px" }}>
                  <div>
                    <div style={{ fontWeight: 700, fontSize: "0.9rem" }}>{teams.team_1?.name || "Team 1"}</div>
                    <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Primary Kit: Royal Blue (RGB: 37, 99, 235)</div>
                  </div>
                  <div style={{ width: "32px", height: "32px", borderRadius: "6px", background: "#2563eb", border: "2px solid #fff" }}></div>
                </div>

                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0.75rem", background: "var(--bg-subtle)", borderRadius: "6px" }}>
                  <div>
                    <div style={{ fontWeight: 700, fontSize: "0.9rem" }}>{teams.team_2?.name || "Team 2"}</div>
                    <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Primary Kit: Crimson (RGB: 127, 29, 29)</div>
                  </div>
                  <div style={{ width: "32px", height: "32px", borderRadius: "6px", background: "#7f1d1d", border: "2px solid #fff" }}></div>
                </div>
              </>
            )}

            <div style={{ padding: "0.75rem", borderTop: "1px solid var(--border-light)", marginTop: "0.5rem" }}>
              <div style={{ fontSize: "0.85rem", fontWeight: 700, marginBottom: "0.5rem" }}>Log Real-time Event</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem", marginBottom: "0.5rem" }}>
                <select className="select-input" value={newEventType} onChange={(e) => setNewEventType(e.target.value)}>
                  <option value="shot">Shot</option>
                  <option value="pass">Pass</option>
                  <option value="tackle">Tackle / Interception</option>
                  <option value="goal">Goal</option>
                  <option value="sprint">HSR Sprint</option>
                  <option value="fatigue">Fatigue Spike</option>
                </select>

                <input
                  type="number"
                  className="select-input"
                  placeholder="Jersey #"
                  value={newEventPlayer}
                  onChange={(e) => setNewEventPlayer(Number(e.target.value))}
                />
              </div>

              <input
                type="text"
                className="select-input"
                style={{ width: "100%", marginBottom: "0.5rem" }}
                value={newEventDesc}
                onChange={(e) => setNewEventDesc(e.target.value)}
              />

              <button className="btn-solid" style={{ width: "100%" }} onClick={handleAddEvent}>
                + Log Match Event Tag
              </button>
            </div>
          </div>
        </div>

        {/* Right: Tagged Event History */}
        <div className="card-clean">
          <h3 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "1rem" }}>
            Match Timeline Events ({events.length})
          </h3>

          <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", maxHeight: "420px", overflowY: "auto" }}>
            {events.map((ev) => (
              <div
                key={ev.id}
                style={{
                  padding: "0.65rem 0.85rem",
                  borderRadius: "6px",
                  background: "var(--bg-subtle)",
                  border: "1px solid var(--border-light)",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <div>
                  <div style={{ fontWeight: 600, fontSize: "0.85rem" }}>{ev.description}</div>
                  <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                    {ev.time_str} · Player #{ev.player_jersey} ({ev.team === 1 ? "Team 1" : "Team 2"})
                  </div>
                </div>
                <span className={`tag-pill ${ev.type === "goal" ? "primary" : ""}`}>
                  {ev.type.toUpperCase()}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
