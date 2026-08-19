import React from "react";

export const AnalyticsPage: React.FC = () => {
  return (
    <div>
      <div className="card-header" style={{ marginBottom: "1.25rem" }}>
        <div>
          <h2 style={{ fontSize: "1.4rem", fontWeight: 700 }}>
            📊 Post-Match &amp; Tactical Deep-Dive Analytics
          </h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem", marginTop: "0.25rem" }}>
            Spatial heatmaps, passing networks, xG shot maps, and Catapult speed zone distributions.
          </p>
        </div>
        <button className="btn btn-secondary">📥 Export JSON / CSV Report</button>
      </div>

      <div className="grid-2">
        <div className="card">
          <h3 className="card-title" style={{ marginBottom: "1rem" }}>
            🔥 2D Spatial Density Heatmap
          </h3>
          <div
            style={{
              height: "240px",
              background: "#050811",
              borderRadius: "6px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              border: "1px solid var(--border-subtle)",
            }}
          >
            <span style={{ color: "var(--accent-cyan)", fontSize: "0.9rem" }}>
              Gaussian Kernel Density Pitch Map · 105m × 68m
            </span>
          </div>
        </div>

        <div className="card">
          <h3 className="card-title" style={{ marginBottom: "1rem" }}>
            🎯 Expected Goals (xG) &amp; Shot Locations
          </h3>
          <div
            style={{
              height: "240px",
              background: "#050811",
              borderRadius: "6px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              border: "1px solid var(--border-subtle)",
            }}
          >
            <span style={{ color: "var(--accent-rose)", fontSize: "0.9rem" }}>
              Team 1 xG: 1.42 (8 shots) · Team 2 xG: 0.88 (5 shots)
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
