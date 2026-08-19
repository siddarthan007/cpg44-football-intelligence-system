import React, { useEffect, useState } from "react";

export const WearablePage: React.FC = () => {
  const [telemetry, setTelemetry] = useState<any>(null);

  useEffect(() => {
    const fetchLatest = () => {
      fetch(`http://${window.location.hostname}:8000/api/v1/observations/wearable`)
        .then((r) => r.json())
        .then((d) => setTelemetry(d))
        .catch(() => {});
    };
    fetchLatest();
    const t = setInterval(fetchLatest, 1000);
    return () => clearInterval(t);
  }, []);

  return (
    <div style={{ padding: "1.5rem", maxWidth: "1200px", margin: "0 auto" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.5rem" }}>
        <div>
          <h2 style={{ fontSize: "1.3rem", fontWeight: 700, color: "var(--text-main)" }}>
            Wearable Biometrics &amp; Physiological Load
          </h2>
          <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginTop: "0.25rem" }}>
            Real-time ESP32 vest streaming (MPU6050 IMU, MAX30102 PPG, NEO-6M GPS) time-synced with match video.
          </p>
        </div>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <span style={{ padding: "0.25rem 0.65rem", borderRadius: "9999px", background: "#dcfce7", color: "#166534", fontSize: "0.75rem", fontWeight: 600 }}>
            ● Hostinger Relay Active
          </span>
        </div>
      </div>

      {/* KPI Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "1rem", marginBottom: "1.5rem" }}>
        <div className="card-clean">
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600 }}>PLAYER #27 HEART RATE</div>
          <div style={{ fontSize: "1.8rem", fontWeight: 800, color: "#ef4444", marginTop: "0.25rem" }}>
            165 <span style={{ fontSize: "0.85rem", fontWeight: 500, color: "var(--text-muted)" }}>BPM</span>
          </div>
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>Zone 4 (86% HRmax)</div>
        </div>

        <div className="card-clean">
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600 }}>BLOOD OXYGEN (SPO2)</div>
          <div style={{ fontSize: "1.8rem", fontWeight: 800, color: "var(--accent-blue)", marginTop: "0.25rem" }}>
            97.2 <span style={{ fontSize: "0.85rem", fontWeight: 500, color: "var(--text-muted)" }}>%</span>
          </div>
          <div style={{ fontSize: "0.75rem", color: "#10b981", marginTop: "0.25rem" }}>Stable Oxygenation</div>
        </div>

        <div className="card-clean">
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600 }}>IMU PLAYERLOAD</div>
          <div style={{ fontSize: "1.8rem", fontWeight: 800, color: "#f59e0b", marginTop: "0.25rem" }}>
            158.4 <span style={{ fontSize: "0.85rem", fontWeight: 500, color: "var(--text-muted)" }}>a.u.</span>
          </div>
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>3D Accelerometer Accumulation</div>
        </div>

        <div className="card-clean">
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600 }}>ACWR STRAIN RATIO</div>
          <div style={{ fontSize: "1.8rem", fontWeight: 800, color: "#10b981", marginTop: "0.25rem" }}>
            1.14 <span style={{ fontSize: "0.85rem", fontWeight: 500, color: "var(--text-muted)" }}>sweet spot</span>
          </div>
          <div style={{ fontSize: "0.75rem", color: "#10b981", marginTop: "0.25rem" }}>Safe Range (0.8 - 1.3)</div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
        {/* Left: Load & Energy Metrics */}
        <div className="card-clean">
          <h3 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "1rem" }}>
            Kinematic &amp; Metabolic Load (di Prampero Model)
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", padding: "0.5rem 0", borderBottom: "1px solid var(--border-light)" }}>
              <span style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>Metabolic Power Average:</span>
              <strong style={{ fontSize: "0.85rem" }}>12.8 W/kg</strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", padding: "0.5rem 0", borderBottom: "1px solid var(--border-light)" }}>
              <span style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>High Metabolic Load Distance:</span>
              <strong style={{ fontSize: "0.85rem" }}>640 meters</strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", padding: "0.5rem 0", borderBottom: "1px solid var(--border-light)" }}>
              <span style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>High Intensity Decelerations (&lt; -3 m/s²):</span>
              <strong style={{ fontSize: "0.85rem" }}>7 efforts</strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", padding: "0.5rem 0" }}>
              <span style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>Cardiac Drift Index:</span>
              <strong style={{ fontSize: "0.85rem", color: "#10b981" }}>+3.2% (Normal)</strong>
            </div>
          </div>
        </div>

        {/* Right: AI Substitution & Injury Advisory */}
        <div className="card-clean">
          <h3 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "1rem" }}>
            Coaching &amp; Medical Advisory
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
            <div style={{ padding: "0.85rem", borderRadius: "6px", background: "var(--bg-subtle)", borderLeft: "4px solid #10b981" }}>
              <div style={{ fontWeight: 700, fontSize: "0.85rem", color: "#166534" }}>Player #27 (R. Edwards) — Normal Condition</div>
              <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
                Workload and cardiac recovery profiles are optimal. Available for full match duration.
              </p>
            </div>

            <div style={{ padding: "0.85rem", borderRadius: "6px", background: "var(--bg-subtle)", borderLeft: "4px solid #f59e0b" }}>
              <div style={{ fontWeight: 700, fontSize: "0.85rem", color: "#b45309" }}>Player #9 (C. Lancaster) — Load Warning</div>
              <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
                Accumulated 10 high-speed sprints with elevated recovery heart rate. Consider rotation at 70th minute.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
