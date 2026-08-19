import React, { useEffect, useState } from "react";

export const WearablePage: React.FC = () => {
  const [telemetry, setTelemetry] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);

  useEffect(() => {
    const fetchLatest = () => {
      fetch(`http://${window.location.hostname}:8000/api/v1/observations/wearable`)
        .then((res) => res.json())
        .then((data) => {
          if (data && typeof data === "object") {
            setTelemetry(data);
          }
        })
        .catch(() => {});
    };

    fetchLatest();
    const timer = setInterval(fetchLatest, 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div>
      <div className="card-header" style={{ marginBottom: "1.25rem" }}>
        <div>
          <h2 style={{ fontSize: "1.4rem", fontWeight: 700 }}>
            🫀 Wearable Biometrics &amp; Injury Prevention Engine
          </h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem", marginTop: "0.25rem" }}>
            Real-time ESP32 Telemetry (MPU6050 IMU, MAX30102 PPG, NEO-6M GPS) fused with Computer Vision Load.
          </p>
        </div>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <span className="badge-pill badge-low">● Hostinger Relay Synced</span>
          <span className="badge-pill badge-wearable">TCP Port 9000</span>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid-4" style={{ marginBottom: "1.25rem" }}>
        <div className="stat-box">
          <div className="stat-label">Player #7 Heart Rate</div>
          <div className="stat-val" style={{ color: "var(--accent-rose)" }}>
            168 <span className="stat-unit">BPM</span>
          </div>
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
            Max: 188 BPM · Zone 4 (88% HRmax)
          </div>
        </div>

        <div className="stat-box">
          <div className="stat-label">Blood Oxygen (SpO2)</div>
          <div className="stat-val" style={{ color: "var(--accent-cyan)" }}>
            97.4 <span className="stat-unit">%</span>
          </div>
          <div style={{ fontSize: "0.75rem", color: "var(--accent-emerald)", marginTop: "0.25rem" }}>
            ✓ Stable Oxygenation
          </div>
        </div>

        <div className="stat-box">
          <div className="stat-label">IMU PlayerLoad</div>
          <div className="stat-val" style={{ color: "var(--accent-amber)" }}>
            184.2 <span className="stat-unit">a.u.</span>
          </div>
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
            3D Accelerometer Accumulation
          </div>
        </div>

        <div className="stat-box">
          <div className="stat-label">ACWR Injury Risk</div>
          <div className="stat-val" style={{ color: "var(--accent-emerald)" }}>
            1.12 <span className="stat-unit">sweet spot</span>
          </div>
          <div style={{ fontSize: "0.75rem", color: "var(--accent-emerald)", marginTop: "0.25rem" }}>
            Safe Range: 0.8 - 1.3
          </div>
        </div>
      </div>

      <div className="grid-2">
        {/* Live PPG & Cardiac Waveform */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">📈 Real-time PPG Heart Rate &amp; Cardiac Drift</h3>
            <span className="badge-pill badge-low">Maxim Ref Algorithm</span>
          </div>
          <div
            style={{
              height: "220px",
              background: "var(--bg-secondary)",
              borderRadius: "6px",
              border: "1px solid var(--border-subtle)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexDirection: "column",
              gap: "0.5rem",
            }}
          >
            <div style={{ color: "var(--accent-rose)", fontSize: "1.8rem", fontWeight: 700 }}>
              ❤️ ~168 BPM (Active Sprint)
            </div>
            <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
              Motion Artifact Filter: Optical Noise Rejected During &gt;3.5G Impacts
            </div>
          </div>
        </div>

        {/* Metabolic Power vs Cardiac Cost */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">⚡ di Prampero Metabolic Power vs Internal Load</h3>
            <span className="badge-pill badge-wearable">Vision + Sensor Fusion</span>
          </div>
          <div
            style={{
              height: "220px",
              background: "var(--bg-secondary)",
              borderRadius: "6px",
              border: "1px solid var(--border-subtle)",
              padding: "1rem",
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-around",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span>Metabolic Power (Vision):</span>
              <strong style={{ color: "var(--accent-cyan)" }}>12.4 W/kg</strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span>Equivalent Flat Speed:</span>
              <strong>4.2 m/s</strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span>High Intensity Decelerations:</span>
              <strong style={{ color: "var(--accent-amber)" }}>6 reps (&lt; -3.0 m/s²)</strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span>Predicted Fatigue Level:</span>
              <strong style={{ color: "var(--accent-emerald)" }}>Low (28%)</strong>
            </div>
          </div>
        </div>
      </div>

      {/* AI Injury Risk & Tactical Substitution Advisor */}
      <div className="card" style={{ marginTop: "1.25rem" }}>
        <div className="card-header">
          <h3 className="card-title">🤖 AI Injury Risk &amp; Substitution Advisor</h3>
          <span className="badge-pill badge-low">XGBoost Multimodal Predictor</span>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
          <div style={{ background: "var(--bg-secondary)", padding: "1rem", borderRadius: "8px" }}>
            <h4 style={{ color: "var(--accent-cyan)", marginBottom: "0.5rem" }}>
              Player #7 (A. Silva) — Recommended Action
            </h4>
            <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)" }}>
              Workload is within optimal ACWR sweet spot (1.12). Cardiac response matches metabolic demands with no significant cardiac drift. Player is fit for full 90-minute load.
            </p>
          </div>

          <div style={{ background: "var(--bg-secondary)", padding: "1rem", borderRadius: "8px" }}>
            <h4 style={{ color: "var(--accent-amber)", marginBottom: "0.5rem" }}>
              Player #9 (E. Haaland) — Fatigue Advisory
            </h4>
            <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)" }}>
              Elevated heart rate (178 BPM) observed during recovery phases after sprint #11. Recommend reducing high pressing intensity or scheduling substitution at 70th minute.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
