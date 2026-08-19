import React, { useState } from "react";

export const SettingsPage: React.FC = () => {
  const [relayUrl, setRelayUrl] = useState("http://127.0.0.1:8081");
  const [modelWeight, setModelWeight] = useState("soccernet_v2/best.pt");
  const [confThresh, setConfThresh] = useState(0.25);

  return (
    <div>
      <div className="card-header" style={{ marginBottom: "1.25rem" }}>
        <div>
          <h2 style={{ fontSize: "1.4rem", fontWeight: 700 }}>⚙️ System Settings &amp; Server Config</h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem", marginTop: "0.25rem" }}>
            Configure Hostinger KVM 2 VPS relay, local GPU acceleration, and YOLO detector parameters.
          </p>
        </div>
      </div>

      <div className="card" style={{ maxWidth: "680px" }}>
        <div style={{ marginBottom: "1.25rem" }}>
          <label style={{ display: "block", fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "0.35rem" }}>
            Hostinger KVM 2 Sensor Relay URL
          </label>
          <input
            type="text"
            value={relayUrl}
            onChange={(e) => setRelayUrl(e.target.value)}
            style={{
              width: "100%",
              padding: "0.6rem 0.75rem",
              borderRadius: "6px",
              background: "var(--bg-secondary)",
              border: "1px solid var(--border-subtle)",
              color: "var(--text-primary)",
            }}
          />
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
            VPS Ingestion Port: 8081 (HTTP/WS) / 9000 (ESP32 TCP)
          </div>
        </div>

        <div style={{ marginBottom: "1.25rem" }}>
          <label style={{ display: "block", fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "0.35rem" }}>
            YOLOv8 Detection Model Weights
          </label>
          <select
            value={modelWeight}
            onChange={(e) => setModelWeight(e.target.value)}
            style={{
              width: "100%",
              padding: "0.6rem 0.75rem",
              borderRadius: "6px",
              background: "var(--bg-secondary)",
              border: "1px solid var(--border-subtle)",
              color: "var(--text-primary)",
            }}
          >
            <option value="soccernet_v2/best.pt">runs/detect/soccernet_v2/best.pt (Fine-Tuned Base)</option>
            <option value="yolov8m.pt">yolov8m.pt (Standard Pretrained)</option>
            <option value="yolov8s.pt">yolov8s.pt (Fast Lightweight)</option>
          </select>
        </div>

        <div style={{ marginBottom: "1.5rem" }}>
          <label style={{ display: "block", fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "0.35rem" }}>
            Detection Confidence Threshold: {confThresh}
          </label>
          <input
            type="range"
            min="0.05"
            max="0.80"
            step="0.05"
            value={confThresh}
            onChange={(e) => setConfThresh(parseFloat(e.target.value))}
            style={{ width: "100%" }}
          />
        </div>

        <button className="btn btn-primary">Save Settings</button>
      </div>
    </div>
  );
};
