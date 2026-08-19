import React, { useState } from "react";

export const SettingsPage: React.FC = () => {
  const [relayUrl, setRelayUrl] = useState("http://127.0.0.1:8081");
  const [modelWeight, setModelWeight] = useState("soccernet_v2/best.pt");

  return (
    <div style={{ padding: "1.5rem", maxWidth: "800px", margin: "0 auto" }}>
      <div style={{ marginBottom: "1.5rem" }}>
        <h2 style={{ fontSize: "1.3rem", fontWeight: 700, color: "var(--text-main)" }}>System Configuration</h2>
        <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginTop: "0.25rem" }}>
          Configure Hostinger VPS relay endpoint, local sensor hub, and GPU inference.
        </p>
      </div>

      <div className="card-clean">
        <div style={{ marginBottom: "1.25rem" }}>
          <label style={{ display: "block", fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "0.25rem" }}>
            Hostinger KVM 2 Sensor Relay URL
          </label>
          <input
            type="text"
            className="select-input"
            style={{ width: "100%" }}
            value={relayUrl}
            onChange={(e) => setRelayUrl(e.target.value)}
          />
        </div>

        <div style={{ marginBottom: "1.5rem" }}>
          <label style={{ display: "block", fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "0.25rem" }}>
            Active Detection Model
          </label>
          <select
            className="select-input"
            style={{ width: "100%" }}
            value={modelWeight}
            onChange={(e) => setModelWeight(e.target.value)}
          >
            <option value="soccernet_v2/best.pt">runs/detect/soccernet_v2/best.pt (SoccerNet Base)</option>
            <option value="yolov8m.pt">yolov8m.pt (Standard Pretrained)</option>
          </select>
        </div>

        <button className="btn-solid">Save Configuration</button>
      </div>
    </div>
  );
};
