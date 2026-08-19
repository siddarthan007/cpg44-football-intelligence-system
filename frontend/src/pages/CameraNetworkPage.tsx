import React, { useEffect, useState } from "react";

export const CameraNetworkPage: React.FC = () => {
  const [cameras, setCameras] = useState<any[]>([]);

  useEffect(() => {
    fetch(`http://${window.location.hostname}:8000/api/v1/cameras`)
      .then((r) => r.json())
      .then((d) => setCameras(d))
      .catch(() => {});
  }, []);

  return (
    <div style={{ padding: "1.5rem", maxWidth: "1100px", margin: "0 auto" }}>
      <div style={{ marginBottom: "1.5rem" }}>
        <h2 style={{ fontSize: "1.3rem", fontWeight: 700, color: "var(--text-main)" }}>
          Camera Network &amp; Mobile Streaming
        </h2>
        <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginTop: "0.25rem" }}>
          Single and multi-camera RTSP/WebRTC feeds and instant smartphone camera ingest.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: "1.5rem" }}>
        <div className="card-clean">
          <h3 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "1rem" }}>Connected Cameras</h3>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
            {cameras.map((c) => (
              <div
                key={c.id}
                style={{
                  padding: "0.85rem",
                  borderRadius: "6px",
                  background: "var(--bg-subtle)",
                  border: "1px solid var(--border-light)",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <div>
                  <div style={{ fontWeight: 600, fontSize: "0.9rem" }}>{c.name}</div>
                  <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>{c.type} · {c.source}</div>
                </div>
                <div style={{ textAlign: "right" }}>
                  <span style={{ fontSize: "0.75rem", fontWeight: 600, color: "#10b981" }}>ONLINE</span>
                  <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>{c.fps} FPS · {c.latency_ms} ms</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="card-clean">
          <h3 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "0.5rem" }}>Smartphone Camera Ingest</h3>
          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "1rem" }}>
            Open this URL on any phone browser to stream live video directly to the gateway.
          </p>
          <div style={{ background: "var(--bg-subtle)", padding: "1rem", borderRadius: "6px", textAlign: "center", border: "1px solid var(--border-light)" }}>
            <div style={{ color: "var(--accent-blue)", fontWeight: 700, fontSize: "0.95rem" }}>
              http://{window.location.hostname}:8000/camera
            </div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
              Direct browser WebRTC/MJPEG streaming
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
