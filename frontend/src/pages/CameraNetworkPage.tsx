import React, { useEffect, useState } from "react";

export const CameraNetworkPage: React.FC = () => {
  const [cameras, setCameras] = useState<any[]>([]);

  useEffect(() => {
    fetch(`http://${window.location.hostname}:8000/api/v1/cameras`)
      .then((res) => res.json())
      .then((data) => setCameras(data))
      .catch(() => {});
  }, []);

  return (
    <div>
      <div className="card-header" style={{ marginBottom: "1.25rem" }}>
        <div>
          <h2 style={{ fontSize: "1.4rem", fontWeight: 700 }}>
            📡 Multi-Camera Network Gateway &amp; Smartphone Streaming
          </h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem", marginTop: "0.25rem" }}>
            Connect RTSP cameras, IP streams, and mobile phone cameras via WebRTC for multi-view match tracking.
          </p>
        </div>
        <button className="btn btn-primary">+ Register Camera</button>
      </div>

      <div className="grid-2">
        {/* Active Cameras List */}
        <div className="card">
          <h3 className="card-title" style={{ marginBottom: "1rem" }}>
            🎥 Connected Video Sources
          </h3>

          <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
            {cameras.map((cam) => (
              <div
                key={cam.id}
                style={{
                  background: "var(--bg-secondary)",
                  padding: "1rem",
                  borderRadius: "8px",
                  border: "1px solid var(--border-subtle)",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <div>
                  <div style={{ fontWeight: 600 }}>{cam.name}</div>
                  <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                    {cam.source} · {cam.type}
                  </div>
                </div>
                <div style={{ textAlign: "right" }}>
                  <span className="badge-pill badge-low">{cam.status.toUpperCase()}</span>
                  <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: "0.25rem" }}>
                    {cam.fps} FPS · {cam.latency_ms} ms
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Smartphone Camera Stream QR Ingest */}
        <div className="card">
          <h3 className="card-title" style={{ marginBottom: "1rem" }}>
            📱 Smartphone Camera Quick Stream
          </h3>
          <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "1rem" }}>
            Turn any mobile phone into a sideline camera. Open the URL below on the smartphone browser to stream video directly into the CPG44 Gateway.
          </p>

          <div
            style={{
              background: "var(--bg-secondary)",
              padding: "1.25rem",
              borderRadius: "8px",
              border: "1px solid var(--border-subtle)",
              textAlign: "center",
            }}
          >
            <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--accent-cyan)", marginBottom: "0.5rem" }}>
              http://{window.location.hostname}:8000/camera
            </div>
            <p style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
              Direct browser-to-gateway WebRTC / MJPEG streaming. No mobile app installation required.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
