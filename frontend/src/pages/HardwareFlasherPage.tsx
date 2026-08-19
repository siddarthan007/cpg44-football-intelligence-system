import React, { useEffect, useState } from "react";

export const HardwareFlasherPage: React.FC = () => {
  const [ports, setPorts] = useState<any[]>([]);
  const [selectedPort, setSelectedPort] = useState("/dev/ttyUSB0");
  const [playerId, setPlayerId] = useState(27);
  const [wifiSsid, setWifiSsid] = useState("Field_Hotspot");
  const [wifiPass, setWifiPass] = useState("FieldPass123");
  const [endpoint, setEndpoint] = useState(`http://${window.location.hostname}:8000/ingest`);
  const [flashStatus, setFlashStatus] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [chipInfo, setChipInfo] = useState<any>(null);

  const fetchPorts = () => {
    fetch(`http://${window.location.hostname}:8000/api/v1/hardware/ports`)
      .then((r) => r.json())
      .then((d) => {
        setPorts(d);
        if (d.length > 0 && !selectedPort) setSelectedPort(d[0].device);
      })
      .catch(() => {});
  };

  const fetchChip = () => {
    fetch(`http://${window.location.hostname}:8000/api/v1/hardware/chip-info?port=${selectedPort}`)
      .then((r) => r.json())
      .then((d) => setChipInfo(d))
      .catch(() => {});
  };

  const fetchStatus = () => {
    fetch(`http://${window.location.hostname}:8000/api/v1/hardware/flash/status`)
      .then((r) => r.json())
      .then((d) => setFlashStatus(d))
      .catch(() => {});
  };

  useEffect(() => {
    fetchPorts();
    fetchChip();
    fetchStatus();
    const interval = setInterval(fetchStatus, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleFlash = async () => {
    setLoading(true);
    await fetch(`http://${window.location.hostname}:8000/api/v1/hardware/flash`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        port: selectedPort,
        player_id: playerId,
        wifi_ssid: wifiSsid,
        wifi_pass: wifiPass,
        endpoint,
      }),
    });
    setLoading(false);
    fetchStatus();
  };

  const activeJob = flashStatus?.active_job;
  const isFlashing = activeJob && activeJob.status === "flashing";

  return (
    <div style={{ padding: "1.5rem", maxWidth: "1100px", margin: "0 auto" }}>
      <div style={{ marginBottom: "1.5rem" }}>
        <h2 style={{ fontSize: "1.3rem", fontWeight: 700, color: "var(--text-main)" }}>
          ESP32 Hardware Provisioning &amp; Serial Flasher
        </h2>
        <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginTop: "0.25rem" }}>
          Flash and configure soccer wearable vests (ESP32-S3, MPU6050, MAX30102, NEO-6M) directly from the dashboard.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1.1fr 1fr", gap: "1.5rem" }}>
        {/* Left: Configuration & Flash Options */}
        <div className="card-clean">
          <h3 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "1rem" }}>
            Device Configuration &amp; Port Selection
          </h3>

          <div style={{ marginBottom: "1rem" }}>
            <label style={{ display: "block", fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "0.25rem" }}>
              Target Serial Port
            </label>
            <div style={{ display: "flex", gap: "0.5rem" }}>
              <select
                className="select-input"
                style={{ flex: 1 }}
                value={selectedPort}
                onChange={(e) => {
                  setSelectedPort(e.target.value);
                  fetchChip();
                }}
              >
                {ports.map((p) => (
                  <option key={p.device} value={p.device}>
                    {p.name} — {p.description}
                  </option>
                ))}
              </select>
              <button className="btn-outline" onClick={fetchPorts}>Rescan</button>
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginBottom: "1rem" }}>
            <div>
              <label style={{ display: "block", fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "0.25rem" }}>
                Player Jersey # (Device ID)
              </label>
              <input
                type="number"
                className="select-input"
                style={{ width: "100%" }}
                value={playerId}
                onChange={(e) => setPlayerId(Number(e.target.value))}
              />
            </div>

            <div>
              <label style={{ display: "block", fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "0.25rem" }}>
                Wi-Fi SSID
              </label>
              <input
                type="text"
                className="select-input"
                style={{ width: "100%" }}
                value={wifiSsid}
                onChange={(e) => setWifiSsid(e.target.value)}
              />
            </div>

            <div>
              <label style={{ display: "block", fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "0.25rem" }}>
                Wi-Fi Password
              </label>
              <input
                type="password"
                className="select-input"
                style={{ width: "100%" }}
                value={wifiPass}
                onChange={(e) => setWifiPass(e.target.value)}
              />
            </div>

            <div>
              <label style={{ display: "block", fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "0.25rem" }}>
                Telemetry Ingest URL
              </label>
              <input
                type="text"
                className="select-input"
                style={{ width: "100%" }}
                value={endpoint}
                onChange={(e) => setEndpoint(e.target.value)}
              />
            </div>
          </div>

          <button
            className="btn-solid"
            style={{ width: "100%", marginTop: "0.5rem" }}
            disabled={isFlashing || loading}
            onClick={handleFlash}
          >
            {isFlashing ? "Flashing ESP32 Binary..." : `Flash Firmware to Player #${playerId} Vest`}
          </button>
        </div>

        {/* Right: Chip Info & Live Serial Monitor Logs */}
        <div className="card-clean" style={{ display: "flex", flexDirection: "column" }}>
          <h3 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "0.75rem" }}>
            Hardware Diagnostics &amp; Flash Logs
          </h3>

          {chipInfo && (
            <div style={{ background: "var(--bg-subtle)", padding: "0.75rem", borderRadius: "6px", marginBottom: "1rem", fontSize: "0.8rem" }}>
              <div><strong>Target Chip:</strong> {chipInfo.chip}</div>
              <div><strong>MAC Address:</strong> {chipInfo.mac}</div>
              <div><strong>Flash Memory:</strong> {chipInfo.flash_size}</div>
            </div>
          )}

          {isFlashing && (
            <div style={{ marginBottom: "1rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", fontWeight: 600, marginBottom: "0.35rem" }}>
                <span>Writing Firmware</span>
                <span style={{ color: "var(--accent-blue)" }}>{activeJob?.progress_pct}%</span>
              </div>
              <div className="scrubber-track">
                <div className="scrubber-fill" style={{ width: `${activeJob?.progress_pct}%` }}></div>
              </div>
            </div>
          )}

          <div
            style={{
              flex: 1,
              minHeight: "180px",
              background: "#0f172a",
              color: "#e2e8f0",
              fontFamily: "monospace",
              fontSize: "0.75rem",
              padding: "0.75rem",
              borderRadius: "6px",
              overflowY: "auto",
            }}
          >
            <div style={{ color: "#38bdf8" }}>-- CPG44 ESP32 Provisioning Console --</div>
            {activeJob?.logs ? (
              activeJob.logs.map((log: string, i: number) => <div key={i}>&gt; {log}</div>)
            ) : (
              <div style={{ color: "#64748b", marginTop: "0.5rem" }}>Ready to flash. Select serial port and click flash.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
