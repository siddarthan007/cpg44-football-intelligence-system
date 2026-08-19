import React, { useState } from "react";

export const MatchUploadPage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [matchName, setMatchName] = useState("College Championship Match");
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setProgress(15);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("match_name", matchName);

    try {
      await fetch(`http://${window.location.hostname}:8000/api/v1/matches/upload`, {
        method: "POST",
        body: formData,
      });
      setProgress(100);
    } catch (e) {
      alert("Upload failed. Ensure backend is running.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={{ padding: "1.5rem", maxWidth: "1100px", margin: "0 auto" }}>
      <div style={{ marginBottom: "1.5rem" }}>
        <h2 style={{ fontSize: "1.3rem", fontWeight: 700, color: "var(--text-main)" }}>
          Video Ingest &amp; Pitch Calibration
        </h2>
        <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginTop: "0.25rem" }}>
          Upload match footage, calibrate camera landmarks, and run GPU inference without terminal commands.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: "1.5rem" }}>
        <div className="card-clean">
          <h3 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "1rem" }}>Upload Video Clip</h3>

          <div style={{ marginBottom: "1rem" }}>
            <label style={{ display: "block", fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "0.25rem" }}>
              Match Name
            </label>
            <input
              type="text"
              className="select-input"
              style={{ width: "100%" }}
              value={matchName}
              onChange={(e) => setMatchName(e.target.value)}
            />
          </div>

          <div
            style={{
              border: "2px dashed var(--border-light)",
              borderRadius: "6px",
              padding: "2rem",
              textAlign: "center",
              background: "var(--bg-subtle)",
              marginBottom: "1.25rem",
              cursor: "pointer",
            }}
            onClick={() => document.getElementById("vid-upload")?.click()}
          >
            <input
              id="vid-upload"
              type="file"
              accept="video/*"
              style={{ display: "none" }}
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
            <div style={{ fontSize: "1.8rem", color: "var(--accent-blue)", marginBottom: "0.5rem" }}>📁</div>
            <div style={{ fontWeight: 600, fontSize: "0.9rem" }}>
              {file ? file.name : "Select or drag match video (.mp4, .mov)"}
            </div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
              {file ? `${(file.size / (1024 * 1024)).toFixed(1)} MB` : "720p / 1080p footage"}
            </div>
          </div>

          <button className="btn-solid" style={{ width: "100%" }} disabled={!file || uploading} onClick={handleUpload}>
            {uploading ? "Processing Match Video..." : "Start Video Processing"}
          </button>
        </div>

        <div className="card-clean">
          <h3 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "0.5rem" }}>Pitch Calibration</h3>
          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "1rem" }}>
            4 anchor points mapping pixels to standard 105m × 68m coordinates.
          </p>

          <div style={{ height: "200px", background: "var(--bg-subtle)", border: "1px solid var(--border-light)", borderRadius: "6px", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)", fontSize: "0.85rem" }}>
            Homography Matrix: Calibrated (SNMOT-060)
          </div>

          <div style={{ display: "flex", gap: "0.75rem", marginTop: "1rem" }}>
            <button className="btn-outline" style={{ flex: 1 }}>Auto-Detect</button>
            <button className="btn-outline" style={{ flex: 1 }}>Reset Anchors</button>
          </div>
        </div>
      </div>
    </div>
  );
};
