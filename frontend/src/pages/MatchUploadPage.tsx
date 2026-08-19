import React, { useState } from "react";

export const MatchUploadPage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [matchName, setMatchName] = useState("College Championship Match");
  const [mode, setMode] = useState("inference");
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [result, setResult] = useState<any>(null);

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setProgress(10);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("match_name", matchName);
    formData.append("mode", mode);

    try {
      const res = await fetch(`http://${window.location.hostname}:8000/api/v1/matches/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult(data);
      setProgress(100);
    } catch (e) {
      alert("Upload failed. Check backend connection.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div>
      <div className="card-header" style={{ marginBottom: "1.25rem" }}>
        <div>
          <h2 style={{ fontSize: "1.4rem", fontWeight: 700 }}>
            📤 Upload Match Video &amp; College Training Engine
          </h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem", marginTop: "0.25rem" }}>
            Upload recorded college or tournament matches for live YOLOv8 inference, 2D pitch projection, or fine-tuning.
          </p>
        </div>
      </div>

      <div className="grid-2">
        {/* Upload Form Card */}
        <div className="card">
          <h3 className="card-title" style={{ marginBottom: "1rem" }}>
            📁 Select Match Video File
          </h3>

          <div style={{ marginBottom: "1rem" }}>
            <label style={{ display: "block", fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "0.35rem" }}>
              Match Name
            </label>
            <input
              type="text"
              value={matchName}
              onChange={(e) => setMatchName(e.target.value)}
              style={{
                width: "100%",
                padding: "0.6rem 0.75rem",
                borderRadius: "6px",
                background: "var(--bg-secondary)",
                border: "1px solid var(--border-subtle)",
                color: "var(--text-primary)",
              }}
            />
          </div>

          <div style={{ marginBottom: "1rem" }}>
            <label style={{ display: "block", fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "0.35rem" }}>
              Processing Pipeline Mode
            </label>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value)}
              style={{
                width: "100%",
                padding: "0.6rem 0.75rem",
                borderRadius: "6px",
                background: "var(--bg-secondary)",
                border: "1px solid var(--border-subtle)",
                color: "var(--text-primary)",
              }}
            >
              <option value="inference">Live GPU Inference (YOLOv8 + ByteTrack + Radar)</option>
              <option value="train">Fine-Tune Detector on College Footage (2-Stage Augmentation)</option>
              <option value="batch">Offline Batch Analysis &amp; Annotated MP4 Export</option>
            </select>
          </div>

          <div
            style={{
              border: "2px dashed var(--border-subtle)",
              borderRadius: "8px",
              padding: "2rem",
              textAlign: "center",
              background: "var(--bg-secondary)",
              marginBottom: "1rem",
              cursor: "pointer",
            }}
            onClick={() => document.getElementById("file-input")?.click()}
          >
            <input
              id="file-input"
              type="file"
              accept="video/*"
              style={{ display: "none" }}
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
            <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>📹</div>
            <div style={{ fontWeight: 600 }}>{file ? file.name : "Click to select or drop video file (.mp4, .avi)"}</div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
              {file ? `${(file.size / (1024 * 1024)).toFixed(1)} MB` : "Supports 720p / 1080p university footage"}
            </div>
          </div>

          <button
            className="btn btn-primary"
            style={{ width: "100%", justifyContent: "center" }}
            disabled={!file || uploading}
            onClick={handleUpload}
          >
            {uploading ? "Processing Match..." : "Start Processing Pipeline"}
          </button>
        </div>

        {/* Calibration & Homography Tool Card */}
        <div className="card">
          <h3 className="card-title" style={{ marginBottom: "1rem" }}>
            📐 Interactive Pitch Calibration
          </h3>
          <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "1rem" }}>
            Anchor 4 pitch landmarks (touchline corners / center circle) to map camera pixels to the 105m × 68m standard pitch.
          </p>

          <div
            style={{
              background: "#050811",
              borderRadius: "6px",
              height: "220px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              border: "1px solid var(--border-subtle)",
            }}
          >
            <span style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
              Pitch Calibration Matrix Ready · 4 Landmark Anchors
            </span>
          </div>

          <div style={{ display: "flex", gap: "0.75rem", marginTop: "1rem" }}>
            <button className="btn btn-secondary" style={{ flex: 1, justifyContent: "center" }}>
              Auto-Detect Pitch Keypoints
            </button>
            <button className="btn btn-secondary" style={{ flex: 1, justifyContent: "center" }}>
              Load SNMOT-060 Preset
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
