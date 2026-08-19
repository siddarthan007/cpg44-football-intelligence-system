import React, { useEffect, useState } from "react";

export const TrainingHubPage: React.FC = () => {
  const [status, setStatus] = useState<any>(null);
  const [epochs, setEpochs] = useState(50);
  const [batch, setBatch] = useState(4);
  const [imgsz, setImgsz] = useState(1280);
  const [model, setModel] = useState("yolov8m.pt");
  const [loading, setLoading] = useState(false);

  const fetchStatus = () => {
    fetch(`http://${window.location.hostname}:8000/api/v1/training/status`)
      .then((r) => r.json())
      .then((d) => setStatus(d))
      .catch(() => {});
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 1500);
    return () => clearInterval(interval);
  }, []);

  const startYolo = async () => {
    setLoading(true);
    await fetch(`http://${window.location.hostname}:8000/api/v1/training/start-yolo`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        data_path: "/home/siddartha/SoccerNet_YOLO/data.yaml",
        model_name: model,
        epochs,
        imgsz,
        batch,
      }),
    });
    setLoading(false);
    fetchStatus();
  };

  const stopYolo = async () => {
    await fetch(`http://${window.location.hostname}:8000/api/v1/training/stop-yolo`, {
      method: "POST",
    });
    fetchStatus();
  };

  const trainStrain = async () => {
    setLoading(true);
    await fetch(`http://${window.location.hostname}:8000/api/v1/training/train-strain-model`, {
      method: "POST",
    });
    setLoading(false);
    fetchStatus();
  };

  const activeJob = status?.active_job;
  const isRunning = activeJob && activeJob.status === "running";

  return (
    <div style={{ padding: "1.5rem", maxWidth: "1200px", margin: "0 auto" }}>
      <div style={{ marginBottom: "1.5rem" }}>
        <h2 style={{ fontSize: "1.3rem", fontWeight: 700, color: "var(--text-main)" }}>
          Model Training &amp; Physiological Strain Intelligence
        </h2>
        <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginTop: "0.25rem" }}>
          Train YOLOv8 on the SoccerNet dataset and retrain XGBoost/RandomForest strain prediction models.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: "1.5rem" }}>
        {/* Left: YOLO Training Card */}
        <div className="card-clean">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
            <h3 style={{ fontSize: "1rem", fontWeight: 700 }}>YOLOv8 Detection Training</h3>
            <span style={{ fontSize: "0.75rem", fontWeight: 600, color: status?.dataset_ready ? "#10b981" : "#ef4444" }}>
              {status?.dataset_ready ? "✓ SoccerNet_YOLO Ready" : "✗ Dataset Missing"}
            </span>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginBottom: "1rem" }}>
            <div>
              <label style={{ display: "block", fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "0.25rem" }}>
                Architecture
              </label>
              <select
                className="select-input"
                style={{ width: "100%" }}
                value={model}
                disabled={isRunning}
                onChange={(e) => setModel(e.target.value)}
              >
                <option value="yolov8m.pt">YOLOv8 Medium (Balanced)</option>
                <option value="yolov8s.pt">YOLOv8 Small (Fast)</option>
                <option value="yolo26n.pt">YOLO Nano (Edge)</option>
              </select>
            </div>

            <div>
              <label style={{ display: "block", fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "0.25rem" }}>
                Image Resolution
              </label>
              <select
                className="select-input"
                style={{ width: "100%" }}
                value={imgsz}
                disabled={isRunning}
                onChange={(e) => setImgsz(Number(e.target.value))}
              >
                <option value="1280">1280 px (Small Ball Accuracy)</option>
                <option value="960">960 px (Fast 25+ FPS)</option>
              </select>
            </div>

            <div>
              <label style={{ display: "block", fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "0.25rem" }}>
                Epochs
              </label>
              <input
                type="number"
                className="select-input"
                style={{ width: "100%" }}
                value={epochs}
                disabled={isRunning}
                onChange={(e) => setEpochs(Number(e.target.value))}
              />
            </div>

            <div>
              <label style={{ display: "block", fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "0.25rem" }}>
                Batch Size
              </label>
              <input
                type="number"
                className="select-input"
                style={{ width: "100%" }}
                value={batch}
                disabled={isRunning}
                onChange={(e) => setBatch(Number(e.target.value))}
              />
            </div>
          </div>

          {/* Active Job Progress */}
          {isRunning && (
            <div style={{ background: "var(--bg-subtle)", padding: "1rem", borderRadius: "6px", marginBottom: "1rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", fontWeight: 600, marginBottom: "0.5rem" }}>
                <span>Training Epoch {activeJob.current_epoch} / {activeJob.epochs}</span>
                <span style={{ color: "var(--accent-blue)" }}>{activeJob.progress_pct}%</span>
              </div>
              <div className="scrubber-track" style={{ marginBottom: "0.75rem" }}>
                <div className="scrubber-fill" style={{ width: `${activeJob.progress_pct}%` }}></div>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "var(--text-muted)" }}>
                <span>Loss: {activeJob.loss}</span>
                <span>mAP50: {activeJob.mAP50}</span>
                <span>Target: RTX 5060</span>
              </div>
            </div>
          )}

          <div style={{ display: "flex", gap: "0.75rem" }}>
            {isRunning ? (
              <button className="btn-outline" style={{ color: "#ef4444", borderColor: "#ef4444" }} onClick={stopYolo}>
                Stop Training
              </button>
            ) : (
              <button className="btn-solid" disabled={loading} onClick={startYolo}>
                {loading ? "Starting..." : "Start YOLOv8 Training"}
              </button>
            )}
          </div>
        </div>

        {/* Right: Strain & Injury Risk ML Model Card */}
        <div className="card-clean">
          <h3 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "0.5rem" }}>
            Physiological Strain ML Model
          </h3>
          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "1rem" }}>
            XGBoost classifier predicting non-contact soft-tissue injury risk from ACWR, HR-drift, and kinematic load.
          </p>

          <div style={{ background: "var(--bg-subtle)", padding: "1rem", borderRadius: "6px", marginBottom: "1.25rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.5rem", fontSize: "0.85rem" }}>
              <span>Validation Accuracy:</span>
              <strong style={{ color: "#10b981" }}>{(status?.strain_model?.val_accuracy * 100).toFixed(1)}%</strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.5rem", fontSize: "0.85rem" }}>
              <span>Model Algorithm:</span>
              <strong>XGBoost Gradient Boosting</strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem" }}>
              <span>Top Feature Driver:</span>
              <strong style={{ color: "var(--accent-blue)" }}>ACWR Load Ratio (35%)</strong>
            </div>
          </div>

          <button className="btn-solid" style={{ width: "100%" }} disabled={loading} onClick={trainStrain}>
            {loading ? "Retraining..." : "Retrain Strain Model"}
          </button>
        </div>
      </div>
    </div>
  );
};
