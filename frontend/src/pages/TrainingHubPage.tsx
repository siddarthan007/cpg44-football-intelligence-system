import { useEffect, useState } from 'react';
import { EmptyState, ErrorBanner, PageHeader, StatusBadge, Value } from '../components/common';
import { api } from '../lib/api';

type Metrics = { epoch?: number; precision?: number; recall?: number; map50?: number; map50_95?: number; box_loss?: number };
type TrainingJob = { id: string; status: string; epochs: number; progress_pct: number; metrics?: Metrics; error?: string; log_path?: string; best_weights?: string };
type Run = { name: string; metrics: Metrics; best_weights?: string | null; updated_at: number };
type TrainingStatus = {
  active_job?: TrainingJob | null;
  dataset_path?: string;
  dataset_ready?: boolean;
  runs?: Run[];
  gpu_required_for_practical_training?: boolean;
  strain_model?: { trained?: boolean; backend?: string | null; validation?: Record<string, number> | null; training_source?: string };
  outcome_labels?: { collected_samples: number; collected_player_sessions: number; labelled_samples: number; positive_outcomes: number; ready: boolean };
  outcome_window_days?: number;
};

export const TrainingHubPage = () => {
  const [status, setStatus] = useState<TrainingStatus>({});
  const [dataset, setDataset] = useState('');
  const [model, setModel] = useState('yolov8m.pt');
  const [epochs, setEpochs] = useState(50);
  const [batch, setBatch] = useState(4);
  const [imgsz, setImgsz] = useState(1280);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [matchId, setMatchId] = useState('live');
  const [playerId, setPlayerId] = useState('');
  const [outcome, setOutcome] = useState(0);
  const [labelSource, setLabelSource] = useState('');
  const [outcomeWindow, setOutcomeWindow] = useState(7);

  const refresh = () => api.trainingStatus().then((body) => {
    const next = body as TrainingStatus;
    setStatus(next);
    setDataset((current) => current || next.dataset_path || '');
    setOutcomeWindow(next.outcome_window_days ?? 7);
    setError(null);
  }).catch((reason: unknown) => setError(reason instanceof Error ? reason.message : 'Training API unavailable'));

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(refresh, 2000);
    return () => window.clearInterval(timer);
  }, []);

  const start = async () => {
    setBusy(true); setError(null);
    try {
      const response = await api.startTraining({ data_path: dataset, model_name: model, epochs, batch, imgsz });
      if (response.ok === false) throw new Error(String(response.error ?? 'Training did not start'));
      await refresh();
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Training did not start'); }
    finally { setBusy(false); }
  };

  const stop = async () => {
    try {
      const response = await api.stopTraining();
      if (response.ok === false) throw new Error(String(response.error));
      await refresh();
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Unable to stop job'); }
  };

  const trainOutcomeModel = async () => {
    setBusy(true); setError(null);
    try {
      const response = await api.trainStrain();
      if (response.ok === false) throw new Error(String(response.error));
      await refresh();
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Outcome model did not train'); }
    finally { setBusy(false); }
  };

  const saveOutcome = async () => {
    setBusy(true); setError(null);
    try {
      await api.labelPlayerSession({
        match_id: matchId,
        player_id: Number(playerId),
        injury_label: outcome,
        outcome_window_days: outcomeWindow,
        label_source: labelSource,
      });
      setPlayerId(''); setLabelSource('');
      await refresh();
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Outcome label was not saved'); }
    finally { setBusy(false); }
  };

  const active = status.active_job;
  const running = active?.status === 'running' || active?.status === 'starting' || active?.status === 'stopping';
  const labels = status.outcome_labels ?? { collected_samples: 0, collected_player_sessions: 0, labelled_samples: 0, positive_outcomes: 0, ready: false };

  return (
    <div className="page">
      <PageHeader eyebrow="MODEL OPERATIONS" title="Training and evaluation" description="Launch real Ultralytics jobs, inspect artifact metrics, and keep outcome models gated on labelled evidence." />
      <ErrorBanner message={error} />

      <div className="two-column">
        <section className="card-clean">
          <div className="section-heading">
            <div><span className="eyebrow">DETECTOR</span><h2>SoccerNet training</h2></div>
            <StatusBadge status={status.dataset_ready ? 'ready' : 'unavailable'} />
          </div>
          <div className="form-grid">
            <label className="field wide"><span>Dataset YAML</span><input value={dataset} onChange={(event) => setDataset(event.target.value)} disabled={running} /></label>
            <label className="field"><span>Checkpoint</span><select value={model} onChange={(event) => setModel(event.target.value)} disabled={running}><option value="yolov8m.pt">YOLOv8 medium</option><option value="yolov8s.pt">YOLOv8 small</option><option value="yolo11m.pt">YOLO11 medium</option></select></label>
            <label className="field"><span>Image size</span><select value={imgsz} onChange={(event) => setImgsz(Number(event.target.value))} disabled={running}><option value={960}>960 px</option><option value={1280}>1280 px</option><option value={1536}>1536 px</option></select></label>
            <label className="field"><span>Epochs</span><input type="number" min="1" max="500" value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} disabled={running} /></label>
            <label className="field"><span>Batch</span><input type="number" min="1" max="128" value={batch} onChange={(event) => setBatch(Number(event.target.value))} disabled={running} /></label>
          </div>
          {active && (
            <div className="job-card">
              <div className="section-heading"><strong>{active.id}</strong><StatusBadge status={active.status} /></div>
              <div className="progress"><span style={{ width: `${active.progress_pct ?? 0}%` }} /></div>
              <div className="compact-metrics">
                <span>Epoch <strong>{active.metrics?.epoch ?? '-'} / {active.epochs}</strong></span>
                <span>mAP50 <strong><Value value={active.metrics?.map50} digits={3} /></strong></span>
                <span>mAP50-95 <strong><Value value={active.metrics?.map50_95} digits={3} /></strong></span>
              </div>
              {active.error && <p className="field-error">{active.error}</p>}
            </div>
          )}
          <div className="button-row">
            {running ? <button className="btn-danger" onClick={stop}>Stop process</button> : <button className="btn-solid" disabled={busy || !status.dataset_ready} onClick={start}>Start training process</button>}
          </div>
        </section>

        <section className="card-clean">
          <div className="section-heading">
            <div><span className="eyebrow">PLAYER AVAILABILITY</span><h2>Outcome model gate</h2></div>
            <StatusBadge status={status.strain_model?.trained ? 'ready' : 'unavailable'} />
          </div>
          <p className="body-copy">The model trains only after clinician-labelled outcomes exist. A chronological holdout reports ROC AUC, average precision and Brier score; heuristic ACWR values are never used as labels.</p>
          <dl className="definition-grid outcome-grid">
            <div><dt>Measured rows</dt><dd>{labels.collected_samples}</dd></div>
            <div><dt>Player-sessions</dt><dd>{labels.collected_player_sessions}</dd></div>
            <div><dt>Labelled sessions</dt><dd>{labels.labelled_samples}</dd></div>
            <div><dt>Positive outcomes</dt><dd>{labels.positive_outcomes}</dd></div>
            <div><dt>Minimum gate</dt><dd>100 / 10 per class</dd></div>
            <div><dt>Validation</dt><dd>Temporal 20%</dd></div>
          </dl>
          {status.strain_model?.validation && (
            <div className="compact-metrics stacked">
              {Object.entries(status.strain_model.validation).map(([key, value]) => <span key={key}>{key.replaceAll('_', ' ')} <strong>{value}</strong></span>)}
            </div>
          )}
          <button className="btn-solid" disabled={busy || !labels.ready} onClick={trainOutcomeModel}>Train from labelled outcomes</button>
          {!labels.ready && <p className="form-help">Add labels with provenance before this action becomes available.</p>}
          <div className="section-divider" />
          <h3>Record reviewed session outcome</h3>
          <p className="form-help">One label represents one player-session, not every five-second sample. Re-entering the same match and player updates that outcome.</p>
          <div className="form-grid">
            <label className="field"><span>Match ID</span><input value={matchId} onChange={(event) => setMatchId(event.target.value)} /></label>
            <label className="field"><span>Wearable player ID</span><input type="number" min="1" value={playerId} onChange={(event) => setPlayerId(event.target.value)} /></label>
            <label className="field wide"><span>Reviewed outcome</span><select value={outcome} onChange={(event) => setOutcome(Number(event.target.value))}><option value={0}>No documented time-loss injury</option><option value={1}>Documented time-loss injury</option></select></label>
            <label className="field"><span>Fixed outcome window</span><input type="text" value={`${outcomeWindow} days`} readOnly /></label>
            <label className="field"><span>Reviewer or record reference</span><input value={labelSource} onChange={(event) => setLabelSource(event.target.value)} placeholder="Physio register 2026-08-20" /></label>
          </div>
          <button className="btn-outline" disabled={busy || !matchId || !playerId || labelSource.trim().length < 4} onClick={saveOutcome}>Save reviewed outcome</button>
        </section>
      </div>

      <section className="card-clean">
        <div className="section-heading"><div><span className="eyebrow">ARTIFACTS</span><h2>Recorded runs</h2></div><span className="muted">Metrics are read from results.csv</span></div>
        {!status.runs?.length ? <EmptyState title="No completed training artifacts found." /> : (
          <div className="table-scroll"><table><thead><tr><th>Run</th><th>Epoch</th><th>Precision</th><th>Recall</th><th>mAP50</th><th>mAP50-95</th><th>Weights</th></tr></thead><tbody>
            {status.runs.map((run) => <tr key={run.name}><td><strong>{run.name}</strong></td><td>{run.metrics.epoch}</td><td><Value value={run.metrics.precision} digits={3} /></td><td><Value value={run.metrics.recall} digits={3} /></td><td><Value value={run.metrics.map50} digits={3} /></td><td><Value value={run.metrics.map50_95} digits={3} /></td><td className="mono small">{run.best_weights ?? 'missing'}</td></tr>)}
          </tbody></table></div>
        )}
      </section>
    </div>
  );
};
