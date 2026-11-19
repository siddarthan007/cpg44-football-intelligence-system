import { useEffect, useState } from 'react';
import { Glyph } from '../components/Glyph';
import { api } from '../lib/api';
import type { SystemInfo } from '../lib/types';

type Metrics = { epoch?: number; precision?: number; recall?: number; map50?: number; map50_95?: number };
type Job = { id: string; status: string; epochs: number; progress_pct: number; metrics?: Metrics; error?: string };
type Run = { name: string; metrics: Metrics; best_weights?: string | null; updated_at: number };
type Status = {
  active_job?: Job | null; dataset_path?: string; dataset_ready?: boolean; runs?: Run[];
  strain_model?: { trained?: boolean; validation?: Record<string, number | string> | null };
  outcome_labels?: { collected_samples: number; collected_player_sessions: number; labelled_samples: number; positive_outcomes: number; ready: boolean };
  outcome_window_days?: number;
};

const fmt = (v?: number) => (v == null || !Number.isFinite(v) ? '' : v.toFixed(3));

export const TrainingHubPage = () => {
  const [st, setSt] = useState<Status>({});
  const [info, setInfo] = useState<SystemInfo | null>(null);
  const [form, setForm] = useState({ dataset: '', model: 'yolov8m.pt', epochs: 50, batch: 2, imgsz: 1280 });
  const [label, setLabel] = useState({ match: 'live', player: '', outcome: 0, source: '' });
  const [msg, setMsg] = useState<{ ok: boolean; text: string }>();
  const [busy, setBusy] = useState(false);

  const refresh = () => api.trainingStatus().then(b => {
    const s = b as Status; setSt(s); setForm(f => ({ ...f, dataset: f.dataset || s.dataset_path || '' }));
  }).catch(() => undefined);
  useEffect(() => {
    void refresh(); api.systemInfo().then(setInfo).catch(() => setInfo(null));
    const t = window.setInterval(refresh, 3000); return () => window.clearInterval(t);
  }, []);

  const act = async (fn: () => Promise<Record<string, unknown>>, done: string) => {
    setBusy(true); setMsg(undefined);
    try { const r = await fn(); if (r.ok === false) throw new Error(String(r.error ?? 'failed')); setMsg({ ok: true, text: done }); await refresh(); }
    catch (e) { setMsg({ ok: false, text: e instanceof Error ? e.message : String(e) }); }
    finally { setBusy(false); }
  };

  const job = st.active_job;
  const running = job && ['running', 'starting', 'stopping'].includes(job.status);
  const inUse = info?.active_config.detector;
  const labels = st.outcome_labels ?? { collected_samples: 0, collected_player_sessions: 0, labelled_samples: 0, positive_outcomes: 0, ready: false };
  const need = 100;

  return <div className="page">
    <header className="page-header"><h1>Models</h1><p>The detector the tracker uses, and the outcome model trained on your own sessions.</p></header>
    {msg && <div className={`notice ${msg.ok ? '' : 'bad'}`}>{msg.text}</div>}

    <section className="panel sys-strip" aria-label="This computer">
      <span><Glyph g="runtime" size={16} /><b>{info?.gpu.available ? info.gpu.name : 'No GPU visible'}</b></span>
      <span><Glyph g="model" size={16} />Detector <b>{inUse ?? 'none'}</b></span>
      <span><Glyph g="eye" size={16} />Tracker <b>{info?.active_config.tracker ?? 'unknown'}</b></span>
      <span className={info?.hub_connected ? 'state-ok' : 'muted'}><Glyph g="vest" size={16} />Sensor hub {info?.hub_connected ? 'connected' : 'off'}</span>
    </section>

    <section className="panel" aria-label="Detector runs">
      <div className="panel-head"><h2>Detector runs</h2><span className="muted" style={{ fontSize: 12.5 }}>Each run scored on its own validation set. The Evidence page compares them on the same test clips.</span></div>
      {!st.runs?.length ? <div className="empty">No trained runs yet</div> : <div className="table-scroll"><table className="runs">
        <thead><tr><th scope="col">Run</th><th scope="col">Epochs</th><th scope="col">mAP50</th><th scope="col">mAP50-95</th><th scope="col"><span className="sr-only">State</span></th></tr></thead>
        <tbody>{st.runs.map(r => <tr key={r.name} className={r.name === inUse ? 'active' : ''}>
          <th scope="row">{r.name}</th><td className="num">{r.metrics.epoch ?? ''}</td><td className="num">{fmt(r.metrics.map50)}</td><td className="num">{fmt(r.metrics.map50_95)}</td>
          <td>{r.name === inUse && <span className="chip-in-use"><Glyph g="check" size={13} />in use</span>}</td></tr>)}</tbody>
      </table></div>}
      {job && <div className="job"><b>{job.id}</b><div className="rate-bar"><i className="ok" style={{ width: `${job.progress_pct ?? 0}%` }} /></div>
        <span className="num muted">epoch {job.metrics?.epoch ?? 0} of {job.epochs}{job.metrics?.map50_95 != null ? `, mAP50-95 ${fmt(job.metrics.map50_95)}` : ''}</span>
        {running && <button className="btn ghost small" onClick={() => void act(() => api.stopTraining(), 'Stopping')}>Stop</button>}</div>}
      <details className="more"><summary>Train a new detector</summary>
        <div className="row-fields">
          <label>Dataset<input style={{ minWidth: 320 }} value={form.dataset} onChange={e => setForm({ ...form, dataset: e.target.value })} disabled={Boolean(running)} /></label>
          <label>Start from<select value={form.model} onChange={e => setForm({ ...form, model: e.target.value })}><option value="yolov8m.pt">YOLOv8 medium</option><option value="yolov8s.pt">YOLOv8 small</option><option value="yolo11m.pt">YOLO11 medium</option></select></label>
          <label>Image size<select value={form.imgsz} onChange={e => setForm({ ...form, imgsz: Number(e.target.value) })}><option value={960}>960</option><option value={1280}>1280</option><option value={1536}>1536</option></select></label>
          <label>Epochs<input type="number" min={1} max={500} value={form.epochs} onChange={e => setForm({ ...form, epochs: Number(e.target.value) })} /></label>
          <button className="btn small" disabled={busy || Boolean(running) || !st.dataset_ready} onClick={() => void act(() => api.startTraining({ data_path: form.dataset, model_name: form.model, epochs: form.epochs, batch: form.batch, imgsz: form.imgsz }), 'Training started')}>Start</button>
        </div></details>
    </section>

    <section className="panel" aria-label="Outcome model">
      <div className="panel-head"><h2>Outcome model</h2>{st.strain_model?.trained && <span className="chip-in-use"><Glyph g="check" size={13} />trained</span>}</div>
      <p className="muted" style={{ margin: '0 0 10px', fontSize: 13.5 }}>A gradient-boosted classifier on each player's measured load, trained only once enough sessions carry a reviewed outcome. Until then the app gives load cues, not predictions.</p>
      <div className="rate" style={{ maxWidth: 560 }}><span>Labelled sessions</span>
        <div className="rate-bar"><i className={labels.ready ? 'ok' : 'warn'} style={{ width: `${Math.min(100, labels.labelled_samples / need * 100)}%` }} /></div>
        <b className="num">{labels.labelled_samples} / {need}</b></div>
      {st.strain_model?.validation && <div className="nums-row">{Object.entries(st.strain_model.validation).slice(0, 4).map(([k, v]) => <div key={k}><b className="num">{String(v)}</b><small>{k.replaceAll('_', ' ')}</small></div>)}</div>}
      <div className="row-fields">
        <button className="btn small" disabled={busy || !labels.ready} onClick={() => void act(() => api.trainStrain(), 'Model trained')}>Train</button>
      </div>
      <details className="more"><summary>Record a session outcome</summary>
        <div className="row-fields">
          <label>Session<input value={label.match} onChange={e => setLabel({ ...label, match: e.target.value })} /></label>
          <label>Vest<input type="number" min={1} value={label.player} onChange={e => setLabel({ ...label, player: e.target.value })} /></label>
          <label>Outcome within {st.outcome_window_days ?? 7} days<select value={label.outcome} onChange={e => setLabel({ ...label, outcome: Number(e.target.value) })}>
            <option value={0}>No time-loss injury</option><option value={1}>Time-loss injury</option></select></label>
          <label>Recorded by<input value={label.source} onChange={e => setLabel({ ...label, source: e.target.value })} placeholder="Physio register, 20 Aug" /></label>
          <button className="btn ghost small" disabled={busy || !label.player || label.source.trim().length < 4}
            onClick={() => void act(() => api.labelPlayerSession({ match_id: label.match, player_id: Number(label.player), injury_label: label.outcome, outcome_window_days: st.outcome_window_days ?? 7, label_source: label.source }), 'Outcome saved')}>Save</button>
        </div></details>
    </section>
  </div>;
};
