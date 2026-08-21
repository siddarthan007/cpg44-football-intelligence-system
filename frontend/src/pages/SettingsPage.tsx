import { useEffect, useState } from 'react';
import { EmptyState, ErrorBanner, PageHeader, StatusBadge } from '../components/common';
import { api } from '../lib/api';
import type { SystemInfo } from '../lib/types';

export const SettingsPage = () => {
  const [info, setInfo] = useState<SystemInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.systemInfo().then(setInfo).catch((reason: unknown) => setError(reason instanceof Error ? reason.message : 'System API unavailable'));
  }, []);
  return (
    <div className="page">
      <PageHeader eyebrow="RUNTIME" title="System evidence" description="Read-only state from the local backend. Runtime URLs and model paths are configured with environment variables, not silently stored in the browser." actions={<StatusBadge status={info ? 'online' : 'offline'} />} />
      <ErrorBanner message={error} />
      <div className="two-column">
        <section className="card-clean">
          <div className="section-heading"><div><span className="eyebrow">COMPUTE</span><h2>Local workstation</h2></div></div>
          <dl className="definition-grid">
            <div><dt>Backend</dt><dd>{info?.version ?? '-'}</dd></div>
            <div><dt>GPU</dt><dd>{info?.gpu.available ? info.gpu.name : 'Unavailable to WSL'}</dd></div>
            <div><dt>Detector</dt><dd>{info?.active_config.detector ?? '-'}</dd></div>
            <div><dt>Tracker</dt><dd>{info?.active_config.tracker ?? '-'}</dd></div>
            <div><dt>Sensor hub</dt><dd>{info?.hub_connected ? 'Connected' : 'Offline'}</dd></div>
            <div><dt>Mode</dt><dd>{info?.active_mode ?? '-'}</dd></div>
          </dl>
          <p className="artifact-path"><strong>Hub:</strong> {info?.hub_url ?? 'not configured'}</p>
          <p className="artifact-path"><strong>VPS relay:</strong> {info?.hostinger_relay_url ?? 'not configured'}</p>
        </section>
        <section className="card-clean">
          <div className="section-heading"><div><span className="eyebrow">MODEL REGISTRY</span><h2>Detector artifacts</h2></div></div>
          {!info?.models.length ? <EmptyState title="No best.pt artifacts found." /> : <div className="model-list">{info.models.map((model) => <article key={model.path}><strong>{model.name}</strong><span>{model.path} · {model.size_mb} MB</span></article>)}</div>}
        </section>
      </div>
      <div className="quality-callout neutral"><strong>Configuration</strong><p>Set <code>CPG44_HUB_URL</code>, <code>CPG44_RELAY_URL</code>, <code>CPG44_RELAY_TOKEN</code>, <code>CPG44_MODEL_PATH</code> and <code>CPG44_CORS_ORIGINS</code> before launching the services. Secrets stay in environment files outside version control.</p></div>
    </div>
  );
};
