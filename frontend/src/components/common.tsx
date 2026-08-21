import type { ReactNode } from 'react';
import type { LivePayload } from '../lib/types';

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: string;
  title: string;
  description: string;
  actions?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        {eyebrow && <span className="eyebrow">{eyebrow}</span>}
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </header>
  );
}

export function Panel({
  title,
  subtitle,
  children,
}: {
  title?: string;
  subtitle?: string;
  children: ReactNode;
}) {
  return (
    <section className="panel">
      {(title || subtitle) && (
        <header className="panel-head">
          <div>
            {title && <h2>{title}</h2>}
            {subtitle && <p className="muted small">{subtitle}</p>}
          </div>
        </header>
      )}
      <div className="panel-body">{children}</div>
    </section>
  );
}

export function Stat({
  label,
  value,
  hint,
  tone = 'default',
}: {
  label: string;
  value: ReactNode;
  hint?: string;
  tone?: 'default' | 'good' | 'warn' | 'bad';
}) {
  return (
    <div className={`stat tone-${tone}`}>
      <span className="stat-label">{label}</span>
      <strong className="stat-value">{value}</strong>
      {hint && <span className="stat-hint">{hint}</span>}
    </div>
  );
}

export function ErrorBanner({ message }: { message?: string | null }) {
  if (!message) return null;
  return <div className="error-banner">{message}</div>;
}

export function EmptyState({ title }: { title: string }) {
  return <div className="empty-state">{title}</div>;
}

export function StatusBadge({ status }: { status: string }) {
  const tone =
    status === 'online' || status === 'live' || status === 'ready' || status === 'ok' || status === 'usable' || status === 'completed'
      ? 'good'
      : status === 'paused' || status === 'degraded' || status === 'review' || status === 'recorded' || status === 'stale'
        ? 'warn'
        : status === 'offline' || status === 'error' || status === 'failed' || status === 'unavailable'
          ? 'bad'
          : 'default';
  return <span className={`badge tone-${tone}`}>{status}</span>;
}

export function ProvenanceBanner({ snapshot }: { snapshot: LivePayload | null }) {
  if (!snapshot) return null;
  const p = snapshot.provenance;
  const source = p.live ? 'Live pipeline' : p.kind === 'recorded_analysis' ? 'Recorded analysis' : 'No source';
  return (
    <div className={`provenance ${p.live ? 'is-live' : ''}`}>
      <div>
        <strong>{source}</strong>
        <span>
          {p.source_file ? ` · ${p.source_file}` : ''}
          {typeof p.age_s === 'number' ? ` · ${p.age_s.toFixed(p.live ? 1 : 0)}s old` : ''}
        </span>
      </div>
      <StatusBadge status={snapshot.data_quality?.status ?? 'unreported'} />
    </div>
  );
}

export function Value({ value, digits = 1, suffix = '-' }: { value: unknown; digits?: number; suffix?: string }) {
  return typeof value === 'number' && Number.isFinite(value) ? <>{value.toFixed(digits)}</> : <>{suffix}</>;
}
