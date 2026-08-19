import type { ReactNode } from 'react';

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
  return <div className="empty">{title}</div>;
}

export function StatusBadge({ status }: { status: string }) {
  const tone =
    status === 'online' || status === 'live' || status === 'ready' || status === 'ok'
      ? 'good'
      : status === 'paused' || status === 'degraded'
        ? 'warn'
        : status === 'offline' || status === 'error'
          ? 'bad'
          : 'default';
  return <span className={`badge tone-${tone}`}>{status}</span>;
}
