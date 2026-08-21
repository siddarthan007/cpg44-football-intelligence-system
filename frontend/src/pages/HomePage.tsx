import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { api } from '../lib/api';
import type { Match, SystemInfo } from '../lib/types';
import { ErrorBanner, Stat } from '../components/common';

export function HomePage() {
  const navigate = useNavigate();
  const [matches, setMatches] = useState<Match[]>([]);
  const [info, setInfo] = useState<SystemInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.listMatches().then(setMatches).catch((e) => setError(e.message));
    api.systemInfo().then(setInfo).catch(() => undefined);
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">
          <span className="brand-mark">◎</span>
          <span>CPG44</span>
        </span>
        <div className="topbar-right">
          <Link to="/settings" className="chip">Settings</Link>
        </div>
      </header>
      <main className="content">
        <div className="page-head">
          <div>
            <h1>Matches</h1>
            <p className="muted small">
              Vision + wearable intelligence for a campus pitch. Inspired by the
              Football CV dashboard (REST, WebSocket, join-key observations):
              wired to this WSL pipeline and the ESP32 hub.
            </p>
          </div>
        </div>
        <ErrorBanner message={error} />
        {info && (
          <div className="grid cols-4" style={{ marginBottom: 16 }}>
            <Stat label="Compute" value={info.gpu.available ? 'GPU' : 'CPU'} hint={info.gpu.name ?? 'WSL'} tone={info.gpu.available ? 'good' : 'warn'} />
            <Stat label="Detector" value={info.active_config.detector} />
            <Stat label="Wearable hub" value={info.hub_connected ? 'connected' : 'offline'} tone={info.hub_connected ? 'good' : 'warn'} hint={info.hub_url} />
            <Stat label="Uptime" value={`${Math.round(info.uptime_s)} s`} />
          </div>
        )}
        <div className="grid cols-2">
          {matches.map((m) => (
            <button
              key={m.match_id}
              className="match-card"
              onClick={() => navigate(`/matches/${m.match_id}/live`)}
              style={{ textAlign: 'left' }}
            >
              <strong>{m.name}</strong>
              <p className="muted small">{m.venue} · {m.pitch_length_m}×{m.pitch_width_m} m · {m.status}</p>
              <p className="muted small">{m.notes}</p>
            </button>
          ))}
        </div>
      </main>
    </div>
  );
}
