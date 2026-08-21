import { useEffect, useMemo, useState } from 'react';
import { EmptyState, ErrorBanner, PageHeader, ProvenanceBanner, StatusBadge, Value } from '../components/common';
import { api, websocketUrl } from '../lib/api';
import { useLive } from '../lib/useLive';

type TelemetryRow = {
  id: string;
  timestamp?: number;
  hr?: number;
  spo2?: number;
  playerLoad?: number;
  speed?: number;
  speedSource?: string;
  signalQuality?: number;
  gps?: unknown;
  source?: string;
};

const number = (value: unknown) => typeof value === 'number' && Number.isFinite(value) ? value : undefined;

export const WearablePage = () => {
  const { data, error: liveError } = useLive('live');
  const [relayRows, setRelayRows] = useState<Record<string, unknown> | unknown[]>([]);
  const [relayError, setRelayError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let socket: WebSocket | null = null;
    let retry: number | undefined;
    const pull = () => api.wearables().then((rows) => {
      if (!cancelled) { setRelayRows(rows); setRelayError(null); }
    }).catch((error: unknown) => {
      if (!cancelled) setRelayError(error instanceof Error ? error.message : 'Wearable endpoint unavailable');
    });
    const connect = () => {
      if (cancelled) return;
      socket = new WebSocket(websocketUrl('/ws/wearables'));
      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as { wearables?: Record<string, unknown> | unknown[] };
          setRelayRows(message.wearables ?? []);
          setRelayError(null);
        } catch {
          setRelayError('Wearable feed returned invalid data');
        }
      };
      socket.onerror = () => socket?.close();
      socket.onclose = () => {
        if (cancelled) return;
        void pull();
        retry = window.setTimeout(connect, 1500);
      };
    };
    void pull();
    connect();
    const timer = window.setInterval(() => {
      if (!socket || socket.readyState !== WebSocket.OPEN) void pull();
    }, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
      if (retry !== undefined) window.clearTimeout(retry);
      socket?.close();
    };
  }, []);

  const rows = useMemo<TelemetryRow[]>(() => {
    const byId = new Map<string, TelemetryRow>();
    for (const player of data?.players ?? []) {
      const metrics = player.wearable_metrics;
      if (!metrics) continue;
      const id = String(player.player_id ?? player.track_id);
      byId.set(id, {
        id,
        timestamp: number(metrics.timestamp),
        hr: number(metrics.hr ?? metrics.heart_rate_bpm ?? metrics.bpm),
        spo2: number(metrics.spo2 ?? metrics.spo2_pct ?? metrics.spo2_estimate_pct),
        playerLoad: number(metrics.player_load_imu ?? metrics.player_load),
        speed: number(player.speed_mps ?? metrics.speed_mps),
        speedSource: player.speed_mps == null ? 'wearable' : 'calibrated vision',
        signalQuality: number(metrics.signal_quality),
        gps: metrics.gps,
        source: 'vision-linked vest',
      });
    }
    const candidates = Array.isArray(relayRows)
      ? relayRows
      : Object.entries(relayRows).map(([id, value]) => ({ id, ...((value as Record<string, unknown>) ?? {}) }));
    for (const raw of candidates) {
      if (!raw || typeof raw !== 'object') continue;
      const item = raw as Record<string, unknown>;
      const metrics = (item.metrics as Record<string, unknown> | undefined) ?? item;
      const id = String(item.player_id ?? item.global_player_id ?? item.id ?? 'unknown');
      const timestampNs = number(item.source_timestamp_ns);
      byId.set(id, {
        id,
        timestamp: timestampNs === undefined
          ? number(item.timestamp ?? item.t ?? item.received_at)
          : timestampNs / 1e9,
        hr: number(metrics.hr ?? metrics.heart_rate_bpm ?? metrics.bpm),
        spo2: number(metrics.spo2 ?? metrics.spo2_pct ?? metrics.spo2_estimate_pct),
        playerLoad: number(metrics.player_load_imu ?? metrics.player_load),
        speed: number(metrics.speed_mps),
        speedSource: 'wearable GPS',
        signalQuality: number(metrics.signal_quality),
        gps: metrics.gps,
        source: String(item.source ?? 'wearable endpoint'),
      });
    }
    return [...byId.values()].sort((a, b) => a.id.localeCompare(b.id));
  }, [data, relayRows]);

  return (
    <div className="page">
      <PageHeader
        eyebrow="WEARABLE EVIDENCE"
        title="Biometrics and external load"
        description="ESP32 samples are time-stamped, attributed to a player, and kept separate from vision estimates."
        actions={<StatusBadge status={rows.length ? 'online' : 'offline'} />}
      />
      <ErrorBanner message={liveError ?? relayError} />
      <ProvenanceBanner snapshot={data} />

      <div className="quality-callout neutral">
        <strong>Decision-support boundary</strong>
        <p>Wrist or vest optical PPG is motion-sensitive. Load indicators support coaching review; they are not diagnoses or injury probabilities.</p>
      </div>

      {!rows.length ? (
        <section className="card-clean"><EmptyState title="No recent wearable samples. Check the relay, match ID and player ID." /></section>
      ) : (
        <div className="telemetry-grid">
          {rows.map((row) => {
            const age = row.timestamp ? Math.max(0, Date.now() / 1000 - row.timestamp) : undefined;
            return (
              <article className="card-clean telemetry-card" key={row.id}>
                <div className="section-heading">
                  <div><span className="eyebrow">PLAYER</span><h2>#{row.id}</h2></div>
                  <StatusBadge status={age !== undefined && age < 10 ? 'live' : 'stale'} />
                </div>
                <dl className="definition-grid">
                  <div><dt>Heart rate</dt><dd><Value value={row.hr} digits={0} /> <small>bpm</small></dd></div>
                  <div><dt>SpO₂</dt><dd><Value value={row.spo2} digits={1} /> <small>%</small></dd></div>
                  <div><dt>IMU load</dt><dd><Value value={row.playerLoad} digits={2} /> <small>a.u.</small></dd></div>
                  <div><dt>Speed</dt><dd><Value value={row.speed} digits={2} /> <small>m/s · {row.speedSource}</small></dd></div>
                  <div><dt>Signal quality</dt><dd><Value value={row.signalQuality} digits={2} /></dd></div>
                </dl>
                <footer className="evidence-footer">
                  <span>{row.source}</span><span>{age === undefined ? 'timestamp unavailable' : `${age.toFixed(1)}s old`}</span>
                </footer>
              </article>
            );
          })}
        </div>
      )}
    </div>
  );
};
