import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { Glyph, type GlyphName } from '../components/Glyph';
import { LiveBoard } from '../components/LiveBoard';
import { api, apiUrl } from '../lib/api';
import { clock, TEAM_HEX } from '../lib/analysis';
import type { Insight } from '../lib/analysis';
import { useLive } from '../lib/useLive';
import type { LivePlayer } from '../lib/types';

type Camera = { id: string; name: string; type: string };
type Calib = { id: string; name: string; has_thumb?: boolean };
type Vision = { running?: boolean; status?: string; camera_id?: string | null; error?: string | null };
type Vest = { connected?: boolean; clock_synced?: boolean; battery?: { pct?: number } | null };
const KIND_GLYPH: Record<Insight['kind'], GlyphName> = { tactical: 'tactics', physical: 'run', fused: 'pulse', data: 'eye' };

const num = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v : undefined);
const label = (p: LivePlayer) => p.name || (p.jersey != null ? `#${p.jersey}` : `Player ${p.track_id}`);
const kmh = (v: number | null | undefined) => (v == null ? undefined : v * 3.6);
const fmt = (v: number | undefined, digits = 0) => (v == null ? 'n/a' : v.toFixed(digits));

/** Start panel shown while nothing is being tracked: everything needed to go live on one card. */
function StartPanel({ vision, onStarted }: { vision: Vision; onStarted: () => void }) {
  const [cams, setCams] = useState<Camera[]>();
  const [calibs, setCalibs] = useState<Calib[]>();
  const [vest, setVest] = useState<Vest>();
  const [cam, setCam] = useState('');
  const [calib, setCalib] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  useEffect(() => {
    api.cameras().then(rows => { const c = rows as unknown as Camera[]; setCams(c); if (c[0]) setCam(v => v || c[0].id); }).catch(() => setCams([]));
    fetch(apiUrl('/api/v1/calibrations')).then(r => r.json()).then((rows: Calib[]) => { setCalibs(rows); if (rows[0]) setCalib(v => v || rows[0].id); }).catch(() => setCalibs([]));
    api.wearableStatus().then(v => setVest(v as Vest)).catch(() => setVest({ connected: false }));
  }, []);
  const start = async () => {
    setBusy(true); setError(undefined);
    try { await api.startCamera(cam, calib ? { calibration_id: calib } : undefined); onStarted(); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  };
  const starting = vision.running && vision.status !== 'running';
  const rows: { g: GlyphName; title: string; state: 'done' | 'todo' | 'warn'; text: string; to: string; action: string }[] = [
    { g: 'camera', title: 'Camera', to: '/workshop/cameras', action: cams?.length ? 'Manage' : 'Connect',
      state: cams?.length ? 'done' : 'todo', text: cams == null ? 'checking' : cams.length ? `${cams.length} ready` : 'none connected' },
    { g: 'pitch', title: 'Ground', to: '/prepare/pitch', action: calibs?.length ? 'Edit' : 'Calibrate',
      state: calibs?.length ? 'done' : 'warn', text: calibs == null ? 'checking' : calibs.length ? `${calibs.length} saved` : 'no metres without it' },
    { g: 'vest', title: 'Vests', to: '/vest', action: 'Open',
      state: vest?.connected ? 'done' : 'warn', text: vest == null ? 'checking' : vest.connected ? (vest.clock_synced ? 'streaming' : 'syncing clock') : 'video only' },
  ];
  return <section className="panel live-start" aria-label="Start live tracking">
    <div className="live-start-copy">
      <h2>{starting ? 'Starting the camera' : 'Not tracking'}</h2>
      <p className="muted">{starting ? 'The first processed frame appears in a few seconds.' : 'Pick a camera and the ground it films.'}</p>
      {cams && cams.length > 0 && !starting && <div className="live-start-form">
        <label><span>Camera</span><select value={cam} onChange={e => setCam(e.target.value)}>{cams.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}</select></label>
        <label><span>Ground</span><select value={calib} onChange={e => setCalib(e.target.value)}>
          <option value="">Not calibrated</option>{calibs?.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}</select></label>
        <button className="btn" disabled={busy || !cam} onClick={() => void start()}><Glyph g="play" size={15} />Start</button>
      </div>}
      {cams && !cams.length && <Link className="btn" to="/workshop/cameras"><Glyph g="sources" size={15} />Connect a camera</Link>}
      {starting && <progress className="spinner" aria-label="Starting the camera" />}
      {(error || vision.error) && <p className="notice bad">{error || vision.error}</p>}
    </div>
    <ul className="live-checks">
      {rows.map(r => <li key={r.title} className={r.state}>
        <span className="ico"><Glyph g={r.g} size={18} /></span>
        <div><strong>{r.title}</strong><small>{r.text}</small></div>
        <Link className="btn ghost small" to={r.to}>{r.action}</Link>
      </li>)}
    </ul>
  </section>;
}

function PlayerPanel({ p, team }: { p: LivePlayer; team: string }) {
  const w = p.wearable_metrics ?? {};
  const hr = num(w.hr), trimp = num(p.load?.trimp);
  const r = p.readiness;
  return <aside className="panel live-player" aria-label="Selected player">
    <div className="panel-head"><h2><i className="dot" style={{ background: TEAM_HEX[p.team] ?? TEAM_HEX[0], marginRight: 8 }} />{label(p)}</h2>
      <span className="muted">{team}{p.wearable ? ', vest' : ''}</span></div>
    <dl className="live-nums">
      <div><dt>Now</dt><dd>{fmt(kmh(p.speed_mps))}<small>km/h</small></dd></div>
      <div><dt>Top</dt><dd>{fmt(kmh(p.top_speed_mps), 1)}<small>km/h</small></dd></div>
      <div><dt>Distance</dt><dd>{fmt(num(p.distance_m))}<small>m</small></dd></div>
      <div><dt>High speed</dt><dd>{fmt(num(p.load?.hsr_m))}<small>m</small></dd></div>
      {p.wearable && <div><dt>Heart rate</dt><dd>{fmt(hr)}<small>bpm</small></dd></div>}
      {p.wearable && trimp != null && <div><dt>Heart load</dt><dd>{fmt(trimp)}<small>TRIMP</small></dd></div>}
      {num(p.load?.metabolic_power_wkg) != null && <div><dt>Power</dt><dd>{fmt(num(p.load?.metabolic_power_wkg), 1)}<small>W/kg</small></dd></div>}
    </dl>
    {r && r.status !== 'signal_limited' && <div className={`cue ${r.status}`}><strong>{r.action}</strong><span>{r.reason}</span></div>}
  </aside>;
}

export const LivePage = () => {
  const { data, connected } = useLive('live');
  const [vision, setVision] = useState<Vision>({});
  const [view, setView] = useState<'positions' | 'heat'>('positions');
  const [selected, setSelected] = useState<number>();
  const [frameOk, setFrameOk] = useState(false);
  const [busy, setBusy] = useState(false);
  const boardPanel = useRef<HTMLElement>(null);
  const pollVision = useCallback(() => api.visionStatus().then(v => setVision(v as Vision)).catch(() => setVision({})), []);
  useEffect(() => { void pollVision(); const t = window.setInterval(() => void pollVision(), 2000); return () => window.clearInterval(t); }, [pollVision]);

  const live = data?.provenance.live === true;
  const players = useMemo(() => [...(data?.players ?? [])].sort((a, b) => (a.team || 9) - (b.team || 9) || (Number(a.jersey ?? 999) - Number(b.jersey ?? 999)) || a.track_id - b.track_id), [data]);
  const player = players.find(p => p.track_id === selected);
  const m = data?.match;
  const L = m?.pitch_length_m ?? 105, W = m?.pitch_width_m ?? 68;
  const kind = (m as { pitch_kind?: 'football' | 'futsal' } | undefined)?.pitch_kind;
  const teamName = (k: number) => (k === 1 ? m?.home_team : k === 2 ? m?.away_team : undefined) || (k ? `Team ${k}` : 'Unassigned');
  const p1 = num(data?.possession_pct?.['1']), p2 = num(data?.possession_pct?.['2']);
  const hasControl = p1 != null && p2 != null && p1 + p2 > 0;
  const metric = data?.metric === true;
  const elapsed = (data?.frame_index ?? 0) / (data?.source_fps || 25);
  const insights = data?.coach_insights ?? [];
  const warnings = data?.data_quality?.warnings ?? [];
  const vests = players.filter(p => p.wearable).length;
  const stop = async () => { setBusy(true); try { await api.stopVision(); await pollVision(); } finally { setBusy(false); } };
  const full = () => { if (document.fullscreenElement) void document.exitFullscreen(); else void boardPanel.current?.requestFullscreen?.(); };

  if (!live) return <div className="page live">
    <div className="top"><h1>Live</h1>
      {vision.running && <button className="btn ghost small" disabled={busy} onClick={() => void stop()}><Glyph g="pause" size={14} />Stop</button>}</div>
    <StartPanel vision={vision} onStarted={() => void pollVision()} />
  </div>;

  return <div className="page live">
    <div className="top">
      <div className="live-title"><h1>Live</h1><span className={`live-pill ${connected ? 'on' : ''}`}><i />{connected ? 'on air' : 'reconnecting'}</span>
        <time className="num" aria-label="Elapsed">{clock(elapsed)}</time></div>
      <div className="health" aria-label="Feed quality">
        <span title="Frames processed per second" className={(data?.processing_fps ?? 0) >= 10 ? 'state-ok' : 'state-warn'}><Glyph g="camera" size={16} />{fmt(data?.processing_fps, 0)} fps</span>
        <span title={metric ? 'Pitch calibrated' : 'Pitch not calibrated: no metres'} className={metric ? 'state-ok' : 'state-warn'}><Glyph g="pitch" size={16} />{metric ? 'metres' : 'pixels'}</span>
        <span title="Players wearing a linked vest"><Glyph g="vest" size={16} />{vests}</span>
        <button className="btn ghost small" disabled={busy} onClick={() => void stop()}><Glyph g="pause" size={14} />Stop</button>
      </div>
    </div>

    <div className="scoreline" aria-label="Ball control">
      <span className="side"><i className="dot a" />{teamName(1)}</span>
      {hasControl ? <><span className="ctrl-bar" aria-hidden="true"><i style={{ width: `${p1}%` }} /><i style={{ width: `${p2}%` }} /></span>
        <span className="ctrl-val">{Math.round(p1!)} : {Math.round(p2!)}</span></> : <span className="muted">control builds as play runs</span>}
      <span className="side">{teamName(2)}<i className="dot b" /></span>
    </div>

    <div className="stage">
      <section className="panel" aria-label="Camera" style={{ padding: 10 }}>
        <div className="film">
          <img src={apiUrl('/api/v1/live/mjpeg')} alt="Processed camera view" onLoad={() => setFrameOk(true)} onError={() => setFrameOk(false)}
            style={{ width: '100%', height: '100%', objectFit: 'contain', display: frameOk ? 'block' : 'none' }} />
          {!frameOk && <progress className="spinner light" aria-label="Waiting for the camera" />}
        </div>
      </section>
      <section className="panel board-panel" aria-label="Pitch" style={{ padding: 10 }} ref={boardPanel}>
        <div className="panel-head" style={{ marginBottom: 8 }}>
          <div className="seg" role="group" aria-label="Pitch view">
            <button aria-pressed={view === 'positions'} onClick={() => setView('positions')}><Glyph g="moment" size={15} />Positions</button>
            <button aria-pressed={view === 'heat'} disabled={!data?.heatmap} onClick={() => setView('heat')}><Glyph g="heat" size={15} />Heat</button>
          </div>
          <button className="icon-btn" onClick={full} aria-label="Full screen" title="Full screen"><Glyph g="expand" size={16} /></button>
        </div>
        {metric ? <div className="board-wrap"><LiveBoard L={L} W={W} kind={kind} players={players} ball={data?.ball ?? null} heat={data?.heatmap}
          mode={view} selected={player?.track_id} onSelect={setSelected} /></div>
          : <div className="empty"><Glyph g="pitch" size={36} /><span>Calibrate the ground to place players</span><Link className="btn ghost small" to="/prepare/pitch">Pitch setup</Link></div>}
      </section>
    </div>

    {warnings.length > 0 && <div className="live-warnings">{warnings.map(w => <p key={w} className="notice">{w}</p>)}</div>}

    <div className="lower">
      <section className="panel" aria-label="Players">
        <div className="panel-head"><h2>Players</h2><span className="muted num">{players.length} in view</span></div>
        {!players.length ? <div className="empty"><Glyph g="squad" size={30} />Waiting for players</div>
          : <div style={{ overflowX: 'auto' }}><table className="players">
            <thead><tr><th>Player</th><th className="num">km/h</th><th className="num">Distance</th><th className="num">Top</th><th className="num">bpm</th></tr></thead>
            <tbody>{players.map(p => <tr key={p.track_id} aria-selected={player?.track_id === p.track_id} onClick={() => setSelected(p.track_id)}
              tabIndex={0} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setSelected(p.track_id); } }}>
              <td><i className="dot" style={{ background: TEAM_HEX[p.team] ?? TEAM_HEX[0], marginRight: 8 }} />{label(p)}{p.wearable && <Glyph g="vest" size={13} />}</td>
              <td className="num">{fmt(kmh(p.speed_mps))}</td>
              <td className="num">{p.distance_m != null ? `${p.distance_m.toFixed(0)} m` : 'n/a'}</td>
              <td className="num">{fmt(kmh(p.top_speed_mps), 1)}</td>
              <td className="num">{p.wearable ? fmt(num(p.wearable_metrics?.hr)) : ''}</td>
            </tr>)}</tbody>
          </table></div>}
      </section>
      {player ? <PlayerPanel p={player} team={teamName(player.team)} /> : <aside className="panel"><div className="empty"><Glyph g="run" size={30} />Pick a player</div></aside>}
    </div>

    {insights.length > 0 && <section className="panel" aria-label="Worth acting on">
      <div className="panel-head"><h2>Worth acting on</h2></div>
      {insights.map(c => <div key={c.id} className={`finding ${c.kind}`}>
        <span className="kind"><Glyph g={KIND_GLYPH[c.kind] ?? 'tactics'} size={16} /></span>
        <div><h3>{c.title}{c.team ? <i className="dot" style={{ background: TEAM_HEX[c.team], marginLeft: 8 }} /> : null}</h3>
          <p>{c.finding}</p><p className="act"><Glyph g="fwd" size={14} />{c.action}</p></div>
        <span />
      </div>)}
    </section>}
  </div>;
};
