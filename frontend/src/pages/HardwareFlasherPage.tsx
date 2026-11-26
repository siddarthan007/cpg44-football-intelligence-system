import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { Glyph } from '../components/Glyph';
import { api } from '../lib/api';
import { getSquad, type Squad } from '../lib/squad';

type Port = { device: string; name?: string; description?: string; hwid?: string; windows_only?: boolean };
type Toolchain = {
  ready?: boolean; arduino_cli?: string | null;
  relay?: { token_configured?: boolean; ca_configured?: boolean; ready?: boolean };
  last_flash?: { player_id?: number; match_id?: string; wifi_ssid?: string };
};
type FlashJob = { id: string; status: string; logs?: string[]; error?: string; return_code?: number; player_id: number; port: string };
type CheckRow = { key: string; label: string; state: 'ok' | 'warn' | 'bad'; value: string; hint: string };
type Monitor = { port?: string; running?: boolean; lines?: Array<{ t: number; text: string }>; check?: CheckRow[] | null;
  check_age_s?: number | null; notes?: string[]; crashed?: boolean };

const ROW_GLYPH: Record<string, Parameters<typeof Glyph>[0]['g']> = {
  bus: 'chip', imu: 'run', mag: 'target', gps: 'pitch', ecg: 'pulse', battery: 'vest', wifi: 'sources', relay: 'live',
};

/** Live checklist from the vest's own self-test line, read over USB. */
function BenchCheck({ monitor, port, onStart }: { monitor: Monitor; port: string; onStart: () => void }) {
  const fresh = monitor.check && (monitor.check_age_s ?? 99) < 8;
  const bad = monitor.check?.filter(r => r.state === 'bad').length ?? 0;
  const lastWifiNote = monitor.notes?.filter(n => n.startsWith('Wi-Fi:')).slice(-1)[0];
  return <aside className="panel bench" aria-label="Bench check">
    <div className="panel-head">
      <h2>Bench check</h2>
      {monitor.running ? <span className={`live-pill ${fresh ? 'on' : ''}`}><i />{fresh ? 'live' : 'waiting'}</span>
        : <button className="btn ghost small" disabled={!port} onClick={onStart}>Start</button>}
    </div>
    {monitor.crashed && <p className="notice bad">The vest restarted after an error. The serial output below shows where.</p>}
    {!monitor.check ? <div className="empty"><Glyph g="chip" size={28} />
      <span>{port ? (monitor.running ? 'Waiting for the vest to report' : 'Plug in the vest to see each part') : 'Connect the vest by USB'}</span></div>
      : <>
        <p className="bench-summary">{bad ? `${bad} part${bad > 1 ? 's' : ''} need${bad > 1 ? '' : 's'} attention` : 'Every part reports in'}</p>
        <ul className="bench-rows">{monitor.check.map(r => <li key={r.key} className={r.state}>
          <span className="ico"><Glyph g={ROW_GLYPH[r.key] ?? 'info'} size={16} /></span>
          <div><strong>{r.label}</strong><span>{r.value}</span>{r.hint && <small>{r.hint}</small>}</div>
          <i className="dot" aria-label={r.state} />
        </li>)}</ul>
        {lastWifiNote && monitor.check.find(r => r.key === 'wifi')?.state === 'bad' && <p className="bench-note">{lastWifiNote.replace('Wi-Fi: ', '')}</p>}
      </>}
    <details className="more"><summary>Serial output</summary>
      <SerialLog lines={monitor.lines ?? []} />
    </details>
  </aside>;
}

function SerialLog({ lines }: { lines: Array<{ t: number; text: string }> }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const stick = useRef(true);
  useEffect(() => { if (stick.current && ref.current) ref.current.scrollTop = ref.current.scrollHeight; }, [lines.length]);
  return <div className="console" ref={ref} style={{ maxHeight: 300 }}
    onScroll={e => { const el = e.currentTarget; stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24; }}>
    {lines.length ? lines.slice(-200).map((line, i) => <div key={`${i}-${line.t}`}>{line.text}</div>) : <span>No serial output yet.</span>}
  </div>;
}

export const HardwareFlasherPage = () => {
  const [ports, setPorts] = useState<Port[]>([]);
  const [toolchain, setToolchain] = useState<Toolchain>({});
  const [port, setPort] = useState('');
  const [playerId, setPlayerId] = useState(1);
  const [matchId, setMatchId] = useState('live');
  const [ssid, setSsid] = useState('');
  const [password, setPassword] = useState('');
  const [showPass, setShowPass] = useState(false);
  const [job, setJob] = useState<FlashJob | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [monitor, setMonitor] = useState<Monitor>({});
  const [squad, setSquad] = useState<Squad | null>(null);
  const prefilled = useRef(false);
  useEffect(() => { void getSquad().then(setSquad); }, []);

  const refreshHardware = async () => {
    try {
      const [portBody, toolBody] = await Promise.all([api.serialPorts(), api.hardwareStatus()]);
      const next = portBody as Port[];
      const tool = toolBody as Toolchain;
      setPorts(next);
      setToolchain(tool);
      const usable = next.filter(p => !p.windows_only);
      setPort(current => current && usable.some(p => p.device === current) ? current : usable[0]?.device ?? '');
      // Start from the last build's settings (never its password).
      if (!prefilled.current && tool.last_flash) {
        prefilled.current = true;
        if (tool.last_flash.wifi_ssid) setSsid(tool.last_flash.wifi_ssid);
        if (tool.last_flash.player_id) setPlayerId(tool.last_flash.player_id);
        if (tool.last_flash.match_id) setMatchId(tool.last_flash.match_id);
      }
      setError(null);
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Hardware API unavailable'); }
  };

  useEffect(() => { void refreshHardware(); }, []);
  useEffect(() => {
    const pull = () => api.flashStatus().then(body => setJob((body.active_job as FlashJob | null) ?? null)).catch(() => undefined);
    void pull();
    const timer = window.setInterval(pull, 1200);
    return () => window.clearInterval(timer);
  }, []);
  useEffect(() => {
    const pull = () => api.monitorStatus().then(m => setMonitor(m as Monitor)).catch(() => undefined);
    void pull();
    const timer = window.setInterval(pull, 1000);
    return () => window.clearInterval(timer);
  }, []);

  const running = Boolean(job && ['starting', 'compile', 'upload', 'running'].includes(job.status));
  // Open the serial monitor as soon as a vest is attached, unless a flash owns the port.
  const autoStarted = useRef('');
  useEffect(() => {
    if (!port || running || monitor.running || autoStarted.current === port) return;
    autoStarted.current = port;
    void api.monitorStart(port).catch(() => undefined);
  }, [port, running, monitor.running]);

  const flash = async () => {
    setBusy(true); setError(null);
    try {
      const response = await api.flash({ port, player_id: playerId, match_id: matchId, wifi_ssid: ssid, wifi_pass: password });
      if (response.ok === false) throw new Error(String(response.error ?? 'Flash did not start'));
      setJob((response.job as FlashJob) ?? null);
      setPassword('');
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Flash did not start'); }
    finally { setBusy(false); }
  };

  const attachUsb = async () => {
    setBusy(true); setError(null);
    try {
      const response = await api.usbAttach();
      if (response.ok === false) throw new Error(String(response.error ?? 'USB attach failed'));
      await refreshHardware();
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'USB attach failed'); }
    finally { setBusy(false); }
  };

  const realPorts = ports.filter(p => !p.windows_only);
  const hintPorts = ports.filter(p => p.windows_only);
  const current = realPorts.find(p => p.device === port);
  const espressif = (current?.hwid ?? '').toUpperCase().includes('303A');
  const wearer = squad?.players.find(p => p.vest === playerId);
  const missing = [
    !toolchain.arduino_cli && 'Arduino CLI is not installed',
    toolchain.relay && !toolchain.relay.token_configured && 'Relay token is not set (CPG44_RELAY_TOKEN)',
    toolchain.relay && !toolchain.relay.ca_configured && 'Relay certificates are missing (run scripts/prepare_relay_ca.sh)',
  ].filter(Boolean) as string[];
  const done = job?.status === 'completed';
  const failed = job?.status === 'failed' || Boolean(job?.error);
  const passOk = !password || (password.length >= 8 && password.length <= 63);
  const canFlash = Boolean(port) && !missing.length && Boolean(ssid) && passOk && !running && !busy;

  return <div className="page">
    <div className="top"><h1>Flash a vest</h1>
      <span className={`port-chip ${port ? 'on' : ''}`}><Glyph g="chip" size={15} />{port ? `${espressif ? 'ESP32-S3' : 'Board'} on ${port}` : 'No vest connected'}</span>
    </div>
    {error && <div className="notice bad">{error}</div>}
    {missing.map(m => <div key={m} className="notice bad">{m}</div>)}

    <div className="flash-layout">
      <div className="flash-steps-col">
        <section className={`panel step-panel ${port ? 'done' : ''}`}>
          <div className="panel-head"><h2><span className="step-n">1</span>Connect the vest</h2>
            <button className="btn ghost small" onClick={() => void refreshHardware()}>Rescan</button></div>
          {!realPorts.length ? <div className="notice">
            {hintPorts.length ? `Windows sees the vest on ${hintPorts[0].device}. Attach it to this app and accept the administrator prompt.` : 'Plug the vest in with a USB data cable (the port marked USB).'}
            <div style={{ marginTop: 8 }}><button className="btn small" disabled={busy} onClick={() => void attachUsb()}>Attach USB</button></div>
          </div> : realPorts.length > 1 ? <div className="row-fields">
            <label>Port<select value={port} onChange={e => setPort(e.target.value)}>{realPorts.map(p => <option key={p.device} value={p.device}>{p.device}, {p.description ?? p.name}</option>)}</select></label>
          </div> : <p className="muted step-note">Found {espressif ? 'an ESP32-S3' : current?.description} on {port}. Switch the vest's battery off while it is on USB.</p>}
        </section>

        <section className="panel step-panel">
          <div className="panel-head"><h2><span className="step-n">2</span>Vest number and Wi-Fi</h2></div>
          <div className="row-fields">
            <label>Vest number<input type="number" min="1" max="9999" value={playerId} onChange={e => setPlayerId(Number(e.target.value))} /></label>
            <span className="muted step-note">{wearer ? `Worn by ${wearer.number != null ? wearer.number + ' ' : ''}${wearer.name}` : <>Not in the squad yet. <Link to="/squad">Assign it</Link></>}</span>
          </div>
          <div className="row-fields">
            <label>Wi-Fi name<input autoComplete="off" spellCheck={false} value={ssid} onChange={e => setSsid(e.target.value)} placeholder="2.4 GHz network or hotspot" /></label>
            <label>Password
              <span className="pass-field">
                <input type={showPass ? 'text' : 'password'} autoComplete="new-password" value={password} onChange={e => setPassword(e.target.value)}
                  aria-invalid={!passOk} placeholder="blank for an open network" />
                <button type="button" className="icon-btn" onClick={() => setShowPass(v => !v)} aria-label={showPass ? 'Hide password' : 'Show password'}><Glyph g={showPass ? 'x' : 'eye'} size={14} /></button>
              </span>
            </label>
          </div>
          {!passOk && <p className="state-bad step-note">Wi-Fi passwords are 8 to 63 characters.</p>}
          <p className="muted step-note">The vest's radio is 2.4 GHz only. If it cannot join, the Bench check lists the networks it can see.</p>
          <details className="more"><summary>Session name</summary>
            <div className="row-fields"><label>Match ID<input value={matchId} maxLength={64} pattern="[A-Za-z0-9._:-]+" onChange={e => setMatchId(e.target.value)} /></label></div></details>
        </section>

        <section className={`panel step-panel ${done ? 'done' : ''}`}>
          <div className="panel-head"><h2><span className="step-n">3</span>Flash</h2>
            <button className="btn" disabled={!canFlash} onClick={() => void flash()}><Glyph g="chip" size={15} />{running ? 'Flashing' : 'Compile and flash'}</button></div>
          {job ? <div className={`flash-state ${done ? 'ok' : failed ? 'bad' : ''}`} aria-live="polite">
            <div className="flash-steps">{(['compile', 'upload', 'completed'] as const).map((k, n) => {
              const order = ['starting', 'compile', 'upload', 'completed'];
              const reached = order.indexOf(job.status) >= order.indexOf(k) || done;
              return <span key={k} className={reached ? 'on' : ''}>{n + 1}. {k === 'completed' ? 'running' : k}</span>; })}</div>
            {running && <progress className="spinner" aria-label="Flashing" />}
            {done && <p>Flashed. Watch the Bench check on the right, then <Link to="/vest">open Vest readings</Link>.</p>}
            {failed && <p>{job.error ?? 'The build or upload failed; the log below says where.'}</p>}
          </div> : <p className="muted step-note">Takes about a minute: compile, then upload over USB.</p>}
          {job?.logs?.length ? <details className="more" open={failed}><summary>Build log</summary>
            <div className="console">{job.logs.slice(-120).map((line, i) => <div key={`${i}-${line}`}>{line}</div>)}</div></details> : null}
        </section>
      </div>

      <BenchCheck monitor={monitor} port={port} onStart={() => { if (port) void api.monitorStart(port).catch(() => undefined); }} />
    </div>
  </div>;
};
