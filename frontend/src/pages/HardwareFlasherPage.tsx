import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { Glyph } from '../components/Glyph';
import { api } from '../lib/api';
import { getSquad, type Squad } from '../lib/squad';

type Port = { device: string; name?: string; description?: string; hwid?: string; windows_only?: boolean };
type Toolchain = {
  ready?: boolean;
  arduino_cli?: string | null;
  esptool_available?: boolean;
  wsl_usb_required?: boolean;
  message?: string;
  relay?: { endpoint?: string; token_configured?: boolean; ca_configured?: boolean; ready?: boolean };
};
type FlashJob = { id: string; status: string; logs?: string[]; error?: string; return_code?: number; player_id: number; port: string };
type MonitorLine = { t: number; text: string };
type MonitorState = { port?: string; running?: boolean; lines?: MonitorLine[] };

export const HardwareFlasherPage = () => {
  const [ports, setPorts] = useState<Port[]>([]);
  const [toolchain, setToolchain] = useState<Toolchain>({});
  const [selectedPort, setSelectedPort] = useState('');
  const [playerId, setPlayerId] = useState(1);
  const [matchId, setMatchId] = useState('live');
  const [ssid, setSsid] = useState('EACCESS-M1');
  const [password, setPassword] = useState('');
  const [chip, setChip] = useState<Record<string, unknown> | null>(null);
  const [job, setJob] = useState<FlashJob | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [monitor, setMonitor] = useState<MonitorState>({});
  const [monitorAuto, setMonitorAuto] = useState(true);
  const [squad, setSquad] = useState<Squad | null>(null);
  useEffect(() => { void getSquad().then(setSquad); }, []);
  const consoleRef = useRef<HTMLDivElement | null>(null);
  const monitorRef = useRef<HTMLDivElement | null>(null);

  const refreshHardware = async () => {
    try {
      const [portBody, toolBody] = await Promise.all([api.serialPorts(), api.hardwareStatus()]);
      const nextPorts = portBody as Port[];
      setPorts(nextPorts);
      setToolchain(toolBody as Toolchain);
      // Windows-only ports (COM7) are hints, not usable devices inside WSL.
      const usable = nextPorts.filter((port) => !port.windows_only);
      setSelectedPort((current) => current && usable.some((port) => port.device === current) ? current : usable[0]?.device ?? '');
      setError(null);
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Hardware API unavailable'); }
  };

  useEffect(() => { void refreshHardware(); }, []);
  useEffect(() => {
    const pull = () => api.flashStatus().then((body) => setJob((body.active_job as FlashJob | null) ?? null)).catch(() => undefined);
    void pull();
    const timer = window.setInterval(pull, 1200);
    return () => window.clearInterval(timer);
  }, []);
  useEffect(() => {
    const pull = () => api.monitorStatus().then(setMonitor).catch(() => undefined);
    void pull();
    const timer = window.setInterval(pull, 1000);
    return () => window.clearInterval(timer);
  }, []);
  useEffect(() => {
    setChip(null);
    if (!selectedPort) return;
    void api.chipInfo(selectedPort).then(setChip).catch((reason: unknown) => setError(reason instanceof Error ? reason.message : 'Chip probe failed'));
  }, [selectedPort]);

  // Auto-scroll both consoles unless the reader scrolled up.
  useEffect(() => {
    if (consoleRef.current) consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
  }, [job?.logs?.length]);
  useEffect(() => {
    if (monitorAuto && monitorRef.current) monitorRef.current.scrollTop = monitorRef.current.scrollHeight;
  }, [monitor.lines?.length, monitorAuto]);

  // Auto-start the monitor when the flash job finishes so the boot log appears.
  const lastJobStatus = useRef<string>('');
  useEffect(() => {
    const status = job?.status ?? '';
    const finished = (lastJobStatus.current === 'compile' || lastJobStatus.current === 'upload') && status === 'completed';
    lastJobStatus.current = status;
    if (finished && selectedPort && monitorAuto) {
      void api.monitorStart(selectedPort).catch(() => undefined);
    }
  }, [job?.status, selectedPort, monitorAuto]);

  const flash = async () => {
    setBusy(true); setError(null);
    try {
      const response = await api.flash({ port: selectedPort, player_id: playerId, match_id: matchId, wifi_ssid: ssid, wifi_pass: password });
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

  const toggleMonitor = async () => {
    if (monitor.running) {
      await api.monitorStop().catch(() => undefined);
      return;
    }
    if (!selectedPort) return;
    try {
      const response = await api.monitorStart(selectedPort);
      if (response.ok === false) throw new Error(String(response.error ?? 'Monitor did not start'));
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Monitor did not start'); }
  };

  const running = job && ['starting', 'compile', 'upload', 'running'].includes(job.status);
  const realPorts = ports.filter((port) => !port.windows_only);
  const hintPorts = ports.filter((port) => port.windows_only);
  const wearer = squad?.players.find(p => p.vest === playerId);
  const checks: [string, boolean][] = [
    ['Arduino CLI', Boolean(toolchain.arduino_cli)], ['esptool', Boolean(toolchain.esptool_available)],
    ['Relay token', Boolean(toolchain.relay?.token_configured)], ['Relay certificate', Boolean(toolchain.relay?.ca_configured)],
  ];
  const connected = Boolean(selectedPort) && checks.every(([, ok]) => ok);
  const done = job?.status === 'completed';
  const failed = job?.status === 'failed' || Boolean(job?.error);
  return (
    <div className="page">
      <header className="page-header"><h1>Flash a vest</h1><p>Load the relay firmware onto a vest and give it its number and Wi-Fi.</p></header>
      {error && <div className="notice bad">{error}</div>}

      <section className={`panel step-panel ${connected ? 'done' : ''}`}>
        <div className="panel-head"><h2><span className="step-n">1</span>Connect the vest</h2>
          <button className="btn ghost small" onClick={() => void refreshHardware()}>Rescan</button></div>
        <ul className="checks">{checks.map(([label, ok]) => <li key={label} className={ok ? 'ok' : 'bad'}><Glyph g={ok ? 'check' : 'warn'} size={14} />{label}</li>)}</ul>
        {!realPorts.length ? <div className="notice" style={{ marginTop: 10 }}>
          {hintPorts.length ? `Windows sees the vest on ${hintPorts[0].device}. Attach it to WSL and accept the administrator prompt.` : 'Plug the vest in with a data USB cable, then attach it to WSL.'}
          <div style={{ marginTop: 8 }}><button className="btn small" disabled={busy} onClick={() => void attachUsb()}>Attach USB to WSL</button></div>
        </div> : <div className="row-fields">
          <label>Port<select value={selectedPort} onChange={(e) => setSelectedPort(e.target.value)}>{realPorts.map((port) => <option key={port.device} value={port.device}>{port.device}, {port.description ?? port.name}</option>)}</select></label>
          {chip && <span className="muted" style={{ fontSize: 12.5 }}>{chip.ok === true ? 'ESP32 answered' : String(chip.error ?? 'Chip probe unavailable')}</span>}
        </div>}
      </section>

      <section className="panel step-panel">
        <div className="panel-head"><h2><span className="step-n">2</span>Vest number and Wi-Fi</h2></div>
        <div className="row-fields">
          <label>Vest number<input type="number" min="1" max="9999" value={playerId} onChange={(e) => setPlayerId(Number(e.target.value))} /></label>
          <span className="muted" style={{ fontSize: 13 }}>{wearer ? `Worn by ${wearer.number != null ? wearer.number + ' ' : ''}${wearer.name}` : <>Not in the squad yet. <Link to="/squad">Assign it</Link></>}</span>
        </div>
        <div className="row-fields">
          <label>Wi-Fi name<input autoComplete="off" value={ssid} onChange={(e) => setSsid(e.target.value)} /></label>
          <label>Wi-Fi password<input type="password" autoComplete="new-password" value={password} onChange={(e) => setPassword(e.target.value)} /></label>
        </div>
        <details className="more"><summary>Session name</summary>
          <div className="row-fields"><label>Match ID<input value={matchId} maxLength={64} pattern="[A-Za-z0-9._:-]+" onChange={(e) => setMatchId(e.target.value)} /></label></div></details>
        <p className="muted" style={{ fontSize: 12 }}>The password goes only into the firmware build and is never sent back to the browser.</p>
      </section>

      <section className={`panel step-panel ${done ? 'done' : ''}`}>
        <div className="panel-head"><h2><span className="step-n">3</span>Flash</h2>
          <button className="btn" disabled={busy || !toolchain.ready || !selectedPort || !ssid || !matchId || Boolean(running)} onClick={() => void flash()}>
            <Glyph g="chip" size={15} />{running ? 'Flashing' : 'Compile and flash'}</button></div>
        {job && <div className={`flash-state ${done ? 'ok' : failed ? 'bad' : ''}`} aria-live="polite">
          <div className="flash-steps">{['compile', 'upload', 'completed'].map((k, n) => {
            const order = ['starting', 'compile', 'upload', 'completed'];
            const reached = order.indexOf(job.status) >= order.indexOf(k) || done;
            return <span key={k} className={reached ? 'on' : ''}>{n + 1}. {k === 'completed' ? 'running' : k}</span>; })}</div>
          {done && <p>Flashed. The vest restarts and sends within a few seconds: <Link to="/vest">watch it arrive</Link>.</p>}
          {failed && <p>{job.error ?? 'The build or upload failed; the log below says where.'}</p>}
        </div>}
        <details className="more" open={failed}><summary>Build log</summary>
          <div className="console" ref={consoleRef}>{job?.logs?.length ? job.logs.map((line, index) => <div key={`${index}-${line}`}>{line}</div>) : <span>No build yet.</span>}</div></details>
        <details className="more"><summary>Serial monitor {monitor.running ? `(reading ${monitor.port})` : ''}</summary>
          <div className="row-fields" style={{ marginTop: 6 }}>
            <button className="btn ghost small" onClick={() => void toggleMonitor()} disabled={!selectedPort}>{monitor.running ? 'Stop' : 'Start'}</button>
            <label className="inline"><input type="checkbox" checked={monitorAuto} onChange={(e) => setMonitorAuto(e.target.checked)} /> open after flashing</label>
          </div>
          <div className="console" ref={monitorRef} style={{ maxHeight: 320 }}>{monitor.lines?.length ? monitor.lines.map((line, index) => <div key={`${index}-${line.t}`}>{line.text}</div>) : <span>No serial output yet.</span>}</div>
        </details>
      </section>
    </div>
  );
};
