import { useEffect, useState } from 'react';
import { ErrorBanner, PageHeader, StatusBadge } from '../components/common';
import { api } from '../lib/api';

type Port = { device: string; name?: string; description?: string; hwid?: string };
type Toolchain = {
  ready?: boolean;
  arduino_cli?: string | null;
  esptool_available?: boolean;
  wsl_usb_required?: boolean;
  message?: string;
  relay?: { endpoint?: string; token_configured?: boolean; ca_configured?: boolean; ready?: boolean };
};
type FlashJob = { id: string; status: string; logs?: string[]; error?: string; return_code?: number; player_id: number; port: string };

export const HardwareFlasherPage = () => {
  const [ports, setPorts] = useState<Port[]>([]);
  const [toolchain, setToolchain] = useState<Toolchain>({});
  const [selectedPort, setSelectedPort] = useState('');
  const [playerId, setPlayerId] = useState(1);
  const [matchId, setMatchId] = useState('live');
  const [ssid, setSsid] = useState('');
  const [password, setPassword] = useState('');
  const [chip, setChip] = useState<Record<string, unknown> | null>(null);
  const [job, setJob] = useState<FlashJob | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshHardware = async () => {
    try {
      const [portBody, toolBody] = await Promise.all([api.serialPorts(), api.hardwareStatus()]);
      const nextPorts = portBody as Port[];
      setPorts(nextPorts);
      setToolchain(toolBody as Toolchain);
      setSelectedPort((current) => current && nextPorts.some((port) => port.device === current) ? current : nextPorts[0]?.device ?? '');
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
    setChip(null);
    if (!selectedPort) return;
    void api.chipInfo(selectedPort).then(setChip).catch((reason: unknown) => setError(reason instanceof Error ? reason.message : 'Chip probe failed'));
  }, [selectedPort]);

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

  const running = job && ['starting', 'compile', 'upload', 'running'].includes(job.status);
  return (
    <div className="page">
      <PageHeader eyebrow="DEVICE PROVISIONING" title="ESP32 wearable flasher" description="Flash the checked-in sensor firmware with Wi-Fi, player and relay settings. The wearable publishes only through the HTTPS relay." actions={<StatusBadge status={toolchain.ready ? 'ready' : 'unavailable'} />} />
      <ErrorBanner message={error ?? job?.error} />

      <div className="two-column">
        <section className="card-clean">
          <div className="section-heading"><div><span className="eyebrow">TOOLCHAIN</span><h2>WSL readiness</h2></div><button className="btn-outline" onClick={refreshHardware}>Rescan</button></div>
          <dl className="definition-grid">
            <div><dt>Arduino CLI</dt><dd>{toolchain.arduino_cli ? 'installed' : 'missing'}</dd></div>
            <div><dt>esptool</dt><dd>{toolchain.esptool_available ? 'installed' : 'missing'}</dd></div>
            <div><dt>Serial devices</dt><dd>{ports.length}</dd></div>
            <div><dt>Target</dt><dd>ESP32-S3</dd></div>
            <div><dt>Relay token</dt><dd>{toolchain.relay?.token_configured ? 'configured' : 'missing'}</dd></div>
            <div><dt>Relay CA</dt><dd>{toolchain.relay?.ca_configured ? 'configured' : 'missing'}</dd></div>
          </dl>
          {!ports.length && (
            <div className="quality-callout neutral">
              <strong>No ESP32 is attached to WSL</strong>
              <p>In an elevated Windows terminal run <code>usbipd list</code>, then attach the board bus ID with <code>usbipd attach --wsl --busid &lt;BUSID&gt;</code>. Rescan after <code>/dev/ttyACM*</code> or <code>/dev/ttyUSB*</code> appears.</p>
            </div>
          )}
          <div className="form-grid">
            <label className="field wide"><span>Serial port</span><select value={selectedPort} onChange={(event) => setSelectedPort(event.target.value)} disabled={!ports.length}>{!ports.length && <option value="">No attached serial device</option>}{ports.map((port) => <option key={port.device} value={port.device}>{port.device} · {port.description ?? port.name}</option>)}</select></label>
            <label className="field"><span>Player ID</span><input type="number" min="1" max="9999" value={playerId} onChange={(event) => setPlayerId(Number(event.target.value))} /></label>
            <label className="field"><span>Match ID</span><input value={matchId} maxLength={64} pattern="[A-Za-z0-9._:-]+" onChange={(event) => setMatchId(event.target.value)} /></label>
            <label className="field"><span>Wi-Fi SSID</span><input autoComplete="off" value={ssid} onChange={(event) => setSsid(event.target.value)} /></label>
            <label className="field wide"><span>Wi-Fi password</span><input type="password" autoComplete="new-password" value={password} onChange={(event) => setPassword(event.target.value)} /></label>
          </div>
          <p className="form-help">Target: <code>{toolchain.relay?.endpoint ?? 'https://cpg44.nivaspms.com/api/v1/sensors/ingest'}</code>. Secrets are written only to the ignored build header and are not returned to the browser.</p>
          <button className="btn-solid" disabled={busy || !toolchain.ready || !selectedPort || !ssid || !matchId || Boolean(running)} onClick={flash}>{running ? `${job?.status}…` : 'Compile and flash relay firmware'}</button>
        </section>

        <section className="card-clean console-card">
          <div className="section-heading"><div><span className="eyebrow">SERIAL PROBE</span><h2>Hardware and build log</h2></div>{job && <StatusBadge status={job.status} />}</div>
          {chip && <div className="chip-summary">{chip.ok === true ? String(chip.output ?? 'ESP32 detected') : String(chip.error ?? 'Chip probe unavailable')}</div>}
          <div className="console" aria-live="polite">
            {job?.logs?.length ? job.logs.map((line, index) => <div key={`${index}-${line}`}>{line}</div>) : <span>Waiting for a compile or upload job.</span>}
          </div>
        </section>
      </div>
    </div>
  );
};
