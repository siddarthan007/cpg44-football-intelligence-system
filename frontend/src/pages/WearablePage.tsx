import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Glyph } from '../components/Glyph';
import { api, apiUrl } from '../lib/api';

type Status = {
  connected?: boolean; clock_synced?: boolean; transport_delay_ms?: number | null; dropped_samples?: number | null;
  gyro_calibrated?: boolean; accel_scale_calibrated?: boolean; ppg_reason?: string | null; rssi_dbm?: number | null;
  rates_hz?: Record<string, number>; expected_hz?: Record<string, number | null | undefined>;
  clock_offset?: { offset_ms: number; uncertainty_ms: number; samples: number } | null;
  battery?: { pct: number; volts: number | null; minutes_left: number | null; level: string } | null;
  gps?: { fix?: boolean; sat?: number; hdop?: number; filtered?: { speed_mps: number; distance_m: number; pos_sigma_m: number } };
  imu?: { accel_cal_poses?: number; accel_calibrated?: boolean; player_load?: number; player_load_per_min?: number | null; stationary?: boolean };
  pulse?: { bpm?: number | null; spo2?: number | null; quality?: number | null; reason?: string | null; spo2_calibration?: { mode: string; pairs: number } };
};

const DESIGN = { imu: 50, ppg: 25, gps: 5 };
const REASON: Record<string, string> = {
  stale_or_discontinuous_pulse: 'samples missing in the pulse window',
  no_or_weak_contact: 'sensor not pressed on the skin',
  poor_optical_quality: 'weak optical signal',
  motion_artifact: 'too much movement for a clean pulse',
  hr_estimators_disagree: 'two pulse estimates disagree',
  collecting: 'collecting the first seconds',
};

function Tile({ label, value, unit, why, accent }: { label: string; value?: string | null; unit?: string; why?: string; accent?: boolean }) {
  return <div className={`vtile ${value ? '' : 'missing'}`}>
    <small>{label}</small>
    {value ? <b className={`num ${accent ? 'pulse' : ''}`}>{value}<i>{unit}</i></b> : <b className="num none">no value</b>}
    {why && <span>{why}</span>}
  </div>;
}

function Rate({ kind, got, want }: { kind: string; got?: number; want: number }) {
  const share = got == null ? 0 : Math.min(1, got / want);
  const tone = got == null ? 'off' : share >= 0.95 ? 'ok' : share >= 0.7 ? 'warn' : 'bad';
  return <div className="rate" title={`${kind}: ${got ?? 0} of ${want} samples per second arrive`}>
    <span>{kind}</span>
    <div className="rate-bar"><i className={tone} style={{ width: `${share * 100}%` }} /></div>
    <b className="num">{got == null ? 'none' : `${Math.round(share * 100)} %`}</b>
  </div>;
}

export const WearablePage = () => {
  const [s, setS] = useState<Status>({});
  const [down, setDown] = useState(false);
  const [ref, setRef] = useState('');
  const [calMsg, setCalMsg] = useState<{ ok: boolean; text: string }>();

  useEffect(() => {
    let live = true;
    const pull = () => api.wearableStatus().then(r => { if (live) { setS(r as Status); setDown(false); } }).catch(() => live && setDown(true));
    void pull();
    const t = window.setInterval(pull, 1000);
    return () => { live = false; window.clearInterval(t); };
  }, []);

  const on = Boolean(s.connected) && !down;
  const want = { imu: s.expected_hz?.imu || DESIGN.imu, ppg: s.expected_hz?.ppg || DESIGN.ppg, gps: s.expected_hz?.gps || DESIGN.gps };
  const off = s.clock_offset;
  const poses = s.imu?.accel_cal_poses ?? 0;
  const cal = s.pulse?.spo2_calibration;
  const still = Boolean(s.imu?.stationary);

  const calibrate = async () => {
    setCalMsg(undefined);
    const r = await fetch(apiUrl('/api/v1/wearables/calibrate/spo2'), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ reference_pct: Number(ref) }) });
    const body = await r.json().catch(() => ({}));
    setCalMsg(r.ok ? { ok: true, text: `Saved. ${body.calibration?.pairs ?? 1} reading${(body.calibration?.pairs ?? 1) > 1 ? 's' : ''} paired.` }
      : { ok: false, text: (body as { detail?: string }).detail ?? 'Not saved' });
    if (r.ok) setRef('');
  };

  return <div className="page">
    <header className="page-header"><h1>Vest readings</h1>
      <p>What the vest measures right now, and whether each number can be trusted.</p></header>

    <section className="panel" aria-label="Link">
      <div className="chain">
        <div className={`node ${on ? 'ok' : 'off'}`}><Glyph g="vest" size={22} /><b>Vest</b>
          <small>{on ? (s.battery ? `${Math.round(s.battery.pct)} % battery${s.battery.minutes_left != null ? `, ${s.battery.minutes_left} min left` : ''}` : 'battery not reported') : 'not heard from'}</small></div>
        <div className="link"><span>{s.rssi_dbm != null && on ? `Wi-Fi ${s.rssi_dbm} dBm` : ''}</span></div>
        <div className={`node ${on ? 'ok' : 'off'}`}><Glyph g="sources" size={22} /><b>Relay</b>
          <small>{s.transport_delay_ms != null && on ? `${Math.round(s.transport_delay_ms)} ms from vest to here` : 'no samples passing'}</small></div>
        <div className="link"><span>{off ? `clock ${off.offset_ms > 0 ? '+' : ''}${off.offset_ms} ms, ±${off.uncertainty_ms}` : ''}</span></div>
        <div className={`node ${down ? 'off' : 'ok'}`}><Glyph g="runtime" size={22} /><b>This laptop</b>
          <small>{down ? 'API not running' : off ? 'clock aligned to the relay' : 'measuring clock offset'}</small></div>
      </div>
      {!on && <div className="notice" style={{ marginTop: 14 }}>
        {down ? 'Start the system with scripts/start_system.sh.' : <>No vest data yet. Switch the vest on near Wi-Fi; it sends through the relay within a few seconds. A new vest needs <Link to="/vest/setup">flashing</Link> first.</>}
      </div>}
      {on && <div className="rates">
        <h3>Samples arriving</h3>
        <Rate kind="Movement (IMU)" got={s.rates_hz?.imu} want={want.imu} />
        <Rate kind="Pulse (PPG)" got={s.rates_hz?.ppg} want={want.ppg} />
        <Rate kind="GNSS fixes" got={s.rates_hz?.gps} want={want.gps} />
        {(s.dropped_samples ?? 0) > 0 && <p className="muted" style={{ fontSize: 12.5 }}>The vest reports {s.dropped_samples} samples dropped since it started.</p>}
      </div>}
    </section>

    {on && <div className="vgrid">
      <Tile label="Heart rate" accent value={s.pulse?.bpm != null ? `${Math.round(s.pulse.bpm)}` : null} unit=" bpm"
        why={s.pulse?.bpm != null ? `signal quality ${Math.round((s.pulse.quality ?? 0) * 100)} %` : on ? REASON[s.pulse?.reason ?? ''] ?? 'waiting for a clean pulse window' : undefined} />
      <Tile label="SpO2" accent value={s.pulse?.spo2 != null ? `${s.pulse.spo2.toFixed(0)}` : null} unit=" %"
        why={s.pulse?.spo2 != null ? (cal?.mode === 'uncalibrated' ? 'uncalibrated estimate' : `calibrated, ${cal?.pairs} readings`) : on ? 'only measured while standing still' : undefined} />
      <Tile label="PlayerLoad per minute" value={s.imu?.player_load_per_min != null ? s.imu.player_load_per_min.toFixed(1) : null}
        why={s.imu?.player_load != null ? `${s.imu.player_load.toFixed(1)} this session` : undefined} />
      <Tile label="Speed" value={s.gps?.filtered ? (s.gps.filtered.speed_mps * 3.6).toFixed(1) : null} unit=" km/h"
        why={s.gps?.fix ? `${s.gps.sat ?? 0} satellites, HDOP ${s.gps.hdop ?? '?'}` : on ? 'GNSS has no fix; the camera still measures speed' : undefined} />
    </div>}

    <section className="panel" aria-label="Calibration">
      <div className="panel-head"><h2>Calibration</h2></div>
      <div className="cal-rows">
        <div className="cal-row"><Glyph g={s.gyro_calibrated ? 'check' : 'warn'} size={16} />
          <div><b>Gyroscope</b><span>{s.gyro_calibrated ? 'Offset measured while still; tracked as the vest warms up.' : 'Lay the vest flat and still for two seconds.'}</span></div></div>
        <div className="cal-row"><Glyph g={s.accel_scale_calibrated ? 'check' : 'warn'} size={16} />
          <div><b>Accelerometer</b><span>{s.accel_scale_calibrated ? 'Offset and scale fitted from resting poses.' : `Rest the pod still on each of its six faces for two seconds. ${poses} of 6 poses so far.`}</span>
            {!s.accel_scale_calibrated && <div className="poses" aria-hidden="true">{Array.from({ length: 6 }, (_, k) => <i key={k} className={k < poses ? 'got' : ''} />)}</div>}</div></div>
        <div className="cal-row"><Glyph g={cal && cal.mode !== 'uncalibrated' ? 'check' : 'warn'} size={16} />
          <div><b>SpO2</b><span>{cal && cal.mode !== 'uncalibrated' ? `Matched to a finger oximeter (${cal.pairs} reading${cal.pairs > 1 ? 's' : ''}).` : 'Clip a finger pulse oximeter on, sit still, and enter its reading.'}</span>
            <div className="cal-form">
              <input aria-label="Finger oximeter reading" inputMode="decimal" placeholder="98" value={ref} onChange={e => setRef(e.target.value)} />
              <button className="btn small" disabled={!on || !still || !ref} onClick={() => void calibrate()} title={!still ? 'The player must be still' : undefined}>Pair reading</button>
              {calMsg && <span className={calMsg.ok ? 'state-ok' : 'state-bad'}>{calMsg.text}</span>}
            </div></div></div>
      </div>
    </section>
    <p className="muted" style={{ fontSize: 12 }}>Coaching measurements, not medical readings. SpO2 from a chest sensor is an estimate even when calibrated.</p>
  </div>;
};
