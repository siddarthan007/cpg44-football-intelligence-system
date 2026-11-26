import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Glyph } from '../components/Glyph';
import { api, apiUrl } from '../lib/api';

type Status = {
  connected?: boolean; clock_synced?: boolean; transport_delay_ms?: number | null; dropped_samples?: number | null;
  gyro_calibrated?: boolean; accel_scale_calibrated?: boolean; rssi_dbm?: number | null;
  rates_hz?: Record<string, number>; expected_hz?: Record<string, number | null | undefined>;
  clock_offset?: { offset_ms: number; uncertainty_ms: number; samples: number } | null;
  battery?: { pct: number; volts: number | null; minutes_left: number | null; level: string } | null;
  gps?: { fix?: boolean; sat?: number; hdop?: number; filtered?: { speed_mps: number; distance_m: number; pos_sigma_m: number } };
  imu?: { accel_cal_poses?: number; accel_calibrated?: boolean; player_load?: number; player_load_per_min?: number | null; stationary?: boolean };
  parts?: { imu?: boolean | null; magnetometer?: boolean | null; ecg?: boolean | null; gnss?: boolean | null };
  heart?: { bpm?: number | null; steady_bpm?: number | null; rmssd_ms?: number | null; quality?: number | null; reason?: string | null; electrodes?: 'on' | 'off' | null };
};
type Trace = { ok: boolean; fs?: number; values?: number[]; range?: [number, number]; beats?: number[]; lead_off?: boolean };

const DESIGN = { imu: 50, ecg: 5, gps: 5 };
const REASON: Record<string, string> = {
  electrode_off: 'an electrode is off the skin',
  flat_signal: 'no signal: check the cable into the ECG board',
  noisy_ecg: 'too much interference for clean beats',
  too_few_beats: 'finding the first beats',
  ecg_stale: 'the ECG stopped arriving',
  no_ecg: 'waiting for the ECG',
};

/** Live chest ECG on a paper grid, with each detected beat marked. */
function EcgStrip({ trace }: { trace?: Trace }) {
  const W = 1000, H = 220, pad = 18;
  if (!trace?.ok || !trace.values?.length) return <div className="ecg-strip empty"><span>Waiting for the ECG</span></div>;
  // With an electrode off the AD8232 swings between its limits; drawing that is
  // noise, not a heart, so the strip says so instead.
  if (trace.lead_off) return <div className="ecg-strip empty off"><span><strong>Electrodes off</strong>Press the three pads onto the skin; the trace starts by itself.</span></div>;
  const [lo, hi] = trace.range ?? [Math.min(...trace.values), Math.max(...trace.values)];
  const span = Math.max(hi - lo, 1), n = trace.values.length, secs = n / (trace.fs || 125);
  const x = (i: number) => (i / Math.max(n - 1, 1)) * W;
  const y = (v: number) => H - pad - ((Math.min(Math.max(v, lo), hi) - lo) / span) * (H - 2 * pad);
  const points = trace.values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  return <div className={`ecg-strip ${trace.lead_off ? 'off' : ''}`}>
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label={`Chest ECG, last ${secs.toFixed(0)} seconds, ${trace.beats?.length ?? 0} beats`}>
      {(trace.beats ?? []).map(b => <line key={b} x1={(b / secs) * W} x2={(b / secs) * W} y1={4} y2={12} className="beat" />)}
      <polyline points={points} className="trace" />
    </svg>
    <span className="ecg-scale">{secs.toFixed(0)} s</span>
  </div>;
}

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
  return <div className="rate" title={`${kind}: ${got ?? 0} of ${want} per second arrive`}>
    <span>{kind}</span>
    <div className="rate-bar"><i className={tone} style={{ width: `${share * 100}%` }} /></div>
    <b className="num">{got == null ? 'none' : `${Math.round(share * 100)} %`}</b>
  </div>;
}

export const WearablePage = () => {
  const [s, setS] = useState<Status>({});
  const [trace, setTrace] = useState<Trace>();
  const [down, setDown] = useState(false);

  useEffect(() => {
    let live = true;
    const pull = () => api.wearableStatus().then(r => { if (live) { setS(r as Status); setDown(false); } }).catch(() => live && setDown(true));
    void pull();
    const t = window.setInterval(pull, 1000);
    return () => { live = false; window.clearInterval(t); };
  }, []);
  const on = Boolean(s.connected) && !down;
  useEffect(() => {
    if (!on) return;
    let live = true;
    const pull = () => fetch(apiUrl('/api/v1/wearables/ecg?seconds=5')).then(r => r.json()).then(tr => { if (live) setTrace(tr as Trace); }).catch(() => undefined);
    void pull();
    const t = window.setInterval(pull, 500);
    return () => { live = false; window.clearInterval(t); };
  }, [on]);

  const want = { imu: s.expected_hz?.imu || DESIGN.imu, ecg: s.expected_hz?.ecg || DESIGN.ecg, gps: s.expected_hz?.gps || DESIGN.gps };
  const off = s.clock_offset;
  const poses = s.imu?.accel_cal_poses ?? 0;
  const h = s.heart ?? {};
  const electrodesOff = h.electrodes === 'off' || trace?.lead_off;

  return <div className="page">
    <header className="page-header"><h1>Vest readings</h1></header>

    <section className="panel" aria-label="Link">
      <div className="chain">
        <div className={`node ${on ? 'ok' : 'off'}`}><Glyph g="vest" size={22} /><b>Vest</b>
          <small>{on ? (s.battery ? `${Math.round(s.battery.pct)} % battery${s.battery.minutes_left != null ? `, ${s.battery.minutes_left} min left` : ''}` : 'battery not measured') : 'not heard from'}</small></div>
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
    </section>

    {on && <section className="panel heart-panel" aria-label="Heart">
      <div className="heart-figures">
        <div className="heart-now"><small>Heart rate</small>
          <b className="num">{h.bpm != null ? Math.round(h.bpm) : '--'}<i>bpm</i></b>
          <span>{h.bpm != null ? (h.steady_bpm != null ? `steady ${Math.round(h.steady_bpm)} bpm` : 'settling') : REASON[h.reason ?? ''] ?? 'waiting for clean beats'}</span></div>
        <div><small>Variability</small><b className="num">{h.rmssd_ms != null ? Math.round(h.rmssd_ms) : '--'}<i>ms</i></b>
          <span>{h.rmssd_ms != null ? 'RMSSD, still' : 'measured when still for 30 s'}</span></div>
        <div><small>Electrodes</small><b className={`word ${electrodesOff ? 'state-bad' : h.electrodes === 'on' ? 'state-ok' : ''}`}>{electrodesOff ? 'Off' : h.electrodes === 'on' ? 'On' : '--'}</b>
          <span>{electrodesOff ? 'press the three pads onto the skin' : h.quality != null ? `${Math.round(h.quality * 100)} % clean beats` : ''}</span></div>
      </div>
      <EcgStrip trace={trace} />
    </section>}

    {on && <div className="vgrid">
      <Tile label="PlayerLoad per minute" value={s.imu?.player_load_per_min != null ? s.imu.player_load_per_min.toFixed(1) : null}
        why={s.imu?.player_load != null ? `${s.imu.player_load.toFixed(1)} this session` : undefined} />
      <Tile label="Speed" value={s.gps?.filtered ? (s.gps.filtered.speed_mps * 3.6).toFixed(1) : null} unit=" km/h"
        why={s.gps?.fix ? `${s.gps.sat ?? 0} satellites, HDOP ${s.gps.hdop ?? '?'}` : s.parts?.gnss === false ? 'the GPS module sends nothing: check its wiring' : 'no GPS fix yet; the camera still measures speed'} />
      <Tile label="Distance" value={s.gps?.filtered ? s.gps.filtered.distance_m.toFixed(0) : null} unit=" m"
        why={s.gps?.filtered ? `position within ${s.gps.filtered.pos_sigma_m.toFixed(1)} m` : undefined} />
    </div>}

    {on && <section className="panel" aria-label="Samples arriving">
      <div className="panel-head"><h2>Samples arriving</h2></div>
      <div className="rates">
        <Rate kind="Movement" got={s.rates_hz?.imu} want={want.imu} />
        <Rate kind="ECG" got={s.rates_hz?.ecg} want={want.ecg} />
        {s.parts?.gnss === false ? <div className="rate"><span>GPS</span><div className="rate-bar"><i className="bad" style={{ width: '0%' }} /></div><b className="num">no data</b></div>
          : <Rate kind="GPS fixes" got={s.gps?.fix ? s.rates_hz?.gps : 0} want={want.gps} />}
        {(s.dropped_samples ?? 0) > 0 && <p className="muted" style={{ fontSize: 12.5 }}>The vest reports {s.dropped_samples} samples dropped since it started.</p>}
      </div>
    </section>}

    <section className="panel" aria-label="Calibration">
      <div className="panel-head"><h2>Calibration</h2></div>
      <div className="cal-rows">
        <div className="cal-row"><Glyph g={s.gyro_calibrated ? 'check' : 'warn'} size={16} />
          <div><b>Gyroscope</b><span>{s.gyro_calibrated ? 'Offset measured while still; tracked as the vest warms up.' : 'Lay the vest flat and still for two seconds.'}</span></div></div>
        <div className="cal-row"><Glyph g={s.accel_scale_calibrated ? 'check' : 'warn'} size={16} />
          <div><b>Accelerometer</b><span>{s.accel_scale_calibrated ? 'Offset and scale fitted from resting poses.' : `Rest the pod still on each of its six faces for two seconds. ${poses} of 6 poses so far.`}</span>
            {!s.accel_scale_calibrated && <div className="poses" aria-hidden="true">{Array.from({ length: 6 }, (_, k) => <i key={k} className={k < poses ? 'got' : ''} />)}</div>}</div></div>
      </div>
    </section>
    <p className="muted" style={{ fontSize: 12 }}>Coaching measurements, not medical readings.</p>
  </div>;
};
