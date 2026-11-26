import { useEffect, useState } from 'react';
import { Glyph } from '../components/Glyph';
import { apiUrl } from '../lib/api';

type J = Record<string, any>;
interface Row { label: string; before: number; after: number; unit?: string; note?: string }

/** Two-series horizontal bars on one 0..100 scale: before in grey, now in ink. HTML, so text stays readable. */
function BeforeAfter({ rows }: { rows: Row[] }) {
  return <figure className="ev-chart" style={{ margin: 0 }}>
    <div className="ev-legend"><span><i style={{ background: '#b9b3a5' }} />before</span><span><i style={{ background: 'var(--ink)' }} />now</span></div>
    <div className="ba">
      {rows.map(r => <div key={r.label} className="ba-row" title={`${r.label}: ${r.before.toFixed(1)} to ${r.after.toFixed(1)} (${r.note})`}>
        <span className="ba-label">{r.label}</span>
        <span className="ba-bars">
          <i className="before" style={{ width: `${r.before}%` }} />
          <i className="after" style={{ width: `${r.after}%` }} />
        </span>
        <span className="ba-val num"><small>{r.before.toFixed(0)}</small>{r.after.toFixed(1)}</span>
      </div>)}
      <div className="ba-row ba-axis"><span /><span className="ba-ticks">{[0, 25, 50, 75, 100].map(v => <em key={v} style={{ left: `${v}%` }}>{v}</em>)}</span><span /></div>
    </div>
    <details className="ev-table"><summary>Table</summary><table><thead><tr><th>Measure</th><th className="num">Before</th><th className="num">Now</th><th>Data</th></tr></thead>
      <tbody>{rows.map(r => <tr key={r.label}><td>{r.label}</td><td className="num">{r.before.toFixed(1)}</td><td className="num">{r.after.toFixed(1)}</td><td>{r.note}</td></tr>)}</tbody></table></details>
  </figure>;
}

function Tile({ g, value, label }: { g: Parameters<typeof Glyph>[0]['g']; value: string; label: string }) {
  return <div className="panel" style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
    <span className="muted"><Glyph g={g} size={22} /></span>
    <div><div className="num" style={{ fontSize: value.length > 9 ? 17 : 22, fontWeight: 700, whiteSpace: 'nowrap' }}>{value}</div><div className="muted" style={{ fontSize: 12.5 }}>{label}</div></div>
  </div>;
}

export const EvidencePage = () => {
  const [e, setE] = useState<J>();
  const [err, setErr] = useState<string>();
  useEffect(() => { fetch(apiUrl('/api/v1/evidence')).then(r => r.json()).then(setE).catch(x => setErr(String(x))); }, []);
  if (err) return <div className="page"><p className="notice bad">{err}</p></div>;
  if (!e) return <div className="page"><p className="loading-line">Reading results</p></div>;

  const trk = (e[Object.keys(e).find(k => k.startsWith('tracking_test')) ?? ''] ?? {}) as J;
  const tr = trk.tracking ?? {};
  const old = tr['bytetrack (production)'], now = tr['imm-eiou+cmc+reid +link'];
  const det = e.detector_test_soccernet ?? {}, cmp = e.detector_compare ?? {};
  const bestDet = Object.entries(cmp).sort((a: any, b: any) => (b[1].map50_95 ?? 0) - (a[1].map50_95 ?? 0))[0]?.[1] as J | undefined;
  const nowDet = bestDet && bestDet.map50_95 > (det.soccernet?.map50_95 ?? 0) ? bestDet : det.soccernet;
  const reid = e.reid_val, teams = e.team_separation_val, phys = e.physics_synthetic, vest = e.vest_recorded_sessions, today = e.vest_delivery_now;

  const rows: Row[] = [];
  if (old && now) {
    rows.push({ label: 'Tracking HOTA', before: old.HOTA * 100, after: now.HOTA * 100, note: '49 test clips' });
    rows.push({ label: 'Identity IDF1', before: old.IDF1 * 100, after: now.IDF1 * 100, note: '49 test clips' });
  }
  if (reid) rows.push({ label: 'Re-identification mAP', before: reid.colour_histogram.mAP * 100, after: reid.reid_r18.mAP * 100, note: '9 validation clips' });
  if (teams) rows.push({ label: 'Side assignment', before: teams.frame_vote.accuracy * 100, after: teams.identity_cluster.accuracy * 100, note: '9 validation clips' });
  if (trk.ball) rows.push({ label: 'Ball located (F1)', before: trk.ball['top1>=0.1 (production)'].f1 * 100, after: trk.ball['dp-path'].f1 * 100, note: '49 test clips' });
  if (det.soccernet_v2 && nowDet) {
    rows.push({ label: 'Detector mAP50-95', before: det.soccernet_v2.map50_95 * 100, after: nowDet.map50_95 * 100, note: '49 test clips' });
    rows.push({ label: 'Ball AP50', before: det.soccernet_v2.per_class.ball.ap50 * 100, after: (nowDet.per_class.ball.ap50) * 100, note: '49 test clips' });
  }

  const b3 = phys?.ball3d_monocular as J | undefined;
  const noise = b3 ? Object.entries(b3).map(([k, v]: [string, any]) => ({ k: k.replace('noise_', '').replace('px', ''), med: v.median_3d_error_m, ground: v.ground_homography_error_m })) : [];
  const maxE = Math.max(6, ...noise.map(n => n.ground));

  return <div className="page">
    <div className="top"><h1>Evidence</h1><span className="muted">held-out data and known ground truth</span></div>

    {old && now && <div className="ev-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))' }}>
      <Tile g="squad" value={`${Math.round((1 - now.IDSW / old.IDSW) * 100)}% fewer`} label={`ID switches (${old.IDSW} to ${now.IDSW})`} />
      <Tile g="eye" value={`${old.pred_ids} to ${now.pred_ids}`} label={`identities created, true ${now.gt_ids}`} />
      {phys?.ecg_heart_rate && <Tile g="pulse" value={`${phys.ecg_heart_rate.naive_peak_mae_bpm} to ${phys.ecg_heart_rate.hr_mae_bpm}`} label="ECG heart rate error, bpm" />}
      {b3 && <Tile g="ball" value={`${b3['noise_1.0px'].median_3d_error_m} m`} label="3D ball error at 1 px noise" />}
    </div>}

    <div className="ev-grid">
      <section className="panel"><div className="panel-head"><h2>Before and after</h2></div><BeforeAfter rows={rows} /></section>

      {noise.length > 0 && <section className="panel ev-chart"><div className="panel-head"><h2>Ball height from one camera</h2></div>
        <div className="ev-legend"><span><i style={{ background: 'var(--ink)' }} />physics fit</span><span><i style={{ background: '#b9b3a5' }} />ground projection</span></div>
        <svg viewBox="0 0 420 210" role="img" aria-label="3D ball error by pixel noise">
          {[0, 2, 4, 6].map(v => <g key={v}><line x1={40} x2={410} y1={180 - (v / maxE) * 160} y2={180 - (v / maxE) * 160} className="gl" /><text x={34} y={184 - (v / maxE) * 160} textAnchor="end">{v} m</text></g>)}
          {noise.map((n, i) => { const cx = 70 + i * 92; return <g key={n.k}>
            <rect x={cx} y={180 - (n.ground / maxE) * 160} width={26} height={(n.ground / maxE) * 160} rx={3} className="before"><title>{`ground projection ${n.ground} m`}</title></rect>
            <rect x={cx + 30} y={180 - (n.med / maxE) * 160} width={26} height={Math.max(2, (n.med / maxE) * 160)} rx={3} className="after"><title>{`physics fit ${n.med} m`}</title></rect>
            <text x={cx + 43} y={176 - (n.med / maxE) * 160} textAnchor="middle" className="val">{n.med}</text>
            <text x={cx + 28} y={198} textAnchor="middle">{n.k} px</text></g>; })}
        </svg>
        <p className="muted" style={{ margin: 0, fontSize: 12.5 }}>median error, 40 synthetic shots per noise level</p>
      </section>}

      {phys?.ecg_heart_rate && <section className="panel"><div className="panel-head"><h2>Chest ECG</h2><span className="muted">{phys.ecg_heart_rate.cases} recordings, rest to hard running</span></div>
        <div className="ev-grid" style={{ gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 10 }}>
          <Tile g="pulse" value={`${Math.round(phys.ecg_heart_rate.beat_sensitivity * 100)}%`} label="beats found" />
          <Tile g="check" value={phys.ecg_heart_rate.hr_within_3bpm} label="heart rate within 3 bpm" />
          <Tile g="run" value={`${phys.ecg_heart_rate.hard_running_mae_bpm} bpm`} label="error while sprinting" />
          <Tile g="x" value={`${phys.ecg_heart_rate.naive_peak_mae_bpm} bpm`} label="simple peak counting" />
        </div></section>}

      {vest && <section className="panel"><div className="panel-head"><h2>Vest delivery</h2>
        <span className="muted">first prototype{today ? ' against today' : ''}</span></div>
        <div className="ba">
          <div className="ba-row"><span>Motion samples arrived</span>
            <div className="ba-bars"><i className="before" style={{ width: `${vest.imu_completeness_median * 100}%` }} />
              {today && <i className="after" style={{ width: `${today.imu_completeness_median * 100}%` }} />}</div>
            <span className="ba-val num"><small>{Math.round(vest.imu_completeness_median * 100)}%</small>{today ? `${Math.round(today.imu_completeness_median * 100)}%` : ''}</span></div>
          <div className="ba-row"><span>Median delay to the laptop</span>
            <span className="num">{(vest.transport_delay_ms_p50_median / 1000).toFixed(0)} s{today?.transport_delay_ms_p50 != null ? ` then, ${(today.transport_delay_ms_p50 / 1000).toFixed(2)} s now` : ''}</span><span /></div>
        </div>
        <p className="muted" style={{ margin: '10px 0 0', fontSize: 12.5 }}>
          {today ? `Today: ${today.readings} s of live vest data, ${today.dropped_during_run ?? 0} samples dropped, 90% of samples within ${((today.transport_delay_ms_p90 ?? 0) / 1000).toFixed(2)} s. `
            : ''}The prototype sent small uncompressed batches and lost samples on slow Wi-Fi; the vest now queues, compresses and batches them.</p></section>}
    </div>
  </div>;
};
