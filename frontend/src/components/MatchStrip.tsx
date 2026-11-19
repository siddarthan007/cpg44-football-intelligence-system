import { useMemo, useRef, useState } from 'react';
import type { Analysis, AnalysisEvent } from '../lib/analysis';
import { clock, TEAM_HEX } from '../lib/analysis';

interface Props {
  analysis: Analysis;
  t: number;
  onSeek: (t: number) => void;
  focus?: string;
  onEvent?: (e: AnalysisEvent) => void;
}

const ALL_LANES = [
  { key: 'control', label: 'Control', h: 14 },
  { key: 'moments', label: 'Moments', h: 26 },
  { key: 'effort', label: 'Running', h: 40 },
  { key: 'pulse', label: 'Vest pulse', h: 18 },
  { key: 'review', label: 'To check', h: 18 },
];
const LABEL_W = 70;
const W = 1000;

/**
 * The match strip: one horizontal reading of the session. Lanes stack what the
 * camera saw (control, moments, effort) above what the vest felt (pulse) and
 * what still needs a coach's decision (review).
 */
export function MatchStrip({ analysis, t, onSeek, focus, onEvent }: Props) {
  const ref = useRef<SVGSVGElement>(null);
  const LANES = ALL_LANES.filter(l => l.key !== 'pulse' || analysis.provenance.wearable);
  const [hover, setHover] = useState<number | null>(null);
  const dur = Math.max(analysis.match.duration_s, 1);
  const x = (s: number) => LABEL_W + ((W - LABEL_W - 8) * s) / dur;
  const tops = LANES.reduce<number[]>((acc, _l, i) => [...acc, i ? acc[i - 1] + LANES[i - 1].h + 6 : 18], []);
  const H = tops[tops.length - 1] + LANES[LANES.length - 1].h + 4;
  const top = (k: string) => tops[LANES.findIndex(l => l.key === k)];
  const lane = (k: string) => LANES.find(l => l.key === k)!;

  const control = useMemo(() => {
    const out: { a: number; b: number; team: number }[] = [];
    const fr = analysis.timeline.frames;
    for (let i = 0; i < fr.length; i++) {
      const s = fr[i], e = fr[i + 1]?.t ?? s.t + 1 / analysis.timeline.rate_hz;
      const last = out[out.length - 1];
      if (last && last.team === s.pos && Math.abs(last.b - s.t) < 1e-6) last.b = e; else out.push({ a: s.t, b: e, team: s.pos });
    }
    return out.filter(r => r.team);
  }, [analysis]);

  const effort = useMemo(() => {
    const fr = analysis.timeline.frames;
    const bucket = Math.max(1, Math.round(analysis.timeline.rate_hz * 2));
    const rows: { t: number; a: number; b: number }[] = [];
    for (let i = 0; i < fr.length; i += bucket) {
      const seg = fr.slice(i, i + bucket);
      const mean = (team: number) => {
        const v = seg.flatMap(s => s.p.filter(p => p[3] === team && p[4] != null).map(p => p[4]!));
        return v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0;
      };
      rows.push({ t: seg[0].t, a: mean(1), b: mean(2) });
    }
    return rows;
  }, [analysis]);
  const effortMax = Math.max(4, ...effort.map(r => Math.max(r.a, r.b)));

  const seek = (clientX: number) => {
    const r = ref.current?.getBoundingClientRect();
    if (!r) return;
    const px = ((clientX - r.left) / r.width) * W;
    const s = ((px - LABEL_W) / (W - LABEL_W - 8)) * dur;
    onSeek(Math.max(0, Math.min(dur, s)));
  };

  const events = analysis.events.filter(e => e.t != null);
  const mid = (k: string) => top(k) + lane(k).h / 2;
  const ticks = Array.from({ length: 7 }, (_, i) => (dur * i) / 6);

  return <div className="match-strip">
    <svg ref={ref} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="slider" tabIndex={0}
      aria-label="Match timeline" aria-valuemin={0} aria-valuemax={dur} aria-valuenow={t} aria-valuetext={clock(t)}
      onPointerDown={e => { (e.target as Element).setPointerCapture?.(e.pointerId); seek(e.clientX); }}
      onPointerMove={e => {
        const r = ref.current!.getBoundingClientRect();
        const px = ((e.clientX - r.left) / r.width) * W;
        setHover(px > LABEL_W ? ((px - LABEL_W) / (W - LABEL_W - 8)) * dur : null);
        if (e.buttons === 1) seek(e.clientX);
      }}
      onPointerLeave={() => setHover(null)}
      onKeyDown={e => {
        if (e.key === 'ArrowRight') onSeek(Math.min(dur, t + (e.shiftKey ? 10 : 1)));
        if (e.key === 'ArrowLeft') onSeek(Math.max(0, t - (e.shiftKey ? 10 : 1)));
      }}>
      {ticks.map(s => <g key={s}><line x1={x(s)} x2={x(s)} y1={10} y2={H} className="strip-grid" />
        <text x={x(s)} y={9} className="strip-tick" textAnchor="middle">{clock(s)}</text></g>)}
      {LANES.map((l, i) => <g key={l.key}>
        <rect x={LABEL_W} y={tops[i]} width={W - LABEL_W - 8} height={l.h} className="strip-lane" />
        <text x={4} y={tops[i] + l.h / 2 + 3.5} className="strip-label">{l.label}</text>
      </g>)}

      {control.map((c, i) => <rect key={i} x={x(c.a)} y={top('control') + 2} width={Math.max(0.6, x(c.b) - x(c.a))}
        height={lane('control').h - 4} style={{ fill: TEAM_HEX[c.team] }} className="strip-control" />)}

      {effort.length > 1 && <>
        <path className="strip-effort a" d={`M${x(effort[0].t)},${mid('effort')} ` + effort.map(r => `L${x(r.t)},${mid('effort') - (r.a / effortMax) * (lane('effort').h / 2 - 2)}`).join(' ') + ` L${x(effort[effort.length - 1].t)},${mid('effort')} Z`} />
        <path className="strip-effort b" d={`M${x(effort[0].t)},${mid('effort')} ` + effort.map(r => `L${x(r.t)},${mid('effort') + (r.b / effortMax) * (lane('effort').h / 2 - 2)}`).join(' ') + ` L${x(effort[effort.length - 1].t)},${mid('effort')} Z`} />
        <line x1={LABEL_W} x2={W - 8} y1={mid('effort')} y2={mid('effort')} className="strip-axis" />
      </>}


      {events.map(e => {
        const ex = x(e.t!);
        const sel = focus === e.id;
        const click = (ev: React.PointerEvent) => { ev.stopPropagation(); onEvent?.(e); onSeek(e.t!); };
        if (e.status === 'check' || e.status === 'confirmed' || e.status === 'dismissed') {
          if (e.type === 'ball_unseen') return null;
          const y = mid('review');
          return <path key={e.id} d={`M${ex},${y - 6} L${ex + 5},${y} L${ex},${y + 6} L${ex - 5},${y} Z`}
            className={`strip-check ${e.status} ${sel ? 'sel' : ''}`} onPointerDown={click}><title>{e.detail ?? e.type}</title></path>;
        }
        const y = mid('moments');
        const col = TEAM_HEX[e.team] ?? TEAM_HEX[0];
        if (e.type === 'shot') return <path key={e.id} d={`M${ex},${y - 9} L${ex + 6},${y + 6} L${ex - 6},${y + 6} Z`}
          style={{ stroke: col }} className={`strip-shot ${sel ? 'sel' : ''}`} onPointerDown={click}><title>Shot</title></path>;
        if (e.type === 'turnover') return <rect key={e.id} x={ex - 3} y={y - 3} width={6} height={6} transform={`rotate(45 ${ex} ${y})`}
          style={{ fill: col }} className="strip-turnover" onPointerDown={click}><title>Ball won</title></rect>;
        if (e.type === 'sprint') return <rect key={e.id} x={ex} y={y + 6} width={Math.max(2, x(e.t! + (e.duration_s ?? 1)) - ex)} height={3}
          style={{ fill: col }} className="strip-sprint" onPointerDown={click}><title>Sprint</title></rect>;
        if (e.type === 'pass') return <line key={e.id} x1={ex} x2={ex} y1={y - 8} y2={y + 2} style={{ stroke: col }}
          className="strip-pass" onPointerDown={click}><title>Pass</title></line>;
        return null;
      })}

      {hover != null && <line x1={x(hover)} x2={x(hover)} y1={12} y2={H} className="strip-hover" />}
      <line x1={x(t)} x2={x(t)} y1={12} y2={H} className="strip-playhead" />
      <rect x={x(t) - 18} y={0} width={36} height={11} rx={2} className="strip-playhead-tag" />
      <text x={x(t)} y={8.5} textAnchor="middle" className="strip-playhead-text">{clock(t)}</text>
    </svg>
  </div>;
}
