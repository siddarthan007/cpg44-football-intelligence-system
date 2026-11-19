import { apiUrl } from './api';

export type TeamId = 0 | 1 | 2;

export interface AnalysisPlayer {
  id: number;
  team: TeamId;
  role: string;
  label: string;
  observed_s: number;
  distance_m: number;
  top_speed_kmh: number | null;
  hsr_m: number;
  avg_x: number;
  avg_y: number;
  heat: number[][];
  vest?: number;
  number?: number;
  number_share?: number;
}

export interface LoadPlayer {
  id: number; team: TeamId; score: number; status: 'fresh' | 'watch' | 'tiring'; metric: string;
  now: number; peak: number; minutes_seen: number; factors: Record<string, number>; notes: string[];
  minutes_to_half_peak: number | null; role: string; curve: [number, number][];
}

export interface LoadReport {
  available: boolean; reason?: string; duration_s?: number; min_duration_s?: number; window_s?: number;
  players?: LoadPlayer[]; subs?: { team: TeamId; out: number; role: string; score: number; why: string[]; when: string }[];
  team_curve?: Record<'1' | '2', [number, number][]>;
}

export interface VestLink {
  vest: number; track: number | null; confident: boolean; score?: number; margin?: number; lag_s?: number;
  overlap_s?: number; candidates?: [number, number][]; reason?: string;
}

export interface AnalysisEvent {
  id: string;
  type: 'pass' | 'turnover' | 'shot' | 'sprint' | 'identity' | 'team' | 'ball_unseen' | string;
  t: number | null;
  frame: number | null;
  team: TeamId;
  players: number[];
  x?: number;
  y?: number;
  from?: [number, number];
  status: 'observed' | 'check' | 'confirmed' | 'dismissed' | 'info';
  confidence: 'high' | 'medium' | 'low';
  detail?: string;
  reason?: string | null;
  xg?: number;
  duration_s?: number;
  lost_by?: number;
  breakdown?: {
    shot_speed_kmh: number | null;
    distance_to_goal_m: number;
    goal_angle_deg: number;
    players_in_the_way: number | null;
    peak_height_m: number | null;
  };
  decision?: { decision: string; note: string; at: number };
}

export interface Insight {
  id: string;
  kind: 'tactical' | 'physical' | 'fused' | 'data';
  team: TeamId;
  title: string;
  finding: string;
  action: string;
  confidence: 'high' | 'medium' | 'low';
  evidence: { label: string; value: string }[];
  moments: number[];
  players: number[];
  objective: string;
  priority: number;
}

export interface TimelineSample {
  t: number;
  p: [number, number, number, TeamId | 3, number | null, number?, number?][];   // id, x, y, team, speed, vx, vy
  o?: [number, number, number, number, number][];                              // pass options: to, x, y, p_arrive, value
  d?: { call: 'shoot' | 'pass' | 'recycle' | 'carry'; to?: number; p?: number; gain?: number; xg: number; team: number; carrier: number; pressure: number };
  b: [number, number, number] | null;
  v?: [number, number][] | null;
  pos: TeamId;
  h: number | null;
}

export interface Analysis {
  id: string;
  schema: string;
  video_url: string | null;
  match: { name: string; video: string; pitch_length_m: number; pitch_width_m: number; pitch_kind?: 'football' | 'futsal'; goal_width_m?: number; duration_s: number; fps: number; width: number; height: number };
  provenance: { generated_at: number; detector: string; tracker: string; ball: string; wearable: boolean; synthetic: boolean };
  quality: {
    position_funnel?: Record<string, number>; camera_mode?: string; lens_k1?: number;
    frames: number; fps: number; metric: boolean; calibration_coverage: number; ball_seen: number;
    ball_located_incl_interpolated: number; identities: number; raw_tracks: number; links: number;
    uncertain_links: number; appearance_model: string; checks_pending: number; processing_s: number;
    team_separation: { method: string; assigned: number; abstained?: number; kit_separation?: number };
  };
  teams: Record<'1' | '2', { name: string; colour: string; attack_dir: number }>;
  players: AnalysisPlayer[];
  timeline: { rate_hz: number; frames: TimelineSample[] };
  possession: { team1_pct: number | null; team2_pct: number | null; coverage: number; buckets: { t: number; team1: number | null; coverage: number }[] };
  events: AnalysisEvent[];
  ball3d: { segments: { start: number; end: number; kind: string; rmse_px: number; launch_speed_mps: number | null; peak_height_m: number | null }[] };
  team_stats: Record<'1' | '2', { players_seen: number; passes: number; distance_m: number; hsr_m: number; length_m: number | null; width_m: number | null; centroid_x_m: number | null }>;
  insights: Insight[];
  load?: LoadReport;
  /** every frame: [t, x, y, z, observed] */
  ball_track?: [number, number, number, number, number][];
  vests?: VestLink[];
  squad_issues?: string[];
  plan?: { teams: Record<'1' | '2', {
    name: string;
    formation: Record<'with_ball' | 'without_ball', { shape: string | null; players_seen: number; positions: [number, number, number][] }>;
    space_conceded: number[][] | null;
    progression: { channel: string; share: number; mean_success: number } | null;
  }> };
}

export interface AnalysisSummary {
  id: string; name: string; duration_s: number; generated_at: number; metric: boolean;
  ball_seen: number; checks_pending: number; has_video: boolean;
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(apiUrl(path));
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json() as Promise<T>;
}

export const listAnalyses = () => get<AnalysisSummary[]>('/api/v1/analyses');
export const getAnalysis = (id: string) => get<Analysis>(`/api/v1/analyses/${encodeURIComponent(id)}`);
export const decide = async (id: string, eventId: string, decision: 'confirm' | 'dismiss' | 'reset') => {
  const r = await fetch(apiUrl(`/api/v1/analyses/${encodeURIComponent(id)}/review`), {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ event_id: eventId, decision }),
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
};

export const clock = (s: number | null | undefined) => {
  if (s == null || !Number.isFinite(s)) return '--:--';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
};

export const clockFine = (s: number) => `${clock(s)}.${Math.floor((s % 1) * 10)}`;

export const TEAM_INK: Record<number, string> = { 1: 'var(--team-a)', 2: 'var(--team-b)', 0: 'var(--team-none)' };
export const TEAM_HEX: Record<number, string> = { 1: '#2251a3', 2: '#d1492e', 0: '#6b706b', 3: '#1b2320' };

export const eventLabel: Record<string, string> = {
  pass: 'Pass', turnover: 'Ball won', shot: 'Shot', sprint: 'Sprint', identity: 'Same player?',
  team: 'Which side?', ball_unseen: 'Ball out of sight',
};

/* Pitch control, same model as soccer_analytics/decision.py (Spearman 2018). */
const REACTION = 0.7, VMAX = 5.0, SIGMA = 0.45;
export function pitchControl(sample: TimelineSample, team: number, L: number, W: number, nx = 42, ny = 27): number[][] {
  const att = sample.p.filter(p => p[3] === team), def = sample.p.filter(p => p[3] !== team && (p[3] === 1 || p[3] === 2));
  const arrive = (ps: typeof att, x: number, y: number) => {
    let best = Infinity;
    for (const p of ps) {
      const sx = p[1] + (p[5] ?? 0) * REACTION, sy = p[2] + (p[6] ?? 0) * REACTION;
      const t = REACTION + Math.hypot(x - sx, y - sy) / VMAX;
      if (t < best) best = t;
    }
    return best;
  };
  const k = Math.PI / Math.sqrt(3) / SIGMA;
  const grid: number[][] = [];
  for (let j = 0; j < ny; j++) {
    const row: number[] = [];
    for (let i = 0; i < nx; i++) {
      const x = ((i + 0.5) * L) / nx, y = ((j + 0.5) * W) / ny;
      const d = Math.max(-10, Math.min(10, arrive(def, x, y) - arrive(att, x, y)));
      row.push(1 / (1 + Math.exp(-k * d)));
    }
    grid.push(row);
  }
  return grid;
}

/* ---------------- interpolation, shared by the 2D board and the 3D replay ---------------- */
export type BallRow = [number, number, number, number, number];

/** Rows of the full-rate ball track with t0 < t <= t1 (binary search, no full scan). */
export function trackWindow(track: BallRow[], t0: number, t1: number): BallRow[] {
  const first = (t: number) => { let lo = 0, hi = track.length; while (lo < hi) { const m = (lo + hi) >> 1; if (track[m][0] <= t) lo = m + 1; else hi = m; } return lo; };
  return track.slice(first(t0), first(t1));
}

/** Ball at time t from the full-rate track: linear between frames 40 ms apart,
 *  nothing across a real gap (the ball was not located there). */
export function ballAt(track: BallRow[], t: number): [number, number, number] | null {
  if (!track.length || t < track[0][0] || t > track[track.length - 1][0]) return null;
  let lo = 0, hi = track.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (track[m][0] <= t) lo = m; else hi = m; }
  const a = track[lo], b = track[hi];
  if (b[0] - a[0] > 0.12) return Math.abs(t - a[0]) < 0.02 ? [a[1], a[2], a[3]] : Math.abs(t - b[0]) < 0.02 ? [b[1], b[2], b[3]] : null;
  const u = b[0] > a[0] ? (t - a[0]) / (b[0] - a[0]) : 0;
  return [a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u, a[3] + (b[3] - a[3]) * u];
}

/** Players and ball at time t, linear between the two timeline samples around it. */
export function interpolateAt(frames: TimelineSample[], t: number) {
  let lo = 0, hi = frames.length - 1;
  if (hi < 0) return null;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (frames[m].t <= t) lo = m; else hi = m - 1; }
  const a = frames[lo], b = frames[Math.min(frames.length - 1, lo + 1)];
  const u = b.t > a.t ? Math.min(1, Math.max(0, (t - a.t) / (b.t - a.t))) : 0;
  const nb = new Map(b.p.map(p => [p[0], p]));
  const players = a.p.map(p => {
    const q = nb.get(p[0]);
    const x = q ? p[1] + (q[1] - p[1]) * u : p[1], y = q ? p[2] + (q[2] - p[2]) * u : p[2];
    const dx = q ? q[1] - p[1] : 0, dy = q ? q[2] - p[2] : 0;
    return { id: p[0], x, y, team: p[3], sp: p[4] ?? 0, dx, dy };
  });
  let ball = a.b;
  if (a.b && b.b) ball = [a.b[0] + (b.b[0] - a.b[0]) * u, a.b[1] + (b.b[1] - a.b[1]) * u, a.b[2] + (b.b[2] - a.b[2]) * u];
  return { players, ball, holder: a.h };
}
