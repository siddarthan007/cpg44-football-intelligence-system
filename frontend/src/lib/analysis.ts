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

/** Binary search for the timeline sample at or just before t. */
export const sampleAt = (frames: TimelineSample[], t: number): TimelineSample | undefined => {
  if (!frames.length) return undefined;
  let lo = 0, hi = frames.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (frames[mid].t <= t) lo = mid; else hi = mid - 1;
  }
  return frames[lo];
};

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
