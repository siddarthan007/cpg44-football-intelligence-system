export interface SystemInfo {
  version: string;
  gpu: { available: boolean; name: string | null };
  active_config: { detector: string; tracker: string };
  hub_connected: boolean;
  hub_url: string;
  running_matches: string[];
  uptime_s: number;
}

export interface Match {
  match_id: string;
  name: string;
  status: string;
  pitch_length_m: number;
  pitch_width_m: number;
  venue: string | null;
  notes: string | null;
  created_at: number;
  engine_running?: boolean;
  player_count?: number;
}

export interface LivePlayer {
  global_player_id: string;
  team_id: string;
  x: number;
  y: number;
  jersey: string;
  distance_m: number;
  top_speed_mps: number;
  hsr_m: number;
  sprints: number;
  metabolic_wkg: number;
  wearable: boolean;
  team: number;
}

export interface WearableLive {
  connected: boolean;
  ip?: string | null;
  hr?: number | null;
  spo2?: number | null;
  hr_valid?: boolean;
  reason?: string;
  quality?: number;
  speed_mps?: number | null;
  clock?: { valid?: boolean; last_rtt_ms?: number | null };
}

export interface LivePayload {
  match: Match;
  analytics: {
    match_id: string;
    metric: boolean;
    possession: Record<string, number>;
    passes: Record<string, number>;
    players: LivePlayer[];
    source_file: string | null;
  };
  wearable: WearableLive;
  ts: number;
}

export interface WearableObservation {
  match_id: string;
  global_player_id: string;
  timestamp: number;
  source: string;
  metrics: Record<string, unknown>;
}
