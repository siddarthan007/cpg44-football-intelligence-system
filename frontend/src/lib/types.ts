export interface SystemInfo {
  version: string;
  gpu: { available: boolean; name: string | null };
  active_config: { detector: string; tracker: string };
  hub_connected: boolean;
  hub_url: string;
  hostinger_relay_url?: string | null;
  active_mode: string;
  cameras_online: number;
  running_matches: string[];
  uptime_s: number;
  models: Array<{ name: string; path: string; size_mb: number; modified_at: number }>;
}

export interface Match {
  match_id: string;
  name: string;
  status: string;
  mode?: string;
  home_team?: string;
  away_team?: string;
  pitch_length_m: number;
  pitch_width_m: number;
  venue: string | null;
  notes: string | null;
  created_at: number;
  engine_running?: boolean;
}

export interface LoadIndicator {
  score: number | null;
  severity: string | null;
  factors?: Record<string, unknown>;
  model: string;
  evidence_ready?: boolean;
  medical_prediction: false;
}

export interface LivePlayer {
  global_player_id: string;
  track_id: number;
  player_id?: number | null;
  name?: string;
  jersey?: string | number | null;
  team_id: string;
  team: number;
  x: number | null;
  y: number | null;
  speed_mps?: number | null;
  top_speed_mps?: number | null;
  distance_m?: number | null;
  hsr_m?: number | null;
  sprints?: number | null;
  metabolic_power?: number | null;
  player_load?: number | null;
  wearable: boolean;
  position_status?: 'observed' | 'rejected' | 'unprojectable' | 'unknown';
  identity_status?: 'roster_bound' | 'track_only';
  wearable_metrics?: Record<string, unknown> | null;
  load?: Record<string, number> | null;
  load_indicator?: LoadIndicator;
  readiness?: {
    status: 'signal_limited' | 'building_baseline' | 'monitoring' | 'check_in';
    action: string;
    reason: string;
    sources: string[];
    observed_seconds?: number;
    hr_drift_bpm?: number;
    work_rate_change_pct?: number | null;
  };
}

export interface DataQuality {
  status: string;
  metric_calibration?: boolean;
  unique_track_ids?: number;
  simultaneous_players?: number;
  speed_cap_fraction?: number;
  ball_observed?: boolean;
  positioned_players?: number;
  team_assigned_players?: number;
  roster_bound_players?: number;
  ambiguous_reidentifications?: number;
  calibration?: { valid: boolean; flow_inliers?: number; flow_inlier_ratio?: number };
  warnings: string[];
}

export interface LiveEvent {
  frame: number;
  type: 'pass' | 'throw_in' | string;
  team: number;
  from_track?: number;
  to_track?: number;
  x?: number;
  y?: number;
  side?: string;
}

export interface HeatmapGrid {
  rows: number;
  cols: number;
  length_m: number;
  width_m: number;
  cells: number[];
  seconds?: number[];
  observed_player_seconds?: number;
  unit?: 'player_seconds';
}

export interface LivePayload {
  coach_insights?: import('./analysis').Insight[];
  performance?: { stages: Record<string, { p50_ms: number; p95_ms: number; samples: number }> };
  timestamp: number;
  source_kind?: 'recorded_file' | 'live_camera' | 'unknown';
  frame_index?: number;
  loop_count?: number;
  source_fps?: number;
  processing_fps?: number;
  match_id: string;
  match_name?: string;
  match: Match;
  mode?: string;
  metric?: boolean;
  players: LivePlayer[];
  ball: { x: number; y: number } | null;
  possession_pct: Record<string, number>;
  passes: Record<string, number>;
  events?: LiveEvent[];
  throw_ins?: number;
  heatmap?: HeatmapGrid | null;
  passing_network: Record<string, unknown>;
  timeline_tags: Array<{ id: string; time: number; label: string }>;
  shots_xg: Record<string, { shots?: number; xg?: number }>;
  tactics: Record<string, Record<string, unknown>>;
  pass_options?: Array<{
    carrier: number;
    receiver: number;
    score: number;
    openness: number;
    lane: number;
    progress: number;
    x: number;
    y: number;
    model: string;
  }>;
  substitution_watch: Array<Record<string, unknown>>;
  recommendations?: string[];
  data_quality: DataQuality;
  wearables?: {
    connected_players: number;
    players: Record<string, WearableObservation | Record<string, unknown>>;
  };
  provenance: {
    kind: string;
    live: boolean;
    input_kind?: 'recorded_file' | 'live_camera' | 'unknown';
    source_file?: string | null;
    generated_at?: number | null;
    received_at?: number;
    age_s?: number | null;
    warnings?: string[];
  };
}

export interface WearableObservation {
  match_id: string;
  global_player_id: string;
  player_id?: number | null;
  timestamp: number;
  source: string;
  metrics: Record<string, unknown>;
  received_at?: number;
}

export interface ProcessingJob {
  match_id: string;
  status: string;
  progress_pct: number;
  frames_processed?: number;
  total_frames?: number;
  fps?: number;
  error?: string;
  stats_path?: string;
  output_path?: string;
}
