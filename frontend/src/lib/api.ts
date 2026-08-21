import type { LivePayload, Match, ProcessingJob, SystemInfo, WearableObservation } from './types';

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');

export const apiUrl = (path: string) => `${API_BASE}${path}`;

export const websocketUrl = (path: string) => {
  const configured = import.meta.env.VITE_WS_BASE_URL as string | undefined;
  if (configured) return `${configured.replace(/\/$/, '')}${path}`;
  if (API_BASE) {
    const url = new URL(API_BASE, window.location.href);
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    url.pathname = path;
    return url.toString();
  }
  if (import.meta.env.DEV) return `ws://${window.location.hostname}:8000${path}`;
  return `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}${path}`;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), init);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: string; error?: string };
      detail = body.detail ?? body.error ?? detail;
    } catch {
      // Keep the HTTP status when the body is not JSON.
    }
    throw new Error(detail);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

const json = (method: string, body?: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: body === undefined ? undefined : JSON.stringify(body),
});

export const api = {
  health: () => request<{ status: string; version: string; mode: string }>('/api/v1/health'),
  systemInfo: () => request<SystemInfo>('/api/v1/system/info'),
  listMatches: () => request<Match[]>('/api/v1/matches'),
  getMatch: (matchId: string) => request<Match>(`/api/v1/matches/${matchId}`),
  startMatch: (matchId: string) => request(`/api/v1/matches/${matchId}/start`, json('POST')),
  stopMatch: (matchId: string) => request(`/api/v1/matches/${matchId}/stop`, json('POST')),
  live: () => request<LivePayload>('/api/v1/live'),
  analytics: (matchId: string) => request<LivePayload>(`/api/v1/matches/${matchId}/analytics`),
  matchStatus: (matchId: string) => request<Record<string, unknown>>(`/api/v1/matches/${matchId}/status`),
  progress: (matchId: string) => request<ProcessingJob>(`/api/v1/matches/${matchId}/progress`),
  uploadMatch: (form: FormData) => request<{ ok: boolean; match_id: string; status: string }>('/api/v1/matches/upload', { method: 'POST', body: form }),
  cameras: () => request<Array<Record<string, unknown>>>('/api/v1/cameras'),
  registerCamera: (body: Record<string, unknown>) => request('/api/v1/cameras/register', json('POST', body)),
  wearables: () => request<Array<WearableObservation> | Record<string, unknown>>('/api/v1/observations/wearable'),
  postWearable: (body: WearableObservation) => request<WearableObservation>('/api/v1/observations/wearable', json('POST', body)),
  trainingStatus: () => request<Record<string, unknown>>('/api/v1/training/status'),
  startTraining: (body: Record<string, unknown>) => request<Record<string, unknown>>('/api/v1/training/start-yolo', json('POST', body)),
  stopTraining: () => request<Record<string, unknown>>('/api/v1/training/stop-yolo', json('POST')),
  trainStrain: () => request<Record<string, unknown>>('/api/v1/training/train-strain-model', json('POST')),
  labelPlayerSession: (body: Record<string, unknown>) => request<Record<string, unknown>>('/api/v1/training/outcome-label', json('POST', body)),
  hardwareStatus: () => request<Record<string, unknown>>('/api/v1/hardware/status'),
  serialPorts: () => request<Array<Record<string, unknown>>>('/api/v1/hardware/ports'),
  chipInfo: (port: string) => request<Record<string, unknown>>(`/api/v1/hardware/chip-info?port=${encodeURIComponent(port)}`),
  flashStatus: () => request<Record<string, unknown>>('/api/v1/hardware/flash/status'),
  flash: (body: Record<string, unknown>) => request<Record<string, unknown>>('/api/v1/hardware/flash', json('POST', body)),
  teamProfiles: () => request<Record<string, unknown>>('/api/v1/tagging/teams'),
  saveTeamProfiles: (body: Record<string, unknown>) => request<Record<string, unknown>>('/api/v1/tagging/teams', json('POST', body)),
  events: () => request<Array<Record<string, unknown>>>('/api/v1/tagging/events'),
  addEvent: (body: Record<string, unknown>) => request<Record<string, unknown>>('/api/v1/tagging/events', json('POST', body)),
};
