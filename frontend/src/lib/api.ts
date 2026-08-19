import type { LivePayload, Match, SystemInfo, WearableObservation } from './types';

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? '';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

const post = <T>(path: string, body?: unknown) =>
  request<T>(path, { method: 'POST', body: body ? JSON.stringify(body) : undefined });

export const api = {
  health: () => request<{ status: string; version: string }>('/api/v1/health'),
  systemInfo: () => request<SystemInfo>('/api/v1/system/info'),
  listMatches: () => request<Match[]>('/api/v1/matches'),
  getMatch: (matchId: string) => request<Match>(`/api/v1/matches/${matchId}`),
  startMatch: (matchId: string) => post<{ ok: boolean; message: string }>(`/api/v1/matches/${matchId}/start`),
  stopMatch: (matchId: string) => post<{ ok: boolean; message: string }>(`/api/v1/matches/${matchId}/stop`),
  analytics: (matchId: string) => request<LivePayload['analytics']>(`/api/v1/matches/${matchId}/analytics`),
  players: (matchId: string) => request<LivePayload['analytics']['players']>(`/api/v1/matches/${matchId}/players`),
  matchStatus: (matchId: string) => request<Record<string, unknown>>(`/api/v1/matches/${matchId}/status`),
  live: () => request<LivePayload>('/api/v1/live'),
  postWearable: (body: WearableObservation) => post<WearableObservation>('/api/v1/observations/wearable', body),
};
