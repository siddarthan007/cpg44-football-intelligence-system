import { useEffect, useState } from 'react';
import type { LivePayload } from './types';
import { api, websocketUrl } from './api';

export function useLive(_matchId: string) {
  const [data, setData] = useState<LivePayload | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let socket: WebSocket | null = null;
    let retry: number | undefined;
    let lastCommit = 0;
    const pull = () =>
      api
        .live()
        .then((p) => {
          if (cancelled) return;
          setData(p);
          setConnected(true);
          setError(null);
        })
        .catch((e) => {
          if (cancelled) return;
          setConnected(false);
          setError(e instanceof Error ? e.message : 'live poll failed');
        });
    const connect = () => {
      if (cancelled) return;
      socket = new WebSocket(websocketUrl('/ws/live'));
      socket.onopen = () => { if (!cancelled) { setConnected(true); setError(null); } };
      socket.onmessage = (event) => {
        if (cancelled || Date.now() - lastCommit < 180) return;
        try {
          setData(JSON.parse(event.data) as LivePayload);
          setConnected(true);
          setError(null);
          lastCommit = Date.now();
        } catch {
          setError('Live feed returned invalid JSON');
        }
      };
      socket.onerror = () => socket?.close();
      socket.onclose = () => {
        if (cancelled) return;
        setConnected(false);
        retry = window.setTimeout(connect, 1500);
      };
    };
    void pull();
    connect();
    const poll = window.setInterval(() => { if (!socket || socket.readyState !== WebSocket.OPEN) void pull(); }, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(poll);
      if (retry !== undefined) window.clearTimeout(retry);
      socket?.close();
    };
  }, [_matchId]);

  return { data, connected, error };
}
