import { useEffect, useState } from 'react';
import type { LivePayload } from './types';
import { api } from './api';

/** HTTP poll only. Vite's WS proxy to WSL uvicorn dies with EPIPE. */
export function useLive(_matchId: string) {
  const [data, setData] = useState<LivePayload | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
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
    pull();
    const poll = window.setInterval(pull, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(poll);
    };
  }, [_matchId]);

  return { data, connected, error };
}
