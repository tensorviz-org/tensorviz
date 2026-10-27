import { useEffect, useState } from 'react';

import { API_BASE_URL, localApiHeaders } from '../config/apiBase';

/**
 * The backend's own account of itself, for the sidebar's environment readout.
 *
 * The UI tells the user the backend is local and that nothing leaves the
 * machine. That is only a checkable claim if the address, the reachability and
 * the versions come from the running process — so this reads /health, which now
 * reports them, and renders exactly what came back. A hardcoded "torch 2.4.1"
 * would be decoration that happens to look like a fact.
 */


export type BackendEnvironment = {
  /** 'up' once /health answered, 'down' when it did not, 'checking' before. */
  state: 'checking' | 'up' | 'down';
  /** Host:port the frontend is actually configured to call. */
  address: string;
  torch?: string;
  device?: string;
  python?: string;
  libraries: string[];
};

function displayAddress(): string {
  if (!API_BASE_URL) return `${window.location.host} (same origin)`;
  try {
    const url = new URL(API_BASE_URL);
    return url.port ? `${url.hostname}:${url.port}` : url.hostname;
  } catch {
    return API_BASE_URL;
  }
}

export function useBackendEnvironment(): BackendEnvironment {
  const [env, setEnv] = useState<BackendEnvironment>({
    state: 'checking',
    address: displayAddress(),
    libraries: [],
  });

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    fetch(`${API_BASE_URL}/health`, { signal: controller.signal, headers: localApiHeaders() })
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(String(res.status)))))
      .then((body) => {
        if (cancelled) return;
        const e = body?.environment ?? {};
        setEnv({
          state: 'up',
          address: displayAddress(),
          torch: typeof e.torch === 'string' ? e.torch : undefined,
          device: typeof e.device === 'string' ? e.device : undefined,
          python: typeof e.python === 'string' ? e.python : undefined,
          libraries: Array.isArray(e.libraries) ? e.libraries.filter((l: unknown) => typeof l === 'string') : [],
        });
      })
      // Fail loud, quietly: 'down' is a real state the rail must show, because a
      // silently missing readout is indistinguishable from a working backend.
      .catch(() => {
        if (!cancelled) {
          setEnv({ state: 'down', address: displayAddress(), libraries: [] });
        }
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, []);

  return env;
}
