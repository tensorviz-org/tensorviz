/**
 * Where the backend is, decided once.
 *
 * API clients share the same-origin default and explicit hosted API override.
 *
 * IDE mode (task.md D2, "Stage A") is the reason this file exists: the backend
 * serves the React build itself, so the app is same-origin with the API and the
 * base is '' whatever REACT_APP_API_URL happens to say. REACT_APP_IDE is a
 * BUILD-time flag — CRA inlines it — so this resolves before any module that
 * imports it runs, which `window.__TENSORVIZ_IDE__` (set by the IDE entry point)
 * could not promise.
 */

const IDE = process.env.REACT_APP_IDE === 'true';

const CONFIGURED = process.env.REACT_APP_API_URL;

/** apiService.ts and useBackendEnvironment.ts. */
export const API_BASE_URL = IDE ? '' : CONFIGURED || '';

/**
 * The per-launch shared secret the extension gives this app in its iframe URL
 * (task.md D12). Held here rather than in a service so every caller of the API
 * picks it up from one place; empty in the hosted product, where the middleware
 * that would check it is not even registered.
 */
let localToken = '';

export function setLocalApiToken(token: string): void {
  localToken = token;
}

/** Headers every /api/* request must carry in IDE mode, and nothing at all in web mode. */
export function localApiHeaders(): Record<string, string> {
  return localToken ? { 'X-tensorViz-Token': localToken } : {};
}
