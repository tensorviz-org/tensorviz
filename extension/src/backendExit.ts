/** How an unexpected uvicorn exit is described, and whether this process may try again. */

export const BACKEND_RESTART_LIMIT = 2;

export function backendExitNotice(code: number | null, signal: NodeJS.Signals | null): string {
  const how = signal ? `signal ${signal}` : `code ${code ?? 'null'}`;
  return `The tensorViz backend exited unexpectedly (${how}).`;
}

export function shouldRestartBackend(restarts: number): boolean {
  return restarts < BACKEND_RESTART_LIMIT;
}

export function planBackendExit(code: number | null, signal: NodeJS.Signals | null, restarts: number): {
  notice: string;
  restart: boolean;
} {
  return { notice: backendExitNotice(code, signal), restart: shouldRestartBackend(restarts) };
}
