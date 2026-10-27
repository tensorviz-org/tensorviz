/** The one place the backend's launch contract lives.
 *
 * Everything here used to be about two servers on two fixed ports. Stage A
 * (task.md D2) removed the second one: the backend serves the production React
 * build itself, so there is no CRA dev server, no npm, no Node requirement at
 * runtime, and no port 3000. The remaining port is allocated per launch rather
 * than fixed at 8765, so opening the canvas can never collide with ./start.sh,
 * `make e2e`, a second window, or the VS Code plugin helpers that squat 8000
 * (start.sh:5-7).
 */

/** Bind and probe loopback only. The extension forces AUTH_DISABLED (see
 *  backendEnv) and a per-launch token. start.sh also defaults to 127.0.0.1 and
 *  refuses a non-loopback host unless TENSORVIZ_LOCAL_TOKEN is set. */
export const HOST = '127.0.0.1';

/** A directory is a usable BACKEND directory iff this exists as a file inside it.
 *
 *  It used to be two markers relative to a repository root — `backend/server_v2.py`
 *  AND `canvas/package.json` — which made a canvas checkout a hard requirement
 *  for opening any folder (task.md P0). The React build has shipped inside the vsix since
 *  Stage A, so the frontend marker was checking for something the extension no longer
 *  needs, and the backend marker is now resolved against a backend directory that may be
 *  bundled with the app, staged in the vsix, or inside a checkout. */
export const MARKER_FILES = ['server_v2.py'] as const;

export function backendOrigin(port: number): string {
  return `http://${HOST}:${port}`;
}

type Env = Record<string, string | undefined>;

export interface BackendLaunch {
  /** The loopback port uvicorn is told to bind. */
  port: number;
  /** Directory holding the built React app, served at / by StaticFiles. */
  staticDir: string;
  /** Per-launch shared secret required on every /api/* request. */
  token: string;
  /** VS Code-owned state storage, separate from the user's source workspace. */
  stateDbPath: string;
  /** Authoritative source workspace; older launchers can still infer it from the DB. */
  workspaceDir?: string;
  /** Every folder the class index may scan, `path.delimiter`-separated. */
  allowedRoots?: string;
}

/** Environment for `uvicorn server_v2:app`.
 *
 * Everything here beats the user's own .env: server_v2.py:50-52 calls load_dotenv
 * on the project and backend .env files, and load_dotenv defaults to
 * override=False, so values already present in the process environment win. That
 * is what makes injection authoritative and why the extension never writes a
 * .env.local into the user's checkout. */
export function backendEnv(parent: Env, launch: BackendLaunch): Env {
  const env: Env = {
    ...parent,
    // auth.py:34-35 accepts 1|true|yes|on; get_current_user then returns
    // CurrentUser(uid="local-dev-user") with no token (auth.py:78-79). What
    // stands in for it on the wire is TENSORVIZ_LOCAL_TOKEN below.
    AUTH_DISABLED: 'true',
    // The app is same-origin with the API in Stage A, so this allows nothing new;
    // it is set explicitly so a user's .env pointing at a production origin cannot
    // quietly widen what this loopback process accepts.
    CORS_ORIGINS: backendOrigin(launch.port),
    // Only read on the non-uvicorn path (server_v2.py:1258-1264); harmless
    // belt-and-braces now that the port is chosen per launch.
    BACKEND_PORT: String(launch.port),
    // The two switches that make this an IDE backend: serve the built app at /,
    // and refuse any /api/* request that does not carry the launch secret.
    TENSORVIZ_STATIC_DIR: launch.staticDir,
    TENSORVIZ_LOCAL_TOKEN: launch.token,
    // Keep source identity independent of the private state storage location.
    TENSORVIZ_WORKSPACE_DIR: launch.workspaceDir ?? '',
    TENSORVIZ_ALLOWED_ROOTS: launch.allowedRoots ?? launch.workspaceDir ?? '',
    USER_HISTORY_DB_PATH: launch.stateDbPath,
    // Makefile:8 exports this; without it torch + MKL aborts on some machines.
    KMP_DUPLICATE_LIB_OK: 'TRUE',
    // Not optional. Through a pipe Python's stdout is block-buffered, so without
    // this the output channel stays empty until 4 KB accumulates and a crash
    // loses its traceback.
    PYTHONUNBUFFERED: '1',
  };
  // The worker's parent-death pipe must not be inherited by uvicorn. A closed
  // stdin plus that flag asks the worker lifetime guard to kill its process group.
  delete env.TENSORVIZ_PARENT_PIPE;
  return env;
}

/* ── The dev loop's overrides ──────────────────────────────────────────────────
 *
 * ide/dev.sh runs the fork against a live `npm start` so a frontend edit shows up
 * without `build:webview`, and against uvicorn's reloader so a backend edit shows
 * up without a window reload. Three things have to line up for that, and they
 * reach the extension host through its environment because a setting would be
 * written into the user's profile and outlive the dev session that wanted it:
 *
 *   TENSORVIZ_DEV_SERVER_URL  the frame URL, in place of the app the backend serves
 *   TENSORVIZ_BACKEND_PORT    a FIXED port, so the CRA proxy (canvas/src/
 *                             setupProxy.js) knows where to forward /api and /health
 *   TENSORVIZ_BACKEND_RELOAD  pass --reload to uvicorn
 *
 * None of these is read by a packaged build's users: the .app inherits a login
 * shell's environment, where nothing sets them. */

export const DEV_SERVER_URL_ENV = 'TENSORVIZ_DEV_SERVER_URL';
export const BACKEND_PORT_ENV = 'TENSORVIZ_BACKEND_PORT';
export const BACKEND_RELOAD_ENV = 'TENSORVIZ_BACKEND_RELOAD';

/** The `tensorviz.devServerUrl` setting wins; the environment is the fallback. */
export function devServerUrl(configured: string, env: Env): string {
  return (configured.trim() || env[DEV_SERVER_URL_ENV]?.trim()) ?? '';
}

/** The frame URL for a dev server, carrying the same per-launch token the served
 *  app gets. Without it every /api request from the frame is a 401
 *  (middleware.py LocalTokenMiddleware), which reads as "the backend is down". A
 *  URL that already names a token is left alone. */
export function withLaunchToken(url: string, token: string): string {
  if (/[?&]token=/.test(url)) {
    return url;
  }
  return `${url}${url.includes('?') ? '&' : '?'}token=${encodeURIComponent(token)}`;
}

/** A port pinned by the environment, or undefined to allocate one per launch. */
export function fixedBackendPort(env: Env): number | undefined {
  const raw = env[BACKEND_PORT_ENV]?.trim();
  if (!raw) {
    return undefined;
  }
  const port = Number(raw);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error(`${BACKEND_PORT_ENV} must be a port number, got "${raw}".`);
  }
  return port;
}

export function backendReload(env: Env): boolean {
  return ['1', 'true', 'yes', 'on'].includes((env[BACKEND_RELOAD_ENV] ?? '').trim().toLowerCase());
}
