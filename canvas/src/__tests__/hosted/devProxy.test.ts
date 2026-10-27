import fs from 'fs';
import path from 'path';

jest.mock('http-proxy-middleware', () => ({ createProxyMiddleware: jest.fn() }));

const setupProxy = require('../../setupProxy');
const { createProxyMiddleware } = require('http-proxy-middleware');

describe('local development API proxy', () => {
  const VARS = ['REACT_APP_DEV_API_PROXY_TARGET', 'TENSORVIZ_BACKEND_PORT'] as const;
  const saved = new Map(VARS.map((name) => [name, process.env[name]]));

  const run = (env: Partial<Record<(typeof VARS)[number], string>>) => {
    VARS.forEach((name) => delete process.env[name]);
    Object.entries(env).forEach(([name, value]) => {
      process.env[name] = value;
    });
    const app = { use: jest.fn() };
    const proxy = { name: 'dev-api-proxy' };
    createProxyMiddleware.mockReturnValue(proxy);
    setupProxy(app);
    return { app, proxy };
  };

  afterEach(() => {
    saved.forEach((value, name) => {
      if (value === undefined) delete process.env[name];
      else process.env[name] = value;
    });
    jest.clearAllMocks();
  });

  it('proxies API, health and local-browser routes to the configured backend port', () => {
    const { app, proxy } = run({ REACT_APP_DEV_API_PROXY_TARGET: 'http://localhost:9012' });

    expect(createProxyMiddleware).toHaveBeenCalledWith(
      expect.objectContaining({ target: 'http://localhost:9012', changeOrigin: false }),
    );
    expect(app.use).toHaveBeenCalledWith(
      ['/api', '/health', '/local'],
      proxy,
    );
  });

  it('upgrades WebSockets so the local-browser broker socket survives the dev server', () => {
    // The view transport (src/ide/localBrowserViewTransport.ts) opens
    // ws://<frontend-origin>/local/ws; without ws:true the CRA dev server answers
    // the upgrade itself and the handshake never reaches the broker.
    run({ TENSORVIZ_BACKEND_PORT: '8801' });

    expect(createProxyMiddleware).toHaveBeenCalledWith(
      expect.objectContaining({ ws: true }),
    );
  });

  it('forwards to the port the IDE pinned, in preference to start.sh own', () => {
    // ide/dev.sh frames this dev server inside the fork and picks the backend port
    // per launch, so its port is the backend actually serving that window. A stale
    // REACT_APP_DEV_API_PROXY_TARGET inherited from the environment must not win.
    run({ TENSORVIZ_BACKEND_PORT: '8801', REACT_APP_DEV_API_PROXY_TARGET: 'http://localhost:9012' });

    expect(createProxyMiddleware).toHaveBeenCalledWith(
      expect.objectContaining({ target: 'http://127.0.0.1:8801', changeOrigin: true }),
    );
  });

  it('streams rather than buffering, so agent turns arrive token by token', () => {
    run({ TENSORVIZ_BACKEND_PORT: '8801' });

    expect(createProxyMiddleware).toHaveBeenCalledWith(
      expect.objectContaining({ selfHandleResponse: false }),
    );
  });

  it('does nothing at all when neither loop asked for a proxy', () => {
    // A bare `npm start` predates both launch loops and must keep behaving as it
    // did: registering a proxy to a guessed port would silently swallow /api calls
    // a developer meant to send somewhere else.
    const { app } = run({});

    expect(createProxyMiddleware).not.toHaveBeenCalled();
    expect(app.use).not.toHaveBeenCalled();
  });

  it('keeps start.sh on same-origin requests unless the caller overrides the API URL', () => {
    const launcher = fs.readFileSync(
      path.resolve(__dirname, '../..', '..', '..', 'start.sh'),
      'utf8',
    );

    expect(launcher).toContain('REACT_APP_DEV_API_PROXY_TARGET="http://localhost:${BACKEND_PORT}"');
    expect(launcher).toContain('REACT_APP_API_URL="${REACT_APP_API_URL:-}"');
    expect(launcher).not.toContain('REACT_APP_API_URL="http://localhost:${BACKEND_PORT}"');
  });

  it('keeps the ui2 gate self-contained: Playwright launches the dev server with auth off and this proxy on', () => {
    // The plain `npm run test:ui2:gate` must work with no operator exports: the
    // test-only env lives in playwright.config.ts webServer.env (which Playwright
    // merges over process.env), never in production defaults or .env.local. If this
    // block goes missing, the gate boots the Firebase sign-in build with no backend
    // proxy again and both Playwright assertions fail for an environment reason.
    const config = fs.readFileSync(
      path.resolve(__dirname, '../..', '..', 'playwright.config.ts'),
      'utf8',
    );

    expect(config).toContain("REACT_APP_AUTH_DISABLED: 'true'");
    expect(config).toContain('REACT_APP_DEV_API_PROXY_TARGET: backendURL');
    expect(config).toContain("const backendURL = process.env.E2E_BACKEND_URL || 'http://127.0.0.1:8765'");
  });
});

describe('API requests behind the local development proxy', () => {
  // apiBase reads process.env once, at module load, because CRA inlines these at
  // build time. Every case therefore needs a fresh module registry.
  const load = (env: Record<string, string | undefined>) => {
    const saved = { ...process.env };
    Object.entries(env).forEach(([key, value]) => {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    });
    let mod: typeof import('../../config/apiBase');
    jest.isolateModules(() => {
      mod = require('../../config/apiBase');
    });
    process.env = saved;
    return mod!;
  };

  it('sends requests to the frontend origin so the proxy can reach the chosen port', () => {
    const { API_BASE_URL } = load({
      REACT_APP_IDE: undefined,
      REACT_APP_API_URL: '',
      REACT_APP_DEV_API_PROXY_TARGET: 'http://localhost:9012',
    });
    expect(API_BASE_URL).toBe('');
  });

  it('keeps same-origin requests when no API override or development proxy is configured', () => {
    const { API_BASE_URL } = load({
      REACT_APP_IDE: undefined,
      REACT_APP_API_URL: '',
      REACT_APP_DEV_API_PROXY_TARGET: undefined,
    });
    expect(API_BASE_URL).toBe('');
  });

  it('an explicit API URL still wins over the proxy', () => {
    const { API_BASE_URL } = load({
      REACT_APP_IDE: undefined,
      REACT_APP_API_URL: 'https://api.example.com',
      REACT_APP_DEV_API_PROXY_TARGET: 'http://localhost:9012',
    });
    expect(API_BASE_URL).toBe('https://api.example.com');
  });
});
