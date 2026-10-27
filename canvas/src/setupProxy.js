/**
 * The dev loop's bridge from `npm start` to whichever backend is running.
 *
 * Two launch paths need it, and they name the backend differently:
 *
 *   TENSORVIZ_BACKEND_PORT          ide/dev.sh, which frames the CRA dev server
 *                                   inside the fork. In IDE mode the app calls the
 *                                   API same-origin (config/apiBase.ts), and "same
 *                                   origin" is then this server, so /api and /health
 *                                   have to reach the backend the IDE spawned on its
 *                                   per-launch port (extension/src/config.ts
 *                                   fixedBackendPort).
 *   REACT_APP_DEV_API_PROXY_TARGET  ./start.sh, which stopped exporting
 *                                   REACT_APP_API_URL so browser calls stay on the
 *                                   frontend origin instead of depending on CORS and
 *                                   private-network access.
 *
 * The IDE's port wins when both are set: it is chosen per launch and it is the
 * backend actually serving that window. With neither set this file does nothing, so
 * a bare `npm start` behaves exactly as it did before either loop existed.
 *
 * /api, /health, and /local are proxied: they are the backend's route prefixes
 * (server_v2.py, api/routers/local_browser.py). /local additionally needs
 * ws:true so the local-browser view transport's WebSocket upgrade
 * (extension/src/localBrowserHostTransport.ts, src/ide/localBrowserViewTransport.ts)
 * reaches the broker instead of stalling on the CRA dev server. Everything else
 * stays with CRA so hot reload keeps working. Cookies and headers pass through
 * unchanged, so the HttpOnly tensorviz_local_browser cookie and the per-launch
 * X-tensorViz-Token reach LocalTokenMiddleware unchanged. CRA loads
 * src/setupProxy.js by convention.
 */
const { createProxyMiddleware } = require('http-proxy-middleware');

/**
 * Where to forward, and null when nothing asked us to. `changeOrigin` differs by
 * path: the IDE backend is bound on 127.0.0.1 and expects that Host, while
 * start.sh's target is the same localhost the browser already named.
 */
const backendTarget = () => {
  const port = process.env.TENSORVIZ_BACKEND_PORT;
  if (port) {
    return { target: `http://127.0.0.1:${port}`, changeOrigin: true };
  }
  const configured = process.env.REACT_APP_DEV_API_PROXY_TARGET;
  return configured ? { target: configured, changeOrigin: false } : null;
};

module.exports = function setupProxy(app) {
  const resolved = backendTarget();
  if (!resolved) {
    return;
  }
  app.use(
    ['/api', '/health', '/local'],
    createProxyMiddleware({
      target: resolved.target,
      changeOrigin: resolved.changeOrigin,
      // /local/ws is a WebSocket endpoint; without this the upgrade never reaches
      // the broker and the view handshake fails at the dev-server boundary.
      ws: true,
      // Agent turns stream; buffering them would turn a token-by-token reply into one lump.
      selfHandleResponse: false,
      logLevel: 'warn',
    }),
  );
  console.log(`setupProxy: forwarding /api, /health and /local (ws) to ${resolved.target}`);
};

module.exports.backendTarget = backendTarget;
