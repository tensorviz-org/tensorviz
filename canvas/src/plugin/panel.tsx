import '../styles/fonts';
import { configureCanvasTheme, setCanvasHostTheme } from '../styles/canvasTheme';
import React from 'react';
import ReactDOM from 'react-dom/client';
import { Maximize2, Minimize2 } from 'lucide-react';
import { App, applyDocumentTheme, applyHostStyleVariables } from '@modelcontextprotocol/ext-apps';
import { getAssetUrls } from '@tldraw/assets/selfHosted';
import '../index.css';
import LocalBrowserApp from '../ide/LocalBrowserApp';
import { CanvasAssetContext } from '../canvas/TldrawCanvas';
import { createPanelTransport } from './panelTransport';
import { panelAssetFetch } from './panelAssets';

declare global { interface Window { __TENSORVIZ_PANEL_FILES__: Record<string, string> } }
window.fetch = panelAssetFetch(window.__TENSORVIZ_PANEL_FILES__, window.fetch.bind(window));
window.__TENSORVIZ_IDE__ = true;
window.__TENSORVIZ_LOCAL_BROWSER__ = true;
configureCanvasTheme('host');
// Canvas fills its viewport; measuring it as document content collapses inline hosts.
const app = new App({ name: 'tensorViz graph', version: process.env.REACT_APP_TENSORVIZ_VERSION! }, {}, { autoResize: false });
const root = ReactDOM.createRoot(document.getElementById('root')!);
const assets = getAssetUrls(asset => window.__TENSORVIZ_PANEL_FILES__[asset.replace(/^\.\//, '')]);
let active: ReturnType<typeof createPanelTransport> | undefined;
let activeId: string | undefined;

function Panel({ panelId }: { panelId: string }) {
  const [waiting, setWaiting] = React.useState(true);
  const [notice, setNotice] = React.useState<string | null>(null);
  const [context, setContext] = React.useState(() => app.getHostContext());
  React.useEffect(() => {
    const changed = () => setContext(app.getHostContext());
    app.addEventListener('hostcontextchanged', changed);
    return () => app.removeEventListener('hostcontextchanged', changed);
  }, []);
  const fullscreen = context?.displayMode === 'fullscreen';
  const displayLabel = fullscreen ? 'Exit full screen' : 'Full screen';
  const displayControl = context?.availableDisplayModes?.includes(fullscreen ? 'inline' : 'fullscreen')
    ? <button type="button" className="local-canvas-control" aria-label={displayLabel} title={displayLabel}
      onClick={() => { void app.requestDisplayMode({ mode: fullscreen ? 'inline' : 'fullscreen' })
        .then(() => setContext(app.getHostContext()))
        .catch(() => setNotice('The host could not change the graph display mode.')); }}>
      {fullscreen ? <Minimize2 size={15} aria-hidden="true" /> : <Maximize2 size={15} aria-hidden="true" />}
    </button> : null;
  const createTransport = React.useCallback(() => {
    active = createPanelTransport(app, panelId, setWaiting, setNotice);
    return active;
  }, [panelId]);
  return <CanvasAssetContext.Provider value={assets}>
    <LocalBrowserApp hostKind="standalone" createTransport={createTransport} hostControls={displayControl} />
    {notice && <div role="status" style={{ position: 'fixed', bottom: 8, left: 16, right: 16,
      padding: 12, background: 'var(--tv-popover)', color: 'var(--tv-warning)', fontFamily: 'Inter, sans-serif', zIndex: 1000 }}>{notice}</div>}
    {waiting && <div role="status" style={{ position: 'fixed', inset: 0, display: 'grid', placeContent: 'center',
      padding: 32, background: 'var(--tv-canvas)', color: 'var(--tv-text)', fontFamily: 'Inter, sans-serif' }}>
      <h2>Model graph</h2><p>Ask your agent to open a local PyTorch model.<br />Its graph will appear here and follow your changes.</p>
    </div>}
  </CanvasAssetContext.Provider>;
}

// The opener's first result already identifies this panel. Do not open it again.
app.ontoolresult = result => {
  const id = result._meta?.panelId;
  if (typeof id !== 'string' || id === activeId) return;
  active?.dispose(); activeId = id;
  root.render(<Panel key={id} panelId={id} />);
};
function theme() {
  const context = app.getHostContext();
  if (context?.displayMode === 'inline') {
    const dimensions = context.containerDimensions;
    void app.sendSizeChanged({ height: Math.min(600, dimensions && 'maxHeight' in dimensions ? dimensions.maxHeight ?? 600 : 600) });
  }
  if (context?.theme) {
    applyDocumentTheme(context.theme);
    setCanvasHostTheme(context.theme);
  }
  if (context?.styles?.variables) applyHostStyleVariables(context.styles.variables);
}
app.onhostcontextchanged = theme;
app.onteardown = async () => { await active?.detach(); return {}; };
root.render(<p style={{ padding: 24, fontFamily: 'system-ui' }}>Connecting to tensorViz…</p>);
// History can remount a live or expired panel. Only a user action may cover the chat.
void app.connect().then(theme).catch(() => root.render(<p role="alert" style={{ padding: 24 }}>This client could not connect the graph panel. Ask your agent to open the browser graph.</p>));
window.addEventListener('pagehide', () => active?.dispose());
