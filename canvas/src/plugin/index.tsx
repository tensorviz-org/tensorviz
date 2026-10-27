import '../styles/fonts';
import React from 'react';
import ReactDOM from 'react-dom/client';
import { getAssetUrls } from '@tldraw/assets/selfHosted';
import '../index.css';
import LocalBrowserApp from '../ide/LocalBrowserApp';
import { CanvasAssetContext } from '../canvas/TldrawCanvas';
import { configureCanvasTheme } from '../styles/canvasTheme';

// Same authenticated live renderer, with the standalone host's capabilities.
window.__TENSORVIZ_IDE__ = true;
window.__TENSORVIZ_LOCAL_BROWSER__ = true;
const assets = getAssetUrls({ baseUrl: './assets' });
configureCanvasTheme('browser');
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode><CanvasAssetContext.Provider value={assets}>
    <LocalBrowserApp hostKind="standalone" />
  </CanvasAssetContext.Provider></React.StrictMode>,
);
