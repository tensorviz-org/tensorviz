/** Extension entry: shared Canvas for native and browser views. */
import React from 'react';

import { createNativeCanvasTransport } from './nativeCanvasTransport';
import { setLocalApiToken } from '../config/apiBase';
import { startBridge } from './bridge';
import LocalBrowserApp from './LocalBrowserApp';
import { CanvasErrorBoundary } from './CanvasErrorBoundary';
import { configureCanvasTheme } from '../styles/canvasTheme';
import { getAssetUrls } from '@tldraw/assets/selfHosted';
import { CanvasAssetContext } from '../canvas/TldrawCanvas';

// The packaged viewer already carries these assets. Dev keeps its existing CDN fallback.
const assets = process.env.NODE_ENV === 'production' ? getAssetUrls({ baseUrl: './viewer/assets' }) : undefined;

window.__TENSORVIZ_IDE__ = true;

const url = new URL(window.location.href);

const view = url.searchParams.get('view');
configureCanvasTheme(view === 'canvas' ? 'host' : 'browser');

const token = url.searchParams.get('token');
if (token) {
  setLocalApiToken(token);
  url.searchParams.delete('token');
  window.history.replaceState(null, '', `${url.pathname}${url.search}${url.hash}`);
}

startBridge();

window.__TENSORVIZ_LOCAL_BROWSER__ = true;

export default function IdeApp(): React.ReactElement {
  return <CanvasAssetContext.Provider value={assets}><CanvasErrorBoundary>
    {view === 'canvas'
      ? <LocalBrowserApp createTransport={createNativeCanvasTransport} /> : <LocalBrowserApp />}
  </CanvasErrorBoundary></CanvasAssetContext.Provider>;
}
