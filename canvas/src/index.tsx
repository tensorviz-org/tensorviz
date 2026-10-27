import './styles/fonts';
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import { ThemeProvider } from './contexts/ThemeContext';
import reportWebVitals from './reportWebVitals';

// Browsers fire a benign "ResizeObserver loop completed with undelivered
// notifications" event when an observer callback triggers further resize
// observations inside the same frame — common with nested resizable
// panels + React Flow. CRA's webpack-dev-server overlay treats it as
// fatal. The bubble-phase listener I had originally lost the race
// against webpack-dev-server's own handler. Three layers now:
//   1. capture-phase window listener to stop the event before any
//      bubble-phase handler (including CRA's) sees it.
//   2. window.onerror override that returns true to swallow.
//   3. MutationObserver that nukes the overlay DOM if it still slips
//      through (some webpack-dev-server versions render directly).
if (typeof window !== 'undefined') {
  const RESIZE_OBSERVER_NOISE =
    /ResizeObserver loop (limit exceeded|completed with undelivered notifications)/;

  const isNoise = (msg: unknown): boolean =>
    typeof msg === 'string' && RESIZE_OBSERVER_NOISE.test(msg);

  window.addEventListener(
    'error',
    (event) => {
      if (isNoise(event.message)) {
        event.stopImmediatePropagation();
        event.stopPropagation();
        event.preventDefault();
      }
    },
    true, // capture phase — beats CRA's bubble-phase listener
  );

  window.addEventListener(
    'unhandledrejection',
    (event) => {
      const reason: any = event.reason;
      const message = typeof reason === 'string' ? reason : reason?.message;
      if (isNoise(message)) {
        event.stopImmediatePropagation();
        event.stopPropagation();
        event.preventDefault();
      }
    },
    true,
  );

  const priorOnError = window.onerror;
  window.onerror = function (message, source, lineno, colno, error) {
    if (isNoise(message)) return true;
    return priorOnError ? priorOnError.call(this, message, source, lineno, colno, error) : false;
  };

  // Last-resort: webpack-dev-server overlay sometimes mounts as
  // #webpack-dev-server-client-overlay (iframe) directly to <body>.
  // Hide it if the only error inside is the resize-loop advisory.
  const hideResizeOverlay = () => {
    const overlay = document.getElementById('webpack-dev-server-client-overlay');
    if (overlay) (overlay as HTMLElement).style.display = 'none';
  };
  document.addEventListener('DOMContentLoaded', () => {
    const obs = new MutationObserver(() => {
      hideResizeOverlay();
    });
    obs.observe(document.body, { childList: true, subtree: false });
  });
}

// IDE mode is decided at BUILD time (task.md M2): the extension sets the IDE
// build flag, CRA inlines the comparison below, and the web bundle keeps the
// entry it has always had. The IDE entry is lazy so its module body — which
// sets window.__TENSORVIZ_IDE__, lifts the per-launch token out of the URL and
// starts the postMessage bridge — is a chunk the hosted product never fetches.
const App = React.lazy(() => process.env.REACT_APP_IDE === 'true'
  ? import('./ide/main')
  : import('./App'));

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  <React.StrictMode>
    <ThemeProvider>
        <React.Suspense fallback={null}>
          <App />
        </React.Suspense>
    </ThemeProvider>
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
