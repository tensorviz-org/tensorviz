/**
 * The app's half of the shell ↔ host bridge (task.md D2 "Stage A", §6.4).
 *
 * In IDE mode this document is served by the backend over http:// and framed by
 * a tiny webview shell. The shell has `acquireVsCodeApi()`; we do not. So
 * everything the app needs from the editor — writing a file, opening a link,
 * the clipboard — is a postMessage to `window.parent`, and the shell relays it.
 *
 * Origin handling, deliberately asymmetric: the shell knows our origin (it built
 * the iframe URL) and always posts with an explicit targetOrigin. We cannot know
 * the shell's `vscode-webview://<uuid>` origin ahead of time, so the first
 * message — `ready`, which carries nothing — goes to '*', and every later
 * message goes to the origin the shell's `hostReady` reply came from. If no
 * shell answers (the app opened in a plain browser tab, which is how the
 * Playwright spec loads it) the bridge simply stays inactive and every host
 * service falls back to its browser behaviour.
 */
import { AppMessage, newLocalBrowserMessageId, PROTOCOL_VERSION } from './protocol.generated';
import { setCanvasHostTheme } from '../styles/canvasTheme';

const FRAMED = typeof window !== 'undefined' && window.parent !== window;

/** How often the canvas liveness beacon goes out. */
const CANVAS_STATE_INTERVAL_MS = 2000;

let hostOrigin: string | null = null;
let started = false;

/** True once a shell has answered, which is the only proof there is a host at all. */
export function bridgeActive(): boolean {
  return hostOrigin !== null;
}

export function postToHost(message: AppMessage): void {
  if (!FRAMED || hostOrigin === null) return;
  window.parent.postMessage(message, hostOrigin);
}

/** Native paste cannot rely on Electron targeting the nested cross-origin frame. */
export function readClipboardText(): Promise<string> {
  if (!bridgeActive()) return Promise.reject(new Error('Canvas host is unavailable'));
  const requestId = newLocalBrowserMessageId();
  return new Promise((resolve, reject) => {
    const finish = (text?: string) => {
      clearTimeout(timeout);
      window.removeEventListener('message', receive);
      if (text === undefined) reject(new Error('Could not read clipboard'));
      else resolve(text);
    };
    const receive = (event: MessageEvent) => {
      if (event.source !== window.parent || event.origin !== hostOrigin) return;
      const data = event.data;
      if (data?.type !== 'clipboardText' || data.requestId !== requestId) return;
      finish(typeof data.text === 'string' && data.text.length <= 16000 ? data.text : undefined);
    };
    const timeout = window.setTimeout(() => finish(), 5000);
    window.addEventListener('message', receive);
    postToHost({ type: 'readClipboard', requestId });
  });
}

/** Explicit response copy, acknowledged by the native clipboard owner. */
export function writeClipboardText(text: string): Promise<void> {
  if (!bridgeActive() || text.length > 64000) return Promise.reject(new Error('Could not copy text'));
  const requestId = newLocalBrowserMessageId();
  return new Promise((resolve, reject) => {
    const finish = (failed: boolean) => {
      clearTimeout(timeout); window.removeEventListener('message', receive);
      if (failed) reject(new Error('Could not copy text')); else resolve();
    };
    const receive = (event: MessageEvent) => {
      if (event.source !== window.parent || event.origin !== hostOrigin) return;
      if (event.data?.type === 'clipboardWritten' && event.data.requestId === requestId) finish(event.data.error === true);
    };
    const timeout = window.setTimeout(() => finish(true), 5000);
    window.addEventListener('message', receive);
    postToHost({ type: 'clipboard', text, requestId });
  });
}

/** tldraw's own root element. Its absence after the licence grace period is the
 *  failure Stage A exists to prevent, so this is measured in the app's document —
 *  the only place that can see it — and reported outward. */
function canvasMounted(): boolean {
  return document.querySelectorAll('.tl-container').length > 0;
}

export function startBridge(): void {
  if (started || !FRAMED) return;
  started = true;

  // Keyboard events in the cross-origin canvas do not bubble to VS Code.
  // Forward this fixed native action even while a canvas input has focus.
  window.addEventListener('keydown', (event: KeyboardEvent) => {
    if (hostOrigin === null || event.altKey || !event.shiftKey ||
        !(event.metaKey || event.ctrlKey) || event.key.toLowerCase() !== 'p') return;
    event.preventDefault();
    event.stopPropagation();
    postToHost({ type: 'showCommandPalette' });
  }, true);

  window.addEventListener('message', (event: MessageEvent) => {
    if (event.source !== window.parent || (hostOrigin !== null && event.origin !== hostOrigin)) return;
    // Theme is shell presentation metadata, outside the graph protocol.
    const data = event.data as { type?: unknown; protocolVersion?: unknown; theme?: unknown } | undefined;
    if (!data || typeof data.type !== 'string') return;
    if (data.type === 'hostReady') {
      if (data.protocolVersion !== PROTOCOL_VERSION) {
        // Refuse loudly rather than half-speak an older protocol: a silently
        // ignored message is indistinguishable from a feature that never worked.
        // eslint-disable-next-line no-console
        console.error(
          `tensorViz IDE bridge: protocol ${data.protocolVersion} from the host, ${PROTOCOL_VERSION} here.`,
        );
        return;
      }
      hostOrigin = event.origin;
      if (data.theme === 'light' || data.theme === 'dark') setCanvasHostTheme(data.theme);
      postToHost({ type: 'canvasState', mounted: canvasMounted(), t: Date.now() });
      return;
    }
    if (data.type === 'canvasTheme' && hostOrigin !== null && (data.theme === 'light' || data.theme === 'dark')) {
      setCanvasHostTheme(data.theme);
    }
  });

  // '*' exactly once, and only for a message with no content: see the note above.
  window.parent.postMessage(
    { type: 'ready', protocolVersion: PROTOCOL_VERSION, t: Date.now() } as AppMessage,
    '*',
  );

  window.setInterval(() => {
    postToHost({ type: 'canvasState', mounted: canvasMounted(), t: Date.now() });
  }, CANVAS_STATE_INTERVAL_MS);
}
