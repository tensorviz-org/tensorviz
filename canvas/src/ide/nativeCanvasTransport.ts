import { captureNativePresentation, restoreNativePresentation } from './nativeCanvasPresentation';
import { setCanvasHostTheme } from '../styles/canvasTheme';
import {
  isLocalBrowserMessage, LOCAL_BROWSER_PROTOCOL_VERSION, LocalBrowserMessage,
  LocalBrowserSessionInfo, newLocalBrowserMessageId,
} from './protocol.generated';
import type {
  LocalBrowserViewMessage, LocalBrowserViewSend, LocalBrowserViewStatus, LocalBrowserViewTransport,
} from './localBrowserViewTransport';

/** LocalBrowserApp's existing controller protocol over the authenticated shell.
 * This transport never bootstraps cookies, opens sockets or calls model APIs. */
export function createNativeCanvasTransport(): LocalBrowserViewTransport {
  let origin: string | null = null;
  let session: LocalBrowserSessionInfo | null = null;
  let status: LocalBrowserViewStatus = { phase: 'idle', hostLink: 'unknown', error: null };
  let violations = 0;
  let disposed = false;
  let connecting: Promise<void> | undefined;
  let resolveConnect: (() => void) | undefined;
  let rejectConnect: ((error: Error) => void) | undefined;
  let timeout: ReturnType<typeof setTimeout> | undefined;
  const messages = new Set<(message: LocalBrowserViewMessage) => void>();
  const statuses = new Set<(status: LocalBrowserViewStatus) => void>();
  const setStatus = (next: LocalBrowserViewStatus) => {
    status = next;
    statuses.forEach(listener => listener(next));
  };
  const receive = (event: MessageEvent) => {
    if (disposed || event.source !== window.parent) return;
    const data = event.data;
    if (data?.type === 'canvasConnected' && typeof data.sessionId === 'string' && data.sessionId &&
        typeof data.viewId === 'string' && data.viewId.length > 0 && data.viewId.length <= 128) {
      if (origin !== null && event.origin !== origin) return;
      origin = event.origin;
      if (data.theme === 'light' || data.theme === 'dark') setCanvasHostTheme(data.theme);
      restoreNativePresentation(data.presentation);
      session = { workspaceSessionId: data.sessionId, viewId: data.viewId, protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
        role: 'view', expiresAt: '', capabilities: { readOnly: true, canNavigateSource: true,
          canWriteSource: false, canRunAgent: false, canEditGraph: false, canProposeSourceEdits: true } };
      clearTimeout(timeout);
      setStatus({ phase: 'connected', hostLink: 'connected', error: null });
      resolveConnect?.();
      return;
    }
    if (data?.type === 'canvasCheckpoint' && session && event.origin === origin && typeof data.requestId === 'string') {
      window.parent.postMessage({ type: 'canvasPresentation', requestId: data.requestId,
        state: captureNativePresentation() }, origin);
      return;
    }
    if (data?.type === 'canvasTheme' && session && event.origin === origin) {
      if (data.theme === 'light' || data.theme === 'dark') setCanvasHostTheme(data.theme);
      return;
    }
    if (data?.type === 'canvasVisible' && session && event.origin === origin) {
      // Re-register selection after native hiding without restoring the camera
      // or asking the host to reparse an unchanged projection.
      setStatus({ phase: 'connected', hostLink: 'connected', error: null });
      return;
    }
    if (data?.type !== 'canvasMessage' || !session || event.origin !== origin) return;
    if (!isLocalBrowserMessage(data.message, session.workspaceSessionId)) { violations++; return; }
    messages.forEach(listener => listener(data.message as LocalBrowserViewMessage));
  };
  window.addEventListener('message', receive);
  return {
    get status() { return status; },
    get session() { return session; },
    get protocolViolations() { return violations; },
    connect() {
      if (disposed) return Promise.reject(new Error('Canvas transport disposed'));
      if (connecting) return connecting;
      setStatus({ phase: 'connecting', hostLink: 'unknown', error: null });
      connecting = new Promise<void>((resolve, reject) => {
        resolveConnect = resolve; rejectConnect = reject;
        timeout = setTimeout(() => {
          setStatus({ phase: 'ended', hostLink: 'disconnected', error: null });
          reject(new Error('Canvas host did not answer. Reopen Canvas to retry.'));
        }, 15_000);
        window.parent.postMessage({ type: 'canvasConnect', protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION }, '*');
      });
      return connecting;
    },
    send(message: LocalBrowserViewSend): LocalBrowserMessage | null {
      if (disposed || status.phase !== 'connected' || !session || !origin) return null;
      const full = { ...message, protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
        workspaceSessionId: session.workspaceSessionId, messageId: newLocalBrowserMessageId() };
      if (!isLocalBrowserMessage(full, session.workspaceSessionId)) throw new Error(`Invalid Canvas request: ${message.type}`);
      window.parent.postMessage({ type: 'canvasMessage', message: full }, origin);
      return full;
    },
    onMessage(listener) { messages.add(listener); return () => { messages.delete(listener); }; },
    onStatusChange(listener) { statuses.add(listener); return () => { statuses.delete(listener); }; },
    dispose() {
      if (disposed) return;
      disposed = true;
      clearTimeout(timeout);
      window.removeEventListener('message', receive);
      rejectConnect?.(new Error('Canvas transport disposed'));
      messages.clear(); statuses.clear();
      // Disposing presentation does not send cancellation to the host.
    },
  };
}
