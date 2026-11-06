/** Publish from a plugin view. The graph file Export graph would save goes to the
 * plugin host over the authenticated local connection; the host signs the member in
 * through tensorviz.dev and uploads it to their account, as private. The view opens the
 * website itself, only when the member clicks: the connect page while signing in, and
 * the graph's Share page afterwards. No token or website credential reaches this page. */
import { LOCAL_BROWSER_PUBLISH_MAX_BYTES, newLocalBrowserMessageId,
  type LocalBrowserPublishedGraph, type LocalBrowserTargetIdentity } from './protocol.generated';
import type { LocalBrowserViewTransport } from './localBrowserViewTransport';
import { sameValidationTarget } from './localBrowserValidationClient';

export type PublishState =
  | { phase: 'idle' }
  | { phase: 'preparing'; detail: string }
  | { phase: 'signing_in'; detail: string; signInUrl: string }
  | { phase: 'uploading'; detail: string }
  | { phase: 'published'; detail: string; graph: LocalBrowserPublishedGraph }
  | { phase: 'failed'; detail: string };

export interface LocalBrowserPublishClient {
  readonly state: PublishState;
  subscribe(listener: () => void): () => void;
  /** The member clicked Publish and the graph file is being prepared; false while one is running. */
  start(): boolean;
  /** Sends the prepared graph file. */
  publish(graphFile: string): void;
  /** Preparing the graph file failed. */
  fail(detail: string): void;
  /** Stops preparing or signing in. An upload already sent finishes. */
  cancel(): void;
  dismiss(): void;
  dispose(): void;
}

export const IDLE_PUBLISH: PublishState = { phase: 'idle' };

// CRA's pinned TypeScript DOM library predates this native Chromium API.
declare const CompressionStream: new (format: 'gzip') => ReadableWritablePair<Uint8Array, Uint8Array>;

/** The graph file gzipped, in base64, as the protocol carries it. */
export async function gzipBase64(text: string): Promise<string> {
  const source = new ReadableStream<Uint8Array>({ start(controller) { controller.enqueue(new TextEncoder().encode(text)); controller.close(); } });
  const reader = source.pipeThrough(new CompressionStream('gzip')).getReader();
  let binary = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    for (let index = 0; index < value.length; index += 0x8000) {
      binary += String.fromCharCode(...Array.from(value.subarray(index, index + 0x8000)));
    }
  }
  return window.btoa(binary);
}

const UNREACHABLE = 'Publish could not reach the tensorViz plugin. Reconnect and try again.';

export function createLocalBrowserPublishClient(transport: LocalBrowserViewTransport,
  activeTarget: () => LocalBrowserTargetIdentity | null,
  compress: (text: string) => Promise<string> = gzipBase64): LocalBrowserPublishClient {
  let state: PublishState = IDLE_PUBLISH;
  let pending: { id: string; target: LocalBrowserTargetIdentity } | null = null;
  let disposed = false;
  const listeners = new Set<() => void>();
  const set = (next: PublishState) => { state = next; listeners.forEach(listener => listener()); };
  const finish = (next: PublishState) => { pending = null; set(next); };
  const running = () => state.phase === 'preparing' || state.phase === 'signing_in' || state.phase === 'uploading';
  const connected = () => transport.status.phase === 'connected' && transport.status.hostLink === 'connected';

  const offMessage = transport.onMessage(message => {
    if (!pending) return;
    if (message.type === 'error') { finish({ phase: 'failed', detail: UNREACHABLE }); return; }
    if ((message.type !== 'graphPublishState' && message.type !== 'graphPublishResult')
      || message.requestId !== pending.id || !sameValidationTarget(message.target, pending.target)) return;
    if (message.type === 'graphPublishState') {
      set(message.phase === 'signing_in' && message.signInUrl
        ? { phase: 'signing_in', detail: message.detail, signInUrl: message.signInUrl }
        : { phase: 'uploading', detail: message.detail });
    } else if (message.status === 'published' && message.graph) {
      finish({ phase: 'published', detail: message.detail, graph: message.graph });
    } else finish(message.status === 'cancelled' ? IDLE_PUBLISH : { phase: 'failed', detail: message.detail });
  });
  const offStatus = transport.onStatusChange(() => {
    if (pending && !connected()) finish({ phase: 'failed', detail: 'The connection to the tensorViz plugin ended while publishing. Reconnect and publish again.' });
  });

  const client: LocalBrowserPublishClient = {
    get state() { return state; },
    subscribe(listener) { listeners.add(listener); return () => { listeners.delete(listener); }; },
    start() {
      if (disposed || running()) return false;
      set({ phase: 'preparing', detail: 'Preparing the graph file…' });
      return true;
    },
    publish(graphFile) {
      if (disposed || state.phase !== 'preparing') return;
      const target = activeTarget();
      if (!target || !connected()) { set({ phase: 'failed', detail: 'Connect to the tensorViz plugin to publish this graph.' }); return; }
      if (new Blob([graphFile]).size > LOCAL_BROWSER_PUBLISH_MAX_BYTES) { set({ phase: 'failed', detail: 'tensorViz takes graph files up to 10 MB.' }); return; }
      const asked = { id: newLocalBrowserMessageId(), target: { ...target } };
      pending = asked;
      void compress(graphFile).then(file => {
        if (pending !== asked) return;
        if (!transport.send({ type: 'graphPublishRequest', target: asked.target, requestId: asked.id, file })) finish({ phase: 'failed', detail: UNREACHABLE });
      }).catch(() => { if (pending === asked) finish({ phase: 'failed', detail: 'This graph could not be prepared for publishing.' }); });
    },
    fail(detail) { if (state.phase === 'preparing' && !pending) set({ phase: 'failed', detail }); },
    cancel() {
      const asked = pending;
      if (asked) {
        try { transport.send({ type: 'graphPublishCancel', target: asked.target, requestId: asked.id }); }
        catch { /* Leaving also cancels this view's request at the broker. */ }
      }
      finish(IDLE_PUBLISH);
    },
    dismiss() { if (!running()) set(IDLE_PUBLISH); },
    dispose() {
      if (disposed) return;
      if (pending) client.cancel();
      disposed = true;
      offMessage(); offStatus(); listeners.clear();
    },
  };
  return client;
}
