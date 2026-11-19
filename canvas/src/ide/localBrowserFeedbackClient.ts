/** Send feedback from the Canvas's feedback button. The member's words, and the open
 * model's class and block count only when they include it, go to the host over the local
 * connection; the host adds its version and operating system and posts it to the
 * tensorViz team. Nothing is sent until the member presses Send. */
import { LOCAL_BROWSER_FEEDBACK_MAX_LENGTH, newLocalBrowserMessageId, type LocalBrowserFeedbackKind } from './protocol.generated';
import type { LocalBrowserViewTransport } from './localBrowserViewTransport';

export type FeedbackState =
  | { phase: 'idle' }
  | { phase: 'sending' }
  | { phase: 'sent'; detail: string }
  | { phase: 'failed'; detail: string };

export interface FeedbackDraft {
  kind: LocalBrowserFeedbackKind;
  message: string;
  graph: { model: string; nodes: number } | null;
}

export interface LocalBrowserFeedbackClient {
  readonly state: FeedbackState;
  subscribe(listener: () => void): () => void;
  send(draft: FeedbackDraft): void;
  reset(): void;
  dispose(): void;
}

export const IDLE_FEEDBACK: FeedbackState = { phase: 'idle' };
const UNREACHABLE = 'Feedback could not reach tensorViz here. Reconnect and try again.';
/** The host answers within the website's own timeout; this only ends a lost reply. */
const REPLY_TIMEOUT_MS = 45_000;

export function createLocalBrowserFeedbackClient(transport: LocalBrowserViewTransport): LocalBrowserFeedbackClient {
  let state: FeedbackState = IDLE_FEEDBACK;
  let pending: { id: string; timer: ReturnType<typeof setTimeout> } | null = null;
  let disposed = false;
  const listeners = new Set<() => void>();
  const set = (next: FeedbackState) => { state = next; listeners.forEach(listener => listener()); };
  const finish = (next: FeedbackState) => { if (pending) clearTimeout(pending.timer); pending = null; set(next); };
  const connected = () => transport.status.phase === 'connected' && transport.status.hostLink === 'connected';

  const offMessage = transport.onMessage(message => {
    if (!pending) return;
    if (message.type === 'error') { finish({ phase: 'failed', detail: UNREACHABLE }); return; }
    if (message.type !== 'feedbackResult' || message.requestId !== pending.id) return;
    finish(message.status === 'sent' ? { phase: 'sent', detail: message.detail } : { phase: 'failed', detail: message.detail });
  });
  const offStatus = transport.onStatusChange(() => {
    if (pending && !connected()) finish({ phase: 'failed', detail: 'The connection to tensorViz ended before your feedback was sent. Reconnect and try again.' });
  });

  const client: LocalBrowserFeedbackClient = {
    get state() { return state; },
    subscribe(listener) { listeners.add(listener); return () => { listeners.delete(listener); }; },
    send(draft) {
      if (disposed || pending) return;
      const message = draft.message.trim();
      if (!message) { set({ phase: 'failed', detail: 'Write a few words of feedback first.' }); return; }
      if (message.length > LOCAL_BROWSER_FEEDBACK_MAX_LENGTH) {
        set({ phase: 'failed', detail: `Keep feedback under ${LOCAL_BROWSER_FEEDBACK_MAX_LENGTH} characters.` }); return;
      }
      if (!connected()) { set({ phase: 'failed', detail: UNREACHABLE }); return; }
      const id = newLocalBrowserMessageId();
      pending = { id, timer: setTimeout(() => { if (pending?.id === id) finish({ phase: 'failed', detail: UNREACHABLE }); }, REPLY_TIMEOUT_MS) };
      set({ phase: 'sending' });
      if (!transport.send({ type: 'feedbackRequest', requestId: id, kind: draft.kind, message, graph: draft.graph })) finish({ phase: 'failed', detail: UNREACHABLE });
    },
    reset() { if (!pending) set(IDLE_FEEDBACK); },
    dispose() {
      if (disposed) return;
      disposed = true;
      if (pending) clearTimeout(pending.timer);
      pending = null;
      offMessage(); offStatus(); listeners.clear();
    },
  };
  return client;
}
