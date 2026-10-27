import type { App } from '@modelcontextprotocol/ext-apps';
import { decodeLocalBrowserMessage } from '../ide/localBrowserWire';
import { isLocalBrowserMessage, LOCAL_BROWSER_PROTOCOL_VERSION, LOCAL_BROWSER_CLOSE_CODES, newLocalBrowserMessageId,
  type LocalBrowserSessionInfo } from '../ide/protocol.generated';
import type { LocalBrowserViewMessage, LocalBrowserViewStatus, LocalBrowserViewTransport } from '../ide/localBrowserViewTransport';

const READ_ONLY_MESSAGES = new Set(['viewSelection', 'viewFocusResult', 'captureInspectRequest', 'captureInspectCancel']);

/** Canvas's existing protocol over the host-owned MCP Apps channel. */
export function createPanelTransport(app: Pick<App, 'callServerTool' | 'updateModelContext'>, panelId: string,
  waiting: (value: boolean) => void, notice: (value: string | null) => void): LocalBrowserViewTransport & { detach(): Promise<void> } {
  const clientId = newLocalBrowserMessageId();
  let session: LocalBrowserSessionInfo | null = null;
  let status: LocalBrowserViewStatus = { phase: 'idle', hostLink: 'unknown', error: null };
  let sequence = 0, violations = 0, disposed = false, running = false, released = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let connected: Promise<void> | undefined;
  let resolveConnect: (() => void) | undefined;
  let rejectConnect: ((error: Error) => void) | undefined;
  let contextUpdates = Promise.resolve();
  let releasePromise = Promise.resolve();
  const pending: Record<string, unknown>[] = [];
  const messages = new Set<(message: LocalBrowserViewMessage) => void>();
  const statuses = new Set<(status: LocalBrowserViewStatus) => void>();
  const ended = () => status.phase === 'ended';
  const setStatus = (next: LocalBrowserViewStatus) => {
    if (next.phase === status.phase && next.hostLink === status.hostLink && next.error === status.error) return;
    status = next; statuses.forEach(listener => listener(next));
  };
  const release = (name: 'close_graph_panel' | 'detach_graph_panel') => {
    if (released) return;
    released = true;
    releasePromise = app.callServerTool({ name, arguments: { panelId, clientId } }).then(() => {}, () => {});
  };
  const stop = () => {
    waiting(false);
    setStatus({ phase: 'ended', hostLink: 'disconnected', error: status.error });
    rejectConnect?.(new Error('The graph panel disconnected. Reopen Model graph from this chat.'));
    release('close_graph_panel');
  };
  async function exchange() {
    if (disposed || running || ended()) return;
    clearTimeout(timer); running = true;
    const outgoing = pending.splice(0, 32);
    try {
      const name = outgoing.every(message => READ_ONLY_MESSAGES.has(String(message.type))) ? 'graph_panel_read' : 'graph_panel_exchange';
      const result = await app.callServerTool({ name, arguments: { panelId, clientId, after: sequence, messages: outgoing } });
      if (disposed) return;
      if (result.isError) throw new Error('Graph panel request failed.');
      const packet = result._meta?.panel as { session: LocalBrowserSessionInfo | null; status: LocalBrowserViewStatus;
        events: { sequence: number; wire: string }[]; waitingForModel: boolean } | undefined;
      if (!packet || !Array.isArray(packet.events) || packet.events.length > 256) throw new Error('Invalid panel response.');
      if (!packet.status || !['idle', 'connecting', 'connected', 'reconnecting', 'ended'].includes(packet.status.phase)
        || !['unknown', 'connected', 'disconnected'].includes(packet.status.hostLink)
        || (packet.status.error !== null && !(packet.status.error in LOCAL_BROWSER_CLOSE_CODES))
        || typeof packet.waitingForModel !== 'boolean') throw new Error('Invalid panel status.');
      waiting(packet.waitingForModel);
      session = packet.session;
      if (session && (session.protocolVersion !== LOCAL_BROWSER_PROTOCOL_VERSION || session.role !== 'view'
        || typeof session.viewId !== 'string' || !session.viewId || typeof session.workspaceSessionId !== 'string'
        || !session.workspaceSessionId)) throw new Error('Invalid panel session.');
      setStatus(packet.status);
      if (session && status.phase === 'connected') resolveConnect?.();
      for (const event of packet.events) {
        if (!Number.isSafeInteger(event.sequence) || event.sequence < 1) throw new Error('Invalid panel sequence.');
        if (event.sequence <= sequence) continue;
        if (event.sequence !== sequence + 1 || !session) throw new Error('Graph panel updates arrived out of order.');
        const message = await decodeLocalBrowserMessage(event.wire);
        if (!isLocalBrowserMessage(message, session.workspaceSessionId)) { violations++; throw new Error('Invalid graph message.'); }
        messages.forEach(listener => listener(message as LocalBrowserViewMessage));
        sequence = event.sequence;
      }
      if (ended()) stop();
    } catch (error) {
      // T3 refuses this call before dispatch when its approval is declined.
      // Keep reading the existing capture, but never replay the declined action.
      if (!disposed && outgoing.length && error instanceof Error && 'code' in error && error.code === -32000
        && error.message === 'MCP error -32000: Declined by the user.') {
        notice('Action cancelled. The current graph is still available.');
        return;
      }
      // Never retry a possibly applied setup/capture action after an uncertain response.
      if (!disposed) stop();
    } finally {
      running = false;
      if (!disposed && !ended()) timer = setTimeout(() => { void exchange(); }, pending.length ? 0 : 350);
    }
  }
  const dispose = () => {
    if (disposed) return;
    disposed = true; clearTimeout(timer); messages.clear(); statuses.clear();
    rejectConnect?.(new Error('Graph panel closed.'));
    release('detach_graph_panel');
  };
  return {
    get status() { return status; }, get session() { return session; }, get protocolViolations() { return violations; },
    connect() {
      if (connected) return connected;
      connected = new Promise<void>((resolve, reject) => { resolveConnect = resolve; rejectConnect = reject; });
      setStatus({ phase: 'connecting', hostLink: 'unknown', error: null });
      void exchange();
      return connected;
    },
    send(message) {
      if (disposed || status.phase !== 'connected' || !session) return null;
      const full = { ...message, protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
        workspaceSessionId: session.workspaceSessionId, messageId: newLocalBrowserMessageId() };
      if (!isLocalBrowserMessage(full, session.workspaceSessionId)) throw new Error('Invalid graph panel message.');
      if (pending.length >= 128) { stop(); return null; }
      pending.push({ ...full });
      if (message.type === 'viewSelection') {
        // Context is tied to the exact displayed revision, not source text or a guess.
        const context = { structuredContent: { tensorviz: { viewId: session.viewId, target: message.target,
          nodeIds: message.nodeIds, edgeIds: message.edgeIds, tooLarge: message.tooLarge === true } } };
        contextUpdates = contextUpdates.then(async () => {
          if (disposed || ended()) return;
          try { await app.updateModelContext(context); notice(null); }
          catch { notice('Selection context could not update. Your agent can still read this selection with tensorViz.'); }
        });
      }
      void exchange();
      return full;
    },
    onMessage(listener) { messages.add(listener); return () => { messages.delete(listener); }; },
    onStatusChange(listener) { statuses.add(listener); return () => { statuses.delete(listener); }; },
    dispose,
    async detach() { dispose(); await releasePromise; },
  };
}
