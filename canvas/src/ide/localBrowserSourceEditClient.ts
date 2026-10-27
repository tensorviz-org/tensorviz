import { LocalBrowserSourceEdit, LocalBrowserTargetIdentity, newLocalBrowserMessageId } from './protocol.generated';
import { LocalBrowserViewTransport } from './localBrowserViewTransport';
import { createLocalBrowserCandidateClient, type LocalBrowserCandidateClient } from './localBrowserCandidateClient';

export interface SourceEditOutcome { status: 'applied' | 'cancelled' | 'refused'; detail: string }
export interface SourceEditHandle { requestId: string; promise: Promise<SourceEditOutcome> }
export interface LocalBrowserSourceEditClient {
  candidates?: LocalBrowserCandidateClient;
  propose(edit: LocalBrowserSourceEdit): SourceEditHandle;
  cancel(): void;
  dispose(): void;
}

/** Review requests share the page transport; this owns correlation, never sockets or retries. */
export function createLocalBrowserSourceEditClient(
  transport: LocalBrowserViewTransport,
  activeTarget: () => LocalBrowserTargetIdentity | null,
): LocalBrowserSourceEditClient {
  const candidates = createLocalBrowserCandidateClient(transport);
  let pending: { id: string; target: LocalBrowserTargetIdentity; resolve(value: SourceEditOutcome): void } | null = null;
  let disposed = false;
  const settle = (value: SourceEditOutcome) => { const entry = pending; pending = null; if (entry) candidates.end(entry.id); entry?.resolve(value); };
  const cancel = () => {
    if (!pending) return;
    transport.send({ type: 'sourceEditCancel', target: pending.target, requestId: pending.id });
    settle({ status: 'cancelled', detail: 'Proposal cancelled. Check VS Code for the current source.' });
  };
  const offMessage = transport.onMessage(message => {
    if (!pending) return;
    if (message.type === 'sourceEditResult' && message.requestId === pending.id) {
      const fields: Array<keyof LocalBrowserTargetIdentity> = ['workspaceFolder', 'file', 'cls', 'documentVersion', 'sourceDigest', 'backendRevision'];
      if (fields.every(key => message.target[key] === pending!.target[key])) settle(message);
    } else if (message.type === 'staleState') {
      cancel();
    } else if (message.type === 'activeTarget' || message.type === 'snapshot') {
      const target = message.type === 'snapshot' ? message.state.activeTarget : message.target;
      if (!target || target.workspaceFolder !== pending.target.workspaceFolder ||
          target.file !== pending.target.file || target.cls !== pending.target.cls) cancel();
    }
    // Same-class reprojection can precede the applied receipt. Keep the request
    // until the host answers; its native apply guard checks exact versions.
  });
  const offStatus = transport.onStatusChange(status => {
    if (status.phase !== 'connected' || status.hostLink !== 'connected') {
      settle({ status: 'refused', detail: 'The connection ended during review. Check the VS Code buffer before retrying.' });
    }
  });
  return {
    candidates: candidates.client,
    propose(edit) {
      const requestId = newLocalBrowserMessageId();
      const target = activeTarget();
      if (disposed || pending || !target || transport.status.phase !== 'connected' || transport.status.hostLink !== 'connected') {
        return { requestId, promise: Promise.resolve({ status: 'refused', detail: 'Connect to the current graph and finish any existing source review first.' }) };
      }
      const promise = new Promise<SourceEditOutcome>(resolve => { pending = { id: requestId, target, resolve }; });
      candidates.begin(requestId, target, 'capabilityId' in edit);
      try {
        const viewId = transport.session?.viewId;
        if (!transport.send({ type: 'sourceEditRequest', target, requestId, edit, ...(viewId ? { viewId } : {}) })) {
          settle({ status: 'refused', detail: 'The source review could not reach VS Code. Reconnect and try again.' });
        }
      } catch {
        settle({ status: 'refused', detail: 'That proposal is invalid or too large. Check its values and try again.' });
      }
      return { requestId, promise };
    },
    cancel,
    dispose() { if (disposed) return; disposed = true; cancel(); candidates.client.dispose(); offMessage(); offStatus(); },
  };
}
