import type { CaptureInspectionPage, CaptureInspectionQuery, LocalBrowserTargetIdentity } from './protocol.generated';
import { newLocalBrowserMessageId } from './protocol.generated';
import type { LocalBrowserViewTransport } from './localBrowserViewTransport';
import { sameValidationTarget } from './localBrowserValidationClient';

export interface CaptureInspectionResult { page: CaptureInspectionPage | null; detail: string }
export interface LocalBrowserCaptureClient {
  inspect(query: CaptureInspectionQuery): Promise<CaptureInspectionResult>;
  cancel(): void;
  dispose(): void;
}

/** One read at a time; request, target and capture identity all belong to the
 * response. Disconnect/reselection cannot attach old details to a new node. */
export function createLocalBrowserCaptureClient(transport: LocalBrowserViewTransport,
  activeTarget: () => LocalBrowserTargetIdentity | null): LocalBrowserCaptureClient {
  let disposed = false;
  let pending: { id: string; target: LocalBrowserTargetIdentity; query: CaptureInspectionQuery;
    timer: ReturnType<typeof setTimeout>; resolve(value: CaptureInspectionResult): void } | null = null;
  const settle = (value: CaptureInspectionResult) => {
    const entry = pending; pending = null;
    if (entry) { clearTimeout(entry.timer); entry.resolve(value); }
  };
  const cancel = (detail = 'Capture inspection closed.') => {
    if (!pending) return;
    try { transport.send({ type: 'captureInspectCancel', target: pending.target, requestId: pending.id }); }
    catch { /* A disconnected broker also drops this pending read. */ }
    finally { settle({ page: null, detail }); }
  };
  const offMessage = transport.onMessage(message => {
    if (!pending) return;
    if (message.type === 'captureInspectResult' && message.requestId === pending.id && sameValidationTarget(message.target, pending.target)) {
      const page = message.page, query = pending.query;
      if (page && (page.captureId !== query.captureId || page.sourceId !== query.sourceId
        || JSON.stringify(page.entity ?? null) !== JSON.stringify(query.entity ?? null)
        || JSON.stringify(page.path) !== JSON.stringify(query.path))) return;
      settle({ page, detail: message.detail });
    } else if (message.type === 'activeTarget' || message.type === 'snapshot' || message.type === 'projection') {
      const target = message.type === 'snapshot' ? message.state.activeTarget : message.target;
      if (!target || target.workspaceFolder !== pending.target.workspaceFolder || target.file !== pending.target.file || target.cls !== pending.target.cls) {
        cancel('The selected model changed. Reopen captured details.');
      }
    } else if (message.type === 'error') cancel('Capture inspection could not reach the TensorViz host. Reconnect and try again.');
  });
  const offStatus = transport.onStatusChange(status => {
    if (status.phase !== 'connected' || status.hostLink !== 'connected') settle({ page: null, detail: 'The connection ended. Reconnect to inspect captured details.' });
  });
  return {
    inspect(query) {
      cancel();
      const target = activeTarget();
      if (disposed || !target || transport.status.phase !== 'connected' || transport.status.hostLink !== 'connected') {
        return Promise.resolve({ page: null, detail: 'Connect to the model host to inspect captured details.' });
      }
      const requestId = newLocalBrowserMessageId();
      const promise = new Promise<CaptureInspectionResult>(resolve => {
        pending = { id: requestId, target: { ...target }, query: JSON.parse(JSON.stringify(query)), resolve,
          timer: setTimeout(() => cancel('Capture inspection timed out. Reconnect and try again.'), 10_000) };
      });
      try {
        if (!transport.send({ type: 'captureInspectRequest', requestId, target, query })) {
          settle({ page: null, detail: 'Capture inspection could not reach the TensorViz host.' });
        }
      } catch { settle({ page: null, detail: 'Capture inspection could not start.' }); }
      return promise;
    },
    cancel,
    dispose() { if (disposed) return; disposed = true; cancel(); offMessage(); offStatus(); },
  };
}
