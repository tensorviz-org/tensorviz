import type { CaptureExportReference, LocalBrowserTargetIdentity } from './protocol.generated';
import { newLocalBrowserMessageId } from './protocol.generated';
import type { LocalBrowserViewTransport } from './localBrowserViewTransport';
import { sameValidationTarget } from './localBrowserValidationClient';
import { requirePortableCapture, type PortableCapture } from '../services/portableCapture';

export interface PortableExportResult { capture: PortableCapture | null; detail: string }
export interface LocalBrowserPortableClient {
  prepare(reference: CaptureExportReference): Promise<PortableExportResult>;
  cancel(): void;
  dispose(): void;
}

/** Export is private to the asking view and exact displayed target. No raw
 * capture source, launch token or model execution enters this client. */
export function createLocalBrowserPortableClient(transport: LocalBrowserViewTransport,
  activeTarget: () => LocalBrowserTargetIdentity | null): LocalBrowserPortableClient {
  let disposed = false;
  let pending: { id: string; target: LocalBrowserTargetIdentity; reference: CaptureExportReference;
    timer: ReturnType<typeof setTimeout>; resolve(value: PortableExportResult): void } | null = null;
  const settle = (value: PortableExportResult) => {
    const entry = pending; pending = null;
    if (entry) { clearTimeout(entry.timer); entry.resolve(value); }
  };
  const cancel = (detail = 'Capture export was cancelled.') => {
    if (!pending) return;
    try { transport.send({ type: 'captureExportCancel', target: pending.target, requestId: pending.id }); }
    catch { /* Disconnection also removes this request from the broker. */ }
    finally { settle({ capture: null, detail }); }
  };
  const offMessage = transport.onMessage(message => {
    if (!pending) return;
    if (message.type === 'captureExportResult' && message.requestId === pending.id && sameValidationTarget(message.target, pending.target)) {
      if (message.reference.captureId !== pending.reference.captureId || message.reference.sourceId !== pending.reference.sourceId) return;
      if (message.capture === null) { settle({ capture: null, detail: message.detail }); return; }
      try {
        const capture = requirePortableCapture(message.capture);
        if (capture.capture.id !== pending.reference.captureId) throw new Error('capture mismatch');
        settle({ capture, detail: message.detail });
      } catch { settle({ capture: null, detail: 'The exported capture was invalid or belonged to another graph. Export refused.' }); }
    } else if (message.type === 'activeTarget' || message.type === 'projection' || message.type === 'snapshot'
      || message.type === 'staleState' || message.type === 'syncState') {
      const target = message.type === 'snapshot' ? message.state.activeTarget : message.target;
      if (!target || !sameValidationTarget(target, pending.target)) cancel('The graph changed during export. Export its current version again.');
    } else if (message.type === 'error') cancel('Export could not reach VS Code. Reconnect and try again.');
  });
  const offStatus = transport.onStatusChange(status => {
    if (status.phase !== 'connected' || status.hostLink !== 'connected') settle({ capture: null, detail: 'The connection ended during export. Reconnect and try again.' });
  });
  return {
    prepare(reference) {
      cancel();
      const target = activeTarget();
      if (disposed || !target || transport.status.phase !== 'connected' || transport.status.hostLink !== 'connected') {
        return Promise.resolve({ capture: null, detail: 'Connect to VS Code to export the retained capture.' });
      }
      const requestId = newLocalBrowserMessageId();
      const promise = new Promise<PortableExportResult>(resolve => {
        pending = { id: requestId, target: { ...target }, reference: { ...reference }, resolve,
          timer: setTimeout(() => cancel('Capture export timed out. Reconnect and try again.'), 15_000) };
      });
      try {
        if (!transport.send({ type: 'captureExportRequest', target, requestId, reference })) settle({ capture: null, detail: 'Capture export could not reach VS Code.' });
      } catch { settle({ capture: null, detail: 'Capture export could not start.' }); }
      return promise;
    },
    cancel,
    dispose() { if (disposed) return; disposed = true; cancel(); offMessage(); offStatus(); },
  };
}
