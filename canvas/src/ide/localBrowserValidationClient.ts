import { LocalBrowserTargetIdentity, LocalBrowserValidationOutcome, LocalBrowserValidationEvidence, newLocalBrowserMessageId } from './protocol.generated';
import { LocalBrowserViewTransport } from './localBrowserViewTransport';

export interface ValidationReceipt extends LocalBrowserValidationOutcome {
  target: LocalBrowserTargetIdentity | null; projectionRevision?: number; runId?: string; evidence?: LocalBrowserValidationEvidence;
}
export interface LocalBrowserValidationClient {
  /** `fresh`: Run fresh — the host never reuses an identical earlier result for it. */
  validate(fresh?: boolean): Promise<ValidationReceipt>;
  cancel(): void;
  dispose(): void;
}

const fields: Array<keyof LocalBrowserTargetIdentity> = ['workspaceFolder', 'file', 'cls', 'documentVersion', 'sourceDigest', 'backendRevision'];
export const sameValidationTarget = (a: LocalBrowserTargetIdentity | null, b: LocalBrowserTargetIdentity | null) =>
  !!a && !!b && fields.every(key => a[key] === b[key]);

/** Request correlation only; the host checks freshness and owns the bounded worker
 *  deadline. The page cannot infer a saved profile's timeout or charge profile
 *  selection/probing against it. Disconnect, cancellation and retarget still settle. */
export function createLocalBrowserValidationClient(
  transport: LocalBrowserViewTransport, activeTarget: () => LocalBrowserTargetIdentity | null,
): LocalBrowserValidationClient {
  let visibleTarget = activeTarget();
  let measuredRevision: number | undefined;
  let pending: { id: string; target: LocalBrowserTargetIdentity; resolve(result: ValidationReceipt): void } | null = null;
  let disposed = false;
  const settle = (outcome: LocalBrowserValidationOutcome) => {
    const entry = pending; pending = null;
    entry?.resolve({ ...outcome, runId: entry.id, target: visibleTarget ?? entry.target, projectionRevision: measuredRevision });
  };
  const cancel = (detail = 'Validation cancelled.') => {
    if (!pending) return;
    try { transport.send({ type: 'validateCancel', requestId: pending.id, target: pending.target }); }
    catch { /* A disconnected broker also cancels the host's request. */ }
    finally { settle({ status: 'cancelled', detail }); }
  };
  const offMessage = transport.onMessage(message => {
    if (message.type === 'projection' || message.type === 'activeTarget' || message.type === 'snapshot') {
      visibleTarget = message.type === 'snapshot' ? message.state.activeTarget : message.target;
      if (pending && (!visibleTarget || ['workspaceFolder', 'file', 'cls', 'documentVersion', 'sourceDigest'].some(
        key => visibleTarget![key as keyof LocalBrowserTargetIdentity] !== pending!.target[key as keyof LocalBrowserTargetIdentity]))) {
        cancel('The model changed. Validate the newly opened model when it is ready.');
      }
    }
    if (!pending) return;
    if (message.type === 'projection' && message.traced && !message.greyed) {
      measuredRevision = message.target.backendRevision;
    }
    if (message.type === 'validateResult' && message.requestId === pending.id && sameValidationTarget(message.target, pending.target)) {
      settle(message);
    } else if ((message.type === 'staleState' || (message.type === 'syncState' && message.state === 'stale'))
      && sameValidationTarget(message.target, visibleTarget)) {
      cancel(message.reason ?? 'Graph refresh failed. Reopen the graph from VS Code and validate again.');
    } else if (message.type === 'error') {
      cancel('Validation could not reach VS Code. Reopen the graph and try again.');
    }
    // A successful check publishes its measured graph before the receipt. The
    // receipt belongs to the new projection, so a later edit can invalidate it.
  });
  const offStatus = transport.onStatusChange(status => {
    if (status.phase !== 'connected' || status.hostLink !== 'connected') {
      settle({ status: 'refused', detail: 'The connection ended during validation. Reconnect and validate again.' });
    }
  });
  return {
    validate(fresh = false) {
      const target = activeTarget();
      if (disposed || pending || !target || transport.status.phase !== 'connected' || transport.status.hostLink !== 'connected') {
        return Promise.resolve({ status: 'refused', detail: 'Connect to the current graph and wait for any running validation.', target });
      }
      visibleTarget = target;
      measuredRevision = undefined;
      const requestId = newLocalBrowserMessageId();
      const promise = new Promise<ValidationReceipt>(resolve => { pending = { id: requestId, target, resolve }; });
      try {
        if (!transport.send({ type: 'validateRequest', requestId, target,
          ...(fresh ? { fresh: true as const } : {}) })) {
          settle({ status: 'refused', detail: 'Validation could not reach VS Code. Reconnect and try again.' });
        }
      } catch {
        settle({ status: 'refused', detail: 'Validation could not start. Reopen the graph and try again.' });
      }
      return promise;
    },
    cancel,
    dispose() { if (disposed) return; disposed = true; cancel(); offMessage(); offStatus(); },
  };
}
