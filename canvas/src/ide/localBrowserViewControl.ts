import type { LocalBrowserTargetIdentity } from './protocol.generated';
import type { LocalBrowserViewTransport } from './localBrowserViewTransport';
import { sameValidationTarget } from './localBrowserValidationClient';

/** Presentation bridge for this exact browser connection. No source writes. */
export function createLocalBrowserViewControl(transport: LocalBrowserViewTransport, target: () => LocalBrowserTargetIdentity | null) {
  const pending = new Set<() => void>();
  let disposed = false;
  const refresh = () => {
    if (!disposed && transport.session?.viewId) window.dispatchEvent(new CustomEvent('builder-command', { detail: { command: 'report-selection' } }));
  };
  const onSelection = (event: Event) => {
    const current = target(), viewId = transport.session?.viewId;
    const detail = (event as CustomEvent).detail;
    if (disposed || document.hidden || !current || !viewId || !Array.isArray(detail?.nodeIds) || !Array.isArray(detail?.edgeIds)) return;
    // Never silently turn a huge selection into a different, smaller one.
    if (detail.nodeIds.length > 100 || detail.edgeIds.length > 100) {
      transport.send({ type: 'viewSelection', viewId, target: current, nodeIds: [], edgeIds: [], tooLarge: true });
      return;
    }
    transport.send({ type: 'viewSelection', viewId, target: current, nodeIds: detail.nodeIds, edgeIds: detail.edgeIds });
  };
  window.addEventListener('tensorviz-selection', onSelection);
  const onVisibility = () => {
    const current = target(), viewId = transport.session?.viewId;
    if (document.hidden && current && viewId) transport.send({ type: 'viewSelection', viewId, target: current, nodeIds: [], edgeIds: [] });
    else refresh();
  };
  document.addEventListener('visibilitychange', onVisibility);
  const offStatus = transport.onStatusChange(status => {
    if (status.phase === 'connected' && status.hostLink === 'connected') refresh();
    else for (const cancel of Array.from(pending)) cancel();
  });
  const offMessage = transport.onMessage(message => {
    if (message.type !== 'focusView' || message.viewId !== transport.session?.viewId) return;
    let settled = false;
    const finish = (ok: boolean) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      pending.delete(cancel);
      const current = target();
      const applied = ok && !disposed && !!current && sameValidationTarget(current, message.target);
      if (!disposed) transport.send({ type: 'viewFocusResult', viewId: message.viewId, target: message.target, requestId: message.requestId,
        ok: applied, detail: applied ? 'Canvas selection focused.' : 'The canvas was not ready or its source changed. Refresh the graph and retry.' });
    };
    const cancel = () => finish(false);
    const timer = setTimeout(cancel, 3000);
    pending.add(cancel);
    const current = target();
    if (!current || !sameValidationTarget(current, message.target)) { cancel(); return; }
    window.dispatchEvent(new CustomEvent('builder-command', { detail: {
      command: 'focus-selection', payload: { nodeIds: message.nodeIds, edgeIds: message.edgeIds, target: message.target, finish },
    } }));
  });
  return { refresh, dispose() {
    disposed = true;
    for (const cancel of Array.from(pending)) cancel();
    window.removeEventListener('tensorviz-selection', onSelection);
    document.removeEventListener('visibilitychange', onVisibility);
    offMessage(); offStatus();
  } };
}
