import { createLocalBrowserViewControl } from '../../ide/localBrowserViewControl';
import { configureCodeFocusOn, setCodeFocusOn } from '../../ide/codeFocusOn';
import type { LocalBrowserViewTransport, LocalBrowserViewMessage } from '../../ide/localBrowserViewTransport';
import type { LocalBrowserTargetIdentity } from '../../ide/protocol.generated';

const TARGET: LocalBrowserTargetIdentity = { workspaceFolder: '/work', file: 'net.py', cls: 'Net', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 };
beforeEach(() => { localStorage.clear(); configureCodeFocusOn('host'); });

test('hiding a view clears its selection and suppresses hidden selection updates', () => {
  const r = rig(), listener = jest.fn();
  window.addEventListener('builder-command', listener);
  try {
    Object.defineProperty(document, 'hidden', { configurable: true, value: true });
    document.dispatchEvent(new Event('visibilitychange'));
    window.dispatchEvent(new CustomEvent('tensorviz-selection', { detail: { nodeIds: ['fc'], edgeIds: [] } }));
    expect(r.sent).toEqual([{ type: 'viewSelection', viewId: 'view-one', target: TARGET, nodeIds: [], edgeIds: [], follow: false }]);
    Object.defineProperty(document, 'hidden', { configurable: true, value: false });
    document.dispatchEvent(new Event('visibilitychange'));
    expect(listener.mock.calls[0][0].detail.command).toBe('report-selection');
  } finally {
    delete (document as any).hidden;
    window.removeEventListener('builder-command', listener); r.control.dispose();
  }
});
function rig() {
  let current = TARGET;
  let receive!: (message: LocalBrowserViewMessage) => void;
  const sent: any[] = [];
  const transport = { session: { viewId: 'view-one' }, send: (message: any) => { sent.push(message); return message; },
    onMessage: (listener: typeof receive) => { receive = listener; return () => {}; }, onStatusChange: () => () => {},
  } as unknown as LocalBrowserViewTransport;
  const control = createLocalBrowserViewControl(transport, () => current);
  return { control, sent, retarget() { current = { ...TARGET, backendRevision: 2 }; },
    focus(fields = {}) { receive({ type: 'focusView', protocolVersion: 1, workspaceSessionId: 's', messageId: 'm', viewId: 'view-one', target: TARGET, requestId: 'r', nodeIds: ['fc'], edgeIds: [], ...fields }); } };
}

test('selection uses the broker-assigned view and current target; oversized selection is explicit', () => {
  const r = rig();
  try {
    window.dispatchEvent(new CustomEvent('tensorviz-selection', { detail: { nodeIds: ['fc'], edgeIds: [] } }));
    expect(r.sent[0]).toMatchObject({ viewId: 'view-one', target: TARGET, nodeIds: ['fc'] });
    window.dispatchEvent(new CustomEvent('tensorviz-selection', { detail: { nodeIds: Array.from({ length: 101 }, (_, i) => String(i)), edgeIds: [] } }));
    expect(r.sent[1]).toMatchObject({ tooLarge: true, nodeIds: [], edgeIds: [] });
  } finally { r.control.dispose(); }
});

test('each selection carries Code focus, and turning it on or off re-reports the selection', () => {
  const r = rig(), listener = jest.fn();
  window.addEventListener('builder-command', listener);
  try {
    window.dispatchEvent(new CustomEvent('tensorviz-selection', { detail: { nodeIds: ['fc'], edgeIds: [] } }));
    expect(r.sent.at(-1)).toMatchObject({ nodeIds: ['fc'], follow: false });
    setCodeFocusOn(true);
    expect(listener.mock.calls.at(-1)[0].detail.command).toBe('report-selection');
    window.dispatchEvent(new CustomEvent('tensorviz-selection', { detail: { nodeIds: ['fc'], edgeIds: [] } }));
    expect(r.sent.at(-1)).toMatchObject({ nodeIds: ['fc'], follow: true });
    r.control.dispose();
    listener.mockClear();
    setCodeFocusOn(false);
    expect(listener).not.toHaveBeenCalled();
  } finally { window.removeEventListener('builder-command', listener); r.control.dispose(); }
});

test('another view or a stale revision cannot focus this canvas', () => {
  const r = rig(), listener = jest.fn();
  window.addEventListener('builder-command', listener);
  try {
    r.focus({ viewId: 'other' });
    r.retarget(); r.focus();
    expect(listener).not.toHaveBeenCalled();
    expect(r.sent).toHaveLength(1);
    expect(r.sent[0]).toMatchObject({ type: 'viewFocusResult', ok: false });
  } finally { window.removeEventListener('builder-command', listener); r.control.dispose(); }
});

test('focus is acknowledged only after the canvas confirms, and a source change wins', () => {
  const r = rig();
  let finish!: (ok: boolean) => void;
  const listener = (event: Event) => { finish = (event as CustomEvent).detail.payload.finish; };
  window.addEventListener('builder-command', listener);
  try {
    r.focus();
    expect(r.sent).toHaveLength(0);
    r.retarget(); finish(true);
    expect(r.sent[0]).toMatchObject({ type: 'viewFocusResult', ok: false });
    finish(true);
    expect(r.sent).toHaveLength(1);
  } finally { window.removeEventListener('builder-command', listener); r.control.dispose(); }
});
