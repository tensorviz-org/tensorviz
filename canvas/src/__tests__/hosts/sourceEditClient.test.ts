import { createLocalBrowserSourceEditClient } from '../../ide/localBrowserSourceEditClient';
import { LocalBrowserViewTransport } from '../../ide/localBrowserViewTransport';

const target = { workspaceFolder: '/work', file: 'net.py', cls: 'Net', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 };
function harness() {
  let message: (value: any) => void = () => undefined;
  let status: (value: any) => void = () => undefined;
  const send = jest.fn(value => value);
  const transport = { status: { phase: 'connected', hostLink: 'connected', error: null }, send,
    onMessage(fn: any) { message = fn; return jest.fn(); }, onStatusChange(fn: any) { status = fn; return jest.fn(); },
  } as unknown as LocalBrowserViewTransport;
  const client = createLocalBrowserSourceEditClient(transport, () => target);
  return { client, send, message: (value: any) => message(value), status: (value: any) => status(value) };
}

test('same-model reprojection before the apply receipt does not report a false cancellation', async () => {
  const h = harness();
  const handle = h.client.propose({ kind: 'remove_layer', nodeId: 'act' });
  h.message({ type: 'projection', target: { ...target, documentVersion: 2, backendRevision: 2 } });
  h.message({ type: 'sourceEditResult', requestId: handle.requestId, target, status: 'applied', detail: 'Applied.' });
  expect(await handle.promise).toMatchObject({ status: 'applied' });
  expect(h.send).toHaveBeenCalledTimes(1);
  expect(h.send.mock.calls[0][0]).toMatchObject({ type: 'sourceEditRequest', target, edit: { kind: 'remove_layer', nodeId: 'act' } });
});

test('retargeting cancels the pending proposal and ignores a late success', async () => {
  const h = harness();
  const handle = h.client.propose({ kind: 'remove_layer', nodeId: 'act' });
  h.message({ type: 'activeTarget', target: { ...target, cls: 'Other' } });
  h.message({ type: 'sourceEditResult', requestId: handle.requestId, target, status: 'applied', detail: 'Late.' });
  expect(await handle.promise).toMatchObject({ status: 'cancelled' });
  expect(h.send.mock.calls[1][0].type).toBe('sourceEditCancel');
});

test('a disconnect reports uncertainty instead of claiming the source was unchanged', async () => {
  const h = harness();
  const handle = h.client.propose({ kind: 'remove_layer', nodeId: 'act' });
  h.status({ phase: 'reconnecting', hostLink: 'disconnected' });
  expect(await handle.promise).toMatchObject({ status: 'refused', detail: expect.stringContaining('Check the VS Code buffer') });
});

test('duplicate concurrent proposals are refused without another outbound message', async () => {
  const h = harness();
  const first = h.client.propose({ kind: 'remove_layer', nodeId: 'act' });
  expect(await h.client.propose({ kind: 'remove_layer', nodeId: 'fc' }).promise).toMatchObject({ status: 'refused' });
  expect(h.send).toHaveBeenCalledTimes(1);
  h.client.dispose();
  expect(await first.promise).toMatchObject({ status: 'cancelled' });
});
