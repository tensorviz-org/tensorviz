import { createLocalBrowserCaptureClient } from '../../ide/localBrowserCaptureClient';
import type { LocalBrowserViewMessage, LocalBrowserViewStatus, LocalBrowserViewTransport } from '../../ide/localBrowserViewTransport';
import type { CaptureInspectionPage, CaptureInspectionQuery } from '../../ide/protocol.generated';

const target = { workspaceFolder: '/ws', file: 'net.py', cls: 'Net', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 };
const query: CaptureInspectionQuery = { captureId: 'capture:1', sourceId: 'source:1', path: ['values'], offset: 0, limit: 2 };
const page: CaptureInspectionPage = { captureId: query.captureId, sourceId: query.sourceId, path: query.path,
  node: { key: null, kind: 'array', count: 0 }, total: 0, items: [], nextOffset: null };
function rig() {
  let receive!: (message: LocalBrowserViewMessage) => void;
  let changeStatus!: (status: LocalBrowserViewStatus) => void;
  const send = jest.fn((message: any) => ({}));
  const transport = { status: { phase: 'connected', hostLink: 'connected' }, send,
    onMessage(listener: typeof receive) { receive = listener; return () => {}; },
    onStatusChange(listener: typeof changeStatus) { changeStatus = listener; return () => {}; },
  } as unknown as LocalBrowserViewTransport;
  const client = createLocalBrowserCaptureClient(transport, () => target);
  return { client, send, changeStatus: (status: Partial<LocalBrowserViewStatus>) => changeStatus(status as LocalBrowserViewStatus),
    receive: (message: any) => receive(message),
    reply(fields: object = {}) { receive({ type: 'captureInspectResult', protocolVersion: 1, workspaceSessionId: 's', messageId: 'm',
      target, requestId: send.mock.calls[0][0].requestId, page, detail: 'Retained capture evidence.', ...fields }); } };
}

beforeEach(() => jest.useFakeTimers());
afterEach(() => jest.useRealTimers());

test('only the exact request, target, capture, entity and path can fulfill inspection', async () => {
  const r = rig();
  const settled = jest.fn(); const promise = r.client.inspect(query).then(settled);
  r.reply({ requestId: 'foreign' }); r.reply({ target: { ...target, cls: 'Other' } });
  r.reply({ page: { ...page, captureId: 'foreign' } }); r.reply({ page: { ...page, path: ['operations'] } });
  r.reply({ page: { ...page, entity: { kind: 'operation', id: 'a' } } });
  await Promise.resolve(); expect(settled).not.toHaveBeenCalled();
  r.reply(); await promise; expect(settled).toHaveBeenCalledWith({ page, detail: 'Retained capture evidence.' });
  r.client.dispose();
});

test('retained details survive a source revision but not retargeting, disconnect or disposal', async () => {
  const r = rig(); const promise = r.client.inspect(query);
  r.receive({ type: 'activeTarget', target: { ...target, documentVersion: 2 } }); r.reply();
  expect((await promise).page).toEqual(page);
  const next = r.client.inspect(query); r.receive({ type: 'activeTarget', target: { ...target, cls: 'Other' } });
  expect((await next).detail).toMatch(/selected model changed/);
  const disconnected = r.client.inspect(query); r.changeStatus({ phase: 'reconnecting' });
  expect((await disconnected).page).toBeNull();
  const disposed = r.client.inspect(query); r.client.dispose(); expect((await disposed).page).toBeNull();
  expect((await r.client.inspect(query)).page).toBeNull();
});

test('a new read cancels the old request; timeout and throwing transport settle without leaked work', async () => {
  const r = rig(); const first = r.client.inspect(query); const second = r.client.inspect({ ...query, offset: 2 });
  expect((await first).page).toBeNull(); expect(r.send.mock.calls[1][0].type).toBe('captureInspectCancel');
  jest.advanceTimersByTime(10_000); expect((await second).detail).toMatch(/timed out/);
  r.send.mockImplementation(() => { throw new Error('closed'); });
  expect((await r.client.inspect(query)).detail).toMatch(/could not start/);
  r.client.dispose(); expect(jest.getTimerCount()).toBe(0);
});
