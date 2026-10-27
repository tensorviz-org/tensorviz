import { createLocalBrowserPortableClient } from '../../ide/localBrowserPortableClient';
import type { LocalBrowserViewMessage, LocalBrowserViewStatus, LocalBrowserViewTransport } from '../../ide/localBrowserViewTransport';
import portable from '../../../../tests/fixtures/contracts/portable-capture.json';

const target = { workspaceFolder: '/ws', file: 'net.py', cls: 'Net', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 };
const reference = { captureId: portable.capture.id, sourceId: 'source:0' };
function rig() {
  let receive!: (message: LocalBrowserViewMessage) => void;
  let changeStatus!: (status: LocalBrowserViewStatus) => void;
  const send = jest.fn((message: any) => ({}));
  const transport = { status: { phase: 'connected', hostLink: 'connected' }, send,
    onMessage(listener: typeof receive) { receive = listener; return () => {}; },
    onStatusChange(listener: typeof changeStatus) { changeStatus = listener; return () => {}; },
  } as unknown as LocalBrowserViewTransport;
  const client = createLocalBrowserPortableClient(transport, () => target);
  return { client, send, changeStatus: (status: Partial<LocalBrowserViewStatus>) => changeStatus(status as LocalBrowserViewStatus),
    receive: (message: any) => receive(message),
    reply(fields: object = {}) { receive({ type: 'captureExportResult', protocolVersion: 1, workspaceSessionId: 's', messageId: 'm',
      target, requestId: send.mock.calls[0][0].requestId, reference, capture: portable, detail: 'Prepared.', ...fields }); } };
}

beforeEach(() => jest.useFakeTimers());
afterEach(() => jest.useRealTimers());

test('only exact request/target/source identity can provide a validated detached payload', async () => {
  const r = rig();
  const settled = jest.fn(); const promise = r.client.prepare(reference).then(settled);
  r.reply({ requestId: 'foreign' }); r.reply({ target: { ...target, backendRevision: 2 } });
  r.reply({ reference: { ...reference, sourceId: 'other' } });
  await Promise.resolve(); expect(settled).not.toHaveBeenCalled();
  r.reply(); await promise;
  expect(settled).toHaveBeenCalledWith({ capture: portable, detail: 'Prepared.' });
  r.client.dispose();
});

test('refuses malformed, private and foreign capture payloads even with a valid response identity', async () => {
  for (const change of [
    (value: any) => { value.capture.source.workspaceId = 'private'; },
    (value: any) => { value.capture.id = 'other'; },
    (value: any) => { value.capture.operations[0].arguments.kwargs.token = { kind: 'str', value: 'SECRET' }; },
  ]) {
    const r = rig(), capture = JSON.parse(JSON.stringify(portable)); change(capture);
    const promise = r.client.prepare(reference); r.reply({ capture });
    expect((await promise).capture).toBeNull(); r.client.dispose();
  }
});

test('source/projection changes, timeout, disconnect and disposal cancel without accepting late data', async () => {
  for (const change of [
    (r: ReturnType<typeof rig>) => r.receive({ type: 'projection', target: { ...target, backendRevision: 2 } }),
    (r: ReturnType<typeof rig>) => r.changeStatus({ phase: 'reconnecting' }),
    (r: ReturnType<typeof rig>) => r.client.dispose(),
    (_r: ReturnType<typeof rig>) => jest.advanceTimersByTime(15_000),
  ]) {
    const r = rig(), promise = r.client.prepare(reference); change(r); r.reply();
    expect((await promise).capture).toBeNull(); r.client.dispose();
  }
  expect(jest.getTimerCount()).toBe(0);
});

test('replacement and transport failures settle with explicit refusal', async () => {
  const r = rig(), first = r.client.prepare(reference), next = r.client.prepare(reference);
  expect((await first).capture).toBeNull(); expect(r.send.mock.calls[1][0].type).toBe('captureExportCancel');
  r.client.cancel(); expect((await next).capture).toBeNull();
  r.send.mockImplementation(() => { throw Error('PRIVATE_URL'); });
  expect((await r.client.prepare(reference)).detail).toBe('Capture export could not start.');
  r.client.dispose();
});
