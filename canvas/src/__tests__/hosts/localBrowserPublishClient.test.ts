import { gunzipSync } from 'zlib';
import { createLocalBrowserPublishClient, gzipBase64 } from '../../ide/localBrowserPublishClient';
import { isLocalBrowserMessage } from '../../ide/protocol.generated';
import type { LocalBrowserViewMessage, LocalBrowserViewStatus, LocalBrowserViewTransport } from '../../ide/localBrowserViewTransport';

const target = { workspaceFolder: '/ws', file: 'net.py', cls: 'Net', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 };
const graph = { id: 'AbCdEfGhIjKlMnOpQrSt', title: 'Net', url: 'https://tensorviz.dev/explore/graph/?id=AbCdEfGhIjKlMnOpQrSt',
  shareUrl: 'https://tensorviz.dev/explore/graph/?id=AbCdEfGhIjKlMnOpQrSt&share=1' };
const signInUrl = 'https://tensorviz.dev/explore/connect/?state=abc&callback=http%3A%2F%2F127.0.0.1%3A52123%2Ftensorviz%2Fauth';
const file = JSON.stringify({ format: 'tensorviz.graph', document: { semantic: { name: 'Net', nodes: [{ node_id: 'fc' }] } } });
const settled = () => new Promise(resolve => setTimeout(resolve, 0));

function rig(status: Partial<LocalBrowserViewStatus> = { phase: 'connected', hostLink: 'connected' }) {
  let receive!: (message: LocalBrowserViewMessage) => void;
  let changeStatus!: (status: LocalBrowserViewStatus) => void;
  const current = { ...status } as LocalBrowserViewStatus;
  const send = jest.fn((message: any): any => {
    // Every message the view sends must be one the broker accepts.
    expect(isLocalBrowserMessage({ ...message, protocolVersion: 1, workspaceSessionId: 's', messageId: 'm' }, 's')).toBe(true);
    return {};
  });
  const transport = { get status() { return current; }, send,
    onMessage(listener: typeof receive) { receive = listener; return () => {}; },
    onStatusChange(listener: typeof changeStatus) { changeStatus = listener; return () => {}; },
  } as unknown as LocalBrowserViewTransport;
  const compress = jest.fn(async (text: string) => Buffer.from(text).toString('base64'));
  const client = createLocalBrowserPublishClient(transport, () => target, compress);
  const phases: string[] = [];
  client.subscribe(() => phases.push(client.state.phase));
  const requestId = () => send.mock.calls.find(([message]) => message.type === 'graphPublishRequest')[0].requestId;
  return {
    client, send, compress, phases,
    receive: (message: any) => receive(message),
    host: (type: string, fields: object) => receive({ type, protocolVersion: 1, workspaceSessionId: 's', messageId: 'm',
      target, requestId: requestId(), ...fields } as LocalBrowserViewMessage),
    changeStatus: (next: Partial<LocalBrowserViewStatus>) => { Object.assign(current, next); changeStatus(current); },
  };
}

test('publishes the prepared graph file and follows only its own request from sign-in to the Share page', async () => {
  const r = rig();
  expect(r.client.start()).toBe(true);
  expect(r.client.start()).toBe(false);
  r.client.publish(file);
  await settled();
  expect(r.send).toHaveBeenCalledWith({ type: 'graphPublishRequest', target, requestId: expect.any(String), file: Buffer.from(file).toString('base64') });
  r.receive({ type: 'graphPublishState', protocolVersion: 1, workspaceSessionId: 's', messageId: 'm', target, requestId: 'another-view',
    phase: 'uploading', detail: 'Not this request.' });
  r.host('graphPublishState', { target: { ...target, backendRevision: 2 }, phase: 'uploading', detail: 'Another graph.' });
  expect(r.client.state.phase).toBe('preparing');
  r.host('graphPublishState', { phase: 'signing_in', detail: 'Sign in to tensorViz in your browser to publish Net.', signInUrl });
  expect(r.client.state).toEqual({ phase: 'signing_in', detail: 'Sign in to tensorViz in your browser to publish Net.', signInUrl });
  r.host('graphPublishState', { phase: 'uploading', detail: 'Publishing Net to tensorViz…' });
  r.host('graphPublishResult', { status: 'published', detail: 'Net is on tensorViz, visible only to you.', graph });
  expect(r.client.state).toEqual({ phase: 'published', detail: 'Net is on tensorViz, visible only to you.', graph });
  r.host('graphPublishResult', { status: 'failed', detail: 'A late answer.', graph: null });
  expect(r.client.state.phase).toBe('published');
  expect(r.phases).toEqual(['preparing', 'signing_in', 'uploading', 'published']);
  r.client.dismiss();
  expect(r.client.state.phase).toBe('idle');
  r.client.dispose();
});

test('cancelling while signing in tells the host once and ignores its late answer', async () => {
  const r = rig();
  r.client.start(); r.client.publish(file); await settled();
  r.host('graphPublishState', { phase: 'signing_in', detail: 'Sign in.', signInUrl });
  r.client.cancel();
  expect(r.send).toHaveBeenLastCalledWith({ type: 'graphPublishCancel', target, requestId: r.send.mock.calls[0][0].requestId });
  r.host('graphPublishResult', { status: 'published', detail: 'Too late.', graph });
  expect(r.client.state.phase).toBe('idle');
  // A cancel before the file is ready sends nothing at all.
  r.client.start(); r.client.cancel(); r.client.publish(file); await settled();
  expect(r.send).toHaveBeenCalledTimes(2);
  r.client.dispose();
});

test('reports why a publish cannot start or finish, without sending a file it should not', async () => {
  const offline = rig({ phase: 'reconnecting', hostLink: 'unknown' });
  offline.client.start(); offline.client.publish(file);
  expect(offline.client.state).toEqual({ phase: 'failed', detail: 'Connect to the tensorViz plugin to publish this graph.' });
  const large = rig();
  large.client.start(); large.client.publish('x'.repeat(10 * 1024 * 1024 + 1));
  expect(large.client.state).toEqual({ phase: 'failed', detail: 'tensorViz takes graph files up to 10 MB.' });
  const unprepared = rig();
  unprepared.client.start(); unprepared.client.fail('Wait for this model to finish opening before exporting its graph.');
  unprepared.client.publish(file); await settled();
  expect(unprepared.client.state).toEqual({ phase: 'failed', detail: 'Wait for this model to finish opening before exporting its graph.' });
  for (const rigged of [offline, large, unprepared]) expect(rigged.send).not.toHaveBeenCalled();

  const refused = rig();
  refused.client.start(); refused.client.publish(file); await settled();
  refused.host('graphPublishResult', { status: 'failed', detail: 'tensorViz takes graphs with 1 to 5,000 blocks.', graph: null });
  expect(refused.client.state).toEqual({ phase: 'failed', detail: 'tensorViz takes graphs with 1 to 5,000 blocks.' });
  const dropped = rig();
  dropped.client.start(); dropped.client.publish(file); await settled();
  dropped.changeStatus({ phase: 'reconnecting', hostLink: 'unknown' });
  expect(dropped.client.state.phase).toBe('failed');
  const brokerError = rig();
  brokerError.client.start(); brokerError.client.publish(file); await settled();
  brokerError.receive({ type: 'error', protocolVersion: 1, workspaceSessionId: 's', messageId: 'm', code: 'INVALID_MESSAGE' });
  expect(brokerError.client.state.phase).toBe('failed');
  const unsent = rig();
  unsent.send.mockReturnValueOnce(null);
  unsent.client.start(); unsent.client.publish(file); await settled();
  expect(unsent.client.state).toEqual({ phase: 'failed', detail: 'Publish could not reach the tensorViz plugin. Reconnect and try again.' });
  // A failed publish can be tried again.
  expect(unsent.client.start()).toBe(true);
});

test('disposing a running publish cancels it', async () => {
  const r = rig();
  r.client.start(); r.client.publish(file); await settled();
  r.client.dispose();
  expect(r.send).toHaveBeenLastCalledWith(expect.objectContaining({ type: 'graphPublishCancel' }));
  expect(r.client.start()).toBe(false);
});

test('the graph file travels gzipped in base64', async () => {
  Object.assign(global, { CompressionStream: require('stream/web').CompressionStream });
  const encoded = await gzipBase64(file);
  expect(gunzipSync(Buffer.from(encoded, 'base64')).toString('utf8')).toBe(file);
});
