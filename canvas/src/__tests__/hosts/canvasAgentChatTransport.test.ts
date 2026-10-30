import { readUIMessageStream } from 'ai';
import { CanvasAgentChatTransport, type AgentUIMessage } from '../../ide/canvasAgentChatTransport';
import type { CanvasAgentState } from '../../ide/protocol.generated';

const action = { action: 'send' as const, requestId: 'u', provider: 'codex' as const, prompt: 'Explain', nodeIds: [],
  target: { workspaceFolder: '/fixture', file: 'net.py', cls: 'Net', sourceDigest: 'a'.repeat(64), backendRevision: 1, documentVersion: 1 } };
const state = (text: string, status: CanvasAgentState['status'] = 'running'): CanvasAgentState => ({
  sessionId: 'chat', revision: 1, provider: 'codex', status, activity: '', requestId: 'u', ack: 'u',
  messages: [{ id: 'u', role: 'user', text: 'Explain' }, { id: 'a', role: 'assistant', text }],
});
const request = (transport: CanvasAgentChatTransport, signal = new AbortController().signal) => transport.sendMessages({
  chatId: 'chat', trigger: 'submit-message', messageId: 'u', messages: [], body: { action }, abortSignal: signal,
});

test('host snapshots produce a valid AI SDK stream without duplicate text', async () => {
  const send = jest.fn(() => true), transport = new CanvasAgentChatTransport(send);
  const stream = await request(transport);
  transport.accept(state('First')); transport.accept(state('First')); transport.accept(state('First second', 'idle'));
  let last: AgentUIMessage | undefined;
  for await (const message of readUIMessageStream<AgentUIMessage>({ stream, terminateOnError: true })) last = message;
  expect(last?.id).toBe('a'); expect(last?.parts).toEqual([{ type: 'text', text: 'First second', state: 'done' }]);
  expect(send).toHaveBeenCalledTimes(1); expect(transport.streaming).toBe(false);
});

test('a refused send settles the SDK stream and preserves the host transcript', async () => {
  const transport = new CanvasAgentChatTransport(() => true), stream = await request(transport);
  transport.accept({ ...state('', 'error'), messages: [], activity: 'Target changed. Refresh first.' });
  await expect(stream.getReader().read()).rejects.toThrow('Target changed');
  expect(transport.streaming).toBe(false);
});

test('SDK abort stops the matching host request; view unmount does not cancel host work', async () => {
  const send = jest.fn(() => true), transport = new CanvasAgentChatTransport(send), controller = new AbortController();
  await request(transport, controller.signal); controller.abort();
  expect(send).toHaveBeenLastCalledWith({ action: 'stop', requestId: 'u' });
  await request(transport); send.mockClear(); transport.disconnect();
  expect(send).not.toHaveBeenCalled(); expect(transport.streaming).toBe(false);
});

test('unavailable bridge fails immediately instead of leaving Send pending forever', async () => {
  const transport = new CanvasAgentChatTransport(() => false), stream = await request(transport);
  await expect(stream.getReader().read()).rejects.toThrow('reconnecting');
  expect(transport.streaming).toBe(false);
});
