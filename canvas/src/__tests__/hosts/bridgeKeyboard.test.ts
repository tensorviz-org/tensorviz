import { PROTOCOL_VERSION } from '../../ide/protocol.generated';

const post = jest.fn();
const origin = 'vscode-webview://keyboard-test';

beforeAll(() => {
  jest.useFakeTimers();
  Object.defineProperty(window, 'parent', { value: { postMessage: post }, configurable: true });
  // The bridge must observe a framed page at import time.
  // eslint-disable-next-line @typescript-eslint/no-var-requires, global-require
  (require('../../ide/bridge') as typeof import('../../ide/bridge')).startBridge();
});
afterAll(() => jest.useRealTimers());

it('leaves browser shortcuts alone until the native shell answers', () => {
  post.mockClear();
  const event = new KeyboardEvent('keydown', { key: 'P', metaKey: true, shiftKey: true, cancelable: true });
  window.dispatchEvent(event);
  expect(event.defaultPrevented).toBe(false);
  expect(post).not.toHaveBeenCalled();
});

it.each(['metaKey', 'ctrlKey'])('opens the native command palette from a focused input with %s+Shift+P', modifier => {
  window.dispatchEvent(new MessageEvent('message', {
    source: window.parent, origin, data: { type: 'hostReady', protocolVersion: PROTOCOL_VERSION },
  }));
  post.mockClear();
  const input = document.createElement('input');
  document.body.append(input);
  input.focus();
  const event = new KeyboardEvent('keydown', { key: 'P', [modifier]: true, shiftKey: true, bubbles: true, cancelable: true });
  input.dispatchEvent(event);
  expect(event.defaultPrevented).toBe(true);
  expect(post).toHaveBeenCalledTimes(1);
  expect(post).toHaveBeenCalledWith({ type: 'showCommandPalette' }, origin);
  input.remove();
});

it('leaves plain graph search and Alt combinations to their existing handlers', () => {
  post.mockClear();
  for (const modifiers of [{ metaKey: true }, { ctrlKey: true }, { metaKey: true, shiftKey: true, altKey: true }]) {
    const event = new KeyboardEvent('keydown', { key: 'p', ...modifiers, cancelable: true });
    window.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
  }
  expect(post).not.toHaveBeenCalled();
});

it('paste accepts only the matching clipboard reply from the authenticated shell', async () => {
  const { readClipboardText } = require('../../ide/bridge') as typeof import('../../ide/bridge');
  post.mockClear();
  const result = readClipboardText();
  const request = post.mock.calls[0][0];
  expect(request).toEqual({ type: 'readClipboard', requestId: expect.any(String) });
  expect(post.mock.calls[0][1]).toBe(origin);
  let settled = false;
  result.then(() => { settled = true; });
  const receive = (source: Window | null, from: string, requestId: string, text: string) => window.dispatchEvent(new MessageEvent('message', {
    source, origin: from, data: { type: 'clipboardText', requestId, text },
  }));
  receive(window.parent, 'https://other.example', request.requestId, 'wrong origin');
  receive(null, origin, request.requestId, 'wrong source');
  receive(window.parent, origin, 'another request', 'wrong request');
  await Promise.resolve();
  expect(settled).toBe(false);
  receive(window.parent, origin, request.requestId, 'plain\ntext');
  await expect(result).resolves.toBe('plain\ntext');
});

it('paste reports host failure and times out rather than silently hanging', async () => {
  const { readClipboardText } = require('../../ide/bridge') as typeof import('../../ide/bridge');
  post.mockClear();
  const failed = readClipboardText();
  const rejected = expect(failed).rejects.toThrow('Could not read clipboard');
  window.dispatchEvent(new MessageEvent('message', { source: window.parent, origin,
    data: { type: 'clipboardText', requestId: post.mock.calls[0][0].requestId, error: true } }));
  await rejected;
  const absent = readClipboardText();
  const timedOut = expect(absent).rejects.toThrow('Could not read clipboard');
  jest.advanceTimersByTime(5000);
  await timedOut;
});

it('copy writes only explicit text and accepts only the authenticated correlated acknowledgement', async () => {
  const { writeClipboardText } = require('../../ide/bridge') as typeof import('../../ide/bridge');
  post.mockClear();
  const copied = writeClipboardText('Selected reply\ntext');
  const request = post.mock.calls[0][0];
  expect(request).toEqual({ type: 'clipboard', text: 'Selected reply\ntext', requestId: expect.any(String) });
  let finished = false; copied.then(() => { finished = true; });
  for (const [source, from, requestId] of [[null, origin, request.requestId], [window.parent, 'https://other.example', request.requestId], [window.parent, origin, 'wrong']]) {
    window.dispatchEvent(new MessageEvent('message', { source, origin: from, data: { type: 'clipboardWritten', requestId } }));
  }
  await Promise.resolve(); expect(finished).toBe(false);
  window.dispatchEvent(new MessageEvent('message', { source: window.parent, origin, data: { type: 'clipboardWritten', requestId: request.requestId } }));
  await expect(copied).resolves.toBeUndefined();
  post.mockClear();
  await expect(writeClipboardText('x'.repeat(64001))).rejects.toThrow();
  expect(post).not.toHaveBeenCalled();
});
