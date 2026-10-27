import React from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { CanvasAgentChat } from '../../ide/CanvasAgentChat';
import * as bridge from '../../ide/bridge';

jest.mock('../../ide/bridge', () => ({ bridgeActive: jest.fn(() => true), readClipboardText: jest.fn(), writeClipboardText: jest.fn() }));
const clipboard = (bridge as unknown as { readClipboardText: jest.Mock }).readClipboardText;
const writeClipboard = (bridge.writeClipboardText as jest.Mock);

function reply() {
  let receive!: (message: any) => void;
  const send = jest.fn();
  const transport = { session: { viewId: 'copy-view' }, status: { phase: 'connected', hostLink: 'connected' }, send,
    onMessage: (listener: typeof receive) => { receive = listener; return () => {}; }, onStatusChange: () => () => {} };
  render(<CanvasAgentChat transport={transport as any} target={null} connected />);
  fireEvent.click(screen.getByRole('button', { name: 'Open agent chat' }));
  act(() => receive({ type: 'agentState', viewId: 'copy-view', state: { sessionId: 'copy', revision: 1,
    provider: 'codex', status: 'idle', activity: '', messages: [{ id: 'answer', role: 'assistant', text: 'Model is blocked.\nUse a new environment.' }] } }));
  return { send, text: screen.getByText(/Model is blocked\./) };
}

function compose(text = '') {
  render(<CanvasAgentChat transport={null} target={null} connected={false} />);
  fireEvent.click(screen.getByRole('button', { name: 'Open agent chat' }));
  const field = screen.getByRole('textbox') as HTMLTextAreaElement;
  fireEvent.change(field, { target: { value: text } });
  return field;
}
beforeEach(() => {
  jest.clearAllMocks(); (bridge.bridgeActive as jest.Mock).mockReturnValue(true);
  writeClipboard.mockResolvedValue(undefined); window.getSelection()?.removeAllRanges();
  // jsdom has no editing engine. Real insertion and Undo/Redo are verified in Chromium.
  Object.defineProperty(document, 'execCommand', { configurable: true, value: jest.fn((_command, _showUI, text) => {
    const field = document.activeElement as HTMLTextAreaElement;
    field.setRangeText(text, field.selectionStart, field.selectionEnd, 'end');
    fireEvent.input(field);
    return true;
  }) });
});

test.each(['metaKey', 'ctrlKey'])('native %s+C copies only selected conversation text without sending a request', async modifier => {
  const { text, send } = reply();
  const range = document.createRange(); range.setStart(text.firstChild!, 0); range.setEnd(text.firstChild!, 17);
  window.getSelection()!.addRange(range);
  await act(async () => { expect(fireEvent.keyDown(document.body, { key: 'c', [modifier]: true })).toBe(false); });
  expect(writeClipboard).toHaveBeenCalledWith('Model is blocked.');
  expect(send.mock.calls.some(([message]) => message.action === 'send')).toBe(false);
});

test('copy button copies the whole response, and a failure is reported without exposing host details', async () => {
  reply();
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Copy agent response' })));
  expect(writeClipboard).toHaveBeenCalledWith('Model is blocked.\nUse a new environment.');
  writeClipboard.mockRejectedValue(new Error('private clipboard failure'));
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Copy agent response' })));
  expect(screen.getByRole('alert')).toHaveTextContent('Could not copy. Try again.');
  expect(screen.queryByText('private clipboard failure')).toBeNull();
});

test('native copy ignores a selection outside the conversation and browser shortcuts remain native', () => {
  const { text } = reply();
  const range = document.createRange(); range.selectNodeContents(text);
  window.getSelection()!.addRange(range);
  (bridge.bridgeActive as jest.Mock).mockReturnValue(false);
  expect(fireEvent.keyDown(document.body, { key: 'c', metaKey: true })).toBe(true);
  (bridge.bridgeActive as jest.Mock).mockReturnValue(true);
  const outside = document.createElement('span'); outside.textContent = 'Outside chat'; document.body.append(outside);
  range.selectNodeContents(outside);
  expect(fireEvent.keyDown(document.body, { key: 'c', metaKey: true })).toBe(true);
  expect(writeClipboard).not.toHaveBeenCalled();
  outside.remove();
});

test.each(['metaKey', 'ctrlKey'])('native %s+V pastes plain text at the selection without sending', async modifier => {
  clipboard.mockResolvedValue('two\r\nlines');
  const field = compose('before OLD after');
  field.setSelectionRange(7, 10);
  await act(async () => { expect(fireEvent.keyDown(field, { key: 'v', [modifier]: true })).toBe(false); });
  expect(clipboard).toHaveBeenCalledTimes(1);
  expect(field).toHaveValue('before two\nlines after');
  expect(field).toHaveFocus();
  expect(document.execCommand).toHaveBeenCalledWith('insertText', false, 'two\nlines');
});

test('browser paste and unrelated shortcuts remain native and do not read the host clipboard', () => {
  const field = compose();
  (bridge.bridgeActive as jest.Mock).mockReturnValue(false);
  expect(fireEvent.keyDown(field, { key: 'v', metaKey: true })).toBe(true);
  (bridge.bridgeActive as jest.Mock).mockReturnValue(true);
  for (const keys of [{ key: 'v' }, { key: 'v', metaKey: true, altKey: true }, { key: 'c', metaKey: true }]) {
    expect(fireEvent.keyDown(field, keys)).toBe(true);
  }
  expect(clipboard).not.toHaveBeenCalled();
});

test('an empty clipboard preserves selected text', async () => {
  clipboard.mockResolvedValue('');
  const field = compose('keep');
  field.setSelectionRange(0, 4);
  await act(async () => { fireEvent.keyDown(field, { key: 'v', metaKey: true }); });
  expect(field).toHaveValue('keep');
});

test('native plain-text paste respects the prompt limit while preserving text after the selection', async () => {
  clipboard.mockResolvedValue('B'.repeat(20000));
  const field = compose('A'.repeat(15990) + 'OLDend');
  field.setSelectionRange(15990, 15993);
  await act(async () => { fireEvent.keyDown(field, { key: 'V', metaKey: true, shiftKey: true }); });
  expect(field).toHaveValue('A'.repeat(15990) + 'B'.repeat(7) + 'end');
});

test.each(['typing', 'selection', 'blur', 'minimize'])('a late clipboard reply cannot overwrite the draft after %s', async change => {
  let answer!: (text: string) => void;
  clipboard.mockImplementation(() => new Promise<string>(resolve => { answer = resolve; }));
  const field = compose('keep');
  field.setSelectionRange(4, 4);
  fireEvent.keyDown(field, { key: 'v', metaKey: true });
  if (change === 'typing') fireEvent.change(field, { target: { value: 'keep typing' } });
  if (change === 'selection') field.setSelectionRange(0, 4);
  if (change === 'blur') field.blur();
  if (change === 'minimize') fireEvent.click(screen.getByRole('button', { name: 'Minimize agent chat' }));
  await act(async () => answer('unexpected'));
  if (change === 'minimize') fireEvent.click(screen.getByRole('button', { name: 'Open agent chat' }));
  expect(screen.getByRole('textbox')).toHaveValue(change === 'typing' ? 'keep typing' : 'keep');
});

test('clipboard failure keeps the draft and shows a retry message without exposing clipboard data', async () => {
  clipboard.mockRejectedValue(new Error('private host failure'));
  const field = compose('keep');
  await act(async () => { fireEvent.keyDown(field, { key: 'v', ctrlKey: true }); });
  expect(field).toHaveValue('keep');
  expect(screen.getByRole('alert')).toHaveTextContent('Could not paste. Try again.');
  expect(screen.queryByText('private host failure')).toBeNull();
});

test('a late clipboard failure does not interrupt continued typing', async () => {
  let fail!: (error: Error) => void;
  clipboard.mockImplementation(() => new Promise<string>((_resolve, reject) => { fail = reject; }));
  const field = compose('keep');
  fireEvent.keyDown(field, { key: 'v', metaKey: true });
  fireEvent.change(field, { target: { value: 'keep typing' } });
  await act(async () => fail(new Error('clipboard unavailable')));
  expect(field).toHaveValue('keep typing');
  expect(screen.queryByRole('alert')).toBeNull();
});
