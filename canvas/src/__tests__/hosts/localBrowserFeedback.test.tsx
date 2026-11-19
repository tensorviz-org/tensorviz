import React from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import FeedbackButton from '../../ide/FeedbackButton';
import { createLocalBrowserFeedbackClient } from '../../ide/localBrowserFeedbackClient';
import { isLocalBrowserMessage } from '../../ide/protocol.generated';
import type { LocalBrowserViewMessage, LocalBrowserViewStatus, LocalBrowserViewTransport } from '../../ide/localBrowserViewTransport';

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
  const client = createLocalBrowserFeedbackClient(transport);
  const request = () => send.mock.calls.find(([message]) => message.type === 'feedbackRequest')?.[0];
  return {
    client, send, request,
    result: (fields: object) => receive({ type: 'feedbackResult', protocolVersion: 1, workspaceSessionId: 's', messageId: 'm',
      requestId: request().requestId, ...fields } as LocalBrowserViewMessage),
    changeStatus: (next: Partial<LocalBrowserViewStatus>) => { Object.assign(current, next); changeStatus(current); },
  };
}

test('sends nothing until Send, then only the words and, when ticked, the model class and block count', () => {
  const r = rig();
  render(<FeedbackButton client={r.client} graph={{ model: 'TinyNet', nodes: 7 }} />);
  expect(screen.queryByTestId('feedback-form')).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Send feedback' }));
  expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled();
  fireEvent.click(screen.getByRole('radio', { name: 'Idea' }));
  fireEvent.change(screen.getByLabelText('Your feedback'), { target: { value: '  Let me pin a layer.  ' } });
  expect(r.send).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Send' }));
  expect(r.request()).toEqual({ type: 'feedbackRequest', requestId: expect.any(String), kind: 'idea', message: 'Let me pin a layer.',
    graph: { model: 'TinyNet', nodes: 7 } });
  expect(screen.getByRole('button', { name: 'Sending…' })).toBeDisabled();
  act(() => r.result({ status: 'sent', detail: 'Thanks. Your feedback reached the tensorViz team.' }));
  expect(screen.getByRole('status')).toHaveTextContent('Thanks. Your feedback reached the tensorViz team.');
});

test('leaves the graph out when unticked and shows why a send failed', () => {
  const r = rig();
  render(<FeedbackButton client={r.client} graph={{ model: 'TinyNet', nodes: 7 }} />);
  fireEvent.click(screen.getByRole('button', { name: 'Send feedback' }));
  fireEvent.click(screen.getByRole('checkbox'));
  fireEvent.change(screen.getByLabelText('Your feedback'), { target: { value: 'Edges overlap.' } });
  fireEvent.click(screen.getByRole('button', { name: 'Send' }));
  expect(r.request()).toEqual(expect.objectContaining({ kind: 'bug', graph: null }));
  act(() => r.result({ status: 'failed', detail: 'You sent feedback a moment ago. Try again shortly.' }));
  expect(screen.getByRole('alert')).toHaveTextContent('You sent feedback a moment ago.');
  expect(screen.getByLabelText('Your feedback')).toHaveValue('Edges overlap.');
});

test('opens the form above the button when it sits in the footer, within the room left', () => {
  const r = rig();
  render(<FeedbackButton client={r.client} graph={null} />);
  const button = screen.getByRole('button', { name: 'Send feedback' });
  // jsdom lays nothing out: place the button at the bottom of the window, where the footer is.
  const top = window.innerHeight - 36;
  button.getBoundingClientRect = () => ({ top, bottom: top + 28, left: 600, right: 631, width: 31, height: 28, x: 600, y: top, toJSON: () => ({}) });
  fireEvent.click(button);
  const form = screen.getByTestId('feedback-form');
  expect(form.style.bottom).toBe('44px');
  expect(form.style.top).toBe('');
  expect(form.style.maxHeight).toBe(`${top - 16}px`);
});

test('ignores other replies and fails instead of waiting when the host goes away', () => {
  const r = rig();
  r.client.send({ kind: 'bug', message: 'Broken.', graph: null });
  expect(r.client.state.phase).toBe('sending');
  r.client.send({ kind: 'bug', message: 'Again.', graph: null });
  expect(r.send).toHaveBeenCalledTimes(1);
  r.changeStatus({ phase: 'reconnecting' } as Partial<LocalBrowserViewStatus>);
  expect(r.client.state).toEqual({ phase: 'failed', detail: expect.stringContaining('connection to tensorViz ended') });

  const offline = rig({ phase: 'connecting' } as Partial<LocalBrowserViewStatus>);
  offline.client.send({ kind: 'bug', message: 'Broken.', graph: null });
  expect(offline.send).not.toHaveBeenCalled();
  expect(offline.client.state.phase).toBe('failed');
});
