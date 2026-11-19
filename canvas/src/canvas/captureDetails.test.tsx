import React from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { CaptureDetails } from './CaptureDetails';
import type { ModelNodeData } from '../types';
import type { CaptureInspectionQuery } from '../ide/protocol.generated';
import type { CaptureInspectionResult, LocalBrowserCaptureClient } from '../ide/localBrowserCaptureClient';

const reference: NonNullable<ModelNodeData['captureRef']> = {
  version: 1, captureId: 'capture-a', sourceId: 'source-a', kind: 'operation', semanticId: 'op:internal-a',
};
const inlineProps = { inline: true };
const response = (query: CaptureInspectionQuery): CaptureInspectionResult => ({
  detail: '', page: { ...query, node: { key: null, kind: 'object' }, total: 26,
    items: [{ key: 'shape', kind: 'array', count: 2 }], nextOffset: query.offset === 0 ? 25 : null },
});
function makeClient() {
  return { inspect: jest.fn(async (query: CaptureInspectionQuery) => response(query)), cancel: jest.fn(), dispose: jest.fn() };
}
const showModal = jest.fn();
beforeEach(() => {
  showModal.mockImplementation(function (this: HTMLDialogElement) { this.setAttribute('open', ''); });
  HTMLDialogElement.prototype.showModal = showModal;
});

it.each([false, true])('does not inspect until explicitly opened (inline=%s)', async inline => {
  const client = makeClient();
  const view = render(<CaptureDetails {...{ inline }} reference={reference} label="Linear" client={client} />);
  expect(client.inspect).not.toHaveBeenCalled();
  view.rerender(<CaptureDetails {...{ inline }} reference={{ ...reference, captureId: 'capture-b' }} label="Linear" client={client} />);
  expect(client.inspect).not.toHaveBeenCalled();
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'View captured details' })); });
  expect(client.inspect).toHaveBeenCalledTimes(1);
});

it('opens and closes inline evidence inside its panel without a modal', async () => {
  const client = makeClient();
  const { container } = render(<CaptureDetails {...inlineProps} reference={reference} label="Linear" client={client} />);
  const trigger = screen.getByRole('button', { name: 'View captured details' });
  expect(trigger).not.toHaveAttribute('aria-haspopup', 'dialog');
  fireEvent.click(trigger);
  const evidence = await screen.findByRole('region', { name: 'Captured details: Linear' });
  expect(container).toContainElement(evidence);
  expect(document.querySelector('dialog')).toBeNull();
  expect(showModal).not.toHaveBeenCalled();
  expect(within(evidence).getByRole('heading', { name: 'Linear' })).toBeVisible();
  fireEvent.click(within(evidence).getByRole('button', { name: 'Close captured details' }));
  expect(screen.queryByRole('region', { name: 'Captured details: Linear' })).not.toBeInTheDocument();
  expect(client.cancel).toHaveBeenCalledTimes(1);
  expect(trigger).toHaveFocus();
  expect(client.inspect).toHaveBeenCalledTimes(1);
});

it('opens directly from an explicit graph action without a second trigger', async () => {
  const client = makeClient(), close = jest.fn();
  render(<CaptureDetails inline initiallyOpen hideTrigger onClose={close} reference={reference} label="Linear" client={client} />);
  expect(screen.queryByRole('button', { name: 'View captured details' })).not.toBeInTheDocument();
  expect(await screen.findByRole('region', { name: 'Captured details: Linear' })).toBeVisible();
  expect(client.inspect).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole('button', { name: 'Close captured details' }));
  expect(close).toHaveBeenCalledTimes(1);
  expect(client.cancel).toHaveBeenCalledTimes(1);
});

it('cancels an inline request on unmount and ignores its late result', async () => {
  let resolve!: (result: CaptureInspectionResult) => void;
  const client: LocalBrowserCaptureClient = { inspect: jest.fn(() => new Promise(done => { resolve = done; })), cancel: jest.fn(), dispose: jest.fn() };
  const { unmount } = render(<CaptureDetails {...inlineProps} reference={reference} label="Linear" client={client} />);
  fireEvent.click(screen.getByRole('button', { name: 'View captured details' }));
  expect(screen.getByRole('region', { name: 'Captured details: Linear' })).toHaveTextContent('Reading captured details');
  unmount();
  expect(client.cancel).toHaveBeenCalledTimes(1);
  await act(async () => resolve({ page: null, detail: 'Late response' }));
  expect(screen.queryByText('Late response')).not.toBeInTheDocument();
});

it.each([
  { ...reference, captureId: 'capture-b', semanticId: 'op:internal-b' },
  { ...reference, semanticId: 'op:internal-b' },
  { ...reference, sourceId: 'source-b' },
])('resets the path and pagination when selection changes to %j', async nextReference => {
  const client = makeClient();
  const { rerender } = render(<CaptureDetails {...inlineProps} reference={reference} label="Linear" client={client} />);
  fireEvent.click(screen.getByRole('button', { name: 'View captured details' }));
  fireEvent.click(await screen.findByRole('button', { name: /array.*2 items/ }));
  await waitFor(() => expect(client.inspect).toHaveBeenLastCalledWith(expect.objectContaining({ path: ['shape'], offset: 0 })));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Next page' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Next page' }));
  await waitFor(() => expect(client.inspect).toHaveBeenLastCalledWith(expect.objectContaining({ path: ['shape'], offset: 25 })));
  rerender(<CaptureDetails {...inlineProps} reference={nextReference} label="ReLU" client={client} />);
  await waitFor(() => expect(client.inspect).toHaveBeenLastCalledWith(expect.objectContaining({ captureId: nextReference.captureId,
    sourceId: nextReference.sourceId, entity: { kind: nextReference.kind, id: nextReference.semanticId }, path: [], offset: 0 })));
  expect(screen.getByRole('button', { name: 'Previous page' })).toBeDisabled();
});

it('uses the display label rather than an internal ID in the section path', async () => {
  const { container } = render(<CaptureDetails inline reference={reference} label="Linear" client={makeClient()} />);
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'View captured details' })); });
  expect(container.querySelector('.capture-details-path')).toHaveTextContent('Linear');
  expect(container.querySelector('.capture-details-path')).not.toHaveTextContent(reference.semanticId);
});

it('retains unavailable and stale evidence warnings inline', async () => {
  const client = makeClient();
  client.inspect.mockResolvedValue({ page: null, detail: 'This capture is no longer retained.' });
  render(<CaptureDetails inline stale reference={reference} label="Linear" client={client} />);
  fireEvent.click(screen.getByRole('button', { name: 'View captured details' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('This capture is no longer retained.');
  expect(screen.getByText(/does not verify the current source/)).toBeVisible();
  expect(screen.getByRole('button', { name: 'Next page' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Previous page' })).toBeDisabled();
});

it('preserves the modal portal by default', async () => {
  const { container } = render(<CaptureDetails reference={reference} label="Linear" client={makeClient()} />);
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'View captured details' })); });
  const dialog = screen.getByRole('dialog', { name: 'Captured details: Linear' });
  expect(container).not.toContainElement(dialog);
  expect(showModal).toHaveBeenCalledTimes(1);
  fireEvent(dialog, new Event('cancel', { bubbles: true }));
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
});
