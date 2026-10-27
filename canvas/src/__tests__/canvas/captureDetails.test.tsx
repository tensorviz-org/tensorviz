import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { CaptureDetails } from '../../canvas/CaptureDetails';
import type { CaptureInspectionPage, CaptureInspectionQuery } from '../../ide/protocol.generated';

const reference = { version: 1 as const, captureId: 'capture:1', sourceId: 'source:1', kind: 'operation' as const, semanticId: 'add2' };
beforeAll(() => { HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', ''); }; });

test('selected-node details page both uses of a tensor, navigate exact records and disclose stale evidence', async () => {
  const inspect = jest.fn(async (query: CaptureInspectionQuery) => {
    const page: CaptureInspectionPage = { captureId: query.captureId, sourceId: query.sourceId, entity: query.entity, path: query.path,
      node: { key: null, kind: 'object', count: 2 }, items: [{ key: 'kind', kind: 'string', value: 'value' }, { key: 'id', kind: 'string', value: 'a' }], total: 2, nextOffset: null };
    if (query.path.join('.') === 'arguments') Object.assign(page, { node: { key: null, kind: 'array', count: 2 },
      items: [{ key: query.offset, kind: 'object', count: 2, label: 'a · value' }], nextOffset: query.offset ? null : 1 });
    return { page, detail: 'Retained capture evidence.' };
  });
  const cancel = jest.fn();
  render(<CaptureDetails reference={reference} label="Second Add" stale client={{ inspect, cancel, dispose() {} }} />);
  const button = screen.getByRole('button', { name: 'View captured details' });
  expect(button).toHaveAttribute('aria-haspopup', 'dialog');
  fireEvent.focus(button);
  expect(screen.getByRole('tooltip')).toHaveTextContent('does not run the model');
  expect(inspect).not.toHaveBeenCalled();
  expect(screen.queryByText('Reading captured details…')).not.toBeInTheDocument();
  fireEvent.click(button);
  expect(screen.getByText('Reading captured details…')).toBeVisible();
  expect(await screen.findByRole('dialog', { name: 'Captured details: Second Add' })).toBeVisible();
  expect(screen.getByText(/does not verify the current source/)).toBeInTheDocument();
  await screen.findByText('Inspect tensor a');
  fireEvent.click(screen.getByText('Arguments'));
  await screen.findByText('1–1 of 2 · array');
  fireEvent.click(screen.getByText('Next page'));
  await screen.findByText('2–2 of 2 · array');
  fireEvent.click(screen.getByText(/a · value/));
  await waitFor(() => expect(inspect).toHaveBeenLastCalledWith(expect.objectContaining({ entity: { kind: 'operation', id: 'add2' }, path: ['arguments', 1], offset: 0 })));
  fireEvent.click(await screen.findByText('Inspect tensor a'));
  await waitFor(() => expect(inspect).toHaveBeenLastCalledWith(expect.objectContaining({ entity: { kind: 'value', id: 'a' }, path: [] })));
  fireEvent.click(screen.getByText('Coverage'));
  await waitFor(() => expect(inspect).toHaveBeenLastCalledWith(expect.objectContaining({ path: ['coverage'] })));
  expect(inspect.mock.calls[inspect.mock.calls.length - 1][0]).not.toHaveProperty('entity');
  fireEvent.click(screen.getByLabelText('Close captured details'));
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument(); expect(cancel).toHaveBeenCalled();
  expect(screen.getByRole('button', { name: 'View captured details' })).toHaveFocus();
});

test('unavailable evidence stays explicit and closing prevents a late page from appearing', async () => {
  const client = { inspect: jest.fn(async () => ({ page: null, detail: 'This capture is no longer retained.' })), cancel: jest.fn(), dispose() {} };
  render(<CaptureDetails reference={reference} label="Add" client={client} />);
  fireEvent.click(screen.getByRole('button', { name: 'View captured details' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('no longer retained');
  expect(screen.getByText('Next page')).toBeDisabled();
  fireEvent.click(screen.getByLabelText('Close captured details'));
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
});
