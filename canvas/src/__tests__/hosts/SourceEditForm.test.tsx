import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { SourceEditForm } from '../../ide/SourceEditForm';
import { LocalBrowserContext } from '../../ide/localBrowserContext';
import { ModelNode } from '../../types';
import { EMPTY_LOCAL_BROWSER_PAGE_STATE, LocalBrowserPageState } from '../../ide/localBrowserState';

const nodes = [{ id: 'fc', data: { parameters: { out_features: 8 } } }, { id: 'act', data: { parameters: {} } }] as ModelNode[];
function setup(selected = nodes.slice(0, 1), page = EMPTY_LOCAL_BROWSER_PAGE_STATE) {
  const propose = jest.fn(() => ({ requestId: 'req', promise: Promise.resolve({ status: 'applied' as const, detail: 'Applied in VS Code.' }) }));
  const cancel = jest.fn();
  render(<LocalBrowserContext.Provider value={{ sourceEdits: { propose, cancel, dispose: jest.fn() },
    page, status: { phase: 'connected', hostLink: 'connected', error: null },
  }}><SourceEditForm nodes={selected} /></LocalBrowserContext.Provider>);
  return { propose, cancel };
}

test('refresh failure disables review even when retained nodes are selected', () => {
  const page: LocalBrowserPageState = { ...EMPTY_LOCAL_BROWSER_PAGE_STATE,
    syncState: 'stale', greyed: true, reason: 'Graph refresh timed out. Reopen from VS Code.' };
  const { propose } = setup(nodes.slice(0, 1), page);
  expect(screen.getByRole('button', { name: 'Review in VS Code' })).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Review in VS Code' }));
  expect(propose).not.toHaveBeenCalled();
});

test('parameter controls send semantic intent and only report application after the host result', async () => {
  const { propose } = setup();
  fireEvent.change(screen.getByLabelText('Parameter value'), { target: { value: '16' } });
  fireEvent.click(screen.getByRole('button', { name: 'Review in VS Code' }));
  expect(propose).toHaveBeenCalledWith({ kind: 'set_param', nodeId: 'fc', param: 'out_features', value: 16 });
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Applied in VS Code.'));
});

test('grouping uses the whole selection and discloses checkpoint key changes', async () => {
  const { propose } = setup(nodes);
  expect(screen.getByText(/checkpoint keys/)).toBeVisible();
  fireEvent.change(screen.getByLabelText('New module name'), { target: { value: 'features' } });
  fireEvent.click(screen.getByRole('button', { name: 'Review in VS Code' }));
  expect(propose).toHaveBeenCalledWith({ kind: 'group_layers', nodeIds: ['fc', 'act'], name: 'features' });
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Applied in VS Code.'));
});

test('adding is proposed without changing selected node data', async () => {
  const { propose } = setup();
  fireEvent.change(screen.getByLabelText('Source operation'), { target: { value: 'add_layer' } });
  fireEvent.change(screen.getByLabelText('New module name'), { target: { value: 'extra' } });
  fireEvent.click(screen.getByRole('button', { name: 'Review in VS Code' }));
  expect(propose).toHaveBeenCalledWith(expect.objectContaining({ kind: 'add_layer', afterNodeId: 'fc', nodeId: 'extra', layerType: 'ReLU' }));
  expect(nodes[0].data.parameters).toEqual({ out_features: 8 });
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Applied in VS Code.'));
});

test('deletion requests a source review rather than deleting the canvas node', async () => {
  const { propose } = setup();
  fireEvent.change(screen.getByLabelText('Source operation'), { target: { value: 'remove_layer' } });
  fireEvent.click(screen.getByRole('button', { name: 'Review in VS Code' }));
  expect(propose).toHaveBeenCalledWith({ kind: 'remove_layer', nodeId: 'fc' });
  expect(nodes[0].id).toBe('fc');
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Applied in VS Code.'));
});

test('captured and mixed selections cannot propose legacy source edits', () => {
  const captured = { ...nodes[0], data: { ...nodes[0].data,
    captureRef: { version: 1, captureId: 'capture', sourceId: 'source', kind: 'invocation', semanticId: 'call' } } } as ModelNode;
  const { propose } = setup([captured, nodes[1]]);
  expect(screen.getByRole('status')).toHaveTextContent('no supported source-edit recipe');
  expect(screen.getByRole('button', { name: 'Review in VS Code' })).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Review in VS Code' }));
  expect(propose).not.toHaveBeenCalled();
});
