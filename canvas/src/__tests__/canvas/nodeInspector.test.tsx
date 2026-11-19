import React from 'react';
import '@testing-library/jest-dom';
import { fireEvent, render, screen } from '@testing-library/react';
import { NodeInspector } from '../../canvas/NodeInspector';
import { LocalBrowserContext } from '../../ide/localBrowserContext';
import { NodeExplanationContext } from '../../canvas/NodeExplanationContext';
import type { ModelNode } from '../../types';

jest.mock('../../ide/projection', () => ({ localBrowserMode: () => false }));

const node: ModelNode = { id: 'fc', type: 'layerNode', position: { x: 0, y: 0 }, data: {
  layerType: { id: 'linear', name: 'Linear', displayName: 'Linear', category: 'linear', description: '', icon: 'Box', parameters: [] },
  parameters: { out_features: 8 }, captureLabel: 'encoder.fc', sourceLocation: { file: 'model.py', line: 12 },
} };
const props = { node, title: 'encoder.fc', nodes: [node], edges: [], onNavigate: jest.fn(), onBack: jest.fn(), onClose: jest.fn() };

it('uses a single labelled return action instead of duplicate back and close buttons', () => {
  render(<NodeInspector {...props} />);
  expect(screen.getByRole('button', { name: 'Back to graph' })).toHaveTextContent('Back to graph');
  expect(screen.queryByRole('button', { name: 'Close inspector' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Back to graph' }));
  expect(props.onBack).toHaveBeenCalledTimes(1);
});

it('requires an explicit Edit action and keeps the source form in the inspector', () => {
  const propose = jest.fn();
  render(<LocalBrowserContext.Provider value={{ sourceEdits: { propose }, page: {} } as any}>
    <NodeInspector {...props} />
  </LocalBrowserContext.Provider>);
  expect(screen.queryByRole('region', { name: 'Edit model source' })).not.toBeInTheDocument();
  expect(screen.queryByRole('textbox')).not.toBeInTheDocument();
  expect(propose).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Edit source' }));
  expect(screen.getByRole('complementary', { name: 'Node inspector' })).toContainElement(screen.getByRole('region', { name: 'Edit model source' }));
  expect(screen.getByLabelText('out_features')).toHaveValue('8');
  expect(screen.getByRole('button', { name: 'Review deletion' })).not.toBeVisible();
  fireEvent.click(screen.getByText('More source actions'));
  expect(screen.getByRole('button', { name: 'Review deletion' })).toBeVisible();
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  expect(propose).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Back to details' }));
  expect(screen.queryByRole('region', { name: 'Edit model source' })).not.toBeInTheDocument();
});

it.each(['boundary', 'unsupported', 'read-only', 'stale', 'review'] as const)('does not offer source edits for %s selections', kind => {
  const selected = { ...node, data: { ...node.data, ...(kind === 'boundary' ? { focusBoundary: { direction: 'in' as const, externalNodeId: 'outside' } }
    : kind === 'unsupported' ? { sourceLocation: undefined } : {}) } };
  render(<LocalBrowserContext.Provider value={{ sourceEdits: kind === 'read-only' ? null : { propose: jest.fn() },
    page: { greyed: kind === 'stale' }, reviewing: kind === 'review' } as any}>
    <NodeInspector {...props} node={selected} />
  </LocalBrowserContext.Provider>);
  expect(screen.queryByRole('button', { name: 'Edit source' })).not.toBeInTheDocument();
  expect(screen.queryByText(/no safe source edit/)).not.toBeInTheDocument();
});

it('keeps preparation, progress and cancellation on the original Explain button', () => {
  const start = jest.fn(() => 'new-request'), cancel = jest.fn();
  const value: any = { states: {}, retry: jest.fn(), reveal: jest.fn(), model: { start, cancel, toggle: jest.fn() } };
  const view = render(<NodeExplanationContext.Provider value={value}><NodeInspector {...props} /></NodeExplanationContext.Provider>);
  const button = screen.getByRole('button', { name: 'Explain this component' });
  fireEvent.click(button);
  expect(screen.getByRole('button', { name: 'Stop explaining' })).toBe(button);
  expect(button).toHaveAttribute('data-state', 'preparing');
  expect(screen.queryByText('Preparing explanation…')).not.toBeInTheDocument();
  expect(screen.queryByRole('region', { name: 'AI explanation' })).not.toBeInTheDocument();
  const loading = { ...value, model: { ...value.model, state: { requestId: 'new-request', revision: 'revision',
    status: 'loading', completed: 0, total: 1, detail: 'Reading model.py' } } };
  view.rerender(<NodeExplanationContext.Provider value={loading}><NodeInspector {...props} /></NodeExplanationContext.Provider>);
  expect(button).toHaveAttribute('data-state', 'working');
  expect(screen.queryByText('Reading model.py')).not.toBeInTheDocument();
  fireEvent.click(button);
  expect(cancel).toHaveBeenCalledTimes(1);
  expect(button).toHaveAttribute('data-state', 'stopped');
  expect(screen.getByRole('button', { name: 'Explain again' })).toBeEnabled();
  expect(start).toHaveBeenCalledTimes(1);
});

it('shows failure on the Explain button with the reason on focus, not as an inspector paragraph', () => {
  const value: any = { states: {}, retry: jest.fn(), reveal: jest.fn(), model: {
    start: jest.fn(() => 'failed-request'), cancel: jest.fn(), toggle: jest.fn() } };
  const view = render(<NodeExplanationContext.Provider value={value}><NodeInspector {...props} /></NodeExplanationContext.Provider>);
  const button = screen.getByRole('button', { name: 'Explain this component' });
  fireEvent.click(button);
  const detail = 'Context exceeds the explanation limit.';
  view.rerender(<NodeExplanationContext.Provider value={{ ...value, model: { ...value.model, state: {
    requestId: 'failed-request', revision: 'revision', status: 'unavailable', completed: 0, total: 1, detail,
  } } }}><NodeInspector {...props} /></NodeExplanationContext.Provider>);
  expect(screen.getByRole('button', { name: 'Retry AI explanation' })).toBe(button);
  expect(button).toHaveAttribute('data-state', 'error');
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  expect(screen.queryByText(detail)).not.toBeInTheDocument();
  fireEvent.focus(button);
  expect(screen.getByRole('tooltip')).toHaveTextContent(detail);
});

it('starts a node explanation only after an explicit Explain action', () => {
  const start = jest.fn();
  render(<NodeExplanationContext.Provider value={{ states: {}, retry: jest.fn(), reveal: jest.fn(),
    model: { start, cancel: jest.fn(), toggle: jest.fn() } }}>
    <NodeInspector {...props} />
  </NodeExplanationContext.Provider>);
  expect(start).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Explain this component' }));
  expect(start).toHaveBeenCalledTimes(1);
  expect(start).toHaveBeenCalledWith(['fc']);
});
