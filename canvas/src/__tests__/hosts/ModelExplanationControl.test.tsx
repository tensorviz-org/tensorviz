import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import ModelExplanationControl from '../../ide/ModelExplanationControl';
import { NodeExplanationContext, type NodeExplanationContextValue } from '../../canvas/NodeExplanationContext';

const value = (extra: Partial<NonNullable<NodeExplanationContextValue['model']>> = {}): NodeExplanationContextValue => ({
  states: {}, annotationsVisible: true, retry: jest.fn(), reveal: jest.fn(),
  model: { start: jest.fn(), cancel: jest.fn(), toggle: jest.fn(), ...extra },
});
const mount = (context: NodeExplanationContextValue | null, compact = false) => render(
  <NodeExplanationContext.Provider value={context}><ModelExplanationControl compact={compact} /></NodeExplanationContext.Provider>,
);

it.each([false, true])('starts the visible graph request without selecting a node, compact=%s', compact => {
  const listener = jest.fn();
  window.addEventListener('builder-command', listener);
  mount(value(), compact);
  fireEvent.click(screen.getByRole('button', { name: 'Explain model' }));
  expect(listener.mock.calls[0][0].detail).toEqual({ command: 'explain-model', payload: undefined });
  window.removeEventListener('builder-command', listener);
});

it('shows real progress and lets the user stop without losing completed work', () => {
  const context = value({ state: { requestId: 'run', revision: 'revision', status: 'loading', completed: 3, total: 7 } });
  mount(context, true);
  expect(screen.getByRole('button', { name: 'Explaining model · 3/7' })).toBeDisabled();
  expect(screen.getByRole('status')).toHaveTextContent('3 of 7');
  fireEvent.click(screen.getByRole('button', { name: 'Stop explaining model' }));
  expect(context.model!.cancel).toHaveBeenCalledTimes(1);
});

it('toggles cached annotations after success and keeps explanatory depth available', () => {
  const context = value({ state: { requestId: 'run', revision: 'revision', status: 'ready', completed: 7, total: 7 } });
  mount(context);
  fireEvent.click(screen.getByRole('button', { name: 'Explanations' }));
  expect(context.model!.toggle).toHaveBeenCalledTimes(1);
  expect(context.model!.start).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Model explanation details' }));
  expect(screen.getByRole('status')).toHaveTextContent('7 of 7 components explained');
  expect(screen.getByRole('button', { name: 'Explain current view' })).toBeVisible();
});

it('retains partial results and exposes the actual failure and retry path', () => {
  mount(value({ state: { requestId: 'run', revision: 'revision', status: 'unavailable', completed: 2, total: 7,
    detail: 'Connect an agent in chat to explain this model.' } }));
  expect(screen.getByRole('button', { name: 'Explanations' })).toHaveAttribute('aria-pressed', 'true');
  fireEvent.click(screen.getByRole('button', { name: 'Model explanation details' }));
  expect(screen.getByText('Connect an agent in chat to explain this model.')).toBeVisible();
  expect(screen.getByRole('button', { name: 'Try again' })).toBeVisible();
  expect(screen.getByRole('button', { name: 'Open agent chat' })).toBeVisible();
});

it('does not offer unsupported hosts an action and explains why a disconnected view is disabled', () => {
  const view = mount(null);
  expect(screen.queryByRole('button')).toBeNull();
  view.rerender(<NodeExplanationContext.Provider value={value({ disabledReason: 'Reconnect to the graph host first.' })}><ModelExplanationControl compact={false} /></NodeExplanationContext.Provider>);
  expect(screen.getByRole('button', { name: 'Explain model' })).toBeDisabled();
  fireEvent.focus(screen.getByRole('button', { name: 'Explain model' }).parentElement!);
  expect(screen.getByRole('tooltip')).toHaveTextContent('Reconnect to the graph host first.');
});
