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
  fireEvent.click(screen.getByRole('button', { name: 'Explain this view' }));
  expect(listener.mock.calls[0][0].detail).toEqual({ command: 'explain-model', payload: undefined });
  window.removeEventListener('builder-command', listener);
});

it('shows real progress and lets the user stop without losing completed work', () => {
  const context = value({ state: { requestId: 'run', revision: 'revision', status: 'loading', completed: 3, total: 7 } });
  mount(context, true);
  expect(screen.getByRole('button', { name: 'Explaining view · 3/7' })).toBeDisabled();
  expect(screen.getByRole('status')).toHaveTextContent('3 of 7');
  fireEvent.click(screen.getByRole('button', { name: 'Stop explaining view' }));
  expect(context.model!.cancel).toHaveBeenCalledTimes(1);
});

it('explains the current viewport again after success and keeps hiding separate', () => {
  const context = value({ state: { requestId: 'run', revision: 'revision', status: 'ready', completed: 7, total: 7 } });
  mount(context);
  const listener = jest.fn();
  window.addEventListener('builder-command', listener);
  fireEvent.click(screen.getByRole('button', { name: 'Explain this view' }));
  expect(listener.mock.calls[0][0].detail.command).toBe('explain-model');
  window.removeEventListener('builder-command', listener);
  expect(context.model!.toggle).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'View explanation details' }));
  expect(screen.getByRole('status')).toHaveTextContent('7 of 7 components explained');
  expect(screen.getByRole('button', { name: 'Explain current view' })).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Hide explanations' }));
  expect(context.model!.toggle).toHaveBeenCalledTimes(1);
});

it('retains partial results and exposes the actual failure and retry path', () => {
  mount(value({ state: { requestId: 'run', revision: 'revision', status: 'unavailable', completed: 2, total: 7,
    detail: 'Connect an agent in chat to explain this model.' } }));
  expect(screen.getByRole('button', { name: 'Retry explanations' })).toBeEnabled();
  fireEvent.click(screen.getByRole('button', { name: 'View explanation details' }));
  expect(screen.getByText('Connect an agent in chat to explain this model.')).toBeVisible();
  expect(screen.getByRole('button', { name: 'Try again' })).toBeVisible();
  expect(screen.getByRole('button', { name: 'Open agent chat' })).toBeVisible();
});

it('shows the reason for a failed start instead of a misleading zero-completed count', () => {
  const detail = 'The configured agent needs sign-in. Open agent chat to connect.';
  mount(value({ state: { requestId: 'run', revision: 'revision', status: 'unavailable', completed: 0, total: 2, detail } }));
  const trigger = screen.getByRole('button', { name: 'Retry explanations' });
  fireEvent.focus(trigger.parentElement!);
  expect(screen.getByRole('tooltip')).toHaveTextContent(detail);
  fireEvent.blur(trigger.parentElement!);
  fireEvent.click(screen.getByRole('button', { name: 'View explanation details' }));
  expect(screen.getByRole('alert')).toHaveTextContent(detail);
  expect(screen.queryByText('0 of 2 components explained.')).not.toBeInTheDocument();
  expect(screen.queryByText(/Up to 128 visible components/)).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Open agent chat' })).toBeVisible();
});

it('does not offer unsupported hosts an action and explains why a disconnected view is disabled', () => {
  const view = mount(null);
  expect(screen.queryByRole('button')).toBeNull();
  view.rerender(<NodeExplanationContext.Provider value={value({ disabledReason: 'Reconnect to the graph host first.' })}><ModelExplanationControl compact={false} /></NodeExplanationContext.Provider>);
  expect(screen.getByRole('button', { name: 'Explain this view' })).toBeDisabled();
  fireEvent.focus(screen.getByRole('button', { name: 'Explain this view' }).parentElement!);
  expect(screen.getByRole('tooltip')).toHaveTextContent('Reconnect to the graph host first.');
});

it('shows the selected explanation model before starting and lets effort be chosen without changing chat', () => {
  const choose = jest.fn(), discover = jest.fn();
  mount(value({ configuration: { settings: { model: 'research-model', effort: 'low' }, followsChat: false,
    choose, discover, availability: { status: 'ready', detail: '', defaultModel: 'research-model', models: [
      { id: 'research-model', label: 'Research model', description: 'Source analysis', efforts: ['low', 'high'], defaultEffort: 'high' },
    ] } } }));
  expect(screen.getByRole('button', { name: 'Explain this view' })).toHaveTextContent('Research model · Low');
  fireEvent.click(screen.getByRole('button', { name: 'Choose explanation model and effort' }));
  expect(discover).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('menuitem', { name: 'Effort for Research model' }));
  fireEvent.click(screen.getByRole('menuitemradio', { name: 'High', exact: true }));
  expect(choose).toHaveBeenCalledWith({ model: 'research-model', effort: 'high' });
  expect(screen.queryByRole('dialog', { name: 'Explanation model' })).toBeNull();
});

it('discovers the model catalogue when opening a cold explanation menu, and prevents unsupported starts', () => {
  const configuration = { settings: {}, followsChat: true, choose: jest.fn(), discover: jest.fn() };
  const view = mount(value({ configuration }));
  fireEvent.click(screen.getByRole('button', { name: 'Choose explanation model and effort' }));
  expect(configuration.discover).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole('button', { name: 'Close explanation model' }));
  view.rerender(<NodeExplanationContext.Provider value={value({ configuration: { ...configuration,
    settings: { model: 'retired' }, availability: { status: 'ready', models: [], detail: '' } } })}>
    <ModelExplanationControl compact={false} />
  </NodeExplanationContext.Provider>);
  expect(screen.getByRole('button', { name: 'Explain this view' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Choose explanation model and effort' })).toBeEnabled();
});

it('keeps the running execution choice visible and disables configuration until it settles', () => {
  mount(value({ state: { requestId: 'run', revision: 'revision', status: 'loading', completed: 1, total: 4,
    execution: { provider: 'codex', model: 'running-model', effort: 'low' } },
    configuration: { settings: { model: 'next-model' }, followsChat: false, choose: jest.fn(), discover: jest.fn() } }));
  expect(screen.getByRole('button', { name: 'Explaining view · 1/4' })).toHaveTextContent('running-model · Low');
  expect(screen.getByRole('button', { name: 'Choose explanation model and effort' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Stop explaining view' })).toBeEnabled();
});
