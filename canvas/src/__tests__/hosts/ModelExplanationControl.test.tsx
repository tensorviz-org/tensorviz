import React from 'react';
import { readFileSync } from 'fs';
import { join } from 'path';
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

it('turns the same primary button into a usable stop action with progress', () => {
  const context = value({ state: { requestId: 'run', revision: 'revision', status: 'loading', completed: 3, total: 7 } });
  const view = mount(value(), true);
  const primary = screen.getByRole('button', { name: 'Explain this view' });
  expect(primary.querySelector('.lucide-sparkles')).toBeInTheDocument();
  view.rerender(<NodeExplanationContext.Provider value={context}><ModelExplanationControl compact /></NodeExplanationContext.Provider>);
  expect(screen.getByRole('button', { name: 'Stop explaining view · 3/7' })).toBe(primary);
  expect(primary).toBeEnabled();
  expect(primary).toHaveAttribute('aria-busy', 'true');
  expect(primary.querySelector('.model-explanation-spinner')).toBeInTheDocument();
  expect(screen.getAllByRole('button').filter(button => /Stop explaining view/.test(button.getAttribute('aria-label') ?? ''))).toEqual([primary]);
  expect(screen.getByRole('status')).toHaveTextContent('3 of 7');
  fireEvent.click(primary);
  expect(context.model!.cancel).toHaveBeenCalledTimes(1);
  expect(context.model!.cancel).toHaveBeenCalledWith();
});

it('explains the current viewport again after success and keeps hiding separate', () => {
  const context = value({ state: { requestId: 'run', revision: 'revision', status: 'ready', completed: 7, total: 7 } });
  const view = mount(value());
  const primary = screen.getByRole('button', { name: 'Explain this view' });
  const overflow = screen.getByRole('button', { name: 'View explanation details' });
  const sparkle = primary.querySelector('.lucide-sparkles');
  expect(overflow).toBeDisabled();
  view.rerender(<NodeExplanationContext.Provider value={context}><ModelExplanationControl compact={false} /></NodeExplanationContext.Provider>);
  expect(screen.getByRole('button', { name: 'View explained · 7/7. Explain this view again' })).toBe(primary);
  expect(primary.querySelector('.lucide-sparkles')).toBe(sparkle);
  expect(sparkle).toHaveClass('model-explanation-complete');
  expect(primary).toHaveAttribute('data-state', 'complete');
  expect(primary.querySelector('.lucide-check')).not.toBeInTheDocument();
  expect(primary).toHaveTextContent('Explained');
  fireEvent.focus(primary);
  expect(screen.getByRole('tooltip')).toHaveTextContent('7 of 7 components explained');
  fireEvent.blur(primary);
  expect(screen.getByRole('button', { name: 'View explanation details' })).toBe(overflow);
  expect(overflow.querySelector('.lucide-ellipsis')).toBeInTheDocument();
  expect(overflow).not.toHaveTextContent('7/7');
  expect(screen.queryByText('Explanation ready')).not.toBeInTheDocument();
  const listener = jest.fn();
  window.addEventListener('builder-command', listener);
  fireEvent.click(primary);
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
  expect(screen.getByRole('button', { name: 'Explanations unavailable. Retry explanations' })).toBeEnabled();
  fireEvent.click(screen.getByRole('button', { name: 'View explanation details' }));
  expect(screen.getByText('Connect an agent in chat to explain this model.')).toBeVisible();
  expect(screen.getByRole('button', { name: 'Try again' })).toBeVisible();
  expect(screen.getByRole('button', { name: 'Open agent chat' })).toBeVisible();
});

it('shows preflight failure on the primary button without inventing a graph revision', () => {
  mount(value({ startFailure: { requestId: 'preflight', detail: 'The graph revision could not be verified.' } }));
  const button = screen.getByRole('button', { name: 'Explanations unavailable. Retry explanations' });
  expect(button.querySelector('.lucide-circle-alert')).not.toBeNull();
  fireEvent.focus(button);
  expect(screen.getByRole('tooltip')).toHaveTextContent('The graph revision could not be verified.');
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
});

it('shows the reason for a failed start instead of a misleading zero-completed count', () => {
  const detail = 'The configured agent needs sign-in. Open agent chat to connect.';
  mount(value({ state: { requestId: 'run', revision: 'revision', status: 'unavailable', completed: 0, total: 2, detail } }));
  const trigger = screen.getByRole('button', { name: 'Explanations unavailable. Retry explanations' });
  fireEvent.focus(trigger.parentElement!);
  expect(screen.getByRole('tooltip')).toHaveTextContent(detail);
  fireEvent.blur(trigger.parentElement!);
  fireEvent.click(screen.getByRole('button', { name: 'View explanation details' }));
  expect(screen.getByRole('alert')).toHaveTextContent(detail);
  expect(screen.queryByText('0 of 2 components explained.')).not.toBeInTheDocument();
  expect(screen.queryByText(/Up to 128 visible components/)).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Open agent chat' })).toBeVisible();
});

it.each([
  ['unavailable', 'Explanations unavailable. Retry explanations', '.lucide-circle-alert', 'Retry explanations'],
  ['cancelled', 'Explanations stopped. Retry explanations', '.lucide-square', 'Stopped · Retry'],
] as const)('keeps %s feedback and diagnostics on the same primary button', (status, label, icon, copy) => {
  const detail = status === 'cancelled' ? 'Stopped by you. Completed results are retained.' : 'Agent sign-in expired.';
  const view = mount(value());
  const primary = screen.getByRole('button', { name: 'Explain this view' });
  const context = value({ state: { requestId: 'run', revision: 'revision', status, completed: 2, total: 7, detail } });
  view.rerender(<NodeExplanationContext.Provider value={context}><ModelExplanationControl compact={false} /></NodeExplanationContext.Provider>);
  expect(screen.getByRole('button', { name: label })).toBe(primary);
  expect(primary.querySelector(icon)).toBeInTheDocument();
  expect(primary.querySelector('.model-explanation-complete')).not.toBeInTheDocument();
  expect(primary).not.toHaveAttribute('data-state', 'complete');
  expect(primary).toHaveTextContent(copy);
  fireEvent.focus(primary);
  expect(screen.getByRole('tooltip')).toHaveTextContent(detail);
  expect(screen.getByRole('tooltip')).toHaveTextContent('2 of 7 components explained');
  fireEvent.blur(primary);
  const overflow = screen.getByRole('button', { name: 'View explanation details' });
  expect(overflow.querySelector('.lucide-ellipsis')).toBeInTheDocument();
  expect(overflow.querySelector(icon)).not.toBeInTheDocument();
  const listener = jest.fn();
  window.addEventListener('builder-command', listener);
  fireEvent.click(primary);
  expect(listener.mock.calls[0][0].detail.command).toBe('explain-model');
  window.removeEventListener('builder-command', listener);
  fireEvent.click(overflow);
  fireEvent.click(screen.getByRole('button', { name: 'Hide explanations' }));
  expect(context.model!.toggle).toHaveBeenCalledTimes(1);
});

it('never claims success for a ready pass with no completed explanations', () => {
  mount(value({ state: { requestId: 'empty', revision: 'revision', status: 'ready', completed: 0, total: 0,
    detail: 'No visible components to explain.' } }));
  const primary = screen.getByRole('button', { name: 'Explain this view' });
  expect(primary.querySelector('.model-explanation-complete')).not.toBeInTheDocument();
  expect(primary).not.toHaveAttribute('data-state', 'complete');
  expect(primary.querySelector('.lucide-sparkles')).toBeInTheDocument();
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
  expect(screen.getByRole('button', { name: 'Stop explaining view · 1/4' })).toHaveTextContent('running-model · Low');
  expect(screen.getByRole('button', { name: 'Choose explanation model and effort' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Stop explaining view · 1/4' })).toBeEnabled();
});

it.each([false, true])('reserves a stable primary and overflow footprint, compact=%s', compact => {
  const view = mount(value(), compact);
  const primary = screen.getByRole('button', { name: 'Explain this view' });
  expect(primary.closest('.model-explanation-control')).toHaveClass(compact ? 'model-explanation-compact' : 'model-explanation-control');
  const css = readFileSync(join(__dirname, '../../ide/modelExplanation.css'), 'utf8');
  const rule = css.match(/\.model-explanation-control \.model-explanation-start \{([^}]+)\}/)?.[1] ?? '';
  expect(rule).toMatch(/width:\s*\d+px/);
  expect(rule).toMatch(/height:\s*\d+px/);
  const choiceRule = css.match(/\.model-explanation-choice \{([^}]+)\}/)?.[1] ?? '';
  expect(choiceRule).toMatch(/white-space:\s*nowrap/);
  expect(choiceRule).toMatch(/text-overflow:\s*ellipsis/);
  const overflow = screen.getByRole('button', { name: 'View explanation details' });
  view.rerender(<NodeExplanationContext.Provider value={value({ state: { requestId: 'run', revision: 'r', status: 'loading', completed: 1, total: 20 } })}>
    <ModelExplanationControl compact={compact} />
  </NodeExplanationContext.Provider>);
  expect(screen.getByRole('button', { name: 'Stop explaining view · 1/20' })).toBe(primary);
  expect(screen.getByRole('button', { name: 'View explanation details' })).toBe(overflow);
  expect(overflow).toBeDisabled();
});

it('keeps progress readable without animation when reduced motion is requested', () => {
  const css = readFileSync(join(__dirname, '../../ide/modelExplanation.css'), 'utf8');
  expect(css).toMatch(/@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{\s*\.model-explanation-spinner\s*\{\s*animation:\s*none/);
});
