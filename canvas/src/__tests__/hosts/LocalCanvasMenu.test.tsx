import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { Wrench } from 'lucide-react';
import LocalCanvasMenu from '../../ide/LocalCanvasMenu';
import type { ModelNode } from '../../types';

jest.mock('../../canvas/GraphQueryPanel', () => ({ GraphOverlayControls: () => <div>Measured overlay controls</div> }));

const names = { 'view-blocks': 'Blocks view', 'view-expanded': 'Expanded view', 'svg-overview': 'SVG overview',
  layout: 'Auto layout', select: 'Select tool', focus: 'Focus selected', 'zoom-in': 'Zoom in', 'zoom-out': 'Zoom out',
  fullscreen: 'Fullscreen canvas', 'download-svg': 'Download SVG' };
function mount(nodes: ModelNode[] = []) {
  const actions = Object.entries(names).map(([id, label]) => ({ label, icon: Wrench, testid: `ui2-action-${id}`,
    run: jest.fn(), hint: id === 'focus' ? 'Select a module/layer first' : label, disabled: id === 'focus' }));
  const result = render(<LocalCanvasMenu actions={actions} nodes={nodes} viewMode="expanded" svgOverview={false} canvasTool="hand" />);
  fireEvent.click(screen.getByRole('button', { name: 'More tools' }));
  return { actions, ...result };
}

it('keeps every existing action reachable in the grouped menu and respects disabled commands', () => {
  const { actions } = mount();
  const menu = screen.getByRole('menu', { name: 'Canvas' });
  for (const name of Object.values(names)) expect(within(menu).getByText(name)).toBeVisible();
  expect(screen.getByRole('menuitemcheckbox', { name: 'Expanded view' })).toHaveAttribute('aria-checked', 'true');
  expect(screen.getByRole('menuitemcheckbox', { name: 'Select tool' })).toHaveAttribute('aria-checked', 'false');
  const disabled = screen.getByRole('menuitem', { name: 'Focus selected' });
  expect(disabled).toBeDisabled();
  fireEvent.focus(disabled.parentElement!);
  expect(screen.getByRole('tooltip')).toHaveTextContent('Select a module/layer first');
  expect(disabled).toHaveAttribute('aria-describedby', screen.getByRole('tooltip').id);
  fireEvent.keyDown(document, { key: 'Escape' });
  expect(screen.queryByRole('tooltip')).toBeNull();
  fireEvent.click(disabled);
  expect(actions.find(action => action.label === 'Focus selected')!.run).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('menuitem', { name: 'Download SVG' }));
  expect(actions.find(action => action.label === 'Download SVG')!.run).toHaveBeenCalledTimes(1);
  expect(screen.queryByRole('menu')).toBeNull();
  expect(screen.getByRole('button', { name: 'More tools' })).toHaveFocus();
});

it('supports keyboard navigation and keeps metric settings behind their explicit action', () => {
  mount();
  expect(screen.queryByText('Measured overlay controls')).toBeNull();
  const menu = screen.getByRole('menu');
  fireEvent.keyDown(menu, { key: 'ArrowUp' });
  expect(screen.getByRole('menuitem', { name: 'Download SVG' })).toHaveFocus();
  fireEvent.keyDown(menu, { key: 'Home' });
  expect(screen.getByRole('menuitemcheckbox', { name: 'Blocks view' })).toHaveFocus();
  fireEvent.keyDown(menu, { key: 'End' });
  expect(screen.getByRole('menuitem', { name: 'Download SVG' })).toHaveFocus();
  fireEvent.click(screen.getByRole('menuitem', { name: 'Display settings' }));
  expect(screen.getByRole('dialog', { name: 'Display settings' })).toHaveTextContent('Measured overlay controls');
  fireEvent.click(screen.getByRole('button', { name: 'Back to canvas actions' }));
  fireEvent.keyDown(screen.getByRole('menu'), { key: 'Escape' });
  expect(screen.queryByRole('menu')).toBeNull();
  expect(screen.getByRole('button', { name: 'More tools' })).toHaveFocus();
});

it('only offers unchanged-call visibility for a graph with foldable calls and dispatches its existing command', () => {
  const { unmount } = mount();
  expect(screen.queryByTestId('toggle-unchanged-calls')).toBeNull();
  unmount();
  mount([{ id: 'identity', type: 'moduleNode', position: { x: 0, y: 0 }, data: {
    parameters: {}, capturePassThrough: { valueId: 'x', inputPortId: 'in', sourceNodeId: 'source', sourcePortId: 'out' },
  } } as ModelNode]);
  const listener = jest.fn();
  window.addEventListener('builder-command', listener);
  try {
    fireEvent.click(screen.getByRole('menuitemcheckbox', { name: 'Show unchanged calls (1)' }));
    expect(listener.mock.calls[0][0].detail).toEqual({ command: 'show-unchanged-calls', payload: { shown: true } });
  } finally { window.removeEventListener('builder-command', listener); }
});
