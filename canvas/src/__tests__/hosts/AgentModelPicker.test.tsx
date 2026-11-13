import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import AgentModelPicker, { AgentModelMenu } from '../../ide/AgentModelPicker';
import type { CanvasAgentAvailability, CanvasAgentSettings } from '../../ide/protocol.generated';

const availability: CanvasAgentAvailability = {
  status: 'ready', detail: '', defaultModel: 'reasoning-model', models: [
    { id: 'reasoning-model', label: 'Research model', description: 'Description supplied by the connected provider.', efforts: ['minimal', 'medium', 'ultra'], defaultEffort: 'medium' },
    { id: 'custom/model-v37', label: 'Custom model v37', description: 'New provider model.', efforts: ['adaptive'], defaultEffort: 'adaptive' },
    { id: 'plain-model', label: 'Plain model', description: '', efforts: [] },
  ],
};

const modelButton = () => screen.getByRole('button', { name: /^Chat model:/ });
const effortButton = () => screen.getByRole('button', { name: /^Chat effort:/ });
function mount(settings: CanvasAgentSettings = {}) {
  const onChange = jest.fn(); const onOpen = jest.fn(); const onRefresh = jest.fn();
  const view = render(<AgentModelPicker availability={availability} settings={settings} onChange={onChange} onOpen={onOpen} onRefresh={onRefresh} labelPrefix="Chat" />);
  return { ...view, onChange, onOpen, onRefresh };
}

test('provider model and default effort are visible before opening or sending', () => {
  const { onChange } = mount();
  expect(modelButton()).toHaveTextContent('Research model');
  expect(effortButton()).toHaveTextContent('Medium');
  fireEvent.focus(modelButton());
  expect(modelButton()).toHaveAccessibleDescription('Research model · Agent default — Description supplied by the connected provider.');
  expect(onChange).not.toHaveBeenCalled();
});

test('search filters provider results by model id or name and clearing recovers the full list', () => {
  mount();
  fireEvent.click(modelButton());
  const search = screen.getByRole('searchbox', { name: 'Search models' });
  expect(search).toHaveFocus();
  fireEvent.change(search, { target: { value: 'CUSTOM/MODEL-' } });
  expect(screen.getByRole('menuitemradio', { name: 'Custom model v37' })).toBeVisible();
  expect(screen.queryByRole('menuitemradio', { name: 'Research model' })).toBeNull();
  fireEvent.change(search, { target: { value: 'missing model' } });
  expect(screen.getByRole('status')).toHaveTextContent('No matching models.');
  fireEvent.change(search, { target: { value: '' } });
  expect(screen.getByRole('menuitemradio', { name: 'Research model' })).toBeVisible();
});

test('choosing a model resets its previous effort and closes without changing the default first', () => {
  const { onChange, onOpen } = mount({ model: 'reasoning-model', effort: 'ultra' });
  fireEvent.click(modelButton());
  expect(onOpen).toHaveBeenCalledTimes(1);
  expect(onChange).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('menuitemradio', { name: 'Plain model' }));
  expect(onChange).toHaveBeenCalledWith({ model: 'plain-model' });
  expect(screen.queryByRole('dialog', { name: 'Choose model' })).toBeNull();
  expect(modelButton()).toHaveFocus();
});

test('per-model effort opens inside the same menu and accepts new capability values without a local list', () => {
  const { onChange } = mount({ model: 'reasoning-model', effort: 'ultra' });
  fireEvent.click(modelButton());
  fireEvent.click(screen.getByRole('menuitem', { name: 'Effort for Custom model v37' }));
  expect(screen.getAllByRole('dialog')).toHaveLength(1);
  expect(screen.queryByRole('searchbox')).toBeNull();
  const menu = screen.getByRole('menu', { name: 'Effort for Custom model v37' });
  expect(within(menu).getAllByRole('menuitemradio').map(button => button.textContent)).toEqual(['Default (Adaptive)', 'Adaptive']);
  expect(screen.queryByText('Thinking')).toBeNull();
  fireEvent.click(within(menu).getByRole('menuitemradio', { name: 'Adaptive' }));
  expect(onChange).toHaveBeenCalledWith({ model: 'custom/model-v37', effort: 'adaptive' });
});

test('effort selection preserves the implicit default model and can restore its default effort', () => {
  const { onChange, rerender } = mount();
  fireEvent.click(effortButton());
  expect(screen.getByRole('menuitemradio', { name: 'Default (Medium)' })).toHaveAttribute('aria-checked', 'true');
  fireEvent.click(screen.getByRole('menuitemradio', { name: 'Minimal' }));
  expect(onChange).toHaveBeenLastCalledWith({ effort: 'minimal' });
  rerender(<AgentModelPicker availability={availability} settings={{ effort: 'minimal' }} onChange={onChange} labelPrefix="Chat" />);
  fireEvent.click(effortButton());
  fireEvent.click(screen.getByRole('menuitemradio', { name: 'Default (Medium)' }));
  expect(onChange).toHaveBeenLastCalledWith({});
});

test('keyboard navigation returns from effort to its model and Escape restores trigger focus', () => {
  const { onOpen, onChange } = mount();
  fireEvent.click(modelButton());
  const search = screen.getByRole('searchbox');
  fireEvent.keyDown(search, { key: 'ArrowDown' });
  expect(screen.getByRole('menuitemradio', { name: /^Agent default/ })).toHaveFocus();
  fireEvent.keyDown(document.activeElement!, { key: 'ArrowDown' });
  expect(screen.getByRole('menuitemradio', { name: 'Research model' })).toHaveFocus();
  fireEvent.click(screen.getByRole('menuitem', { name: 'Effort for Research model' }));
  expect(screen.getByRole('button', { name: 'Models' })).toHaveFocus();
  fireEvent.keyDown(screen.getByRole('button', { name: 'Models' }), { key: 'Escape' });
  expect(screen.getByRole('menuitem', { name: 'Effort for Research model' })).toHaveFocus();
  fireEvent.keyDown(document.activeElement!, { key: 'Escape' });
  expect(screen.queryByRole('dialog')).toBeNull();
  expect(modelButton()).toHaveFocus();
  expect(onOpen).toHaveBeenCalledTimes(1);
  expect(onChange).not.toHaveBeenCalled();
});

test('saved unavailable settings remain visible, and busy disables and dismisses their menu', () => {
  const { rerender, onChange } = mount({ model: 'removed-model', effort: 'old-effort' });
  expect(modelButton()).toHaveTextContent('removed-model');
  expect(effortButton()).toHaveTextContent('Old-effort');
  fireEvent.click(modelButton());
  expect(screen.getByText('removed-model is unavailable. Choose another model.')).toBeVisible();
  rerender(<AgentModelPicker availability={availability} settings={{ model: 'removed-model' }} onChange={onChange} labelPrefix="Chat" disabled />);
  expect(modelButton()).toBeDisabled();
  expect(screen.queryByRole('dialog')).toBeNull();
  expect(onChange).not.toHaveBeenCalled();
});

test('refresh is a separate read action and embeddable content reports completion only on a choice', () => {
  const onChange = jest.fn(); const onDone = jest.fn(); const onRefresh = jest.fn();
  render(<AgentModelMenu availability={availability} settings={{}} onChange={onChange} onDone={onDone} onRefresh={onRefresh} />);
  fireEvent.click(screen.getByRole('button', { name: 'Refresh models' }));
  expect(onRefresh).toHaveBeenCalledTimes(1);
  expect(onChange).not.toHaveBeenCalled();
  expect(onDone).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('menuitemradio', { name: 'Plain model' }));
  expect(onChange).toHaveBeenCalledWith({ model: 'plain-model' });
  expect(onDone).toHaveBeenCalledTimes(1);
});
