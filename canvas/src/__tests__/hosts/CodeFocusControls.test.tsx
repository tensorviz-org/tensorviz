import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import CodeFocusControls from '../../ide/CodeFocusControls';
import { LocalBrowserContext } from '../../ide/localBrowserContext';
import { EMPTY_LOCAL_BROWSER_PAGE_STATE } from '../../ide/localBrowserState';
import { DEFAULT_CODE_FOCUS } from '../../ide/protocol.generated';
import { configureCodeFocusOn, isCodeFocusOn } from '../../ide/codeFocusOn';

beforeEach(() => { localStorage.clear(); configureCodeFocusOn('host'); });

function mount(parameter = false, connected = true, reviewing = false, openOptions = true) {
  const change = jest.fn();
  function Harness() {
    const [options, setOptions] = React.useState(DEFAULT_CODE_FOCUS);
    return <LocalBrowserContext.Provider value={{ page: EMPTY_LOCAL_BROWSER_PAGE_STATE,
      status: { phase: 'connected', hostLink: connected ? 'connected' : 'disconnected', error: null }, reviewing,
      codeFocus: { options, parameter, detail: 'Direct references only.', change: next => { change(next); setOptions(next); } } }}>
      <CodeFocusControls compact />
    </LocalBrowserContext.Provider>;
  }
  render(<Harness />);
  if (!reviewing && openOptions) fireEvent.click(screen.getByRole('button', { name: 'Code focus options' }));
  return change;
}

test('one click turns Code focus on and off, and the choice is kept', () => {
  mount(false, true, false, false);
  const toggle = screen.getByRole('button', { name: 'Code focus', exact: true });
  expect(toggle).toHaveAttribute('aria-pressed', 'false');
  fireEvent.click(toggle);
  expect(toggle).toHaveAttribute('aria-pressed', 'true');
  expect(isCodeFocusOn()).toBe(true);
  expect(localStorage.getItem('tensorviz.canvas.codeFocus.host')).toBe('on');
  fireEvent.click(toggle);
  expect(isCodeFocusOn()).toBe(false);
  fireEvent.focus(toggle);
  expect(screen.getByRole('tooltip')).toHaveTextContent('never opens or dims code');
});

test('scope, dimming and opacity are independent preferences accessible in a compact toolbar', () => {
  const change = mount();
  fireEvent.click(screen.getByRole('checkbox', { name: 'Dim unrelated code' }));
  fireEvent.change(screen.getByRole('combobox', { name: 'Code focus scope' }), { target: { value: 'class' } });
  fireEvent.change(screen.getByRole('slider'), { target: { value: '40' } });
  expect(change).toHaveBeenLastCalledWith({ enabled: true, scope: 'class', references: 'both', opacity: 40 });
  expect(screen.getByRole('combobox', { name: 'Parameter source' })).toBeDisabled();
  expect(screen.getByText(/References apply when the selected node is a parameter/)).toBeVisible();
  expect(screen.getByRole('status')).toHaveTextContent('Direct references only.');
  fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' });
  expect(screen.queryByRole('dialog')).toBeNull();
  expect(isCodeFocusOn()).toBe(false);
});

test('parameter declaration and uses apply within the selected scope', () => {
  const change = mount(true);
  fireEvent.change(screen.getByRole('combobox', { name: 'Code focus scope' }), { target: { value: 'function' } });
  fireEvent.change(screen.getByRole('combobox', { name: 'Parameter source' }), { target: { value: 'uses' } });
  expect(change).toHaveBeenLastCalledWith({ ...DEFAULT_CODE_FOCUS, scope: 'function', references: 'uses' });
});

test('a disconnected view cannot change editor focus, though the choice itself can change', () => {
  mount(true, false);
  for (const control of screen.getAllByRole('combobox')) expect(control).toBeDisabled();
  expect(screen.getByRole('checkbox')).toBeDisabled();
  expect(screen.getByRole('slider')).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Code focus', exact: true })).toBeEnabled();
});

test('graph review does not offer source focus on a candidate graph', () => {
  mount(true, true, true);
  expect(screen.queryByRole('button', { name: 'Code focus', exact: true })).toBeNull();
  expect(screen.queryByRole('button', { name: 'Code focus options' })).toBeNull();
});
