import React from 'react';
import { act, render, screen } from '@testing-library/react';
import {
  configureCanvasTheme, getCanvasTheme, getCanvasThemePreference, onCanvasThemeChange,
  setCanvasHostTheme, setCanvasThemePreference, useCanvasTheme,
} from '../../styles/canvasTheme';

beforeEach(() => { localStorage.clear(); configureCanvasTheme('host'); });

test('the browser starts in Paper while embedded Canvas follows its host', () => {
  expect(getCanvasTheme()).toBe('dark');
  setCanvasHostTheme('light');
  expect(document.documentElement.dataset.tvTheme).toBe('light');
  configureCanvasTheme('browser');
  setCanvasHostTheme('dark');
  expect(getCanvasTheme()).toBe('light');
});

test('the website can default to Ink while a saved Paper choice and the local browser default remain independent', () => {
  configureCanvasTheme('browser', 'dark');
  expect(getCanvasTheme()).toBe('dark');
  expect(document.documentElement.dataset.tvTheme).toBe('dark');
  setCanvasThemePreference('light');
  configureCanvasTheme('browser', 'dark');
  expect(getCanvasTheme()).toBe('light');
  expect(getCanvasThemePreference()).toBe('light');
  setCanvasThemePreference('default');
  expect(getCanvasTheme()).toBe('dark');
  expect(localStorage.getItem('tensorviz.canvas.theme.browser')).toBeNull();
  setCanvasHostTheme('light');
  expect(getCanvasTheme()).toBe('dark');
  configureCanvasTheme('browser');
  expect(getCanvasTheme()).toBe('light');
});

test.each(['light', 'dark'] as const)('a browser default of %s does not replace the embedded host theme', browserDefault => {
  configureCanvasTheme('host', browserDefault);
  expect(getCanvasTheme()).toBe('dark');
  setCanvasHostTheme('light');
  expect(getCanvasTheme()).toBe('light');
});

test('an explicit choice survives reopening and is independent of the other surface', () => {
  configureCanvasTheme('browser');
  setCanvasThemePreference('dark');
  configureCanvasTheme('host');
  setCanvasHostTheme('light');
  expect(getCanvasTheme()).toBe('light');
  setCanvasThemePreference('dark');
  setCanvasHostTheme('light');
  expect(getCanvasTheme()).toBe('dark');
  setCanvasThemePreference('default');
  expect(getCanvasTheme()).toBe('light');
  configureCanvasTheme('browser');
  expect(getCanvasThemePreference()).toBe('dark');
});

test('theme listeners update the mounted view and stop after teardown', () => {
  const listener = jest.fn();
  const stop = onCanvasThemeChange(listener);
  function ReadTheme() { return <output>{useCanvasTheme()}</output>; }
  render(<ReadTheme />);
  act(() => setCanvasThemePreference('light'));
  expect(screen.getByText('light')).toBeInTheDocument();
  expect(listener).toHaveBeenLastCalledWith('light');
  stop(); listener.mockClear();
  act(() => setCanvasThemePreference('dark'));
  expect(screen.getByText('dark')).toBeInTheDocument();
  expect(listener).not.toHaveBeenCalled();
});

test('blocked storage and an invalid saved value still permit a session choice', () => {
  localStorage.setItem('tensorviz.canvas.theme.browser', 'unknown');
  configureCanvasTheme('browser');
  expect(getCanvasThemePreference()).toBe('default');
  const read = jest.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('blocked'); });
  const write = jest.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked'); });
  try {
    configureCanvasTheme('host');
    setCanvasThemePreference('light');
    expect(getCanvasTheme()).toBe('light');
  } finally { read.mockRestore(); write.mockRestore(); }
});
