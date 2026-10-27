import React from 'react';
import { render, screen } from '@testing-library/react';
import { ThemeProvider } from '../../contexts/ThemeContext';

afterEach(() => {
  localStorage.removeItem('canvas-theme.v2');
  document.documentElement.classList.remove('light', 'dark');
});

it.each([null, 'invalid', 'dark', 'light'])('preserves the stored palette when mounting the app (%s)', saved => {
  if (saved !== null) localStorage.setItem('canvas-theme.v2', saved);
  // A remount must clear the opposite palette left by an earlier page.
  document.documentElement.classList.add('light', 'dark');
  render(<ThemeProvider><div>Workbench</div></ThemeProvider>);
  const expected = saved === 'light' ? 'light' : 'dark';
  expect(screen.getByText('Workbench')).toBeVisible();
  expect(document.documentElement).toHaveClass(expected);
  expect(document.documentElement).not.toHaveClass(expected === 'light' ? 'dark' : 'light');
  expect(localStorage.getItem('canvas-theme.v2')).toBe(expected);
});
