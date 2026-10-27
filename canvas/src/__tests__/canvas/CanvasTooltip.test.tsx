import React from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import CanvasTooltip from '../../components/CanvasTooltip';

afterEach(() => { jest.useRealTimers(); jest.restoreAllMocks(); });

it('explains on hover and focus, preserves activation, and dismisses with Escape', () => {
  jest.useFakeTimers();
  const click = jest.fn();
  render(<CanvasTooltip text="Changes the view without running Python."><button title="Old native title" onClick={click}>Layers</button></CanvasTooltip>);
  const button = screen.getByRole('button');
  expect(button).not.toHaveAttribute('title');
  fireEvent.pointerEnter(button.parentElement!);
  expect(screen.queryByRole('tooltip')).toBeNull();
  act(() => jest.advanceTimersByTime(350));
  const tip = screen.getByRole('tooltip');
  expect(button).toHaveAttribute('aria-describedby', tip.id);
  expect(tip).toHaveTextContent('without running Python');
  fireEvent.pointerLeave(button.parentElement!);
  fireEvent.pointerEnter(tip);
  act(() => jest.advanceTimersByTime(200));
  expect(screen.getByRole('tooltip')).toBe(tip);
  fireEvent.keyDown(document, { key: 'Escape' });
  expect(screen.queryByRole('tooltip')).toBeNull();
  fireEvent.focus(button);
  expect(screen.getByRole('tooltip')).toBeVisible();
  fireEvent.click(button);
  expect(click).toHaveBeenCalledTimes(1);
  expect(screen.queryByRole('tooltip')).toBeNull();
  fireEvent.blur(button);
});

it('makes a disabled control’s reason reachable from the keyboard', () => {
  const click = jest.fn();
  render(<CanvasTooltip text="Select a Python environment first."><button disabled onClick={click}>Refresh</button></CanvasTooltip>);
  const anchor = screen.getByRole('button').parentElement!;
  expect(anchor).toHaveAttribute('tabindex', '0');
  fireEvent.focus(anchor);
  expect(screen.getByRole('tooltip')).toHaveTextContent('Select a Python environment');
  expect(anchor).toHaveAttribute('aria-describedby', screen.getByRole('tooltip').id);
  fireEvent.click(screen.getByRole('button'));
  expect(click).not.toHaveBeenCalled();
});

it('keeps help inside a narrow viewport and above a footer control', () => {
  jest.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
    return (this.getAttribute('role') === 'tooltip'
      ? { width: 220, height: 64, top: 0, bottom: 64, left: 0, right: 220 }
      : { width: 30, height: 30, top: window.innerHeight - 35, bottom: window.innerHeight - 5, left: window.innerWidth - 35, right: window.innerWidth - 5 }) as DOMRect;
  });
  render(<CanvasTooltip text="Fit the entire graph."><button>Fit</button></CanvasTooltip>);
  fireEvent.focus(screen.getByRole('button'));
  const tip = screen.getByRole('tooltip');
  expect(Number.parseFloat(tip.style.left) + 220).toBeLessThanOrEqual(window.innerWidth - 8);
  expect(Number.parseFloat(tip.style.top) + 64).toBeLessThan(window.innerHeight - 35);
  fireEvent.resize(window);
  expect(screen.queryByRole('tooltip')).toBeNull();
});
