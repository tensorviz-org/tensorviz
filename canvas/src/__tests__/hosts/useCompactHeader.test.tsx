import React from 'react';
import { act, render } from '@testing-library/react';
import { useCompactHeader } from '../../ide/useCompactHeader';

// jsdom has no layout: the header's measurements are set by hand and the
// ResizeObserver callback is fired as a resize would.
let resize: () => void = () => {};
beforeAll(() => {
  (globalThis as any).ResizeObserver = class { constructor(callback: () => void) { resize = callback; } observe() {} disconnect() {} };
});
afterAll(() => { delete (globalThis as any).ResizeObserver; });

function header({ width, nameCut, actionsHeight }: { width: number; nameCut: boolean; actionsHeight: number }) {
  const element = document.createElement('header');
  element.innerHTML = '<span class="local-browser-model-name"><strong>tensorViz / Net</strong></span><div class="local-browser-header-actions"></div>';
  const strong = element.querySelector('strong')!, actions = element.querySelector('.local-browser-header-actions')!;
  const set = (next: { width: number; nameCut: boolean; actionsHeight: number }) => {
    Object.defineProperty(element, 'clientWidth', { configurable: true, value: next.width });
    Object.defineProperty(strong, 'clientWidth', { configurable: true, value: next.nameCut ? 20 : 120 });
    Object.defineProperty(strong, 'scrollWidth', { configurable: true, value: 120 });
    actions.getBoundingClientRect = () => ({ height: next.actionsHeight } as DOMRect);
  };
  set({ width, nameCut, actionsHeight });
  return { element, set };
}

function Probe({ element, seen }: { element: HTMLElement; seen: boolean[] }) {
  seen.push(useCompactHeader(element));
  return null;
}

it('shows labels while they fit and icons once they would cut the model name or wrap the controls', () => {
  const fake = header({ width: 1800, nameCut: false, actionsHeight: 38 });
  const seen: boolean[] = [];
  render(<Probe element={fake.element} seen={seen} />);
  expect(seen.at(-1)).toBe(false);
  fake.set({ width: 1200, nameCut: true, actionsHeight: 38 });
  act(() => resize());
  expect(seen.at(-1)).toBe(true);
  // No wider than where labels failed: stay compact without trying them again.
  act(() => resize());
  expect(seen.at(-1)).toBe(true);
  // Wider: labels are tried again and kept, because now they fit.
  fake.set({ width: 1800, nameCut: false, actionsHeight: 38 });
  act(() => resize());
  expect(seen.at(-1)).toBe(false);
  fake.set({ width: 1700, nameCut: false, actionsHeight: 76 });
  act(() => resize());
  expect(seen.at(-1)).toBe(true);
});

it('goes back to icons in the same pass when wider is still not wide enough', () => {
  const fake = header({ width: 1000, nameCut: true, actionsHeight: 38 });
  const seen: boolean[] = [];
  render(<Probe element={fake.element} seen={seen} />);
  expect(seen.at(-1)).toBe(true);
  fake.set({ width: 1100, nameCut: true, actionsHeight: 38 });
  act(() => resize());
  expect(seen.at(-1)).toBe(true);
  expect(seen.slice(-2)).toEqual([false, true]);
});
