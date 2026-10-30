import { useSyncExternalStore } from 'react';

/** Code focus on: selecting a node opens its source beside the Canvas, and
 *  double-click or Enter on a layer moves to it. Off: exploring the graph never
 *  opens or dims code; source links still open it. */
type Surface = 'browser' | 'host';

let surface: Surface = 'host';
let on = false;
const listeners = new Set<() => void>();
const storageKey = () => `tensorviz.canvas.codeFocus.${surface}`;
const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
};

export const isCodeFocusOn = (): boolean => on;
export const useCodeFocusOn = () => useSyncExternalStore(subscribe, isCodeFocusOn, isCodeFocusOn);

/** Each surface keeps its own saved choice, like its appearance; off when none is saved. */
export function configureCodeFocusOn(nextSurface: Surface) {
  surface = nextSurface;
  on = false;
  try {
    on = localStorage.getItem(storageKey()) === 'on';
  } catch { /* A private/blocked storage context still permits a session choice. */ }
  listeners.forEach(listener => listener());
}

export function setCodeFocusOn(next: boolean) {
  if (next === on) return;
  on = next;
  try { localStorage.setItem(storageKey(), next ? 'on' : 'off'); } catch { /* Apply the choice even when persistence is unavailable. */ }
  listeners.forEach(listener => listener());
}

/** For code outside React, such as the selection transport. */
export function onCodeFocusOnChange(listener: (on: boolean) => void) {
  return subscribe(() => listener(on));
}
