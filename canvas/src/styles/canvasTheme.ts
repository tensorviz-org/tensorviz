import { useSyncExternalStore } from 'react';

export type CanvasTheme = 'light' | 'dark';
export type CanvasThemePreference = 'default' | CanvasTheme;
type Surface = 'browser' | 'host';

let surface: Surface = 'host';
let hostTheme: CanvasTheme = 'dark';
let browserDefault: CanvasTheme = 'light';
let preference: CanvasThemePreference = 'default';
const listeners = new Set<() => void>();
const storageKey = () => `tensorviz.canvas.theme.${surface}`;
const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
};

export const getCanvasTheme = (): CanvasTheme => preference === 'default'
  ? surface === 'browser' ? browserDefault : hostTheme : preference;
export const getCanvasThemePreference = () => preference;
export const useCanvasTheme = () => useSyncExternalStore(subscribe, getCanvasTheme, getCanvasTheme);
export const useCanvasThemePreference = () => useSyncExternalStore(subscribe, getCanvasThemePreference, getCanvasThemePreference);

function applyTheme() {
  if (typeof document !== 'undefined') {
    document.documentElement.dataset.tvTheme = getCanvasTheme();
    document.documentElement.style.colorScheme = getCanvasTheme();
  }
  listeners.forEach(listener => listener());
}

/** Browser embeds can choose their default; saved choices and host themes take precedence. */
export function configureCanvasTheme(nextSurface: Surface, browserDefaultTheme: CanvasTheme = 'light') {
  surface = nextSurface;
  browserDefault = browserDefaultTheme;
  hostTheme = 'dark';
  preference = 'default';
  try {
    const saved = localStorage.getItem(storageKey());
    if (saved === 'light' || saved === 'dark') preference = saved;
  } catch { /* A private/blocked storage context still permits a session choice. */ }
  applyTheme();
}

export function setCanvasThemePreference(next: CanvasThemePreference) {
  preference = next;
  try {
    if (next === 'default') localStorage.removeItem(storageKey());
    else localStorage.setItem(storageKey(), next);
  } catch { /* Apply the choice even when persistence is unavailable. */ }
  applyTheme();
}

export function setCanvasHostTheme(next: CanvasTheme) {
  hostTheme = next;
  applyTheme();
}

/** Keep an already-mounted tldraw editor in sync without resetting its camera. */
export function onCanvasThemeChange(listener: (theme: CanvasTheme) => void) {
  return subscribe(() => listener(getCanvasTheme()));
}
