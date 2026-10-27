import React, { useState, useEffect } from 'react';

type Theme = 'light' | 'dark';

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme] = useState<Theme>(() => {
    // Storage key is versioned so we ignore values written by the old
    // (light-first) theme system — those would otherwise keep landing
    // returning users in light mode despite the new dark default.
    const saved = localStorage.getItem('canvas-theme.v2');
    return (saved === 'light' ? 'light' : 'dark') as Theme;
  });

  useEffect(() => {
    const root = document.documentElement;
    // Dark is the default surface set on :root, so we toggle a `.light` class.
    // We also keep `.dark` for components that branched on it historically.
    if (theme === 'light') {
      root.classList.add('light');
      root.classList.remove('dark');
    } else {
      root.classList.remove('light');
      root.classList.add('dark');
    }
    localStorage.setItem('canvas-theme.v2', theme);
  }, [theme]);

  return <>{children}</>;
}
