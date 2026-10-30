import React from 'react';
import { render, screen, within } from '@testing-library/react';
import { GraphQueryPanel } from '../../canvas/GraphQueryPanel';

jest.mock('tldraw', () => ({
  atom: (_name: string, initial: unknown) => {
    let value = initial;
    return { get: () => value, set: (next: unknown) => { value = next; } };
  },
  computed: (_name: string, derive: () => unknown) => ({ get: derive }),
  track: (component: unknown) => component,
  useEditor: () => ({}),
}));

/** The graph file viewer's panel lost its overlay controls in 0.1.28. A screen
 *  reader must announce what the panel holds now, not controls that are gone. */
test('announces the viewer panel as the graph search it holds', () => {
  render(<GraphQueryPanel />);
  const panel = screen.getByRole('region', { name: 'Graph search' });
  expect(within(panel).getByRole('searchbox', { name: 'Search the graph' })).toBeVisible();
  expect(screen.queryByRole('region', { name: /overlay/i })).toBeNull();
});
