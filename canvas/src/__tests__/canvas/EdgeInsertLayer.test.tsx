import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { LocalBrowserContext } from '../../ide/localBrowserContext';
import { registerGraphAccess } from '../../canvas/graphAccess';

/**
 * The insert affordance is positioned in the coordinate space of the layer it
 * lives in, and that space is the tldraw CONTAINER, not the browser viewport.
 *
 * `.tl-canvas__in-front` is inset:0 of the container, so a container that starts
 * below a toolbar makes editor.pageToScreen (which adds the container's viewport
 * offset) wrong by exactly that offset: the `+` drew 40px below its own wire, and
 * because moving the pointer down to it left the edge's hit radius, the hover
 * cleared and the button unmounted before the press landed. Unclickable.
 *
 * So the fake editor below deliberately disagrees with itself — pageToScreen
 * shifts, pageToViewport does not — and the assertions read the rendered style.
 * A projection that reaches for screen space puts the `+` at 240px.
 */
const CONTAINER_TOP = 40;
const EDGE_POINT = { x: 300, y: 200 };

let mockEditor: any;

jest.mock('tldraw', () => ({
  // A minimal atom: get/set with no subscription. Every test sets state before
  // rendering, so reactivity is not what is under test here — placement is.
  atom: (_name: string, initial: unknown) => {
    let value = initial;
    return { get: () => value, set: (next: unknown) => { value = next; } };
  },
  // Both call shapes the component uses: useValue(atom) and useValue(name, fn, deps).
  useValue: (...args: any[]) => (typeof args[0] === 'string' ? args[1]() : args[0].get()),
  useEditor: () => mockEditor,
  computed: (_name: string, fn: () => unknown) => ({ get: fn }),
}));

// Imported after the mock so edgeInsert's module-level atoms use the fake.
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { hoveredEdge, insertTarget } = require('../../canvas/edgeInsert');
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { EdgeInsertLayer } = require('../../canvas/EdgeInsertLayer');
const { graphNodes, graphEdges } = require('../../canvas/graphQueryState');

beforeEach(() => {
  mockEditor = {
    getCamera: () => ({ x: 0, y: 0, z: 1 }),
    pageToViewport: (point: { x: number; y: number }) => ({ ...point }),
    pageToScreen: (point: { x: number; y: number }) => ({ x: point.x, y: point.y + CONTAINER_TOP }),
  };
  graphNodes.set([]);
  graphEdges.set([]);
  hoveredEdge.set(null);
  insertTarget.set(null);
});

describe('EdgeInsertLayer placement', () => {
  it.each(['hover', 'picker'])('hides the %s affordance when architecture is read-only', state => {
    registerGraphAccess(mockEditor, { current: true });
    hoveredEdge.set({ edgeId: 'edge-1', point: EDGE_POINT });
    if (state === 'picker') insertTarget.set({ edgeId: 'edge-1', point: EDGE_POINT });
    const { container } = render(<EdgeInsertLayer />);
    expect(container.querySelector('.model-edge-insert-plus')).toBeNull();
    expect(container.querySelector('.model-edge-insert-picker')).toBeNull();
  });

  it('draws the + at the hovered edge point in container space, not viewport space', () => {
    hoveredEdge.set({ edgeId: 'edge-1', point: EDGE_POINT });
    const { container } = render(<EdgeInsertLayer />);

    const plus = container.querySelector('.model-edge-insert-plus') as HTMLElement;
    expect(plus).not.toBeNull();
    expect(plus.style.left).toBe('300px');
    expect(plus.style.top).toBe('200px');
  });

  it('opens the picker at that same point, so it stays anchored to its wire', () => {
    insertTarget.set({ edgeId: 'edge-1', point: EDGE_POINT });
    const { container } = render(<EdgeInsertLayer />);

    const picker = container.querySelector('.model-edge-insert-picker') as HTMLElement;
    expect(picker).not.toBeNull();
    expect(picker.style.left).toBe('300px');
    expect(picker.style.top).toBe('200px');
  });

  it('renders nothing when no edge is hovered', () => {
    const { container } = render(<EdgeInsertLayer />);
    expect(container.querySelector('.model-edge-insert-plus')).toBeNull();
    expect(container.querySelector('.model-edge-insert-picker')).toBeNull();
  });
});

it('local edge insertion opens source review and never uses the builder channel', async () => {
  const propose = jest.fn(() => ({ promise: Promise.resolve({ status: 'refused', detail: 'Cannot rewrite this call safely.' }) }));
  registerGraphAccess(mockEditor, { current: true });
  graphNodes.set([{ id: 'fc', data: { layerType: { id: 'linear' }, parameters: {}, sourceLocation: { file: 'net.py', line: 4 } } }]);
  graphEdges.set([{ id: 'edge-1', source: 'fc', target: 'act' }]);
  hoveredEdge.set({ edgeId: 'edge-1', point: EDGE_POINT });
  const { rerender } = render(<LocalBrowserContext.Provider value={{ sourceEdits: { propose }, page: {} } as any}><EdgeInsertLayer /></LocalBrowserContext.Provider>);
  fireEvent.click(screen.getByLabelText('Insert layer through source review'));
  rerender(<LocalBrowserContext.Provider value={{ sourceEdits: { propose }, page: {} } as any}><EdgeInsertLayer /></LocalBrowserContext.Provider>);
  fireEvent.click(screen.getByText('Review insertion'));
  expect(propose).toHaveBeenCalledWith({ kind: 'add_layer', afterNodeId: 'fc', nodeId: 'new_layer', layerType: 'dropout', params: { p: 0.5 } });
  expect(await screen.findByText('Cannot rewrite this call safely.')).toBeInTheDocument();
});
