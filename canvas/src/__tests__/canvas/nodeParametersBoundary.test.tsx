import React from 'react';
import { render, screen } from '@testing-library/react';
import { LocalBrowserContext } from '../../ide/localBrowserContext';

/**
 * A module-focus boundary card ("From …" / "To …") is presentation-only: it stands
 * for a value outside the focused module, not a layer with source. Clicking it must
 * not open the parameter or source-edit popover, which has nothing it could change.
 */
let mockEditor: any;

jest.mock('tldraw', () => ({
  atom: (_name: string, initial: unknown) => {
    let value = initial;
    return { get: () => value, set: (next: unknown) => { value = next; } };
  },
  useValue: (...args: any[]) => (typeof args[0] === 'string' ? args[1]() : args[0].get()),
  useEditor: () => mockEditor,
  computed: (_name: string, fn: () => unknown) => ({ get: fn }),
}));

// Imported after the mock so the module-level atom uses the fake.
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { parameterTarget } = require('../../canvas/nodeParameters');
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { NodeParametersPopover, TemplateEditingContext } = require('../../canvas/NodeParametersPopover');

const layer = { type: 'layer', nodeData: { layerType: { id: 'conv2d', name: 'Conv2d', displayName: 'Conv2d' },
  parameters: { out_channels: 8 } } };
const boundary = { type: 'input-node', nodeData: { layerType: { id: 'input', name: 'input', displayName: 'input' },
  parameters: {}, captureLabel: 'From activation', focusBoundary: { direction: 'in', externalNodeId: 'activation' } } };

function open(card: { type: string; nodeData: unknown }, local: boolean) {
  const id = 'shape:card';
  mockEditor = {
    getCamera: () => ({ x: 0, y: 0, z: 1 }),
    getShape: (shapeId: string) => (shapeId === id ? { id, type: card.type, props: { nodeData: card.nodeData } } : undefined),
    getShapePageBounds: () => ({ x: 0, y: 0, w: 120, h: 60, maxX: 120 }),
    pageToViewport: (point: { x: number; y: number }) => ({ ...point }),
    getViewportScreenBounds: () => ({ w: 1200, h: 800 }),
    screenToPage: (point: { x: number; y: number }) => ({ ...point }),
    getShapeAtPoint: () => undefined,
    select: jest.fn(),
  };
  parameterTarget.set(id);
  const context = { page: { greyed: false, syncState: 'ready' } } as any;
  return render(<TemplateEditingContext.Provider value={true}>{local
    ? <LocalBrowserContext.Provider value={context}><NodeParametersPopover /></LocalBrowserContext.Provider>
    : <NodeParametersPopover />}</TemplateEditingContext.Provider>);
}

afterEach(() => parameterTarget.set(null));

it('opens the explicitly targeted template editor for a real layer outside the local Canvas', () => {
  open(layer, false);
  expect(screen.getByTestId('node-parameters')).toBeInTheDocument();
});

it.each([true, false])('opens nothing for a module-focus boundary card (local: %s)', local => {
  const { container } = open(boundary, local);
  expect(container).toBeEmptyDOMElement();
});
