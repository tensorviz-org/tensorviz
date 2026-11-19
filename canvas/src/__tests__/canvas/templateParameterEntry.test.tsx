import React from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import fs from 'fs';
import path from 'path';
import ts from 'typescript';
import * as overlay from '../../canvas/GraphFocusOverlay';
import { NodeParametersPopover, TemplateEditingContext } from '../../canvas/NodeParametersPopover';
import { graphNodes, graphSceneNodes } from '../../canvas/graphQueryState';
import { closeNodeParameters, parameterTarget } from '../../canvas/nodeParameters';
import { LocalBrowserContext } from '../../ide/localBrowserContext';
import { nodeIdToShapeId } from '../../canvas/canvasShapeId';
import { getLayerByType } from '../../data/layerTypes';
import type { ModelNode } from '../../types';

jest.mock('tldraw', () => ({
  atom: (_name: string, initial: unknown) => {
    let value = initial;
    const listeners = new Set<() => void>();
    return { get: () => value, set: (next: unknown) => { value = next; listeners.forEach(listener => listener()); },
      subscribe: (listener: () => void) => { listeners.add(listener); return () => listeners.delete(listener); } };
  },
  computed: (_name: string, read: () => unknown) => ({ get: read }),
  track: (component: unknown) => component,
  useEditor: () => mockEditor,
  useValue: (...args: any[]) => {
    // Only the target atom needs reactive subscription in this editor seam.
    const React = require('react');
    const atom = typeof args[0] === 'string' ? null : args[0];
    const value = React.useSyncExternalStore(atom?.subscribe ?? (() => () => {}), atom?.get ?? (() => null));
    return atom ? value : args[1]();
  },
}));
let mockNode: ModelNode;
const mockEditor = {
  getEfficientZoomLevel: () => 1, markEventAsHandled: jest.fn(),
  getContainer: () => document.body, selectNone: jest.fn(),
  getCamera: () => ({ x: 0, y: 0, z: 1 }),
  getShape: (id: string) => id === nodeIdToShapeId(mockNode.id) ? {
    id, type: mockNode.type === 'inputNode' ? 'input-node' : mockNode.type === 'outputNode' ? 'output-node' : 'layer',
    props: { nodeData: mockNode.data },
  } : undefined,
  pageToViewport: (point: object) => point,
  getViewportScreenBounds: () => ({ w: 900, h: 700 }),
  getShapePageBounds: () => ({ x: 100, y: 100, w: 200, h: 100, maxX: 300 }),
};
jest.mock('../../canvas/graphFocus', () => ({ graphFocus: () => ({
  focus: { get: () => ({ active: true, selected: [mockNode.id], edgeIds: new Set(), direction: 'local' }) },
  index: { get: () => ({ edges: new Map() }) },
}), setTraceDirection: jest.fn() }));
jest.mock('../../canvas/BlockSummary', () => ({ FamilyLegend: () => null, BlockSummary: () => null }));
jest.mock('../../canvas/NodeUnderstanding', () => ({ NodeUnderstanding: () => null }));

// Execute the actual canvas component and its capability providers. Only the
// tldraw renderer/shape utilities are replaced; actions and the popover are real.
const source = fs.readFileSync(path.join(__dirname, '../../canvas/TldrawCanvas.tsx'), 'utf8');
const parsed = ts.createSourceFile('TldrawCanvas.tsx', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
const main = parsed.statements.filter(ts.isVariableStatement)
  .flatMap(statement => Array.from(statement.declarationList.declarations))
  .find(node => node.name.getText(parsed) === 'TldrawCanvas')!;
const compiled = ts.transpileModule(`const TldrawCanvas = ${main.initializer!.getText(parsed)};`, {
  compilerOptions: { jsx: ts.JsxEmit.React, target: ts.ScriptTarget.ES2020 },
}).outputText;
const context = React.createContext<any>(undefined);
const noop = () => undefined;
const dependencies = {
  React, useMemo: React.useMemo, useRef: React.useRef, useCallback: React.useCallback,
  useEffect: noop, CanvasAssetContext: context, ConnectionGraphContext: context,
  EdgeInsertContext: context, TemplateEditingContext, ...overlay,
  cancelConnection: noop, cancelEdgeInsert: noop, closeNodeParameters: noop,
  hiddenComponents: {}, CanvasBackground: noop, ConnectionLayer: noop,
  CanvasOverlays: NodeParametersPopover,
  Tldraw: ({ children, components }: any) => <>{children}<components.InFrontOfTheCanvas /></>,
  SelectionWatcher: noop, ToolWatcher: noop, CanvasMinimap: noop, customShapeUtils: [],
};
const Canvas = new Function(...Object.keys(dependencies), `${compiled}; return TldrawCanvas;`)(...Object.values(dependencies));

beforeEach(() => {
  jest.useFakeTimers();
  closeNodeParameters();
  mockNode = { id: 'fc', type: 'layerNode', position: { x: 0, y: 0 }, data: {
    layerType: getLayerByType('linear')!, captureLabel: 'projection', parameters: { in_features: 4, out_features: 8, bias: true },
  } };
  graphNodes.set([mockNode]); graphSceneNodes.set([]);
});
afterEach(() => { jest.runOnlyPendingTimers(); jest.useRealTimers(); });

test('an editable template opens real parameter fields only through More node actions → Edit parameters', () => {
  const command = jest.fn();
  window.addEventListener('builder-command', command);
  render(<Canvas readOnly={false} />);
  expect(parameterTarget.get()).toBeNull();
  expect(screen.queryByTestId('node-parameters')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'More node actions' }));
  expect(screen.queryByRole('button', { name: 'Edit source' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Edit parameters' }));
  expect(parameterTarget.get()).toBe(nodeIdToShapeId('fc'));
  expect(screen.getByTestId('node-parameters')).toBeVisible();
  expect(screen.queryByRole('complementary', { name: 'Node inspector' })).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText('out_features'), { target: { value: '16' } });
  act(() => jest.advanceTimersByTime(449));
  expect(command).not.toHaveBeenCalled();
  act(() => jest.advanceTimersByTime(1));
  expect(command).toHaveBeenCalledTimes(1);
  expect(command.mock.calls[0][0].detail).toEqual({ command: 'set-parameters', payload: { nodeId: 'fc',
    parameters: { in_features: 4, out_features: 16, bias: true } } });
  fireEvent.click(screen.getByRole('button', { name: 'Close parameters' }));
  expect(screen.queryByTestId('node-parameters')).not.toBeInTheDocument();
  window.removeEventListener('builder-command', command);
});

test.each(['readOnly', 'local', 'focusBoundary', 'inputNode', 'outputNode'])('%s does not expose template editing', mode => {
  if (mode === 'focusBoundary') mockNode.data.focusBoundary = { direction: 'in', externalNodeId: 'outside' };
  if (mode === 'inputNode' || mode === 'outputNode') mockNode.type = mode;
  const content = <Canvas readOnly={mode === 'readOnly'} />;
  render(mode === 'local' ? <LocalBrowserContext.Provider value={{ page: {}, sourceEdits: { propose: jest.fn() } } as any}>
    {content}</LocalBrowserContext.Provider> : content);
  fireEvent.click(screen.getByRole('button', { name: 'More node actions' }));
  expect(screen.queryByRole('button', { name: 'Edit parameters' })).not.toBeInTheDocument();
  expect(parameterTarget.get()).toBeNull();
  expect(screen.queryByTestId('node-parameters')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Inspect' }));
  expect(screen.queryByRole('button', { name: 'Edit parameters' })).not.toBeInTheDocument();
  expect(screen.queryByTestId('node-parameters')).not.toBeInTheDocument();
});

test('changing an editable template canvas to readOnly hides its open parameter editor', () => {
  const view = render(<Canvas readOnly={false} />);
  fireEvent.click(screen.getByRole('button', { name: 'More node actions' }));
  fireEvent.click(screen.getByRole('button', { name: 'Edit parameters' }));
  expect(screen.getByTestId('node-parameters')).toBeVisible();
  view.rerender(<Canvas readOnly />);
  expect(screen.queryByTestId('node-parameters')).not.toBeInTheDocument();
});
