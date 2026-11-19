import React, { useEffect } from 'react';
import { act, fireEvent, render } from '@testing-library/react';
import fs from 'fs';
import path from 'path';
import ts from 'typescript';
import { doubleClickedShapeId } from '../../canvas/doubleClickTarget';
import { isModelShape } from '../../canvas/canvasShapePolicy';
import { closeNodeParameters, openNodeParameters, parameterTarget } from '../../canvas/nodeParameters';

jest.mock('tldraw', () => ({
  atom: (_name: string, initial: unknown) => {
    let value = initial;
    return { get: () => value, set: (next: unknown) => { value = next; } };
  },
  getHitShapeOnCanvasPointerDown: (editor: any) => editor.getShapeAtPoint(),
}));

// Execute the shipped watcher with real React effects, without importing the
// tldraw shape renderer/CSS into CRA Jest. AST extraction needs no test export.
const source = fs.readFileSync(path.join(__dirname, '../../canvas/TldrawCanvas.tsx'), 'utf8');
const parsed = ts.createSourceFile('TldrawCanvas.tsx', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
const declaration = parsed.statements.filter(ts.isVariableStatement)
  .flatMap(statement => Array.from(statement.declarationList.declarations))
  .find(node => node.name.getText(parsed) === 'SelectionWatcher')!;
const watcherSource = ts.transpile(`const SelectionWatcher = ${declaration.initializer!.getText(parsed)};`);

let editor: any;
let selected: string[];
let toolPath: string;
let listeners: Record<string, Set<(info: any) => void>>;
let container: HTMLDivElement;
const selectedEdgeId = { set: jest.fn() };
const hoveredEdge = { set: jest.fn() };
const findEdgeAtPoint = jest.fn();
const findEdgeWithMidpoint = jest.fn();
const setTraceDirection = jest.fn();
const dependencies = {
  track: (component: any) => component, useEditor: () => editor, useEffect,
  selectedEdgeId, hoveredEdgeAtom: hoveredEdge, findEdgeAtPoint, findEdgeWithMidpoint,
  EDGE_HIT_PX: 8, setTraceDirection, closeNodeParameters, openNodeParameters,
  parameterTarget, isModelShape, doubleClickedShapeId, cameraDuration: (duration: number) => duration,
};
const SelectionWatcher = new Function(...Object.keys(dependencies), `${watcherSource}; return SelectionWatcher;`)(...Object.values(dependencies));

function emit(channel: string, info: object) {
  act(() => { listeners[channel]?.forEach(listener => listener(info)); });
}

beforeEach(() => {
  jest.clearAllMocks();
  closeNodeParameters();
  selected = ['shape:module'];
  toolPath = 'select.pointing_shape';
  listeners = {};
  container = document.createElement('div');
  editor = {
    getSelectedShapeIds: () => selected,
    getPath: () => toolPath,
    getShape: (id: string) => ({ id, type: 'module-node' }),
    getShapeAtPoint: () => ({ id: 'shape:module', type: 'module-node' }),
    getContainer: () => container,
    getZoomLevel: () => 2,
    screenToPage: (point: object) => point,
    setCursor: jest.fn(),
    on: (channel: string, listener: (info: any) => void) => { (listeners[channel] ??= new Set()).add(listener); },
    off: (channel: string, listener: (info: any) => void) => listeners[channel]?.delete(listener),
  };
});

test('a normal node click reports the click without opening parameters', () => {
  const onShapeClick = jest.fn();
  render(<SelectionWatcher onShapeClick={onShapeClick} />);
  emit('before-event', { name: 'pointer_up' });
  expect(onShapeClick).toHaveBeenCalledWith('shape:module');
  expect(parameterTarget.get()).toBeNull();
});

test.each(['select.translating', 'select.pointing_canvas'])('%s does not report a node click or open parameters', state => {
  toolPath = state;
  const onShapeClick = jest.fn();
  render(<SelectionWatcher onShapeClick={onShapeClick} />);
  emit('before-event', { name: 'pointer_up' });
  expect(onShapeClick).not.toHaveBeenCalled();
  expect(parameterTarget.get()).toBeNull();
});

test('pointer down and multi-selection never report a node click', () => {
  const onShapeClick = jest.fn();
  render(<SelectionWatcher onShapeClick={onShapeClick} />);
  emit('before-event', { name: 'pointer_down' });
  selected = ['shape:a', 'shape:b'];
  emit('before-event', { name: 'pointer_up' });
  expect(onShapeClick).not.toHaveBeenCalled();
});

test('Enter still activates the selected module, but not while typing', () => {
  const onShapeDoubleClick = jest.fn();
  render(<SelectionWatcher onShapeDoubleClick={onShapeDoubleClick} />);
  fireEvent.keyDown(container, { key: 'Enter' });
  expect(onShapeDoubleClick).toHaveBeenCalledWith('shape:module');
  const input = document.createElement('input');
  container.appendChild(input);
  fireEvent.keyDown(input, { key: 'Enter' });
  expect(onShapeDoubleClick).toHaveBeenCalledTimes(1);
  expect(parameterTarget.get()).toBeNull();
});

test('a finished canvas double-click still activates the hit module without opening parameters', () => {
  const onShapeDoubleClick = jest.fn();
  render(<SelectionWatcher onShapeDoubleClick={onShapeDoubleClick} />);
  emit('event', { type: 'click', name: 'double_click', phase: 'up', target: 'canvas' });
  expect(onShapeDoubleClick).toHaveBeenCalledWith('shape:module');
  expect(parameterTarget.get()).toBeNull();
});

test.each(['Enter', 'double-click'])('%s on a leaf requests inspection rather than source navigation', gesture => {
  const onShapeDoubleClick = jest.fn(), inspect = jest.fn();
  editor.getShape = (id: string) => ({ id, type: 'layer' });
  editor.getShapeAtPoint = () => ({ id: 'shape:module', type: 'layer' });
  container.addEventListener('tensorviz-inspect-selection', inspect);
  render(<SelectionWatcher onShapeDoubleClick={onShapeDoubleClick} />);
  if (gesture === 'Enter') fireEvent.keyDown(container, { key: 'Enter' });
  else emit('event', { type: 'click', name: 'double_click', phase: 'up', target: 'canvas' });
  expect(inspect).toHaveBeenCalledTimes(1);
  expect(onShapeDoubleClick).not.toHaveBeenCalled();
  expect(parameterTarget.get()).toBeNull();
});

test('edge clicks keep their zoom-adjusted hit test and do not become node clicks', () => {
  toolPath = 'select.pointing_canvas';
  findEdgeAtPoint.mockReturnValue('edge:a');
  const onShapeClick = jest.fn();
  render(<SelectionWatcher onShapeClick={onShapeClick} />);
  emit('before-event', { name: 'pointer_up', point: { x: 10, y: 20 } });
  expect(findEdgeAtPoint).toHaveBeenCalledWith(editor, { x: 10, y: 20 }, 4);
  expect(selectedEdgeId.set).toHaveBeenLastCalledWith('edge:a');
  expect(onShapeClick).not.toHaveBeenCalled();
});

test('edge hover keeps its hit test and does not interfere with dragging', () => {
  toolPath = 'select.idle';
  const hit = { edgeId: 'edge:a', point: { x: 10, y: 20 } };
  findEdgeWithMidpoint.mockReturnValue(hit);
  render(<SelectionWatcher />);
  emit('before-event', { name: 'pointer_move', point: { x: 10, y: 20 } });
  expect(findEdgeWithMidpoint).toHaveBeenCalledWith(editor, { x: 10, y: 20 }, 4);
  expect(hoveredEdge.set).toHaveBeenCalledWith(hit);
  expect(editor.setCursor).toHaveBeenCalledWith({ type: 'pointer' });
  toolPath = 'select.translating';
  emit('before-event', { name: 'pointer_move', point: { x: 30, y: 40 } });
  expect(findEdgeWithMidpoint).toHaveBeenCalledTimes(1);
});

test('moving selection still closes an explicitly opened editor', () => {
  openNodeParameters('shape:module');
  const view = render(<SelectionWatcher />);
  selected = ['shape:other'];
  view.rerender(<SelectionWatcher />);
  expect(parameterTarget.get()).toBeNull();
});

test('explicit Open in the focus overlay uses the module activation callback', () => {
  const GraphActivationContext = React.createContext<((id: string) => void) | undefined>(undefined);
  const main = parsed.statements.filter(ts.isVariableStatement)
    .flatMap(statement => Array.from(statement.declarationList.declarations))
    .find(node => node.name.getText(parsed) === 'TldrawCanvas')!;
  const compiled = ts.transpileModule(`const TldrawCanvas = ${main.initializer!.getText(parsed)};`, {
    compilerOptions: { jsx: ts.JsxEmit.React, target: ts.ScriptTarget.ES2020 },
  }).outputText;
  const context = React.createContext<any>(undefined);
  const noop = () => undefined;
  const mainDependencies = {
    React, useMemo: React.useMemo, useRef: React.useRef, useCallback: React.useCallback,
    useEffect: noop, CanvasAssetContext: context, ConnectionGraphContext: context,
    EdgeInsertContext: context, GraphNavigationContext: context, GraphActivationContext, TemplateEditingContext: context,
    cancelConnection: noop, cancelEdgeInsert: noop, closeNodeParameters: noop,
    hiddenComponents: {}, CanvasBackground: noop, ConnectionLayer: noop, CanvasOverlays: noop,
    Tldraw: ({ children }: any) => <>{children}</>, SelectionWatcher: noop, ToolWatcher: noop,
    CanvasMinimap: noop, customShapeUtils: [],
    GraphFocusOverlay: () => {
      const activate = React.useContext(GraphActivationContext);
      return <button onClick={() => activate?.('shape:module')}>Open module</button>;
    },
  };
  const Canvas = new Function(...Object.keys(mainDependencies), `${compiled}; return TldrawCanvas;`)(...Object.values(mainDependencies));
  const onShapeDoubleClick = jest.fn();
  const view = render(<Canvas onShapeDoubleClick={onShapeDoubleClick} />);
  fireEvent.click(view.getByRole('button', { name: 'Open module' }));
  expect(onShapeDoubleClick).toHaveBeenCalledWith('shape:module');
});

test('unmount removes gesture and keyboard listeners', () => {
  const onShapeClick = jest.fn();
  const onShapeDoubleClick = jest.fn();
  const view = render(<SelectionWatcher onShapeClick={onShapeClick} onShapeDoubleClick={onShapeDoubleClick} />);
  view.unmount();
  emit('before-event', { name: 'pointer_up' });
  fireEvent.keyDown(container, { key: 'Enter' });
  expect(onShapeClick).not.toHaveBeenCalled();
  expect(onShapeDoubleClick).not.toHaveBeenCalled();
});
