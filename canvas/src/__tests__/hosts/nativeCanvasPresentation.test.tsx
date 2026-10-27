import React from 'react';
import { act, render } from '@testing-library/react';
import { captureNativePresentation, restoreNativePresentation, retainedLayoutChoice, useNativeCanvasPresentation } from '../../ide/nativeCanvasPresentation';
import { postToHost } from '../../ide/bridge';
import { edgeIdToShapeId } from '../../canvas/canvasShapeId';
jest.mock('../../ide/bridge', () => ({ postToHost: jest.fn() }));

test('recreation restores layout, camera, selection and expansion without replacing semantic data or fitting again', () => {
  window.history.replaceState({}, '', '/?view=canvas');
  jest.useFakeTimers();
  const fit = { current: null as string | null };
  const editor = { setCamera: jest.fn(), select: jest.fn(), getShape: () => ({}),
    getCamera: () => ({ x: 20, y: 30, z: 0.7 }), getSelectedShapeIds: () => ['shape:node'],
    store: { listen: () => () => {} } };
  restoreNativePresentation({ graphId: 'Net', camera: { x: 20, y: 30, z: 0.7 }, selected: ['shape:node'],
    nodes: [{ id: 'node', x: 100, y: 200, expanded: true, selected: true, hidden: false, captureDetailLevel: 'operations' }] });
  let latest: any[] = [];
  function Canvas() {
    const [nodes, setNodes] = React.useState<any[]>([{ id: 'node', position: { x: 0, y: 0 }, data: { params: { measured: 42 }, expanded: false } }]);
    latest = nodes;
    useNativeCanvasPresentation('Net', nodes, setNodes, editor as never, fit);
    return null;
  }
  const canvas = render(<Canvas />);
  expect(latest[0].position).toEqual({ x: 100, y: 200 });
  // Reconciliation adds presentation defaults; the host's semantic data stays intact.
  expect(latest[0].data.params).toEqual({ measured: 42 });
  expect(latest[0].data).toMatchObject({ expanded: true, passThroughShown: false, captureDetailLevel: 'operations' });
  expect(editor.setCamera).toHaveBeenCalledWith({ x: 20, y: 30, z: 0.7 });
  expect(editor.select).toHaveBeenCalledWith('shape:node');
  expect(fit.current).toBe('Net');
  act(() => { jest.advanceTimersByTime(100); });
  expect(postToHost).toHaveBeenCalledWith(expect.objectContaining({ type: 'canvasPresentation', state: expect.objectContaining({ graphId: 'Net' }) }));
  canvas.unmount(); jest.useRealTimers(); window.history.replaceState({}, '', '/');
});

test('hide captures the final camera before the 100 ms timer or store listener runs', () => {
  window.history.replaceState({}, '', '/?view=canvas');
  jest.useFakeTimers();
  let camera = { x: 0, y: 0, z: 1 };
  let changed = () => {};
  const editor = { getCamera: () => camera, getSelectedShapeIds: () => ['shape:final'],
    store: { listen: (listener: () => void) => { changed = listener; return () => {}; } } };
  const nodes: any[] = [{ id: 'final', data: { expanded: true } }];
  const fit = { current: null };
  const setNodes = jest.fn();
  function Canvas() { useNativeCanvasPresentation('final-gesture', nodes, setNodes, editor as never, fit); return null; }
  const canvas = render(<Canvas />);
  camera = { x: 33, y: 72, z: 0.63 };
  changed();
  expect(captureNativePresentation()).toMatchObject({ camera, selected: ['shape:final'], nodes: [{ id: 'final', expanded: true }] });
  camera = { x: 44, y: 82, z: 0.83 };
  window.dispatchEvent(new Event('blur'));
  expect(postToHost).toHaveBeenLastCalledWith(expect.objectContaining({ state: expect.objectContaining({ camera }) }));
  canvas.unmount(); jest.useRealTimers(); window.history.replaceState({}, '', '/');
});

test('restored child and edge selection waits for expansion to create their shapes', () => {
  window.history.replaceState({}, '', '/?view=canvas');
  jest.useFakeTimers();
  const shapes = new Set<string>();
  let selected: string[] = [];
  let changed = () => {};
  const editor = { setCamera: jest.fn(), select: jest.fn((...ids: string[]) => { selected = ids; }),
    getShape: (id: string) => shapes.has(id) ? {} : undefined,
    getCamera: () => ({ x: 1, y: 2, z: 0.8 }), getSelectedShapeIds: () => selected,
    store: { listen: (listener: () => void) => { changed = listener; return () => {}; } } };
  const edges: any[] = [{ id: 'inner', source: 'block.child', target: 'block' }];
  const edgeId = edgeIdToShapeId(edges[0]);
  restoreNativePresentation({ graphId: 'expanded-child', camera: { x: 1, y: 2, z: 0.8 },
    selected: ['shape:node:block.child', edgeId], nodes: [{ id: 'block', expanded: true }, { id: 'block.child', hidden: false, selected: true }] });
  const fit = { current: null };
  function Canvas() {
    const [nodes, setNodes] = React.useState<any[]>([{ id: 'block', data: { expanded: false } }, { id: 'block.child', hidden: true, data: {} }]);
    useNativeCanvasPresentation('expanded-child', nodes, setNodes, editor as never, fit, edges);
    return null;
  }
  const canvas = render(<Canvas />);
  expect(editor.select).not.toHaveBeenCalled();
  shapes.add('shape:node:block.child');
  act(() => { changed(); });
  expect(editor.select).not.toHaveBeenCalled();
  shapes.add(edgeId);
  act(() => { changed(); });
  expect(editor.select).toHaveBeenCalledWith('shape:node:block.child', edgeId);
  expect(captureNativePresentation()?.selected).toEqual(['shape:node:block.child', edgeId]);
  canvas.unmount(); jest.useRealTimers(); window.history.replaceState({}, '', '/');
});

test('recreation keeps the Layout choice and tells the canvas which direction the saved positions use', () => {
  window.history.replaceState({}, '', '/?view=canvas');
  jest.useFakeTimers();
  const editor = { setCamera: jest.fn(), select: jest.fn(), getShape: () => ({}),
    getCamera: () => ({ x: 0, y: 0, z: 1 }), getSelectedShapeIds: () => [], store: { listen: () => () => {} } };
  const card = (id: string, parentId?: string) => ({ id, type: id === 'root' ? 'moduleNode' : 'layerNode', position: { x: 0, y: 0 },
    data: { layerType: { id: 'linear', name: 'Linear' }, parameters: {}, ...(parentId ? { parentId } : { expanded: false }) } });
  const state = { graphId: 'RowNet', camera: { x: 0, y: 0, z: 1 }, selected: [],
    nodes: [{ id: 'root', x: 10, y: 20, expanded: true }, { id: 'a', x: 20, y: 52 }, { id: 'b', x: 400, y: 52 }],
    layout: { choice: 'vertical', direction: 'horizontal' } };
  restoreNativePresentation(state);
  expect(retainedLayoutChoice()).toBe('vertical');
  // A malformed layout cannot replace the last valid checkpoint.
  restoreNativePresentation({ ...state, layout: { choice: 'diagonal' } });
  restoreNativePresentation({ ...state, layout: { choice: 'auto', direction: 'LR' } });
  expect(retainedLayoutChoice()).toBe('vertical');
  const onRestoreLayout = jest.fn();
  let latest: any[] = [];
  function Canvas() {
    const [nodes, setNodes] = React.useState<any[]>([card('root'), card('a', 'root'), card('b', 'root')]);
    latest = nodes;
    useNativeCanvasPresentation('RowNet', nodes, setNodes, editor as never, { current: null },
      [{ id: 'ab', source: 'a', target: 'b' }] as never, undefined, undefined, { choice: 'vertical', direction: 'vertical' }, onRestoreLayout);
    return null;
  }
  const canvas = render(<Canvas />);
  expect(onRestoreLayout).toHaveBeenCalledTimes(1);
  expect(onRestoreLayout).toHaveBeenCalledWith({ choice: 'vertical', direction: 'horizontal' });
  // Saved row positions are restored as they were, not re-stacked.
  expect(latest.map(node => [node.id, node.position])).toEqual([['root', { x: 10, y: 20 }], ['a', { x: 20, y: 52 }], ['b', { x: 400, y: 52 }]]);
  act(() => { jest.advanceTimersByTime(100); });
  expect(postToHost).toHaveBeenLastCalledWith(expect.objectContaining({ state: expect.objectContaining({
    graphId: 'RowNet', layout: { choice: 'vertical', direction: 'vertical' } }) }));
  canvas.unmount(); jest.useRealTimers(); window.history.replaceState({}, '', '/');
});
