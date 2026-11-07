import React, { useEffect } from 'react';
import { act, renderHook } from '@testing-library/react';
import { useModuleFocus } from '../../canvas/useModuleFocus';
import { nodeIdToShapeId } from '../../canvas/canvasShapeId';
import { focusLayoutKey, projectModuleFocus, placeModuleFocus } from '../../services/moduleFocus';
import type { FocusLayout, FocusOrientation } from '../../services/moduleFocus';
import type { ModelEdge, ModelGraph, ModelNode } from '../../types';

const mockLayouts: Array<{ key: string; width: number; height: number; orientation: FocusOrientation; passes: number }> = [];
let mockDagrePasses = 0;
jest.mock('dagre', () => {
  const actual = jest.requireActual('dagre');
  return { ...actual, layout: (...args: any[]) => { mockDagrePasses += 1; return actual.layout(...args); } };
});
jest.mock('../../services/moduleFocus', () => {
  const actual = jest.requireActual('../../services/moduleFocus');
  return { ...actual, layoutModuleFocus: (graph: ModelGraph, viewport: { width: number; height: number }, orientation: FocusOrientation) => {
    const before = mockDagrePasses;
    const result = actual.layoutModuleFocus(graph, viewport, orientation);
    mockLayouts.push({ key: actual.focusLayoutKey(graph), ...viewport, orientation, passes: mockDagrePasses - before });
    return result;
  } };
});
jest.mock('tldraw', () => ({ atom: (_name: string, value: any) => ({ get: () => value, set: (next: any) => { value = next; } }) }));
jest.mock('../../canvas/moduleFocusMotion', () => ({ captureFocusFrame: () => undefined }));

const node = (id: string, parentId?: string, module = false): ModelNode => ({
  id, type: module ? 'moduleNode' : 'layerNode', position: { x: 17, y: 23 },
  data: { parentId, layerType: { id: module ? 'module' : 'linear', name: module ? 'Module' : 'Linear' } as any,
    parameters: { name: id }, ...(module ? { expanded: false, moduleInstanceId: id } : {}) },
});
const fixture = () => ({
  nodes: [node('root', undefined, true), node('a', 'root'), node('b', 'root'), node('c', 'root')],
  edges: [{ id: 'ab', source: 'a', target: 'b' }, { id: 'bc', source: 'b', target: 'c' }] as ModelEdge[],
});
const settle = () => act(() => { jest.advanceTimersByTime(64); });

function setup({ attached = true, width = 1100, height = 650, strict = false } = {}) {
  let viewport = { width, height };
  let nextViewport: { width: number; height: number } | undefined;
  let camera = { x: 22, y: 33, z: .6 };
  let selection: string[] = [];
  let zoomSteps = [.05, .1, .25, .5, 1, 2, 4];
  const container = { getBoundingClientRect: () => {
    const result = { ...viewport };
    if (nextViewport) { viewport = nextViewport; nextViewport = undefined; }
    return result;
  } };
  const editor: any = {
    getContainer: () => container, getCamera: () => camera,
    getCameraOptions: () => ({ zoomSteps }), setCameraOptions: (value: { zoomSteps: number[] }) => { zoomSteps = value.zoomSteps; },
    on: jest.fn(), off: jest.fn(), stopCameraAnimation: jest.fn(),
    setCamera: jest.fn((next: typeof camera) => { camera = next; }), zoomToFit: jest.fn(), zoomToBounds: jest.fn(),
    getSelectedShapeIds: () => selection, getShape: () => ({}), select: (...ids: string[]) => { selection = ids; },
    updateViewportScreenBounds: jest.fn(),
  };
  const suspended = new Promise<void>(() => {});
  const props = { ...fixture(), setNodes: jest.fn(), graphId: 'one', sourceKey: 'same-source', enabled: true, sync: true, suspend: false };
  const rendered = renderHook((value: typeof props) => {
    const focus = useModuleFocus(value);
    useEffect(() => { if (value.sync) focus.markSceneCommitted(focus.graph.nodes); }, [value.sync, focus.graph.nodes, focus.markSceneCommitted]);
    if (value.suspend) throw suspended;
    return focus;
  }, { initialProps: props, wrapper: ({ children }: { children: React.ReactNode }) => {
    const content = <React.Suspense fallback={null}>{children}</React.Suspense>;
    return strict ? <React.StrictMode>{content}</React.StrictMode> : content;
  } });
  if (attached) act(() => rendered.result.current.attachEditor(editor));
  return { ...rendered, props, editor, resize: (value: typeof viewport) => { viewport = value; },
    changeAfterNextRead: (value: typeof viewport) => { nextViewport = value; } };
}

function expectedGraph(graph: ModelGraph, viewport: { width: number; height: number }, orientation: FocusOrientation = 'auto') {
  const neutral = graph.nodes.map(node => ({ ...node, selected: false, data: { ...node.data, isSelected: false } }));
  const projected = projectModuleFocus(neutral, graph.edges, 'root')!;
  const layout: FocusLayout = jest.requireActual('../../services/moduleFocus').layoutModuleFocus(projected, viewport, orientation);
  return placeModuleFocus(projected, layout);
}

beforeEach(() => { jest.useFakeTimers(); mockLayouts.length = 0; mockDagrePasses = 0; });
afterEach(() => jest.useRealTimers());

test('first module entry reuses matching measured inputs, retaining exact scene and readiness', () => {
  const { result, props, editor } = setup();
  const before = JSON.stringify(props);
  act(() => { expect(result.current.navigate('root')).toBe(true); }); settle();
  expect(mockLayouts[0]).toMatchObject({ width: 1100, height: 650, orientation: 'auto', passes: 2 });
  expect(result.current.graph).toEqual(expectedGraph(props, { width: 1100, height: 650 }));
  expect(result.current.isReady()).toBe(true);
  expect(editor.zoomToFit).toHaveBeenCalledTimes(1);
  expect(JSON.stringify(props)).toBe(before);
  expect(props.setNodes).not.toHaveBeenCalled();
  expect(mockLayouts).toHaveLength(1);
});

test.each(['LR', 'TB'] as const)('explicit %s flow lays out once and keeps selection restoration', orientation => {
  const { result, props, editor } = setup();
  act(() => result.current.navigate('root')); settle();
  editor.select(nodeIdToShapeId('a')); mockLayouts.length = 0;
  act(() => result.current.setOrientation(orientation)); settle();
  expect(mockLayouts[0]).toMatchObject({ orientation, passes: 1 });
  expect(result.current.graph).toEqual(expectedGraph(props, { width: 1100, height: 650 }, orientation));
  expect(editor.getSelectedShapeIds()).toEqual([nodeIdToShapeId('a')]);
  expect(editor.zoomToFit).toHaveBeenCalledTimes(2);
  expect(mockLayouts).toHaveLength(1);
});

test('a viewport change between render and commit retains the second measured layout', () => {
  const { result, props, changeAfterNextRead } = setup();
  changeAfterNextRead({ width: 350, height: 900 });
  act(() => result.current.navigate('root')); settle();
  expect(mockLayouts.map(({ width, height, orientation }) => ({ width, height, orientation }))).toEqual([
    { width: 1100, height: 650, orientation: 'auto' }, { width: 350, height: 900, orientation: 'auto' },
  ]);
  expect(result.current.graph).toEqual(expectedGraph(props, { width: 350, height: 900 }));
});

test.each([[1000, 700, 1], [800, 500, 2]])('late editor at %i×%i keeps the right layout pass count', (width, height, count) => {
  const { result, props, editor } = setup({ attached: false, width, height });
  act(() => result.current.navigate('root'));
  expect(mockLayouts).toHaveLength(1);
  act(() => result.current.attachEditor(editor)); settle();
  expect(result.current.graph).toEqual(expectedGraph(props, { width, height }));
  expect(result.current.isReady()).toBe(true);
  expect(mockLayouts).toHaveLength(count);
});

test('zero-sized measurements retain fallback recomputation and wait to fit', () => {
  const { result, props, editor, resize } = setup({ width: 0, height: 0 });
  act(() => result.current.navigate('root')); settle();
  expect(mockLayouts.map(({ width, height }) => ({ width, height }))).toEqual([
    { width: 1000, height: 700 }, { width: 0, height: 0 },
  ]);
  expect(result.current.graph).toEqual(expectedGraph(props, { width: 0, height: 0 }));
  expect(result.current.isReady()).toBe(false);
  expect(editor.zoomToFit).not.toHaveBeenCalled();
  resize({ width: 1100, height: 650 }); settle();
  expect(result.current.isReady()).toBe(true);
  expect(editor.zoomToFit).toHaveBeenCalledTimes(1);
});

test.each([[800, 500], [1000, 700]])('new keys and orientation requests stay current before attachment at %i×%i', (width, height) => {
  const { result, rerender, props, editor } = setup({ attached: false, width, height });
  act(() => result.current.navigate('root'));
  const priorDirection = result.current.direction;
  const next = { ...props, nodes: [...props.nodes, node('d', 'root')], edges: [...props.edges, { id: 'cd', source: 'c', target: 'd' }] };
  rerender(next);
  expect(mockLayouts).toHaveLength(2);
  expect(mockLayouts[0].key).not.toEqual(mockLayouts[1].key);
  expect(mockLayouts[1].orientation).toBe(priorDirection);
  act(() => result.current.attachEditor(editor)); settle();
  expect(mockLayouts).toHaveLength(3);
  expect(mockLayouts[2].key).toBe(mockLayouts[1].key);
  expect(mockLayouts[2].orientation).toBe('auto');
  expect(result.current.graph).toEqual(expectedGraph(next, { width, height }));
});

test('saved orientation is measured once and is not serialized with the reuse certificate', () => {
  const { result, props } = setup();
  act(() => result.current.restorePresentation({ scopeId: 'root', views: [{ id: 'root', selected: [],
    orientation: 'auto', direction: 'TB', positions: [] }] })); settle();
  expect(mockLayouts[0]).toMatchObject({ orientation: 'TB', passes: 1 });
  expect(result.current.graph).toEqual(expectedGraph(props, { width: 1100, height: 650 }, 'TB'));
  expect(result.current.capturePresentation().views.some(view => 'layoutRequest' in view)).toBe(false);
  expect(mockLayouts).toHaveLength(1);
});

test('restored unknown fields cannot supply authority to skip measured layout repair', () => {
  const { result, props } = setup();
  const neutral = props.nodes.map(node => ({ ...node, selected: false, data: { ...node.data, isSelected: false } }));
  const projection = projectModuleFocus(neutral, props.edges, 'root')!;
  const key = focusLayoutKey(projection);
  act(() => result.current.restorePresentation({ scopeId: 'root', views: [{ id: 'root', selected: [],
    orientation: 'auto', positions: [], layoutKey: key,
    layout: { direction: 'TB', positions: new Map(projection.nodes.map(node => [node.id, { x: -999, y: -999 }])),
      bounds: { x: 0, y: 0, w: 1, h: 1 } },
    layoutRequest: { key, width: 1100, height: 650, orientation: 'auto' },
  }] })); settle();
  expect(result.current.graph).toEqual(expectedGraph(props, { width: 1100, height: 650 }));
  expect(result.current.isReady()).toBe(true);
  expect(mockLayouts).toHaveLength(1);
  expect(mockLayouts[0]).toMatchObject({ width: 1100, height: 650, orientation: 'auto', passes: 2 });
});

test('StrictMode and ordinary rerenders preserve scene, chosen flow and the fitting contract', () => {
  const { result, rerender, props, editor } = setup({ strict: true });
  act(() => result.current.navigate('root')); settle();
  expect(mockLayouts).toHaveLength(1);
  expect(editor.zoomToFit).toHaveBeenCalledTimes(1);
  expect(result.current.graph).toEqual(expectedGraph(props, { width: 1100, height: 650 }));
  const graph = result.current.graph;
  const passes = mockLayouts.length;
  const fits = editor.zoomToFit.mock.calls.length;
  rerender({ ...props, nodes: props.nodes.map(n => ({ ...n, selected: true, data: { ...n.data, isSelected: true } })) }); settle();
  expect(result.current.graph).toBe(graph);
  expect(focusLayoutKey(result.current.graph)).toBe(focusLayoutKey(graph));
  expect(mockLayouts).toHaveLength(passes);
  expect(editor.zoomToFit).toHaveBeenCalledTimes(fits);
  expect(result.current.isReady()).toBe(true);
});

test('an abandoned suspended render cannot substitute its scene for a later committed graph', () => {
  const { result, rerender, props, editor } = setup({ attached: false, width: 1000, height: 700 });
  act(() => result.current.navigate('root'));
  const before = result.current.graph;
  const abandoned = { ...props, nodes: [...props.nodes, node('abandoned', 'root')], suspend: true };
  act(() => React.startTransition(() => rerender(abandoned)));
  expect(result.current.graph).toBe(before);
  expect(mockLayouts.some(({ key }) => key.includes('abandoned'))).toBe(true);
  act(() => result.current.attachEditor(editor));
  const final = { ...props, nodes: [...props.nodes, node('final', 'root')],
    edges: [...props.edges, { id: 'c-final', source: 'c', target: 'final' }] };
  rerender(final); settle();
  expect(result.current.graph).toEqual(expectedGraph(final, { width: 1000, height: 700 }));
  expect(result.current.graph.nodes.some(node => node.id === 'abandoned')).toBe(false);
  expect(result.current.isReady()).toBe(true);
  expect(final.setNodes).not.toHaveBeenCalled();
});
