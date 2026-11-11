import { act, renderHook } from '@testing-library/react';
import { useEffect } from 'react';
import { useModuleFocus } from '../../canvas/useModuleFocus';
import { validModuleFocusPresentation } from '../../canvas/moduleFocusPresentation';
import { nodeIdToShapeId } from '../../canvas/canvasShapeId';
import { graphSvgBounds } from '../../services/graphSvg';
import type { ModelEdge, ModelNode } from '../../types';

jest.mock('tldraw', () => ({ atom: (_name: string, value: any) => ({ get: () => value, set: (next: any) => { value = next; } }) }));
jest.mock('../../canvas/moduleFocusMotion', () => ({ captureFocusFrame: () => undefined }));
const node = (id: string, parentId?: string, module = false): ModelNode => ({ id,
  type: module ? 'moduleNode' : 'layerNode', position: { x: 17, y: 23 },
  data: { parentId, layerType: { id: module ? 'module' : 'linear', name: module ? 'Module' : 'Linear' } as any,
    parameters: { name: id, _module_path: id }, ...(module ? { expanded: false, moduleInstanceId: id } : {}) } });
const fixture = () => [node('root', undefined, true), node('root.a', 'root', true),
  node('root.a.x', 'root.a'), node('root.b', 'root', true), node('root.b.x', 'root.b')];
const settle = () => act(() => { jest.advanceTimersByTime(40); });
function setup() {
  let camera = { x: 22, y: 33, z: .6 }, selection: string[] = [], width = 1100;
  let displayed: ModelNode[] = [];
  let zoomSteps = [.05, .1, .25, .5, 1, 2, 4];
  const container = { getBoundingClientRect: () => ({ width, height: 650 }) };
  const editor: any = { getContainer: () => container, getCamera: () => camera,
    getCameraOptions: () => ({ zoomSteps }), setCameraOptions: ({ zoomSteps: next }: { zoomSteps: number[] }) => {
      zoomSteps = next; camera = { ...camera, z: Math.max(next[0], Math.min(next[next.length - 1], camera.z)) };
    },
    on: jest.fn(), off: jest.fn(),
    setCamera: jest.fn((value: typeof camera) => { camera = value; }), stopCameraAnimation: jest.fn(),
    zoomToFit: jest.fn(() => {
      const boxes = graphSvgBounds(displayed);
      const x = Math.min(...boxes.map(box => box.x)), y = Math.min(...boxes.map(box => box.y));
      editor.zoomToBounds({ x, y, w: Math.max(...boxes.map(box => box.x + box.w)) - x,
        h: Math.max(...boxes.map(box => box.y + box.h)) - y });
    }),
    zoomToBounds: jest.fn((box: { x: number; y: number; w: number; h: number }, options?: { inset?: number }) => {
      const inset = options?.inset ?? Math.min(128, width * .28);
      const z = Math.max(zoomSteps[0], Math.min(zoomSteps[zoomSteps.length - 1], (width - inset) / box.w, (650 - inset) / box.h));
      camera = { x: width / (2 * z) - box.x - box.w / 2, y: 650 / (2 * z) - box.y - box.h / 2, z };
    }),
    getSelectedShapeIds: () => selection, getShape: () => ({}), select: (...ids: string[]) => { selection = ids; },
    updateViewportScreenBounds: jest.fn() };
  const props = { nodes: fixture(), edges: [] as ModelEdge[], setNodes: jest.fn(), graphId: 'one', sourceKey: 'source-set', enabled: true, sync: true, svg: false };
  const renderedScenes: { active: boolean; ids: string[] }[] = [];
  const hook = renderHook((value: typeof props) => {
    const focus = useModuleFocus(value);
    displayed = focus.graph.nodes;
    renderedScenes.push({ active: focus.active, ids: focus.graph.nodes.map(node => node.id) });
    useEffect(() => { if (value.sync) focus.markSceneCommitted(focus.graph.nodes); }, [value.sync, focus.graph.nodes, focus.markSceneCommitted]);
    return focus;
  }, { initialProps: props });
  act(() => hook.result.current.attachEditor(editor));
  return { ...hook, props, editor, renderedScenes, resize: (value: number) => { width = value; } };
}
beforeEach(() => jest.useFakeTimers());
afterEach(() => jest.useRealTimers());

test('entering, revisiting and returning fit the destination while preserving returning child selection and canonical data', () => {
  const { result, props, editor } = setup();
  const before = JSON.stringify(props.nodes);
  act(() => result.current.navigate('root')); settle();
  const rootCamera = editor.getCamera();
  editor.setCamera({ x: 123, y: 456, z: 1.2 });
  act(() => result.current.navigate('root.a')); settle();
  expect(result.current.graph.nodes.map(node => node.id)).toEqual(['root.a.x']);
  act(() => result.current.back()); settle();
  expect(editor.getCamera()).toEqual(rootCamera);
  expect(editor.getSelectedShapeIds()).toEqual([nodeIdToShapeId('root.a')]);
  act(() => result.current.back()); settle();
  expect(editor.zoomToFit).toHaveBeenCalledTimes(4);
  expect(editor.getCamera()).not.toEqual({ x: 22, y: 33, z: .6 });
  expect(JSON.stringify(props.nodes)).toBe(before);
  expect(props.setNodes).not.toHaveBeenCalled();
});

test('scope drag positions survive visits and native recreation while overview positions remain untouched', () => {
  const first = setup();
  act(() => first.result.current.navigate('root.a')); settle();
  act(() => first.result.current.setDisplayNodes(nodes => nodes.map(node => ({ ...node, position: { x: 250, y: 400 } }))));
  act(() => first.result.current.navigate('root.b')); settle();
  act(() => first.result.current.navigate('root.a')); settle();
  expect(first.result.current.graph.nodes[0].position).toEqual({ x: 250, y: 400 });
  first.editor.setCamera({ x: -220, y: 50, z: 1.1 });
  const saved = first.result.current.capturePresentation(); first.unmount();
  const second = setup();
  act(() => second.result.current.restorePresentation(saved)); settle();
  expect(second.result.current.scopeId).toBe('root.a');
  expect(second.result.current.graph.nodes[0].position).toEqual({ x: 250, y: 400 });
  expect(second.editor.getCamera()).toEqual({ x: -220, y: 50, z: 1.1 });
  expect(first.props.setNodes).not.toHaveBeenCalled(); expect(second.props.setNodes).not.toHaveBeenCalled();
});

test('ordinary renders preserve the chosen direction and camera; explicit orientation requests a fresh layout', () => {
  const { result, rerender, props, editor } = setup();
  act(() => result.current.navigate('root')); settle();
  const direction = result.current.direction;
  editor.setCamera({ x: 42, y: 57, z: 1.3 });
  rerender({ ...props, nodes: [...props.nodes] }); settle();
  expect(result.current.direction).toBe(direction);
  expect(editor.getCamera()).toEqual({ x: 42, y: 57, z: 1.3 });
  act(() => result.current.setOrientation('TB')); settle();
  expect(result.current.direction).toBe('TB');
  expect(editor.getCamera()).not.toEqual({ x: 42, y: 57, z: 1.3 });
});

test('native recreation fits changed contents instead of restoring a camera and positions from the old layout', () => {
  const first = setup();
  act(() => first.result.current.navigate('root.a')); settle();
  act(() => first.result.current.setDisplayNodes(nodes => nodes.map(node => ({ ...node, position: { x: 900, y: 400 } }))));
  first.editor.setCamera({ x: -220, y: 50, z: 1.1 });
  const saved = first.result.current.capturePresentation(); first.unmount();
  const second = setup();
  const nodes = [...second.props.nodes, node('root.a.new', 'root.a')];
  second.rerender({ ...second.props, nodes });
  act(() => second.result.current.restorePresentation(saved)); settle();
  expect(second.result.current.scopeId).toBe('root.a');
  expect(second.result.current.graph.nodes[0].position).not.toEqual({ x: 900, y: 400 });
  expect(second.editor.getCamera()).not.toEqual({ x: -220, y: 50, z: 1.1 });
  expect(second.editor.zoomToFit).toHaveBeenCalledTimes(1);
});

test('rapid navigation settles only the latest destination; an old transition cannot clear it', () => {
  const { result, editor } = setup();
  act(() => result.current.navigate('root')); settle();
  const old = result.current.transition!.revision;
  act(() => { result.current.navigate('root.a'); result.current.navigate('root.b'); });
  act(() => result.current.finishTransition(old));
  expect(result.current.transition?.to).toBe('root.b');
  settle();
  expect(result.current.scopeId).toBe('root.b');
  expect(result.current.isReady()).toBe(true);
  expect(result.current.graph.nodes.map(node => node.id)).toEqual(['root.b.x']);
  expect(editor.zoomToFit).toHaveBeenCalledTimes(2);
});

test('sibling switches fit again after manual zoom, and hidden destinations wait for a measurable viewport', () => {
  const { result, editor, resize } = setup();
  resize(0);
  act(() => result.current.navigate('root.a')); settle();
  expect(editor.zoomToFit).not.toHaveBeenCalled();
  expect(result.current.isReady()).toBe(false);
  resize(1100); settle();
  const fitted = editor.getCamera();
  editor.setCamera({ x: 300, y: 200, z: .2 });
  act(() => result.current.switchSibling('root.a', 'root.b')); settle();
  expect(result.current.scopeId).toBe('root.b');
  expect(editor.zoomToFit).toHaveBeenCalledTimes(2);
  expect(editor.getCamera()).toEqual(fitted);
});

test('same-model deletion falls back to a surviving ancestor; a different model resets all scope state', () => {
  const { result, rerender, props } = setup();
  act(() => result.current.navigate('root.a')); settle();
  rerender({ ...props, nodes: props.nodes.filter(node => !node.id.startsWith('root.a')) }); settle();
  expect(result.current.scopeId).toBe('root'); expect(result.current.isReady()).toBe(true);
  rerender({ ...props, graphId: 'two' }); settle();
  expect(result.current.scopeId).toBeNull();
  expect(result.current.capturePresentation().views.every(view => view.id === null)).toBe(true);
});

test('retained focus state rejects invalid cameras and positions and does not restore missing scopes', () => {
  const { result } = setup();
  const saved = { scopeId: 'missing', views: [{ id: 'missing', selected: [], orientation: 'LR', positions: [] }] };
  expect(validModuleFocusPresentation(saved)).toBe(true);
  act(() => result.current.restorePresentation(saved)); settle();
  expect(result.current.scopeId).toBeNull();
  expect(validModuleFocusPresentation({ ...saved, views: [{ ...saved.views[0], camera: { x: 0, y: 0, z: NaN } }] })).toBe(false);
  expect(validModuleFocusPresentation({ ...saved, views: [{ ...saved.views[0], positions: [['x', { x: Infinity, y: 0 }]] }] })).toBe(false);
});

test('same-source recapture retains module context and orientation, fits the new scene and drops old operation selection', () => {
  const { result, rerender, props, editor, renderedScenes } = setup();
  const captured = props.nodes.map(node => ({ ...node, data: { ...node.data, captureLabel: node.id,
    captureRef: { version: 1, captureId: 'a', sourceId: 'source', kind: node.type === 'moduleNode' ? 'invocation' : 'operation', semanticId: node.id } as any } }));
  rerender({ ...props, nodes: captured });
  act(() => result.current.navigate('root.a')); settle();
  act(() => result.current.setOrientation('TB')); settle();
  editor.setCamera({ x: 42, y: 17, z: 1.3 }); editor.select(nodeIdToShapeId('root.a.x'));
  const next = captured.map(node => ({ ...node, id: `next:${node.id}`, data: { ...node.data,
    parentId: node.data.parentId ? `next:${node.data.parentId}` : undefined,
    captureRef: { ...node.data.captureRef, captureId: 'b', sourceId: 'source:new-run' } } }));
  renderedScenes.length = 0;
  rerender({ ...props, nodes: next }); settle();
  expect(renderedScenes.every(scene => scene.active && scene.ids.length === 1)).toBe(true);
  expect(result.current.scopeId).toBe('next:root.a');
  expect(result.current.direction).toBe('TB');
  expect(editor.getCamera()).not.toEqual({ x: 42, y: 17, z: 1.3 });
  expect(editor.zoomToFit).toHaveBeenCalledTimes(3);
  expect(editor.getSelectedShapeIds()).toEqual([]);
  expect(result.current.capturePresentation().views.some(view => view.id === 'root.a')).toBe(false);
  rerender({ ...props, sourceKey: 'changed-source-set', nodes: next.map(node => ({ ...node, id: `other:${node.id}`, data: { ...node.data,
    parentId: node.data.parentId ? `other:${node.data.parentId}` : undefined,
    captureRef: { ...node.data.captureRef, sourceId: 'different-source' } } })) }); settle();
  expect(result.current.scopeId).toBeNull();
});

test('camera readiness waits for the bridge and no departing-scene positions leak into the overview', () => {
  const { result, rerender, props, editor } = setup();
  act(() => result.current.navigate('root.a')); settle();
  const oldCamera = editor.getCamera();
  rerender({ ...props, sync: false });
  act(() => result.current.navigate(null, { immediate: true })); settle();
  expect(result.current.isReady()).toBe(false);
  expect(editor.getCamera()).toEqual(oldCamera);
  act(() => result.current.setDisplayNodes(nodes => nodes.map(node => ({ ...node, position: { x: 999, y: 999 } }))));
  expect(props.setNodes).not.toHaveBeenCalled();
  act(() => result.current.markSceneCommitted(result.current.graph.nodes)); settle();
  expect(result.current.isReady()).toBe(true);
  expect(editor.zoomToFit).toHaveBeenCalledTimes(2);
  expect(editor.getCamera()).not.toEqual(oldCamera);
  expect(result.current.consumesResize({ width: 1100, height: 650 })).toBe(true);
  expect(result.current.consumesResize({ width: 900, height: 650 })).toBe(false);
});

test('selection changes retain the exact display graph and card data instead of rebuilding a huge scope', () => {
  const { result, rerender, props, editor } = setup();
  act(() => result.current.navigate('root')); settle();
  const graph = result.current.graph;
  rerender({ ...props, nodes: props.nodes.map(node => node.id === 'root.a'
    ? { ...node, selected: true, data: { ...node.data, isSelected: true } } : node) }); settle();
  expect(result.current.graph).toBe(graph);
  expect(editor.zoomToFit).toHaveBeenCalledTimes(1);
});

test('refit preserves selection and direction and measures dragged positions only after a view change', () => {
  const { result, editor, resize } = setup();
  act(() => result.current.navigate('root')); settle();
  const direction = result.current.direction;
  editor.select(nodeIdToShapeId('root.a'));
  editor.setCamera({ x: 100, y: 200, z: 1.7 });
  act(() => result.current.setDisplayNodes(nodes => nodes.map(node => ({ ...node, position: { x: -1400, y: -900 } })))); settle();
  expect(editor.zoomToFit).toHaveBeenCalledTimes(1);
  expect(editor.getCamera()).toEqual({ x: 100, y: 200, z: 1.7 });
  resize(400);
  act(() => result.current.refit()); settle();
  expect(editor.zoomToFit).toHaveBeenCalledTimes(2);
  expect(editor.getSelectedShapeIds()).toEqual([nodeIdToShapeId('root.a')]);
  expect(result.current.direction).toBe(direction);
  for (const box of graphSvgBounds(result.current.graph.nodes)) {
    const { x, y, z } = editor.getCamera();
    expect((box.x + x) * z).toBeGreaterThanOrEqual(31);
    expect((box.x + box.w + x) * z).toBeLessThanOrEqual(400 - 31);
    expect((box.y + y) * z).toBeGreaterThanOrEqual(31);
    expect((box.y + box.h + y) * z).toBeLessThanOrEqual(650 - 31);
  }
});

test('changed visible contents fit once after the bridge commits, while ordinary renders preserve the camera', () => {
  const { result, rerender, props, editor } = setup();
  act(() => result.current.navigate('root.a')); settle();
  const nodes = [...props.nodes, node('root.a.new', 'root.a')];
  rerender({ ...props, nodes, sync: false }); settle();
  expect(editor.zoomToFit).toHaveBeenCalledTimes(1);
  expect(result.current.isReady()).toBe(false);
  act(() => result.current.markSceneCommitted(result.current.graph.nodes)); settle();
  expect(editor.zoomToFit).toHaveBeenCalledTimes(2);
  editor.setCamera({ x: 42, y: 57, z: 1.3 });
  rerender({ ...props, nodes: [...nodes] }); settle();
  expect(editor.zoomToFit).toHaveBeenCalledTimes(2);
  expect(editor.getCamera()).toEqual({ x: 42, y: 57, z: 1.3 });
});

test('switching to SVG fits displayed bounds and switching back waits for native shapes', () => {
  const { result, rerender, props, editor } = setup();
  act(() => result.current.navigate('root.a')); settle();
  editor.zoomToFit.mockClear(); editor.zoomToBounds.mockClear();
  rerender({ ...props, svg: true }); settle();
  expect(editor.zoomToFit).not.toHaveBeenCalled();
  expect(editor.zoomToBounds).toHaveBeenCalledTimes(1);
  act(() => result.current.markSceneCommitted([]));
  rerender({ ...props, sync: false }); settle();
  expect(editor.zoomToFit).not.toHaveBeenCalled();
  act(() => result.current.markSceneCommitted(result.current.graph.nodes)); settle();
  expect(editor.zoomToFit).toHaveBeenCalledTimes(1);
});


test('huge focus scopes automatically fit every child below 5% and overview restores its zoom range', () => {
  const { result, props, rerender, editor, resize } = setup();
  const nodes = [node('root', undefined, true), ...Array.from({ length: 240 }, (_, i) => {
    const child = node(`root.${i}`, 'root');
    return { ...child, data: { ...child.data, shapeEvidence: 'measured' as const,
      inputShape: [2, 512], outputShape: [2, 512] } };
  })];
  const edges = nodes.slice(2).map((target, i) => ({ id: `edge-${i}`, source: nodes[i + 1].id, target: target.id }));
  rerender({ ...props, nodes, edges });
  act(() => result.current.navigate('root')); settle();
  const minimum = editor.getCameraOptions().zoomSteps[0];
  expect(minimum).toBeLessThan(.05);
  expect(minimum).toBeGreaterThan(0);
  expect(editor.getCamera().z).toBeLessThan(.05);
  for (const width of [1100, 400]) {
    if (width === 400) {
      resize(width);
      // The observer can request a fit before tldraw emits its resize event.
      act(() => result.current.refit()); settle();
      expect(editor.getCamera().z).toBeLessThan(minimum);
    }
    for (const box of graphSvgBounds(result.current.graph.nodes)) {
      const { x, y, z } = editor.getCamera();
      expect((box.x + x) * z).toBeGreaterThanOrEqual(31);
      expect((box.x + box.w + x) * z).toBeLessThanOrEqual(width - 31);
      expect((box.y + y) * z).toBeGreaterThanOrEqual(31);
      expect((box.y + box.h + y) * z).toBeLessThanOrEqual(650 - 31);
    }
  }
  // A metrics/selection render must not restore the default and clamp a user's Fit.
  editor.setCamera({ x: 0, y: 0, z: minimum });
  rerender({ ...props, nodes: nodes.map(n => ({ ...n, selected: n.id === 'root.1' })), edges }); settle();
  expect(editor.getCamera().z).toBe(minimum);
  act(() => result.current.setDisplayNodes(nodes => nodes.map(node => ({ ...node, position: { x: 0, y: 0 } })))); settle();
  expect(editor.getCamera().z).toBe(minimum);
  act(() => result.current.back()); settle();
  expect(editor.getCameraOptions().zoomSteps[0]).toBe(.05);
  expect(editor.zoomToFit).toHaveBeenCalledTimes(3);
});


test('deferred bridge sync accepts every drag frame in the current scope and overview', () => {
  const { result, props, rerender } = setup();
  rerender({ ...props, sync: false, nodes: props.nodes.map(node => ({ ...node })) });
  act(() => result.current.setDisplayNodes(nodes => nodes));
  expect(props.setNodes).toHaveBeenCalledTimes(1);
  rerender({ ...props, sync: true });
  act(() => result.current.navigate('root.a')); settle();
  rerender({ ...props, sync: false });
  act(() => result.current.setDisplayNodes(nodes => nodes.map(node => ({ ...node, position: { x: 100, y: 120 } }))));
  act(() => result.current.setDisplayNodes(nodes => nodes.map(node => ({ ...node, position: { x: 300, y: 320 } }))));
  expect(result.current.graph.nodes[0].position).toEqual({ x: 300, y: 320 });
  expect(result.current.isReady()).toBe(true);
  expect(props.setNodes).toHaveBeenCalledTimes(1);
});

test('changing flow preserves the current selected child for the details pane', () => {
  const { result, editor } = setup();
  act(() => result.current.navigate('root')); settle();
  editor.select(nodeIdToShapeId('root.a'));
  act(() => result.current.setOrientation('TB')); settle();
  expect(editor.getSelectedShapeIds()).toEqual([nodeIdToShapeId('root.a')]);
});
