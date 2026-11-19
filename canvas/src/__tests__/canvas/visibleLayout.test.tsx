import React, { useState } from 'react';
import { act, cleanup, fireEvent, render, renderHook, screen } from '@testing-library/react';
import { execFileSync } from 'child_process';
import { RecordingEditor } from '../../test/recordingEditor';
import ModelBuilder from '../../components/ModelBuilder';
import { useTldrawBridge } from '../../canvas/useTldrawBridge';
import { nodeIdToShapeId, edgeIdToShapeId } from '../../canvas/canvasShapeId';
import { endpointOnSide } from '../../canvas/capturePorts';
import { layerCardHeight } from '../../canvas/cardMetrics';
import { layoutTopLevel, reconcileModuleView } from '../../services/canvasLayout';
import { modelGraphToDocumentV2 } from '../../services/graphDocumentAdapter';
import { receiveProjection, setBrowserRevealNodeSink } from '../../ide/projection';
import { LocalBrowserContext } from '../../ide/localBrowserContext';
import { EMPTY_LOCAL_BROWSER_PAGE_STATE } from '../../ide/localBrowserState';
import { DEFAULT_CODE_FOCUS } from '../../ide/protocol.generated';
import { setCodeFocusOn } from '../../ide/codeFocusOn';
import { postToHost } from '../../ide/bridge';
import { ModelGraph, ModelNode } from '../../types';
import type { ProjectionPayload } from '../../ide/protocol.generated';
import metricFixtures from '../../test/fixtures/metricProjections.json';
import { graphNodes } from '../../canvas/graphQueryState';
import { NodeExplanationContext } from '../../canvas/NodeExplanationContext';
import captureDocument from '../../test/fixtures/canvas/capture-document.json';
import { graphDocumentV2ToModelGraph } from '../../services/graphDocumentAdapter';

const mockSignalListeners = new Set<() => void>();
let mockSignalRevision = 0;

jest.mock('tldraw', () => ({
  Box: { Common: (bounds: unknown) => bounds },
  atom: (_name: string, initial: any) => {
    let value = initial;
    return { get: () => value, set: (next: any) => {
      if (Object.is(value, next)) return;
      value = next;
      mockSignalRevision++;
      mockSignalListeners.forEach(listener => listener());
    } };
  },
  computed: (_name: string, read: () => any) => ({ get: read }),
  useValue: (_name: string, read: () => any) => read(),
  useEditor: () => mockEditor,
  track: (Component: any) => (props: any) => {
    require('react').useSyncExternalStore((listener: () => void) => {
      mockSignalListeners.add(listener);
      return () => { mockSignalListeners.delete(listener); };
    }, () => mockSignalRevision);
    return <Component {...props} />;
  },
}));
// Parent discovery needs the real Editor and ShapeUtils; its differential child
// test covers that path. This recording seam keeps only the original create call.
jest.mock('../../canvas/graphShapeCreation', () => ({
  createGraphShapes: (editor: RecordingEditor, shapes: any[]) => editor.createShapes(shapes),
}));
jest.mock('../../ide/bridge', () => ({ postToHost: jest.fn(), bridgeActive: () => true }));
jest.mock('../../services/apiService', () => ({ ApiService: {} }));
jest.mock('../../components/SaveGraphDialog', () => () => null);
jest.mock('../../components/CodeExportDialog', () => () => null);
jest.mock('../../components/CodeImportDialog', () => () => null);
jest.mock('../../components/ModernTemplateGallery', () => () => null);

jest.mock('../../canvas/ValidationSweepOverlay', () => ({ ValidationSweepOverlay: () => null }));
jest.mock('../../canvas/exportImage', () => ({ exportCanvasImage: jest.fn(), saveCanvasImage: jest.fn() }));
// Only the DOM canvas host is replaced: ModelBuilder's command handler, React
// state, projection conversion, layout helper and useTldrawBridge all run.
let mockEditor: RecordingEditor;
let mockCanvasProps: any;
jest.mock('../../canvas/TldrawCanvas', () => ({
  TldrawCanvas: (props: any) => {
    const React = require('react');
    mockCanvasProps = props;
    React.useEffect(() => props.onEditorReady(mockEditor), [props.onEditorReady]);
    return null;
  },
}));

const layer = (id: string, x: number, type = 'linear'): ModelNode => ({
  id, type: 'layerNode', position: { x, y: 200 },
  data: { layerType: { id: type, name: type } as any,
    parameters: { _module_path: id, _shape_provenance: 'observed', _in_shape: id === 'output' ? '3x2' : '3x4',
      _out_shape: id === 'fc2' || id === 'output' ? '3x2' : '3x4',
      ...(type === 'linear' ? { in_features: 4, out_features: id === 'fc2' ? 2 : 4 } : {}) },
    inputShape: id === 'output' ? [3, 2] : [3, 4],
    outputShape: id === 'fc2' || id === 'output' ? [3, 2] : [3, 4], shapeEvidence: 'measured' },
});
// Installed GoodNet reproduction: three width-200 cards touch at x=250/450/650.
const goodNet = (): ModelGraph => ({
  nodes: [
    { ...layer('input', 24, 'input'), type: 'inputNode', position: { x: 24, y: 26 } },
    layer('fc1', 250), layer('relu', 450, 'relu'), layer('fc2', 650),
    { ...layer('output', 1214, 'output'), type: 'outputNode', position: { x: 1214, y: 26 } },
  ],
  edges: [['input', 'fc1'], ['fc1', 'relu'], ['relu', 'fc2'], ['fc2', 'output']].map(([source, target]) =>
    ({ id: `${source}-${target}`, source, target })),
});
const projection = (graph: ModelGraph): ProjectionPayload => ({
  graphId: 'audit:GoodNet', file: 'audit_models.py', cls: 'GoodNet', sourceDigest: 'unchanged-source',
  traced: true, greyed: false, viewRestored: true, diagnostics: [],
  document: modelGraphToDocumentV2(graph, { name: 'GoodNet', revision: 1 }),
} as ProjectionPayload);
const shape = (editor: RecordingEditor, id: string) => editor.getShape(nodeIdToShapeId(id));
const settle = () => act(() => { jest.advanceTimersByTime(100); });
const command = (name: string, payload?: any) => act(() => {
  window.dispatchEvent(new CustomEvent('builder-command', { detail: { command: name, payload } }));
});

it('keeps a draft change readable instead of replacing explicit focus with the pending whole-graph fit', () => {
  const incoming = projection(goodNet());
  act(() => receiveProjection(incoming));
  Object.assign(mockEditor, {
    getShapePageBounds: (id: string) => {
      const record = mockEditor.getShape(id);
      return record && { x: record.x, y: record.y, w: record.props.w, h: record.props.h };
    },
    select: jest.fn(),
  });
  render(<ModelBuilder enableExternalCommands />);
  settle();
  command('focus-selection', { nodeIds: ['fc2', 'output'], target: {
    file: incoming.file, cls: incoming.cls, sourceDigest: incoming.sourceDigest,
    backendRevision: incoming.revision,
  } });
  settle();
  expect(mockEditor.zoomToBounds).toHaveBeenCalledWith(expect.anything(), expect.objectContaining({ targetZoom: 1 }));
  act(() => { jest.advanceTimersByTime(500); });
  expect(mockEditor.zoomToFit).not.toHaveBeenCalled();
  // An explicit Fit still remains available after a focused draft opens.
  command('fit-view');
  expect(mockEditor.zoomToFit).toHaveBeenCalledTimes(1);
});
it('explains only the clicked viewport including boundaries, and camera movement does not start generation', () => {
  act(() => receiveProjection(projection(goodNet())));
  let viewport = { x: 0, y: 0, w: 400, h: 300 };
  const bounds = new Map<string, { x: number; y: number; w: number; h: number }>([
    [nodeIdToShapeId('input'), { x: 0, y: 20, w: 100, h: 80 }],
    [nodeIdToShapeId('fc1'), { x: 120, y: 20, w: 100, h: 80 }],
    [nodeIdToShapeId('output'), { x: 250, y: 20, w: 100, h: 80 }],
    [nodeIdToShapeId('relu'), { x: 800, y: 20, w: 100, h: 80 }],
    [nodeIdToShapeId('fc2'), { x: 1100, y: 20, w: 100, h: 80 }],
  ]);
  Object.assign(mockEditor, { getViewportPageBounds: () => viewport, getShapePageBounds: (id: string) => bounds.get(id) });
  const start = jest.fn();
  render(<NodeExplanationContext.Provider value={{ states: {}, retry: jest.fn(), reveal: jest.fn(),
    model: { start, cancel: jest.fn(), toggle: jest.fn() } }}><ModelBuilder enableExternalCommands /></NodeExplanationContext.Provider>);
  settle();
  command('explain-model');
  expect(start).toHaveBeenLastCalledWith(['input', 'fc1', 'output']);
  viewport = { x: 750, y: 0, w: 250, h: 300 };
  settle();
  expect(start).toHaveBeenCalledTimes(1);
  command('explain-model');
  expect(start).toHaveBeenLastCalledWith(['relu']);
});
it('opens a double-clicked layer\'s source only with Code focus on, wherever Code focus is offered', () => {
  const reveal = jest.fn();
  setBrowserRevealNodeSink(reveal);
  try {
    act(() => receiveProjection(projection(goodNet())));
    const context = { page: EMPTY_LOCAL_BROWSER_PAGE_STATE, status: null,
      codeFocus: { options: DEFAULT_CODE_FOCUS, parameter: false, detail: '', change: jest.fn() } };
    const view = render(<LocalBrowserContext.Provider value={context}><ModelBuilder enableExternalCommands /></LocalBrowserContext.Provider>);
    settle();
    const doubleClick = () => act(() => mockCanvasProps.onShapeDoubleClick(nodeIdToShapeId('fc1')));
    doubleClick();
    expect(reveal).not.toHaveBeenCalled();
    act(() => setCodeFocusOn(true));
    doubleClick();
    expect(reveal).toHaveBeenCalledWith('fc1');
    // Graph review and the copy-only page offer no Code focus; activation keeps its route there.
    act(() => setCodeFocusOn(false));
    reveal.mockClear();
    view.rerender(<LocalBrowserContext.Provider value={{ ...context, codeFocus: undefined }}><ModelBuilder enableExternalCommands /></LocalBrowserContext.Provider>);
    doubleClick();
    expect(reveal).toHaveBeenCalledWith('fc1');
  } finally {
    setBrowserRevealNodeSink(null);
    act(() => setCodeFocusOn(false));
  }
});

function expectAnchors(editor: RecordingEditor, graph: ModelGraph, direction?: 'horizontal' | 'vertical') {
  graph.edges.filter(edge => !edge.hidden).forEach(edge => {
    const from = shape(editor, edge.source);
    const to = shape(editor, edge.target);
    const wire = editor.getShape(edgeIdToShapeId(edge));
    expect(wire).toBeDefined();
    // Edges follow current card geometry while their sides remain tied to the
    // layout direction, or to hierarchy without one: moving a card must not
    // flip its ports at a diagonal threshold. The wide test canvas resolves
    // ModelBuilder's Auto to horizontal; a bare bridge has no direction.
    const source = graph.nodes.find(node => node.id === edge.source)!;
    const target = graph.nodes.find(node => node.id === edge.target)!;
    const column = (node: ModelNode) => direction ? direction === 'vertical' : Boolean(node.parentNode || node.data.parentId);
    const startSide = column(source) ? 'bottom' : 'right';
    const endSide = column(target) ? 'top' : 'left';
    expect(wire.props.startSide).toBe(startSide); expect(wire.props.endSide).toBe(endSide);
    const fromBox = { x: from.x, y: from.y, w: from.props.w, h: from.props.h };
    const toBox = { x: to.x, y: to.y, w: to.props.w, h: to.props.h };
    const start = endpointOnSide(source.data, edge.sourceHandle, fromBox, startSide);
    const end = endpointOnSide(target.data, edge.targetHandle, toBox, endSide);
    expect(wire.x + wire.props.startX).toBeCloseTo(start.x);
    expect(wire.y + wire.props.startY).toBeCloseTo(start.y);
    expect(wire.x + wire.props.endX).toBeCloseTo(end.x);
    expect(wire.y + wire.props.endY).toBeCloseTo(end.y);
  });
}
function expectAtomicNodeUpdates(editor: RecordingEditor) {
  const updates = editor.batches.flat().filter(partial => partial.type !== 'model-edge');
  expect(updates.length).toBeGreaterThan(0);
  expect(new Set(updates.map(partial => partial.id)).size).toBe(updates.length);
  updates.forEach(partial => {
    expect(partial).toEqual(expect.objectContaining({ x: expect.any(Number), y: expect.any(Number),
      props: expect.objectContaining({ nodeData: expect.any(Object), h: expect.any(Number) }) }));
  });
}

function installedUpdates(records: any[], batches: any[][]): any[] {
  // Native Node resolves the installed package without Jest's ESM transforms.
  // No DOM or GUI host. Exit explicitly because the library owns background timers.
  return JSON.parse(execFileSync(process.execPath, ['-e', `
    const { Editor } = require('@tldraw/editor');
    const [records, batches] = JSON.parse(process.argv[1]);
    const store = new Map(records.map(record => [record.id, record]));
    const editor = { getIsReadonly: () => false, run: fn => fn(),
      getShape: id => store.get(id), getShapeUtil: () => ({}), emit: () => {},
      store: { put: updates => updates.forEach(next => store.set(next.id, next)) } };
    for (const batch of batches) Editor.prototype._updateShapes.call(editor, batch);
    process.stdout.write(JSON.stringify([...store.values()]), () => process.exit(0));
  `, JSON.stringify([records, batches])], { cwd: process.cwd(), encoding: 'utf8', timeout: 10000 }));
}

beforeEach(() => {
  jest.useFakeTimers();
  jest.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({ width: 1600, height: 900 } as DOMRect);
  window.__TENSORVIZ_IDE__ = true;
  window.__TENSORVIZ_LOCAL_BROWSER__ = true;
  mockEditor = new RecordingEditor();
  (postToHost as jest.Mock).mockClear();
  graphNodes.set([]);
});

it('updates mounted cards when only profile measurements change, without moving them', () => {
  const measuredProjection = (name: keyof typeof metricFixtures) => ({
    ...projection(goodNet()), document: metricFixtures[name],
  } as unknown as ProjectionPayload);
  act(() => receiveProjection(measuredProjection('batch2')));
  render(<ModelBuilder enableExternalCommands />);
  settle();
  const geometry = mockEditor.getCurrentPageShapes().map(s => [s.id, s.x, s.y, s.props.w, s.props.h]);
  act(() => receiveProjection(measuredProjection('batch3')));
  settle();
  expect(mockEditor.getCurrentPageShapes().map(s => [s.id, s.x, s.y, s.props.w, s.props.h])).toEqual(geometry);
  expect(shape(mockEditor, 'fc1').props.nodeData.outputShape).toEqual([3, 4]);
  expect(shape(mockEditor, 'fc2').props.nodeData.outputShape).toEqual([3, 2]);
  act(() => receiveProjection(measuredProjection('double')));
  settle();
  expect(shape(mockEditor, 'fc1').props.nodeData.outputTensor.dtype).toBe('float64');
  expect(graphNodes.get().find(node => node.id === 'fc1')!.data.parameterCount).toBe(20);
});
afterEach(() => {
  cleanup();
  jest.restoreAllMocks();
  jest.clearAllTimers();
  jest.useRealTimers();
  delete window.__TENSORVIZ_IDE__;
  delete window.__TENSORVIZ_LOCAL_BROWSER__;
});

describe('responsive graph fit', () => {
  const originalObserver = global.ResizeObserver;
  let observers: { callback: () => void; target?: Element; disconnect: jest.Mock }[];
  let size: { width: number; height: number };

  beforeEach(() => {
    observers = [];
    size = { width: 900, height: 700 };
    global.ResizeObserver = jest.fn(callback => {
      const observer = { callback, target: undefined as Element | undefined, disconnect: jest.fn() };
      observers.push(observer);
      return { observe: (target: Element) => { observer.target = target; }, disconnect: observer.disconnect };
    }) as any;
    jest.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(() => ({ ...size } as DOMRect));
  });

  afterEach(() => {
    cleanup();
    global.ResizeObserver = originalObserver;
    jest.restoreAllMocks();
  });

  const mountCanvas = (svg = false, chrome = false) => {
    act(() => receiveProjection(projection(goodNet())));
    const builder = <ModelBuilder enableExternalCommands />;
    const view = render(chrome ? <div className="local-browser-canvas">{builder}</div> : builder);
    if (svg) command('toggle-expanded-svg', { active: true });
    act(() => { jest.advanceTimersByTime(400); });
    mockEditor.zoomToFit.mockClear();
    mockEditor.zoomToBounds.mockClear();
    mockEditor.updateViewportScreenBounds.mockClear();
    const observer = observers.find(item => item.target?.classList.contains('canvas-container') && !item.disconnect.mock.calls.length);
    expect(observer).toBeDefined();
    return { ...view, observer: observer!, fit: svg ? mockEditor.zoomToBounds : mockEditor.zoomToFit };
  };

  it('fits a large traced program below the default minimum zoom', () => {
    const bounds = { x: 0, y: 0, w: 100000, h: 1000 };
    Object.assign(mockEditor, { getCurrentPageBounds: () => bounds });
    mountCanvas();
    const steps = mockEditor.setCameraOptions.mock.calls.at(-1)?.[0].zoomSteps;
    expect(steps?.[0]).toBeLessThanOrEqual((size.width - 128) / bounds.w);
    expect(steps).toContain(1); // Readable detail remains available after Fit.
  });

  it('fits the first rendered graph even when a projection replaces the pending layout', () => {
    act(() => receiveProjection(projection(goodNet())));
    render(<ModelBuilder enableExternalCommands />);
    settle();
    expect(mockEditor.zoomToFit).not.toHaveBeenCalled();
    const next = goodNet();
    next.nodes.push(layer('new-layer', 1800));
    act(() => receiveProjection(projection(next)));
    act(() => { jest.advanceTimersByTime(400); });
    expect(mockEditor.zoomToFit).toHaveBeenCalledTimes(1);
    expect(shape(mockEditor, 'new-layer')).toBeDefined();
    expect(mockEditor.updateViewportScreenBounds).toHaveBeenCalledWith(document.querySelector('.canvas-container'));
  });

  it('fits a different Chrome graph with the same node count, but preserves navigation on ordinary updates', () => {
    const { fit } = mountCanvas();
    const next = { ...projection(goodNet()), graphId: 'audit:AnotherNet', cls: 'AnotherNet' };
    act(() => receiveProjection(next));
    act(() => { jest.advanceTimersByTime(400); });
    expect(fit).toHaveBeenCalledTimes(1);
    fit.mockClear();
    act(() => receiveProjection({ ...next, revision: 2 }));
    act(() => { jest.advanceTimersByTime(400); });
    expect(fit).not.toHaveBeenCalled();
  });

  it.each([false, true])('fits every Blocks selection even when the flat graph is already in Blocks (SVG: %s)', svg => {
    const { fit } = mountCanvas(svg);
    for (let click = 0; click < 2; click++) {
      fit.mockClear();
      command('view-overview');
      expect(fit).not.toHaveBeenCalled();
      act(() => { jest.advanceTimersByTime(400); });
      expect(fit).toHaveBeenCalledTimes(1);
    }
  });

  it('fits Blocks after the expanded children have been removed from the editor', () => {
    const graph = goodNet();
    graph.nodes.push({ ...layer('block', 250), type: 'moduleNode',
      data: { ...layer('block', 250).data, isGroup: true, expanded: true } });
    graph.nodes.find(node => node.id === 'fc1')!.data.parentId = 'block';
    act(() => receiveProjection(projection(graph)));
    render(<ModelBuilder enableExternalCommands />);
    act(() => { jest.advanceTimersByTime(400); });
    expect(shape(mockEditor, 'fc1')).toBeDefined();
    mockEditor.zoomToFit.mockClear();
    const fittedShapes: any[][] = [];
    mockEditor.zoomToFit.mockImplementation(() => fittedShapes.push(mockEditor.getCurrentPageShapes()));
    command('view-overview');
    act(() => { jest.advanceTimersByTime(400); });
    expect(fittedShapes).toHaveLength(1);
    expect(fittedShapes[0].find(node => node.id === nodeIdToShapeId('fc1'))).toBeUndefined();
    expect(fittedShapes[0].find(node => node.id === nodeIdToShapeId('block'))!.props.nodeData.expanded).toBe(false);
  });

  it.each([false, true])('fits the completed validation after the new graph renders (SVG: %s)', svg => {
    const { fit } = mountCanvas(svg);
    const next = goodNet();
    next.nodes.push(layer('validated-layer', 1800));
    const completed = { ...projection(next), revision: 2 };
    act(() => {
      receiveProjection(completed);
      window.dispatchEvent(new CustomEvent('builder-command', {
        detail: { command: 'validation-complete', payload: { projection: completed } },
      }));
    });
    expect(fit).not.toHaveBeenCalled();
    act(() => { jest.advanceTimersByTime(400); });
    expect(fit).toHaveBeenCalledTimes(1);
    if (!svg) expect(shape(mockEditor, 'validated-layer')).toBeDefined();
    fit.mockClear();
    act(() => receiveProjection({ ...completed }));
    act(() => { jest.advanceTimersByTime(400); });
    expect(fit).not.toHaveBeenCalled();
    command('validation-complete', { projection: { ...completed, revision: 1 } });
    act(() => { jest.advanceTimersByTime(400); });
    expect(fit).not.toHaveBeenCalled();
  });

  it.each([false, true])('preserves the graph when header or footer chrome changes the inner height (SVG: %s)', svg => {
    size = { width: 350, height: 560 };
    let surface = { width: 350, height: 800 };
    jest.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
      return (this.classList.contains('local-browser-canvas') ? surface : size) as DOMRect;
    });
    const { observer, fit } = mountCanvas(svg, true);
    command('set-layout-direction', { choice: 'horizontal' });
    act(() => { jest.advanceTimersByTime(400); });
    fit.mockClear();
    const shapes = mockEditor.getCurrentPageShapes();
    for (const height of [530, 560]) {
      size = { ...size, height };
      act(() => observer.callback()); settle();
      expect(fit).not.toHaveBeenCalled();
      expect(mockEditor.getCurrentPageShapes()).toEqual(shapes);
    }
    // An actual sidebar/window height resize must still perform the existing fit.
    surface = { ...surface, height: 700 }; size = { ...size, height: 460 };
    act(() => observer.callback()); settle();
    expect(fit).toHaveBeenCalledTimes(1);
    // And a canvas divider can still change width inside the same outer surface.
    size = { ...size, width: 300 };
    act(() => observer.callback()); settle();
    expect(fit).toHaveBeenCalledTimes(2);
  });

  it.each([false, true])('fits the displayed graph on width and height changes (SVG: %s)', svg => {
    const { observer, fit } = mountCanvas(svg);
    // These sizes cross Auto's aspect boundary; an explicit direction keeps
    // this about fitting. Auto's relayout is covered separately.
    command('set-layout-direction', { choice: 'horizontal' });
    const shapes = mockEditor.getCurrentPageShapes();
    // The initial observer delivery must preserve a restored/manual camera.
    act(() => observer.callback()); settle();
    expect(fit).not.toHaveBeenCalled();
    // A divider drag is not a window resize. Coalesce its changes into one frame.
    act(() => {
      size.width = 600; observer.callback();
      size.width = 320; observer.callback();
    });
    expect(fit).not.toHaveBeenCalled();
    settle();
    expect(fit).toHaveBeenCalledTimes(1);
    expect(fit.mock.calls[0][svg ? 1 : 0]).toMatchObject({ animation: { duration: 0 } });
    expect(mockEditor.updateViewportScreenBounds).toHaveBeenCalledWith(observer.target);
    expect(mockEditor.updateViewportScreenBounds.mock.invocationCallOrder[0]).toBeLessThan(fit.mock.invocationCallOrder[0]);
    expect(mockEditor.getCurrentPageShapes()).toEqual(shapes);
    size.height = 400;
    act(() => observer.callback()); settle();
    expect(fit).toHaveBeenCalledTimes(2);
    // Repeated notifications at the same size must leave manual navigation alone.
    act(() => observer.callback()); settle();
    expect(fit).toHaveBeenCalledTimes(2);
    size.width = 1200;
    act(() => observer.callback()); settle();
    expect(fit).toHaveBeenCalledTimes(3);
  });

  it('Auto lays nested contents out horizontally when wide and vertically once a narrow resize settles; an explicit choice holds', () => {
    const graph = goodNet();
    graph.nodes.push({ ...layer('block', 250), type: 'moduleNode', data: { ...layer('block', 250).data, isGroup: true, expanded: true } });
    for (const id of ['fc1', 'relu', 'fc2']) graph.nodes.find(node => node.id === id)!.data.parentId = 'block';
    let latest: ModelGraph;
    const onGraph = (event: Event) => { latest = (event as CustomEvent).detail.graph; };
    const snapshot = jest.fn();
    window.addEventListener('canvas-graph-updated', onGraph);
    try {
      size = { width: 1600, height: 900 };
      act(() => receiveProjection(projection(graph)));
      render(<ModelBuilder enableExternalCommands onBuilderSnapshot={snapshot} />);
      act(() => { jest.advanceTimersByTime(400); });
      const observer = observers.find(item => item.target?.classList.contains('canvas-container') && !item.disconnect.mock.calls.length)!;
      const along = (axis: 'x' | 'y') => {
        const [a, b, c] = ['fc1', 'relu', 'fc2'].map(id => shape(mockEditor, id));
        return b[axis] >= a[axis] + (axis === 'x' ? a.props.w : a.props.h) && c[axis] >= b[axis] + (axis === 'x' ? b.props.w : b.props.h);
      };
      const lastSnapshot = () => snapshot.mock.calls[snapshot.mock.calls.length - 1][0];
      const semantics = () => JSON.stringify([latest!.nodes.map(node => [node.id, node.data]), latest!.edges.filter(edge => !edge.data?.synthetic)]);
      expect(along('x')).toBe(true);
      expectAnchors(mockEditor, latest!, 'horizontal');
      expect(lastSnapshot()).toMatchObject({ layoutChoice: 'auto', layoutDirection: 'horizontal' });
      const before = semantics();

      // Dragging the divider into a sidebar does not relayout on every frame.
      act(() => { size.width = 700; observer.callback(); size.width = 380; observer.callback(); });
      settle();
      expect(along('x')).toBe(true);
      mockEditor.zoomToFit.mockClear();
      act(() => { jest.advanceTimersByTime(200); });
      expect(along('y')).toBe(true);
      expectAnchors(mockEditor, latest!, 'vertical');
      expect(lastSnapshot()).toMatchObject({ layoutChoice: 'auto', layoutDirection: 'vertical' });
      expect(mockEditor.zoomToFit).not.toHaveBeenCalled();
      act(() => { jest.advanceTimersByTime(400); });
      expect(mockEditor.zoomToFit).toHaveBeenCalledTimes(1);
      expect(semantics()).toBe(before);

      // An ordinary source refresh keeps the resolved direction and its ports.
      act(() => receiveProjection(projection(graph)));
      settle();
      expect(along('y')).toBe(true);
      expectAnchors(mockEditor, latest!, 'vertical');

      // An explicit choice holds across later resizes.
      command('set-layout-direction', { choice: 'horizontal' });
      settle();
      expect(along('x')).toBe(true);
      expectAnchors(mockEditor, latest!, 'horizontal');
      act(() => { size = { width: 300, height: 900 }; observer.callback(); });
      act(() => { jest.advanceTimersByTime(1000); });
      expect(along('x')).toBe(true);
      expect(lastSnapshot()).toMatchObject({ layoutChoice: 'horizontal', layoutDirection: 'horizontal' });

      // Returning to Auto follows the current narrow canvas.
      command('set-layout-direction', { choice: 'auto' });
      settle();
      expect(along('y')).toBe(true);
      expect(semantics()).toBe(before);
    } finally { window.removeEventListener('canvas-graph-updated', onGraph); }
  });

  it('sizes expanded frames before placing output on a fresh source-only graph', () => {
    const graph = goodNet();
    graph.nodes.push({ ...layer('block', 250), type: 'moduleNode', data: { ...layer('block', 250).data, isGroup: true, expanded: true } });
    for (const id of ['fc1', 'relu', 'fc2']) graph.nodes.find(node => node.id === id)!.data.parentId = 'block';
    size = { width: 1600, height: 900 };
    act(() => receiveProjection({ ...projection(graph), traced: false, viewRestored: false }));
    render(<ModelBuilder enableExternalCommands />); settle();
    const block = shape(mockEditor, 'block');
    expect(shape(mockEditor, 'output').x).toBeGreaterThan(block.x + block.props.w);
  });

  it('retains live orientation in captured review and reflows only when Auto changed before returning', () => {
    const graph = goodNet();
    graph.nodes.push({ ...layer('block', 250), type: 'moduleNode', data: { ...layer('block', 250).data, isGroup: true, expanded: true } });
    for (const id of ['fc1', 'relu', 'fc2']) graph.nodes.find(node => node.id === id)!.data.parentId = 'block';
    let latest: ModelGraph;
    const onGraph = (event: Event) => { latest = (event as CustomEvent).detail.graph; };
    window.addEventListener('canvas-graph-updated', onGraph);
    try {
      size = { width: 1600, height: 900 };
      const payload = projection(graph);
      act(() => receiveProjection(payload));
      render(<ModelBuilder enableExternalCommands />);
      act(() => { jest.advanceTimersByTime(400); });
      const live = latest!;
      const review = { ...payload, reviewGraph: live, reviewId: 'layout-review', capturedContext: true };
      act(() => receiveProjection(review)); settle();
      expectAnchors(mockEditor, latest!, 'horizontal');
      expect(latest!.nodes.map(node => node.position)).toEqual(live.nodes.map(node => node.position));
      // A source-only comparison has its own legacy geometry. Switching back
      // to captured context must still recover the original live port sides.
      act(() => receiveProjection({ ...review, capturedContext: false, reviewGraph: reconcileModuleView(graph.nodes, graph.edges, { relayout: true }) })); settle();
      expectAnchors(mockEditor, latest!);
      act(() => receiveProjection(review)); settle();
      expectAnchors(mockEditor, latest!, 'horizontal');
      const observer = observers.find(item => item.target?.classList.contains('canvas-container') && !item.disconnect.mock.calls.length)!;
      act(() => { size = { width: 380, height: 900 }; observer.callback(); });
      act(() => { jest.advanceTimersByTime(1000); });
      expectAnchors(mockEditor, latest!, 'horizontal');
      expect(latest!.nodes.map(node => node.position)).toEqual(live.nodes.map(node => node.position));
      act(() => receiveProjection(payload)); settle();
      expectAnchors(mockEditor, latest!, 'vertical');
      expect(shape(mockEditor, 'fc2').y).toBeGreaterThan(shape(mockEditor, 'fc1').y);
    } finally { window.removeEventListener('canvas-graph-updated', onGraph); }
  });

  it('skips a hidden canvas and fits when it becomes visible again', () => {
    const { observer, fit } = mountCanvas();
    size = { width: 0, height: 0 };
    act(() => observer.callback()); settle();
    expect(fit).not.toHaveBeenCalled();
    size = { width: 900, height: 700 };
    act(() => observer.callback()); settle();
    expect(fit).toHaveBeenCalledTimes(1);
  });

  it('fits a focused scope on sidebar resize, including a return to its original size', () => {
    const graph = goodNet();
    graph.nodes.push({ ...layer('block', 250), type: 'moduleNode',
      data: { ...layer('block', 250).data, isGroup: true, expanded: false, moduleInstanceId: 'block' } });
    graph.nodes.find(node => node.id === 'fc1')!.data.parentId = 'block';
    Object.assign(mockEditor, { getContainer: () => document.querySelector('.canvas-container'),
      stopCameraAnimation: jest.fn(), select: jest.fn() });
    act(() => receiveProjection(projection(graph)));
    render(<ModelBuilder enableExternalCommands />);
    act(() => { jest.advanceTimersByTime(400); });
    act(() => shape(mockEditor, 'block').props.nodeData.onToggleExpand('block'));
    act(() => { jest.advanceTimersByTime(400); });
    expect(document.querySelector('[data-focus-scope="block"]')).toBeInTheDocument();
    const direction = document.querySelector('[data-focus-direction]')!.getAttribute('data-focus-direction');
    const observer = observers.find(item => item.target?.classList.contains('canvas-container') && !item.disconnect.mock.calls.length)!;
    mockEditor.zoomToFit.mockClear();
    act(() => observer.callback()); settle();
    expect(mockEditor.zoomToFit).not.toHaveBeenCalled();
    size.width = 400;
    act(() => observer.callback()); settle(); settle();
    expect(mockEditor.zoomToFit).toHaveBeenCalledTimes(1);
    size.width = 900;
    act(() => observer.callback()); settle(); settle();
    expect(mockEditor.zoomToFit).toHaveBeenCalledTimes(2);
    size.height = 450;
    act(() => observer.callback()); settle(); settle();
    expect(mockEditor.zoomToFit).toHaveBeenCalledTimes(3);
    act(() => observer.callback()); settle();
    expect(mockEditor.zoomToFit).toHaveBeenCalledTimes(3);
    expect(document.querySelector('[data-focus-direction]')!.getAttribute('data-focus-direction')).toBe(direction);
  });

  it('keeps the first fit pending while hidden and consumes it when the graph becomes visible', () => {
    size = { width: 0, height: 0 };
    act(() => receiveProjection(projection(goodNet())));
    render(<ModelBuilder enableExternalCommands />);
    act(() => { jest.advanceTimersByTime(400); });
    expect(mockEditor.zoomToFit).not.toHaveBeenCalled();
    const observer = observers.find(item => item.target?.classList.contains('canvas-container') && !item.disconnect.mock.calls.length)!;
    size = { width: 900, height: 700 };
    act(() => observer.callback()); settle();
    expect(mockEditor.zoomToFit).toHaveBeenCalledTimes(1);
    act(() => receiveProjection(projection(goodNet())));
    act(() => { jest.advanceTimersByTime(400); });
    expect(mockEditor.zoomToFit).toHaveBeenCalledTimes(1);
  });

  it('cancels a queued fit and disconnects when the canvas closes', () => {
    const { observer, fit, unmount } = mountCanvas();
    size.width = 400;
    act(() => observer.callback());
    unmount(); settle();
    expect(fit).not.toHaveBeenCalled();
    expect(observer.disconnect).toHaveBeenCalledTimes(1);
  });
});

it('reproduces lost movement with duplicate IDs using the installed Editor._updateShapes', () => {
  const original = { id: 'shape:audit', type: 'layer', x: 250, y: 200, props: { w: 200, h: 120 } };
  const partials = [{ id: original.id, type: 'layer', x: 299, y: 24 },
    { id: original.id, type: 'layer', props: { h: 160 } }];
  const [actual] = installedUpdates([original], [partials]);
  const editor = new RecordingEditor();
  editor.createShape(original);
  editor.updateShapes(partials);
  expect(editor.getShape(original.id)).toEqual(actual);
  expect(actual).toMatchObject({ x: 250, y: 200, props: { h: 160 } });
});

it('Auto layout moves the installed GoodNet chain through ModelBuilder and the bridge without changing source or evidence', () => {
  const incoming = projection(goodNet());
  const sourceBefore = JSON.stringify(incoming);
  act(() => receiveProjection(incoming));
  let graph: ModelGraph;
  const onGraph = (event: Event) => { graph = (event as CustomEvent).detail.graph; };
  window.addEventListener('canvas-graph-updated', onGraph);
  try {
    render(<ModelBuilder enableExternalCommands />);
    settle();
    expect(shape(mockEditor, 'relu').x - shape(mockEditor, 'fc1').x).toBe(200);
    const before = graph!;
    const expected = layoutTopLevel(before.nodes, before.edges);
    const beforeShapes = mockEditor.getCurrentPageShapes();
    mockEditor.batches = [];
    command('layout');
    settle();
    before.nodes.forEach(node => expect(shape(mockEditor, node.id)).toMatchObject(expected.get(node.id)!));
    expect(shape(mockEditor, 'relu').x - shape(mockEditor, 'fc1').x - shape(mockEditor, 'fc1').props.w).toBeGreaterThanOrEqual(90);
    expect(graph!.nodes.map(node => node.data)).toEqual(before.nodes.map(node => node.data));
    expect(graph!.edges).toEqual(before.edges);
    expectAtomicNodeUpdates(mockEditor);
    const installed = installedUpdates(beforeShapes, mockEditor.batches);
    before.nodes.forEach(node => expect(installed.find(record => record.id === nodeIdToShapeId(node.id)))
      .toEqual(JSON.parse(JSON.stringify(shape(mockEditor, node.id)))));
    expectAnchors(mockEditor, graph!, 'horizontal');

    // Fit waits for the new layout to render.
    act(() => { jest.advanceTimersByTime(400); });
    expect(mockEditor.zoomToFit).toHaveBeenCalled();

    // Drag the actual shape; the bridge writes the presentation position back.
    act(() => {
      mockEditor.updateShape({ id: nodeIdToShapeId('relu'), type: 'layer', x: 812, y: 412 });
      mockEditor.completeUserOperation();
    });
    settle();
    expect(graph!.nodes.find(node => node.id === 'relu')!.position).toEqual({ x: 812, y: 412 });
    expectAnchors(mockEditor, graph!, 'horizontal');
    act(() => receiveProjection(incoming));
    settle();
    expect(shape(mockEditor, 'relu')).toMatchObject({ x: 812, y: 412 });
    expectAnchors(mockEditor, graph!, 'horizontal');

    // A second layout after drag + refresh returns to the same deterministic view.
    command('layout');
    settle();
    before.nodes.forEach(node => expect(shape(mockEditor, node.id)).toMatchObject(expected.get(node.id)!));
    expectAnchors(mockEditor, graph!, 'horizontal');

    const retained = JSON.stringify(graph!);
    command('set-parameters', { nodeId: 'fc1', parameters: { out_features: 900 } });
    command('delete-selected');
    act(() => mockEditor.deleteShapes([nodeIdToShapeId('fc1')]));
    settle();
    expect(JSON.stringify(graph!)).toBe(retained);
    expect(shape(mockEditor, 'fc1')).toBeDefined();
    act(() => { jest.advanceTimersByTime(2000); });
    expect((postToHost as jest.Mock).mock.calls.every(([message]) =>
      ['projectionApplied', 'viewSaved'].includes(message.type))).toBe(true);
    expect(JSON.stringify(incoming)).toBe(sourceBefore);
  } finally { window.removeEventListener('canvas-graph-updated', onGraph); }
});

function mountBridge(initial: ModelGraph, editor = new RecordingEditor()) {
  const hook = renderHook(() => {
    const [nodes, setNodes] = useState(initial.nodes);
    const [edges, setEdges] = useState(initial.edges);
    return { nodes, setNodes, edges, ...useTldrawBridge({ nodes, edges, setNodes, setEdges }) };
  });
  let dispose: () => void;
  act(() => { dispose = hook.result.current.handleEditorReady(editor as any); });
  settle();
  return { ...hook, editor, dispose: () => dispose() };
}

it('applies position, data and measured card height together, including read-only editor synchronization', () => {
  const editor = new RecordingEditor();
  editor.readonly = true;
  const mounted = mountBridge(goodNet(), editor);
  try {
    const previousHeight = shape(editor, 'fc1').props.h;
    editor.batches = [];
    act(() => mounted.result.current.setNodes(nodes => nodes.map(node => node.id === 'fc1' ? {
      ...node, position: { x: 510, y: 73 },
      data: { ...node.data, error: 'Measured run failed here' },
    } : node)));
    settle();
    const node = mounted.result.current.nodes.find(n => n.id === 'fc1')!;
    expect(shape(editor, 'fc1')).toMatchObject({ x: 510, y: 73,
      props: { nodeData: node.data, h: layerCardHeight(node.data) } });
    expect(shape(editor, 'fc1').props.h).toBeGreaterThan(previousHeight);
    expectAtomicNodeUpdates(editor);
    expectAnchors(editor, mounted.result.current);
    expect(editor.getIsReadonly()).toBe(true);
    expect(editor.readonlyChanges.length).toBeGreaterThan(0);
    expect(editor.readonlyChanges.length % 2).toBe(0);
    editor.readonlyChanges.forEach((value, index) => expect(value).toBe(index % 2 === 1));
  } finally { mounted.dispose(); }
});

it('moves and resizes an expanded parent and its child once, then retains relative coordinates while dragging', () => {
  const child = { ...layer('block.fc', 40), data: { ...layer('block.fc', 40).data, parentId: 'block' } };
  const logical: ModelGraph = { nodes: [
    { ...layer('block', 250), type: 'moduleNode', data: { ...layer('block', 250).data, expanded: true, isGroup: true } },
    child, layer('head', 900),
  ], edges: [{ id: 'out', source: child.id, target: 'head' }] };
  const mounted = mountBridge(reconcileModuleView(logical.nodes, logical.edges));
  const { editor } = mounted;
  try {
    editor.batches = [];
    act(() => mounted.result.current.setNodes(nodes => nodes.map(node => node.id === 'block' ? {
      ...node, position: { x: 330, y: 50 }, style: { width: 410, height: 480 },
      data: { ...node.data, moduleLayoutState: 'custom' },
    } : node.id === child.id ? { ...node, position: { x: 60, y: 90 } } : node)));
    settle();
    expect(shape(editor, 'block')).toMatchObject({ x: 330, y: 50, props: { w: 410, h: 480 } });
    expect(shape(editor, child.id)).toMatchObject({ x: 390, y: 140 });
    expectAtomicNodeUpdates(editor);
    expectAnchors(editor, mounted.result.current);
    act(() => {
      editor.updateShape({ id: nodeIdToShapeId('block'), type: 'module-node', x: 430, y: 80 });
      editor.completeUserOperation();
    });
    settle();
    expect(shape(editor, child.id)).toMatchObject({ x: 490, y: 170 });
    expect(mounted.result.current.nodes.find(node => node.id === child.id)!.position).toEqual({ x: 60, y: 90 });
    expectAnchors(editor, mounted.result.current);
  } finally { mounted.dispose(); }
});

it('updates a module width and its edge anchor when data, height and positions are unchanged', () => {
  const module = { ...layer('block', 200), type: 'moduleNode', style: { width: 250, height: 240 } };
  const graph = { nodes: [module, layer('head', 900)], edges: [{ id: 'out', source: 'block', target: 'head' }] };
  const mounted = mountBridge(graph);
  try {
    const before = mounted.result.current.nodes;
    act(() => mounted.result.current.setNodes(nodes => nodes.map(node => node.id === 'block'
      ? { ...node, style: { ...node.style, width: 390 } } : node)));
    settle();
    expect(shape(mounted.editor, 'block')).toMatchObject({ x: 200, y: 200, props: { w: 390, h: 240 } });
    expect(mounted.result.current.nodes.map(node => node.data)).toEqual(before.map(node => node.data));
    expectAnchors(mounted.editor, graph);
  } finally { mounted.dispose(); }
});

it('propagates a single node move to only that node and its incident edges', () => {
  // WHY: a sync used to rewrite every node and delete/recreate every edge, so one
  // move (or one click, via handleSelectionChange) re-wrote the whole graph. This
  // is the guard that the bridge diffs: the changed node and the edges touching it
  // are written; everything else keeps its record untouched.
  const graph = goodNet();
  const mounted = mountBridge(graph);
  try {
    mounted.editor.batches = [];
    act(() => mounted.result.current.setNodes(nodes => nodes.map(node => node.id === 'relu'
      ? { ...node, position: { x: 812, y: 412 } } : node)));
    settle();

    const updated = new Set(mounted.editor.batches.flat().map(partial => partial.id));
    expect(updated.has(nodeIdToShapeId('relu'))).toBe(true);
    expect(updated.has(edgeIdToShapeId(graph.edges[1]))).toBe(true); // fc1 -> relu
    expect(updated.has(edgeIdToShapeId(graph.edges[2]))).toBe(true); // relu -> fc2
    // Untouched cards and edges are not rewritten ...
    expect(updated.has(nodeIdToShapeId('fc1'))).toBe(false);
    expect(updated.has(nodeIdToShapeId('fc2'))).toBe(false);
    expect(updated.has(edgeIdToShapeId(graph.edges[0]))).toBe(false);
    expect(updated.has(edgeIdToShapeId(graph.edges[3]))).toBe(false);
    // ... and no edge was deleted and rebuilt.
    graph.edges.forEach(edge => expect(mounted.editor.getShape(edgeIdToShapeId(edge))).toBeDefined());
    expect(shape(mounted.editor, 'relu')).toMatchObject({ x: 812, y: 412 });
    expectAnchors(mounted.editor, mounted.result.current);
  } finally { mounted.dispose(); }
});

it('switches captured detail without recapturing, restores each layout and works with SVG', () => {
  const captured = graphDocumentV2ToModelGraph(captureDocument as any);
  const root = captured.nodes.find(n => n.data.captureRef?.kind === 'invocation')!;
  const first = captured.nodes.find(n => n.data.captureRef?.kind === 'operation')!;
  const leaf = { ...root, id: 'leaf', data: { ...root.data, parentId: root.id, expanded: false,
    moduleInstanceId: 'leaf', captureRef: { ...root.data.captureRef!, semanticId: 'leaf' } } };
  captured.nodes.push(leaf);
  first.data.parentId = leaf.id;
  act(() => receiveProjection(projection(captured)));
  let latest: ModelGraph = captured;
  const listener = (event: Event) => { latest = (event as CustomEvent).detail.graph; };
  window.addEventListener('canvas-graph-updated', listener);
  try {
  render(<ModelBuilder enableExternalCommands />);
  settle();
  command('view-expanded'); settle();
  expect(shape(mockEditor, root.id)).toBeDefined();
  expect(shape(mockEditor, leaf.id)).toBeDefined();
  expect(shape(mockEditor, first.id)).toBeUndefined();
  const baseline = latest.nodes;
  const positions = baseline.map(n => [n.id, n.position]);
  const semantics = modelGraphToDocumentV2({ nodes: baseline, edges: captured.edges }, { name: 'Model', revision: 1 }).semantic;
  command('set-graph-detail', { level: 'operations' }); settle();
  expect(shape(mockEditor, first.id)).toBeDefined();
  command('set-graph-detail', { level: 'layers' }); settle();
  expect(latest.nodes.map(n => [n.id, n.position])).toEqual(positions);
  expect(modelGraphToDocumentV2({ nodes: latest.nodes, edges: captured.edges }, { name: 'Model', revision: 1 }).semantic).toEqual(semantics);
  command('toggle-expanded-svg', { active: true }); settle();
  command('set-graph-detail', { level: 'operations' }); settle();
  expect(mockEditor.getCurrentPageShapes()).toHaveLength(0);
  expect(screen.getByTestId('graph-svg-layer')).toBeInTheDocument();
  expect(latest.nodes.find(n => n.id === first.id)?.hidden).toBe(false);
  // A repeated projection retains the chosen level and open calls.
  act(() => receiveProjection(projection(captured))); settle();
  expect(latest.nodes.find(n => n.id === root.id)?.data.captureDetailLevel).toBe('operations');
  expect((postToHost as jest.Mock).mock.calls.every(([message]) =>
    ['projectionApplied', 'viewSaved'].includes(message.type))).toBe(true);
  } finally { window.removeEventListener('canvas-graph-updated', listener); }
});

// The installed sidebar driver (extension/scripts/qa-sidebar-canvas.cjs) relies on this.
it('Expanded view shows a nested call folded; an exact focus enters its parent scope', () => {
  const captured = graphDocumentV2ToModelGraph(captureDocument as any);
  const root = captured.nodes.find(n => n.data.captureRef?.kind === 'invocation')!;
  const first = captured.nodes.find(n => n.data.captureRef?.kind === 'operation')!;
  const call = (id: string, parentId: string, label: string) => ({ ...root, id, data: { ...root.data, parentId,
    expanded: false, captureLabel: label, moduleInstanceId: id, captureRef: { ...root.data.captureRef!, semanticId: id } } });
  captured.nodes.push(call('block', root.id, 'block'), call('conv', 'block', 'block.conv'));
  first.data.parentId = 'conv';
  Object.assign(mockEditor, {
    getShapePageBounds: (id: string) => {
      const record = mockEditor.getShape(id);
      return record && { x: record.x, y: record.y, w: record.props.w, h: record.props.h };
    },
    getContainer: () => document.body, select: jest.fn(), selectNone: jest.fn(), stopCameraAnimation: jest.fn(),
  });
  act(() => receiveProjection(projection(captured)));
  render(<ModelBuilder enableExternalCommands />);
  settle();
  const frames = () => { for (let i = 0; i < 5; i++) settle(); };
  command('view-expanded'); frames();
  // Expanded view opens calls that contain calls and leaves leaf calls folded.
  expect(shape(mockEditor, 'block').props.nodeData.expanded).toBe(true);
  expect(shape(mockEditor, 'conv').props.nodeData).toMatchObject({ captureLabel: 'block.conv', expanded: false });
  command('focus-node', { nodeId: 'conv' }); frames();
  // block's scope: its children have cards, block itself does not.
  expect(shape(mockEditor, 'conv')).toBeDefined();
  expect(shape(mockEditor, 'block')).toBeUndefined();
  command('view-expanded'); frames();
  command('focus-node', { nodeId: 'block' }); frames();
  // The root's scope draws block folded, so no block.conv card exists.
  expect(shape(mockEditor, 'block').props.nodeData.expanded).toBe(false);
  expect(shape(mockEditor, 'conv')).toBeUndefined();
});

it('replaces every mounted card with one SVG image in expanded overview', () => {
  act(() => receiveProjection(projection(goodNet())));
  render(<ModelBuilder enableExternalCommands />);
  settle();
  expect(mockEditor.getCurrentPageShapes().length).toBeGreaterThan(0);

  command('toggle-expanded-svg', { active: true });
  settle();
  // The point of the mode: no card or edge shape stays mounted under the image.
  expect(mockEditor.getCurrentPageShapes().length).toBe(0);
  expect(screen.getByTestId('graph-svg-layer')).toBeInTheDocument();

  // Fit must measure the displayed SVG; tldraw has no shapes in this mode.
  mockEditor.zoomToBounds.mockClear();
  mockEditor.zoomToFit.mockClear();
  command('fit-view'); settle();
  expect(mockEditor.zoomToBounds).toHaveBeenCalledWith(expect.objectContaining({
    w: expect.any(Number), h: expect.any(Number),
  }), expect.any(Object));
  expect(mockEditor.zoomToBounds.mock.calls[0][0].w).toBeGreaterThan(0);
  expect(mockEditor.zoomToFit).not.toHaveBeenCalled();

  // Toggling back rebuilds the shapes.
  command('toggle-expanded-svg', { active: false });
  settle();
  expect(mockEditor.getCurrentPageShapes().length).toBeGreaterThan(0);
  mockEditor.zoomToFit.mockClear();
  command('fit-view'); settle();
  expect(mockEditor.zoomToFit).toHaveBeenCalledTimes(1);
});

it('Auto layout carries nested expanded contents and preserves their geometry across a module drag and source refresh', () => {
  const module = (id: string, x: number, parentId?: string): ModelNode => ({
    ...layer(id, x), type: 'moduleNode', data: {
      ...layer(id, x).data, isGroup: true, expanded: true, parentId,
    },
  });
  const leaf = { ...layer('block.inner.fc', 40), data: {
    ...layer('block.inner.fc', 40).data, parentId: 'block.inner',
  } };
  const incoming = projection({ nodes: [layer('input', 24), module('block', 250),
    module('block.inner', 40, 'block'), leaf, layer('head', 450)],
    edges: [{ id: 'in', source: 'input', target: leaf.id }, { id: 'out', source: leaf.id, target: 'head' }] });
  act(() => receiveProjection(incoming));
  let graph: ModelGraph;
  const onGraph = (event: Event) => { graph = (event as CustomEvent).detail.graph; };
  window.addEventListener('canvas-graph-updated', onGraph);
  try {
    render(<ModelBuilder enableExternalCommands />);
    settle();
    const children = graph!.nodes.filter(node => node.parentNode);
    expect(children.map(node => node.id).sort()).toEqual(['block.inner', leaf.id]);
    const expected = layoutTopLevel(graph!.nodes, graph!.edges, 'horizontal');
    command('layout');
    settle();
    expect(shape(mockEditor, 'block')).toMatchObject(expected.get('block')!);
    children.forEach(node => {
      const parent = shape(mockEditor, node.parentNode!);
      const record = shape(mockEditor, node.id);
      expect(record.x - parent.x).toBe(node.position.x);
      expect(record.y - parent.y).toBe(node.position.y);
      expect(record.x + record.props.w).toBeLessThanOrEqual(parent.x + parent.props.w);
      expect(record.y + record.props.h).toBeLessThanOrEqual(parent.y + parent.props.h);
    });
    expectAnchors(mockEditor, graph!, 'horizontal');
    const before = graph!.nodes.map(node => ({ node, record: shape(mockEditor, node.id) }));
    const parent = shape(mockEditor, 'block');
    act(() => {
      mockEditor.updateShape({ id: parent.id, type: parent.type, x: parent.x + 75, y: parent.y + 40 });
      mockEditor.completeUserOperation();
    });
    settle();
    before.forEach(({ node, record }) => {
      const carried = node.id === 'block' || Boolean(node.parentNode);
      expect(shape(mockEditor, node.id)).toMatchObject({ x: record.x + (carried ? 75 : 0), y: record.y + (carried ? 40 : 0) });
    });
    children.forEach(node => expect(graph!.nodes.find(next => next.id === node.id)!.position).toEqual(node.position));
    act(() => receiveProjection(incoming));
    settle();
    before.forEach(({ node, record }) => {
      const carried = node.id === 'block' || Boolean(node.parentNode);
      expect(shape(mockEditor, node.id)).toMatchObject({ x: record.x + (carried ? 75 : 0), y: record.y + (carried ? 40 : 0) });
    });
    children.forEach(node => expect(graph!.nodes.find(next => next.id === node.id)!.position).toEqual(node.position));
    expectAnchors(mockEditor, graph!, 'horizontal');
  } finally { window.removeEventListener('canvas-graph-updated', onGraph); }
});

it('retries a presentation update that arrives during the drag echo guard', () => {
  const mounted = mountBridge(goodNet());
  try {
    act(() => {
      mounted.editor.updateShape({ id: nodeIdToShapeId('fc1'), type: 'layer', x: 600, y: 90 });
      mounted.editor.completeUserOperation();
    });
    // Do not drain the drag's timeout before the app requests another position.
    act(() => mounted.result.current.setNodes(nodes => nodes.map(node => node.id === 'fc1' ? {
      ...node, position: { x: 720, y: 130 }, data: { ...node.data, error: 'Current projection diagnostic' },
    } : node)));
    settle();
    expect(shape(mounted.editor, 'fc1')).toMatchObject({ x: 720, y: 130,
      props: { nodeData: { error: 'Current projection diagnostic' } } });
    expectAnchors(mounted.editor, mounted.result.current);
  } finally { mounted.dispose(); }
});

it('resolves a changed parent even when a child keeps the same relative position and data', () => {
  const parent = (id: string, x: number) => ({ ...layer(id, x), type: 'moduleNode', style: { width: 350, height: 400 } });
  const child = { ...layer('child', 40), position: { x: 40, y: 80 }, parentNode: 'left' };
  const graph = { nodes: [parent('left', 100), parent('right', 700), child, layer('head', 1300)],
    edges: [{ id: 'out', source: child.id, target: 'head' }] };
  const mounted = mountBridge(graph);
  try {
    expect(shape(mounted.editor, child.id)).toMatchObject({ x: 140, y: 280 });
    act(() => mounted.result.current.setNodes(nodes => nodes.map(node => node.id === child.id
      ? { ...node, parentNode: 'right' } : node)));
    settle();
    expect(shape(mounted.editor, child.id)).toMatchObject({ x: 740, y: 280, props: { nodeData: child.data } });
    expectAnchors(mounted.editor, graph);
  } finally { mounted.dispose(); }
});
