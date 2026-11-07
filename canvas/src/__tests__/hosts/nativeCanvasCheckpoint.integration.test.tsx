import { act, cleanup, renderHook } from '@testing-library/react';
import { useRef, useState } from 'react';
import type { Editor, TLShapeId } from 'tldraw';
import { edgeIdToShapeId, nodeIdToShapeId } from '../../canvas/canvasShapeId';
import type { ModuleFocusPresentation } from '../../canvas/moduleFocusPresentation';
import { postToHost } from '../../ide/bridge';
import { restoreNativePresentation, useNativeCanvasPresentation } from '../../ide/nativeCanvasPresentation';
import { createNativeCanvasTransport } from '../../ide/nativeCanvasTransport';
import { LOCAL_BROWSER_PROTOCOL_VERSION } from '../../ide/protocol.generated';
import type { ModelEdge, ModelNode } from '../../types';

// JSDOM is not framed. Observe ordinary publications separately from the real
// transport's postMessage ACK, without mocking capture or restoration.
jest.mock('../../ide/bridge', () => ({ postToHost: jest.fn() }));

const origin = 'vscode-webview://canvas';
const receive = (data: unknown, from = origin, source: Window | null = window.parent) => {
  act(() => { window.dispatchEvent(new MessageEvent('message', { data, origin: from, source })); });
};
const checkpoint = (requestId = 'hide') => receive({ type: 'canvasCheckpoint', requestId });
const node = (id: string): ModelNode => ({ id, position: { x: 10, y: 20 },
  data: { parameters: {}, expanded: true, passThroughShown: false,
    captureDetailLevel: 'operations', operationDetailsShown: true }, hidden: false, selected: true } as ModelNode);
const focusState: ModuleFocusPresentation = { scopeId: 'block', views: [
  { id: null, selected: ['block'], orientation: 'auto', direction: 'LR', layoutKey: 'root',
    camera: { x: 4, y: 5, z: 0.8 }, positions: [['block', { x: 10, y: 20 }]] },
  { id: 'block', selected: ['block.child'], orientation: 'TB', direction: 'TB',
    camera: { x: 8, y: 9, z: 1.2 }, positions: [['block.child', { x: 30, y: 40 }]] },
] };

function makeEditor() {
  let camera = { x: 0, y: 0, z: 1 };
  let selected: TLShapeId[] = [];
  const shapes = new Set<string>();
  const listeners = new Set<() => void>();
  const changed = () => listeners.forEach(listener => listener());
  const editor = {
    setCamera: jest.fn((next: typeof camera) => { camera = next; }),
    getCamera: jest.fn(() => camera),
    getSelectedShapeIds: jest.fn(() => selected),
    getShape: (id: string) => shapes.has(id) ? {} : undefined,
    select: jest.fn((...ids: TLShapeId[]) => { selected = ids; changed(); }),
    store: { listen: jest.fn((listener: () => void) => {
      listeners.add(listener);
      return () => { listeners.delete(listener); };
    }) },
  };
  return { editor: editor as unknown as Editor, calls: editor, shapes, listeners, changed,
    move: (next: typeof camera) => { camera = next; },
    select: (...ids: TLShapeId[]) => { selected = ids; } };
}

let post: jest.SpyInstance;
let transport: ReturnType<typeof createNativeCanvasTransport>;
const published = postToHost as jest.Mock;
const clearPosts = () => { post.mockClear(); published.mockClear(); };
const advance = (ms = 100) => act(() => { jest.advanceTimersByTime(ms); });
async function connect(presentation?: unknown) {
  const connected = transport.connect();
  receive({ type: 'canvasConnected', sessionId: 'native', viewId: 'view', presentation });
  await connected;
  clearPosts();
}
function expectAck(state: unknown, requestId = 'hide') {
  expect(post.mock.calls).toEqual([[{ type: 'canvasPresentation', requestId, state }, origin]]);
  expect(published).not.toHaveBeenCalled();
}

beforeEach(() => {
  window.history.replaceState({}, '', '/?view=canvas');
  jest.useFakeTimers();
  post = jest.spyOn(window.parent, 'postMessage').mockImplementation(() => {});
  published.mockClear();
  transport = createNativeCanvasTransport();
});
afterEach(() => {
  cleanup();
  transport.dispose();
  jest.clearAllTimers();
  jest.useRealTimers();
  post.mockRestore();
  window.history.replaceState({}, '', '/');
});

test('acknowledges empty and retained state when no view is mounted', async () => {
  await connect();
  checkpoint('empty');
  expectAck(undefined, 'empty');
  const retained = { graphId: 'unmounted', camera: { x: 2, y: 3, z: 0.7 }, selected: [], nodes: [] };
  restoreNativePresentation(retained);
  clearPosts();
  checkpoint();
  expectAck(retained);
  advance();
  expectAck(retained);
});

test.each([false, true])('hide sends one complete fresh ACK before the store timer (scheduled: %s)', async scheduled => {
  await connect();
  const view = makeEditor();
  const nodes = [node('block'), { ...node('block.child'), position: { x: 30, y: 40 }, hidden: true,
    selected: false, data: { ...node('block.child').data, expanded: false, passThroughShown: true,
      captureDetailLevel: 'layers' as const, operationDetailsShown: false } }];
  const focus = { isReady: () => true, capturePresentation: () => focusState, restorePresentation: jest.fn() };
  renderHook(() => {
    const fit = useRef<string | null>(null);
    const [current, setNodes] = useState(nodes);
    useNativeCanvasPresentation(`fresh-${scheduled}`, current, setNodes, view.editor, fit,
      undefined, undefined, focus, { choice: 'auto', direction: 'vertical' });
  });
  expect(published).toHaveBeenCalledTimes(1);
  clearPosts();
  if (scheduled) act(() => { view.changed(); });
  const camera = { x: 33, y: 72, z: 0.63 };
  view.move(camera);
  view.select(nodeIdToShapeId('block.child'));
  checkpoint();
  const expected = { graphId: `fresh-${scheduled}`, camera, selected: [nodeIdToShapeId('block.child')],
    moduleFocus: focusState, layout: { choice: 'auto', direction: 'vertical' }, nodes: [
      { id: 'block', x: 10, y: 20, expanded: true, hidden: false, selected: true,
        passThroughShown: false, captureDetailLevel: 'operations', operationDetailsShown: true },
      { id: 'block.child', x: 30, y: 40, expanded: false, hidden: true, selected: false,
        passThroughShown: true, captureDetailLevel: 'layers', operationDetailsShown: false },
    ] };
  expectAck(expected);
  advance(1000);
  expectAck(expected);
});

test.each(['pointerup', 'blur', 'pagehide'])('%s still publishes immediately and cancels its pending store timer', async event => {
  await connect();
  const view = makeEditor();
  renderHook(() => {
    const fit = useRef<string | null>(null);
    const [nodes, setNodes] = useState([node(event)]);
    useNativeCanvasPresentation(event, nodes, setNodes, view.editor, fit);
  });
  clearPosts();
  act(() => { view.changed(); });
  const camera = { x: 44, y: 82, z: 0.83 };
  view.move(camera);
  act(() => { window.dispatchEvent(new Event(event)); });
  expect(published.mock.calls).toEqual([[{ type: 'canvasPresentation', state: expect.objectContaining({ graphId: event, camera }) }]]);
  expect(post).not.toHaveBeenCalled();
  advance();
  expect(published).toHaveBeenCalledTimes(1);
});

test('store changes remain debounced and committed node changes publish without waiting', async () => {
  await connect();
  const view = makeEditor();
  const hook = renderHook(() => {
    const fit = useRef<string | null>(null);
    const [nodes, setNodes] = useState([node('commit')]);
    useNativeCanvasPresentation('committed-nodes', nodes, setNodes, view.editor, fit);
    return setNodes;
  });
  clearPosts();
  act(() => { view.changed(); });
  advance(60);
  act(() => { view.changed(); });
  advance(99);
  expect(published).not.toHaveBeenCalled();
  advance(1);
  expect(published).toHaveBeenCalledTimes(1);
  clearPosts();
  act(() => {
    view.changed();
    hook.result.current(nodes => nodes.map(value => ({ ...value, position: { x: 300, y: 400 },
      data: { ...value.data, expanded: false } })));
  });
  expect(published).toHaveBeenCalledTimes(1);
  expect(published).toHaveBeenLastCalledWith({ type: 'canvasPresentation', state: expect.objectContaining({
    nodes: [expect.objectContaining({ id: 'commit', x: 300, y: 400, expanded: false })],
  }) });
  expect(post).not.toHaveBeenCalled();
  advance();
  expect(published).toHaveBeenCalledTimes(1);
});

test('unauthenticated, wrong-origin, wrong-source and malformed requests cannot capture or cancel a checkpoint', async () => {
  const view = makeEditor();
  renderHook(() => {
    const fit = useRef<string | null>(null);
    const [nodes, setNodes] = useState([node('authenticated')]);
    useNativeCanvasPresentation('authenticated', nodes, setNodes, view.editor, fit);
  });
  clearPosts();
  view.calls.getCamera.mockClear();
  checkpoint('before-connect');
  receive({ type: 'canvasConnected', sessionId: '', viewId: 'view' });
  receive({ type: 'canvasConnected', sessionId: 'native' });
  checkpoint('invalid-session');
  expect(transport.session).toBeNull();
  expect(post).not.toHaveBeenCalled();
  expect(view.calls.getCamera).not.toHaveBeenCalled();
  await connect();
  const messages = jest.fn();
  transport.onMessage(messages);
  act(() => { view.changed(); });
  receive({ type: 'canvasCheckpoint', requestId: 'wrong-origin' }, 'https://other.example');
  receive({ type: 'canvasCheckpoint', requestId: 'wrong-source' }, origin, null);
  receive({ type: 'canvasCheckpoint' });
  receive({ type: 'canvasCheckpoint', requestId: 123 });
  receive({ type: 'canvasMessage', message: { protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
    workspaceSessionId: 'other', messageId: 'foreign', type: 'snapshot', role: 'host', hostSequence: 1,
    state: { activeTarget: null, projection: null, traced: false, greyed: false, syncState: null, connectionState: 'connected' } } });
  expect(messages).not.toHaveBeenCalled();
  expect(transport.protocolViolations).toBe(1);
  expect(post).not.toHaveBeenCalled();
  expect(published).not.toHaveBeenCalled();
  expect(view.calls.getCamera).not.toHaveBeenCalled();
  advance();
  expect(published).toHaveBeenCalledTimes(1);
  clearPosts();
  checkpoint('authenticated-request');
  expectAck(expect.objectContaining({ graphId: 'authenticated' }), 'authenticated-request');
});

test('pending focus returns retained state without publishing a half-restored camera', async () => {
  const retained = { graphId: 'previous-focus', camera: { x: 2, y: 3, z: 0.7 }, selected: [], nodes: [], moduleFocus: focusState };
  await connect(retained);
  const view = makeEditor();
  let ready = false;
  const focus = { isReady: () => ready, capturePresentation: jest.fn(() => focusState), restorePresentation: jest.fn() };
  renderHook(() => {
    const fit = useRef<string | null>(null);
    const [nodes, setNodes] = useState([node('focus')]);
    useNativeCanvasPresentation('pending-focus', nodes, setNodes, view.editor, fit, undefined, undefined, focus);
  });
  expect(published).not.toHaveBeenCalled();
  act(() => { view.changed(); window.dispatchEvent(new Event('blur')); view.changed(); });
  checkpoint();
  expectAck(retained);
  expect(view.calls.getCamera).not.toHaveBeenCalled();
  expect(focus.capturePresentation).not.toHaveBeenCalled();
  advance();
  expectAck(retained);
  clearPosts();
  ready = true;
  const camera = { x: 500, y: 600, z: 1.4 };
  view.move(camera);
  checkpoint('ready');
  expectAck(expect.objectContaining({ graphId: 'pending-focus', camera, moduleFocus: focusState }), 'ready');
});

test('pending child and edge selection survives hide and keeps its normal store notification', async () => {
  const edge = { id: 'inner', source: 'block.child', target: 'block' } as ModelEdge;
  const edges = [edge];
  const selected = [nodeIdToShapeId('block.child'), edgeIdToShapeId(edge)];
  const camera = { x: 1, y: 2, z: 0.8 };
  await connect({ graphId: 'pending-selection', camera, selected,
    nodes: [{ id: 'block', expanded: true }, { id: 'block.child', hidden: false, selected: true }] });
  const view = makeEditor();
  renderHook(() => {
    const fit = useRef<string | null>(null);
    const [nodes, setNodes] = useState([node('block'), node('block.child')]);
    useNativeCanvasPresentation('pending-selection', nodes, setNodes, view.editor, fit, edges);
  });
  expect(view.calls.select).not.toHaveBeenCalled();
  clearPosts();
  checkpoint('waiting-for-shapes');
  expectAck(expect.objectContaining({ camera, selected }), 'waiting-for-shapes');
  clearPosts();
  selected.forEach(id => view.shapes.add(id));
  checkpoint('shapes-ready');
  expect(view.calls.select).toHaveBeenCalledTimes(1);
  expect(view.calls.select).toHaveBeenCalledWith(...selected);
  expectAck(expect.objectContaining({ camera, selected }), 'shapes-ready');
  // Selecting the newly created shapes notifies the store during capture. Keep
  // that ordinary checkpoint, which can also observe later camera changes.
  const laterCamera = { x: 9, y: 10, z: 1.1 };
  view.move(laterCamera);
  advance();
  expect(post).toHaveBeenCalledTimes(1);
  expect(published.mock.calls).toEqual([[{ type: 'canvasPresentation', state: expect.objectContaining({ camera: laterCamera, selected }) }]]);
});

test('rerender replaces the capture and cleanup cancels stale listeners and timers', async () => {
  await connect();
  const first = makeEditor(), second = makeEditor();
  const nodes = [node('transition')];
  const hook = renderHook(({ graphId, editor }: { graphId: string; editor: Editor | null }) => {
    const fit = useRef<string | null>(null);
    const [current, setNodes] = useState(nodes);
    useNativeCanvasPresentation(graphId, current, setNodes, editor, fit);
  }, { initialProps: { graphId: 'before-rerender', editor: first.editor as Editor | null } });
  act(() => { first.changed(); });
  second.move({ x: 700, y: 800, z: 0.5 });
  clearPosts();
  hook.rerender({ graphId: 'after-rerender', editor: second.editor });
  expect(first.listeners.size).toBe(0);
  expect(second.listeners.size).toBe(1);
  expect(published).toHaveBeenCalledTimes(1);
  clearPosts();
  checkpoint();
  const state = expect.objectContaining({ graphId: 'after-rerender', camera: { x: 700, y: 800, z: 0.5 } });
  expectAck(state);
  advance();
  expectAck(state);
  act(() => { second.changed(); });
  hook.rerender({ graphId: 'after-rerender', editor: null });
  expect(second.listeners.size).toBe(0);
  clearPosts();
  checkpoint('no-editor');
  expectAck(state, 'no-editor');
  hook.unmount();
  clearPosts();
  act(() => { first.changed(); second.changed(); window.dispatchEvent(new Event('pagehide')); });
  advance();
  expect(published).not.toHaveBeenCalled();
  checkpoint('unmounted');
  expectAck(state, 'unmounted');
  clearPosts();
  transport.dispose();
  checkpoint('disposed');
  expect(post).not.toHaveBeenCalled();
  expect(published).not.toHaveBeenCalled();
});
