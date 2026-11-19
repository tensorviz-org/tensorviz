import React from 'react';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { NodeExplanationContext, type NodeExplanationContextValue } from '../../canvas/NodeExplanationContext';
import { GraphFocusOverlay } from '../../canvas/GraphFocusOverlay';
import { GraphQueryPanel } from '../../canvas/GraphQueryPanel';
import { graphNodes, graphSceneNodes, graphSearchQuery, graphSearchResult } from '../../canvas/graphQueryState';
import { searchGraph } from '../../canvas/graphSearch';
import * as operationNames from '../../canvas/operationDisplayName';
import * as customCards from '../../canvas/customCard';
import { selectedEdgeId } from '../../canvas/edgeSelection';
import { nodeIdToShapeId } from '../../canvas/canvasShapeId';
import { LocalBrowserContext } from '../../ide/localBrowserContext';
import { ModelEdge, ModelNode } from '../../types';
import { projectModuleFocus } from '../../services/moduleFocus';

// This seam does not assert real camera events or reactive computed caching.
// Each component owns its lazy index for one render; no shared cache is used.
jest.mock('tldraw', () => ({
  atom: (_name: string, initial: unknown) => { let value = initial; return { get: () => value, set: (next: unknown) => { value = next; } }; },
  computed: (_name: string, read: () => unknown) => ({ get: read }),
  track: (component: unknown) => component,
  useEditor: () => mockEditor,
}));
let mockFocus: any;
let mockEdges = new Map<string, any>();
const mockEditor = { getEfficientZoomLevel: () => 1, markEventAsHandled: jest.fn(),
  getShape: jest.fn(), getContainer: jest.fn(), selectNone: jest.fn(),
  pageToViewport: (point: { x: number; y: number }) => point,
  getViewportScreenBounds: () => ({ w: 900, h: 700 }), getSelectionPageBounds: () => ({ x: 0, y: 0, w: 200, h: 100 }),
  getShapePageBounds: jest.fn(() => ({ x: 0, y: 0, w: 200, h: 100 })), select: jest.fn(), zoomToBounds: jest.fn() };
jest.mock('../../canvas/graphFocus', () => ({ graphFocus: () => ({
  focus: { get: () => mockFocus }, index: { get: () => ({ edges: mockEdges }) },
}), setTraceDirection: jest.fn() }));
jest.mock('../../canvas/BlockSummary', () => ({
  FamilyLegend: () => null,
  BlockSummary: ({ node }: any) => <span data-testid="selected-node">{node.data.parameters.marker}</span>,
}));
// Label lookup counts exclude the separate source-understanding inspector.
jest.mock('../../canvas/NodeUnderstanding', () => ({ NodeUnderstanding: () => null }));
jest.mock('../../canvas/CaptureDetails', () => ({ CaptureDetails: ({ label }: any) => <span data-testid="capture-label">{label}</span> }));
const node = (id: string, label = id, extra: Partial<ModelNode> = {}): ModelNode => ({
  id, type: 'layerNode', position: { x: 0, y: 0 },
  data: { layerType: { id: 'linear', name: 'Linear' } as any, parameters: {}, captureLabel: label }, ...extra,
});
const spyLookup = () => jest.spyOn(operationNames, 'operationDisplayName');
// Narrow to the array method under test for the locked Jest typings.
const spyIndexBuild = (nodes: Pick<ModelNode[], 'map'>) => jest.spyOn(nodes, 'map');
beforeEach(() => {
  mockFocus = { active: false, selected: [], edgeIds: new Set(), direction: 'local', truncated: false };
  mockEdges = new Map(); graphNodes.set([]); graphSceneNodes.set([]); graphSearchQuery.set(''); selectedEdgeId.set(null);
  jest.clearAllMocks();
  // CRA resets mock implementations before each test, including this viewport seam.
  mockEditor.getShapePageBounds.mockReturnValue({ x: 0, y: 0, w: 200, h: 100 });
  mockEditor.getContainer.mockReturnValue(document.createElement('div'));
});
afterEach(() => { cleanup(); jest.restoreAllMocks(); });

it('does not request the index or derive labels for empty UI, including repeated camera-like renders of a large graph', () => {
  const nodes = Array.from({ length: 16479 }, (_, i) => node(`n${i}`, `N${i}`, { hidden: i >= 185 }));
  graphNodes.set(nodes);
  const map = spyIndexBuild(nodes);
  const get = spyLookup();
  const view = render(<GraphFocusOverlay />);
  for (let i = 0; i < 4; i++) view.rerender(<GraphFocusOverlay />);
  expect(get).not.toHaveBeenCalled();
  expect(map).not.toHaveBeenCalled();
  expect(screen.getByRole('searchbox', { name: 'Search the graph' })).toBeInTheDocument();
});

it('keeps selection compact until Inspect is explicitly requested', () => {
  const selected = node('layer', 'projection'); selected.data.parameters.marker = 'Full node details';
  graphNodes.set([selected]); mockFocus = { ...mockFocus, active: true, selected: ['layer'] };
  render(<GraphFocusOverlay />);
  expect(screen.getByRole('toolbar', { name: 'Node actions: projection' })).toBeVisible();
  expect(screen.queryByTestId('selected-node')).not.toBeInTheDocument();
  expect(screen.queryByRole('group', { name: 'Trace direction' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Inspect' }));
  expect(screen.getByRole('complementary', { name: 'Node inspector' })).toBeVisible();
  expect(screen.getByTestId('selected-node')).toHaveTextContent('Full node details');
  fireEvent.click(screen.getByRole('button', { name: 'Back to graph' }));
  expect(screen.queryByRole('complementary', { name: 'Node inspector' })).not.toBeInTheDocument();
  expect(mockEditor.zoomToBounds).not.toHaveBeenCalled();
});

it('exposes Explain and Chat directly without mounting the inspector or repeating known shapes', () => {
  const selected = node('fc', 'projection');
  selected.data.inputShape = [1, 8, 768]; selected.data.outputShape = [1, 8, 768]; selected.data.shapeEvidence = 'measured';
  graphNodes.set([selected]); mockFocus = { ...mockFocus, active: true, selected: ['fc'] };
  const start = jest.fn(() => 'node-pass'), compose = jest.fn();
  window.addEventListener('tensorviz-agent-compose', compose);
  render(<NodeExplanationContext.Provider value={{ states: {}, retry: jest.fn(), reveal: jest.fn(),
    model: { start, cancel: jest.fn(), toggle: jest.fn() } }}><GraphFocusOverlay /></NodeExplanationContext.Provider>);
  expect(screen.queryByLabelText('Known tensor shapes')).not.toBeInTheDocument();
  expect(screen.queryByText('(1, 8, 768)')).not.toBeInTheDocument();
  expect(screen.queryByRole('complementary', { name: 'Node inspector' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Discuss in chat' }));
  expect(compose).toHaveBeenCalledTimes(1);
  expect(start).not.toHaveBeenCalled();
  const explain = screen.getByRole('button', { name: 'Explain this component' });
  fireEvent.click(explain);
  expect(start).toHaveBeenCalledTimes(1);
  expect(start).toHaveBeenCalledWith(['fc']);
  expect(screen.getByRole('button', { name: 'Stop explaining' })).toBe(explain);
  expect(explain).toHaveAttribute('data-state', 'preparing');
  expect(screen.queryByRole('complementary', { name: 'Node inspector' })).not.toBeInTheDocument();
  expect(screen.queryByText('Preparing explanation…')).not.toBeInTheDocument();
  expect(mockEditor.zoomToBounds).not.toHaveBeenCalled();
  window.removeEventListener('tensorviz-agent-compose', compose);
});

it('labels clearing the selection instead of showing an unexplained cross', () => {
  graphNodes.set([node('fc', 'projection')]); mockFocus = { ...mockFocus, active: true, selected: ['fc'] };
  render(<GraphFocusOverlay />);
  fireEvent.click(screen.getByRole('button', { name: 'More node actions' }));
  const clear = screen.getByRole('button', { name: 'Clear selection' });
  expect(clear).toHaveTextContent('Clear selection');
  fireEvent.click(clear);
  expect(mockEditor.selectNone).toHaveBeenCalledTimes(1);
});

it('reveals an explicitly clicked connected node in the graph while keeping desktop inspection open', () => {
  const source = node('source', 'Source'), peer = node('peer', 'Destination');
  graphNodes.set([source, peer]); mockFocus = { ...mockFocus, active: true, selected: ['source'] };
  mockEditor.getShape.mockReturnValue({ id: nodeIdToShapeId('peer') });
  jest.spyOn(require('../../canvas/NodeUnderstanding'), 'NodeUnderstanding').mockImplementation(({ onNavigate }: any) =>
    <button onClick={() => onNavigate('peer')}>Show destination in graph</button>);
  render(<GraphFocusOverlay />);
  fireEvent.click(screen.getByRole('button', { name: 'Inspect' }));
  expect(mockEditor.zoomToBounds).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Show destination in graph' }));
  expect(mockEditor.select).toHaveBeenCalledWith(nodeIdToShapeId('peer'));
  expect(mockEditor.zoomToBounds).toHaveBeenCalledWith({ x: 0, y: 0, w: 200, h: 100 },
    expect.objectContaining({ targetZoom: 1 }));
  expect(screen.getByRole('complementary', { name: 'Node inspector' })).toBeVisible();
});

it('returns to the graph before revealing a connected node in a narrow inspector', () => {
  graphNodes.set([node('source', 'Source'), node('peer', 'Destination')]);
  mockFocus = { ...mockFocus, active: true, selected: ['source'] };
  const host = document.createElement('div'); host.className = 'tldraw-canvas-wrapper'; document.body.appendChild(host);
  jest.spyOn(host, 'getBoundingClientRect').mockReturnValue({ width: 480 } as DOMRect);
  mockEditor.getContainer.mockReturnValue(host);
  mockEditor.getShape.mockReturnValue({ id: nodeIdToShapeId('peer') });
  let frame: FrameRequestCallback | undefined;
  jest.spyOn(window, 'requestAnimationFrame').mockImplementation(callback => { frame = callback; return 1; });
  jest.spyOn(require('../../canvas/NodeUnderstanding'), 'NodeUnderstanding').mockImplementation(({ onNavigate }: any) =>
    <button onClick={() => onNavigate('peer')}>Show destination in graph</button>);
  render(<GraphFocusOverlay />, { container: host });
  fireEvent.click(screen.getByRole('button', { name: 'Inspect' }));
  fireEvent.click(screen.getByRole('button', { name: 'Show destination in graph' }));
  expect(screen.queryByRole('complementary', { name: 'Node inspector' })).not.toBeInTheDocument();
  expect(mockEditor.zoomToBounds).not.toHaveBeenCalled();
  act(() => frame?.(0));
  expect(mockEditor.zoomToBounds).toHaveBeenCalledTimes(1);
});

it('requires a separate Trace action and never changes the camera when revealing controls', () => {
  graphNodes.set([node('fc', 'projection')]); mockFocus = { ...mockFocus, active: true, selected: ['fc'] };
  render(<GraphFocusOverlay />);
  expect(screen.queryByRole('group', { name: 'Trace direction' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Trace' }));
  expect(screen.getByRole('group', { name: 'Trace direction' })).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Trace upstream inputs' }));
  expect(require('../../canvas/graphFocus').setTraceDirection).toHaveBeenCalledWith(mockEditor, 'upstream');
  expect(mockEditor.zoomToBounds).not.toHaveBeenCalled();
});

it('updates the open inspector but never carries an edit form to another selection', () => {
  const first = node('fc', 'projection'), second = node('act', 'activation');
  first.data.sourceLocation = { file: 'model.py', line: 4 }; first.data.parameters.out_features = 8;
  second.data.sourceLocation = { file: 'model.py', line: 5 }; second.data.parameters.out_features = 16;
  graphNodes.set([first, second]); mockFocus = { ...mockFocus, active: true, selected: ['fc'] };
  const local = { page: {}, sourceEdits: { propose: jest.fn() } } as any;
  const view = render(<LocalBrowserContext.Provider value={local}><GraphFocusOverlay /></LocalBrowserContext.Provider>);
  fireEvent.click(screen.getByRole('button', { name: 'Inspect' }));
  fireEvent.click(screen.getByRole('button', { name: 'Edit source' }));
  expect(screen.getByLabelText('out_features')).toHaveValue('8');
  mockFocus = { ...mockFocus, selected: ['act'] };
  view.rerender(<LocalBrowserContext.Provider value={local}><GraphFocusOverlay /></LocalBrowserContext.Provider>);
  expect(screen.getByRole('complementary', { name: 'Node inspector' })).toHaveTextContent('activation');
  expect(screen.queryByRole('region', { name: 'Edit model source' })).not.toBeInTheDocument();
  expect(local.sourceEdits.propose).not.toHaveBeenCalled();
});

it('derives one label per render and reveals retained details only after inspection', () => {
  const first = node('same', 'first'); first.data.parameters.marker = 'first node';
  first.data.captureRef = { kind: 'operation', captureId: 'capture', semanticId: 'op', sourceId: 'source' } as any;
  const last = node('same', 'last'); last.data.parameters.marker = 'last node';
  graphNodes.set([first, last]); mockFocus = { ...mockFocus, active: true, selected: ['same'] };
  const get = spyLookup();
  const context = { captures: {}, page: { greyed: false, syncState: 'ready' } } as any;
  render(<LocalBrowserContext.Provider value={context}><GraphFocusOverlay /></LocalBrowserContext.Provider>);
  expect(get).toHaveBeenCalledTimes(1);
  expect(screen.getByRole('toolbar', { name: 'Node actions: last' })).toBeVisible();
  expect(screen.queryByTestId('capture-label')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Inspect' }));
  expect(screen.getByTestId('selected-node')).toHaveTextContent('first node');
  expect(screen.queryByTestId('capture-label')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Captured calls & tensors' }));
  expect(screen.getByTestId('capture-label')).toHaveTextContent('last');
  expect(get).toHaveBeenCalledTimes(2);
});

it('preserves custom-title and ATen label precedence and performs only two selected-edge lookups', () => {
  const custom = node('custom', 'fallback'); custom.data.layerType.id = 'custom_code'; custom.data.parameters = { _class: 'MyBlock' };
  const aten = node('aten', 'aten.add.Tensor');
  graphNodes.set([custom, aten]); mockEdges.set('edge', { source: 'custom', target: 'aten' }); selectedEdgeId.set('edge');
  const get = spyLookup();
  const customLookup = jest.spyOn(customCards, 'customCardModel');
  render(<GraphFocusOverlay />);
  expect(screen.getByText('MyBlock → Add')).toBeInTheDocument();
  expect(customLookup).toHaveBeenCalledTimes(2);
  expect(get).toHaveBeenCalledTimes(1); // Custom title short-circuits the operation label.
});

it('does no label lookup for multi-selection and omits unavailable connection identities', () => {
  graphNodes.set([node('empty', '')]); mockFocus = { ...mockFocus, active: true, selected: ['a', 'b'] };
  const get = spyLookup(); const view = render(<GraphFocusOverlay />);
  expect(screen.getByText('2 selected')).toBeInTheDocument(); expect(get).not.toHaveBeenCalled();
  mockFocus = { ...mockFocus, active: false }; mockEdges.set('edge', { source: 'empty', target: 'missing' }); selectedEdgeId.set('edge');
  view.rerender(<GraphFocusOverlay />);
  expect(within(screen.getByRole('region', { name: 'Selected connection' })).queryByText('missing')).not.toBeInTheDocument();
  expect(screen.getByRole('region', { name: 'Selected connection' })).not.toHaveTextContent('→');
  expect(get).toHaveBeenCalledTimes(1); // The missing node needs no derived label.
});

it('reads changed data at the same IDs and removed IDs on the next render', () => {
  graphNodes.set([node('a', 'Old')]); mockFocus = { ...mockFocus, active: true, selected: ['a'] };
  const view = render(<GraphFocusOverlay />); expect(screen.getByRole('toolbar', { name: 'Node actions: Old' })).toBeInTheDocument();
  graphNodes.set([node('a', 'New')]); view.rerender(<GraphFocusOverlay />); expect(screen.getByRole('toolbar', { name: 'Node actions: New' })).toBeInTheDocument();
  graphNodes.set([]); view.rerender(<GraphFocusOverlay />); expect(screen.queryByText('a')).not.toBeInTheDocument();
});

it('inspects a displayed boundary using its recorded scene data without adding model authority', () => {
  const id = 'focus-boundary:%5B%22invocation%3Aprivate-id%22%5D';
  const boundary = node(id, 'From activation', { type: 'inputNode' });
  boundary.data.focusBoundary = { direction: 'in', externalNodeId: 'activation' };
  boundary.data.layerType = { ...boundary.data.layerType, id: 'input', name: 'input' };
  boundary.data.parameters.marker = 'Recorded boundary';
  graphSceneNodes.set([boundary]);
  mockFocus = { ...mockFocus, active: true, selected: [id] };
  render(<GraphFocusOverlay />);
  expect(screen.getByRole('toolbar', { name: 'Node actions: From activation' })).toBeVisible();
  expect(screen.queryByText(id)).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Inspect' }));
  expect(screen.getByRole('complementary', { name: 'Node inspector' })).toHaveTextContent('From activation');
  expect(screen.queryByRole('button', { name: 'Edit source' })).not.toBeInTheDocument();
  expect(graphNodes.get()).toEqual([]);
});

it('never substitutes a missing boundary ID for its selection title', () => {
  const id = 'focus-boundary:%5B%22invocation%3Aprivate-id%22%5D';
  mockFocus = { ...mockFocus, active: true, selected: [id] };
  render(<GraphFocusOverlay />);
  expect(screen.queryByText(id)).not.toBeInTheDocument();
  expect(screen.getByRole('toolbar', { name: 'Node actions' })).not.toHaveTextContent('Input 1');
});

it('limits search label lookups to eight groups and three inner hits, includes hidden nodes and retains click behavior', () => {
  const nodes: ModelNode[] = [], groups: any[] = [];
  for (let i = 0; i < 10; i++) {
    const id = `group${i}`; nodes.push(node(id, `Group ${i}`));
    const hits = Array.from({ length: 5 }, (_, j) => ({ id: `${id}.child${j}`, field: 'label', matched: 'child', visibleId: id }));
    nodes.push(...hits.map(hit => node(hit.id, hit.id, { hidden: true, parentNode: id })));
    groups.push({ visibleId: id, hits });
  }
  graphNodes.set(nodes); graphSearchQuery.set('child');
  jest.spyOn(graphSearchResult, 'get').mockReturnValue({ query: 'child', groups, hits: groups.flatMap(g => g.hits), visibleIds: new Set(groups.map(g => g.visibleId)) } as any);
  const get = spyLookup(); const map = spyIndexBuild(nodes); render(<GraphQueryPanel />);
  expect(get).toHaveBeenCalledTimes(32);
  expect(map).toHaveBeenCalledTimes(1);
  expect(screen.queryByText('Group 8')).toBeNull();
  expect(screen.getByText(/5 inside · group0.child0, group0.child1, group0.child2/)).toBeInTheDocument();
  fireEvent.click(screen.getByText('Group 0'));
  expect(mockEditor.select).toHaveBeenCalledWith(nodeIdToShapeId('group0'));
  expect(mockEditor.zoomToBounds).toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Clear search' })); expect(graphSearchQuery.get()).toBe('');
});

it('search preserves empty labels and missing scoped-hit fallback without custom-card titles', () => {
  const custom = node('group', ''); custom.data.layerType.id = 'custom_code'; custom.data.parameters = { _class: 'Ignored search title' };
  graphNodes.set([custom]); graphSearchQuery.set('match');
  jest.spyOn(graphSearchResult, 'get').mockReturnValue({ query: 'match', groups: [{ visibleId: 'group', hits: [{ id: 'group.child', matched: 'match' }] }], hits: [{}], visibleIds: new Set(['group']) } as any);
  render(<GraphQueryPanel />);
  expect(screen.queryByText('Ignored search title')).toBeNull();
  expect(within(screen.getByRole('list')).getAllByText(() => true, { selector: 'strong' })[0].textContent).toBe('');
  expect(screen.getByText(/1 inside · child/)).toBeInTheDocument();
});


it('displays actual hidden-node search grouping without expanding the module', () => {
  const parent = node('block', 'Block', { type: 'moduleNode' });
  const child = node('block.child', 'Needle', { hidden: true, parentNode: 'block' });
  graphNodes.set([parent, child]); graphSearchQuery.set('Needle');
  const result = searchGraph(graphNodes.get(), 'Needle');
  expect(result.groups.map(group => group.visibleId)).toEqual(['block']);
  jest.spyOn(graphSearchResult, 'get').mockReturnValue(result);
  const before = JSON.stringify([parent, child]);
  render(<GraphQueryPanel />);
  expect(screen.getByText('Block')).toBeInTheDocument();
  expect(screen.getByText(/1 inside · Needle/)).toBeInTheDocument();
  fireEvent.click(screen.getByText('Block'));
  expect(JSON.stringify([parent, child])).toBe(before);
});

it('preserves an empty selected caption in local capture details without substituting its ID', () => {
  const selected = node('empty', '');
  selected.data.captureRef = { kind: 'operation', captureId: 'capture', semanticId: 'op', sourceId: 'source' } as any;
  graphNodes.set([selected]); mockFocus = { ...mockFocus, active: true, selected: ['empty'] };
  const get = spyLookup();
  const context = { captures: {}, page: { greyed: false, syncState: 'ready' } } as any;
  render(<LocalBrowserContext.Provider value={context}><GraphFocusOverlay /></LocalBrowserContext.Provider>);
  expect(screen.getByRole('toolbar', { name: 'Node actions' })).toBeVisible();
  expect(get).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole('button', { name: 'Captured calls & tensors' }));
  expect(screen.getByTestId('capture-label').textContent).toBe('');
  expect(get).toHaveBeenCalledTimes(2);
});

it('refreshes displayed query labels after graph replacement and clear', () => {
  graphNodes.set([node('group', 'Old')]); graphSearchQuery.set('match');
  jest.spyOn(graphSearchResult, 'get').mockReturnValue({ query: 'match', groups: [{ visibleId: 'group', hits: [{ id: 'group', matched: 'match' }] }], hits: [{}], visibleIds: new Set(['group']) } as any);
  const view = render(<GraphQueryPanel />);
  expect(screen.getByText('Old')).toBeInTheDocument();
  graphNodes.set([node('group', 'New')]); view.rerender(<GraphQueryPanel />);
  expect(screen.getByText('New')).toBeInTheDocument(); expect(screen.queryByText('Old')).toBeNull();
  graphNodes.set([]); view.rerender(<GraphQueryPanel />);
  expect(within(screen.getByRole('list')).getAllByText(() => true, { selector: 'strong' })[0].textContent).toBe('group');
  expect(screen.queryByText('New')).toBeNull();
});

it('uses ready explanation titles lazily while preserving technical labels for other connections', () => {
  const nodes = [node('score', '_score_hidden'), node('output', 'candidate scores')];
  graphNodes.set(nodes);
  const explanation = { requestId: 'explain', nodeId: 'score', revision: 'revision', status: 'ready' as const,
    explanation: { title: 'Score candidates', purpose: 'Compare each candidate with the question.', inputs: [], outputs: [], steps: [], uncertainties: [], citations: [] } };
  const context: NodeExplanationContextValue = { states: {}, annotations: { score: explanation }, retry: jest.fn(), reveal: jest.fn() };
  const map = spyIndexBuild(nodes), get = spyLookup();
  const view = render(<NodeExplanationContext.Provider value={context}><GraphFocusOverlay /></NodeExplanationContext.Provider>);
  expect(map).not.toHaveBeenCalled();
  expect(get).not.toHaveBeenCalled();
  mockEdges.set('edge', { source: 'score', target: 'output' }); selectedEdgeId.set('edge');
  view.rerender(<NodeExplanationContext.Provider value={context}><GraphFocusOverlay /></NodeExplanationContext.Provider>);
  expect(screen.getByText('Score candidates → candidate scores')).toBeInTheDocument();
  expect(map).toHaveBeenCalledTimes(1);
  expect(get).toHaveBeenCalledTimes(1);
});

/** Module focus: the canvas draws the scope's children plus presentation-only boundary
 *  cards, while graphNodes keeps the authoritative graph, which has no boundary nodes. */
function focusedSqueezeExcitation() {
  const module = node('se', 'SqueezeExcitation', { type: 'moduleNode' });
  module.data = { ...module.data, layerType: { id: 'module', name: 'Module' } as any,
    parameters: { name: 'se', _module_path: 'se' }, expanded: false, moduleInstanceId: 'se' };
  const conv = node('se.expand', 'aten.convolution.default');
  conv.data = { ...conv.data, parentId: 'se', parameters: { name: 'expand', _module_path: 'se.expand' } };
  const nodes = [node('activation', 'activation'), module, conv, node('sigmoid', 'sigmoid')];
  const edges: ModelEdge[] = [{ id: 'a>c', source: 'activation', target: 'se.expand' }, { id: 'c>s', source: 'se.expand', target: 'sigmoid' }];
  const scene = projectModuleFocus(nodes, edges, 'se')!;
  const shapes = new Map(scene.nodes.map(item => [nodeIdToShapeId(item.id), { props: { nodeData: item.data } }]));
  mockEditor.getShape.mockImplementation((id: string) => shapes.get(id as any));
  graphNodes.set(nodes);
  graphSceneNodes.set(scene.nodes);
  const boundary = (direction: 'in' | 'out') => scene.nodes.find(item => item.data.focusBoundary?.direction === direction)!;
  return { inbound: boundary('in'), outbound: boundary('out') };
}

it('labels a selected module-focus boundary card by its card, never its internal ID', () => {
  const { inbound } = focusedSqueezeExcitation();
  expect(inbound.id).toMatch(/^focus-boundary:/);
  mockFocus = { ...mockFocus, active: true, selected: [inbound.id], edgeIds: new Set(['a>c']) };
  render(<GraphFocusOverlay />);
  const toolbar = screen.getByRole('toolbar', { name: 'Node actions: From activation' });
  fireEvent.click(within(toolbar).getByRole('button', { name: 'Trace' }));
  expect(screen.getByText('1 connection')).toBeInTheDocument();
  fireEvent.click(within(toolbar).getByRole('button', { name: 'Inspect' }));
  expect(screen.getByRole('heading', { name: 'From activation' })).toBeInTheDocument();
  expect(graphNodes.get().some(item => item.id === inbound.id)).toBe(false);
  expect(document.body.textContent).not.toContain('focus-boundary');
});

it('labels a selected connection that crosses the focused module boundary', () => {
  const { inbound, outbound } = focusedSqueezeExcitation();
  // Keep #106's editor-card fallback covered independently of the scene index.
  graphSceneNodes.set([]);
  mockEdges.set('in', { source: inbound.id, target: 'se.expand' });
  mockEdges.set('out', { source: 'se.expand', target: outbound.id });
  selectedEdgeId.set('in');
  const view = render(<GraphFocusOverlay />);
  expect(screen.getByText('From activation → Convolution')).toBeInTheDocument();
  selectedEdgeId.set('out'); view.rerender(<GraphFocusOverlay />);
  expect(screen.getByText('Convolution → To sigmoid')).toBeInTheDocument();
  expect(document.body.textContent).not.toContain('focus-boundary');
});

it('counts one connection in the singular and several in the plural', () => {
  graphNodes.set([node('a', 'A')]);
  mockFocus = { ...mockFocus, active: true, selected: ['a'], edgeIds: new Set(['e1', 'e2']) };
  render(<GraphFocusOverlay />);
  fireEvent.click(screen.getByRole('button', { name: 'Trace' }));
  expect(screen.getByText('2 connections')).toBeInTheDocument();
});
