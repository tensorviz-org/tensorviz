import React from 'react';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { NodeExplanationContext, type NodeExplanationContextValue } from '../../canvas/NodeExplanationContext';
import { GraphFocusOverlay } from '../../canvas/GraphFocusOverlay';
import { GraphQueryPanel } from '../../canvas/GraphQueryPanel';
import { graphNodes, graphSearchQuery, graphSearchResult } from '../../canvas/graphQueryState';
import { searchGraph } from '../../canvas/graphSearch';
import * as operationNames from '../../canvas/operationDisplayName';
import * as customCards from '../../canvas/customCard';
import { selectedEdgeId } from '../../canvas/edgeSelection';
import { nodeIdToShapeId } from '../../canvas/canvasShapeId';
import { LocalBrowserContext } from '../../ide/localBrowserContext';
import { ModelNode } from '../../types';

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
  mockEdges = new Map(); graphNodes.set([]); graphSearchQuery.set(''); selectedEdgeId.set(null);
  jest.clearAllMocks();
  // CRA resets mock implementations before each test, including this viewport seam.
  mockEditor.getShapePageBounds.mockReturnValue({ x: 0, y: 0, w: 200, h: 100 });
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

it('derives one selected-node label and reuses it for capture details, retaining first-node selection and last-node label semantics', () => {
  const first = node('same', 'first'); first.data.parameters.marker = 'first node';
  first.data.captureRef = { kind: 'operation', captureId: 'capture', semanticId: 'op', sourceId: 'source' } as any;
  const last = node('same', 'last'); last.data.parameters.marker = 'last node';
  graphNodes.set([first, last]); mockFocus = { ...mockFocus, active: true, selected: ['same'] };
  const get = spyLookup();
  const context = { captures: {}, page: { greyed: false, syncState: 'ready' } } as any;
  render(<LocalBrowserContext.Provider value={context}><GraphFocusOverlay /></LocalBrowserContext.Provider>);
  expect(get).toHaveBeenCalledTimes(1);
  expect(screen.getByTestId('selected-node')).toHaveTextContent('first node');
  expect(screen.getByTestId('capture-label')).toHaveTextContent('last');
  expect(within(screen.getByRole('region', { name: 'Connection focus' })).getAllByText(() => true, { selector: 'strong' })[0].textContent).toBe('last');
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

it('does no label lookup for multi-selection and preserves missing-ID and empty-string fallbacks', () => {
  graphNodes.set([node('empty', '')]); mockFocus = { ...mockFocus, active: true, selected: ['a', 'b'] };
  const get = spyLookup(); const view = render(<GraphFocusOverlay />);
  expect(screen.getByText('2 layers')).toBeInTheDocument(); expect(get).not.toHaveBeenCalled();
  mockFocus = { ...mockFocus, active: false }; mockEdges.set('edge', { source: 'empty', target: 'missing' }); selectedEdgeId.set('edge');
  view.rerender(<GraphFocusOverlay />);
  expect(within(screen.getByRole('region', { name: 'Selected connection' })).getAllByText(() => true, { selector: 'strong' })[0].textContent).toBe(' → missing');
  expect(get).toHaveBeenCalledTimes(1); // The missing node needs no derived label.
});

it('reads changed data at the same IDs and removed IDs on the next render', () => {
  graphNodes.set([node('a', 'Old')]); mockFocus = { ...mockFocus, active: true, selected: ['a'] };
  const view = render(<GraphFocusOverlay />); expect(screen.getByText('Old')).toBeInTheDocument();
  graphNodes.set([node('a', 'New')]); view.rerender(<GraphFocusOverlay />); expect(screen.getByText('New')).toBeInTheDocument();
  graphNodes.set([]); view.rerender(<GraphFocusOverlay />); expect(screen.getByText('a')).toBeInTheDocument();
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
  expect(within(screen.getByRole('region', { name: 'Connection focus' })).getAllByText(() => true, { selector: 'strong' })[0].textContent).toBe('');
  expect(screen.getByTestId('capture-label').textContent).toBe('');
  expect(get).toHaveBeenCalledTimes(1);
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
