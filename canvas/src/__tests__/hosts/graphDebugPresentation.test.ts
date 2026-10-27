import { graphDebugPresentation } from '../../ide/graphDebugPresentation';
import { mergeProjection } from '../../ide/applyProjection';
import { modelGraphToDocumentV2 } from '../../services/graphDocumentAdapter';
import { exportCurrentGraphFile } from '../../ide/graphFileExport';
import { EMPTY_LOCAL_BROWSER_PAGE_STATE } from '../../ide/localBrowserState';
import type { ModelGraph, ModelNode } from '../../types';
import type { ProjectionPayload } from '../../ide/protocol.generated';

const node = (id: string): ModelNode => ({ id, type: 'layerNode', position: { x: 20, y: 30 }, data: {
  layerType: { id: 'relu', name: 'ReLU', displayName: 'ReLU', category: 'activation', icon: 'Box', description: '', parameters: [] },
  parameters: {}, shapeEvidence: 'measured', error: 'Old error', sourceLocation: { file: 'model.py', line: 3 },
} });
const before: ModelGraph = { nodes: [node('old:a'), node('old:b')],
  edges: [{ id: 'old:edge', source: 'old:a', target: 'old:b' }] };
const document = (graph: ModelGraph) => modelGraphToDocumentV2(graph, { name: 'Net', graphId: 'g', revision: 1 });
const payload = (traced = false): ProjectionPayload => ({ graphId: 'g', revision: 1, cls: 'Net', file: 'model.py',
  sourceDigest: 'a'.repeat(64), traced, greyed: false, viewRestored: false, spanmap: {},
  document: document({ nodes: [node('new:a')], edges: [] }),
  debug: { previousDocument: document(before), nodePairs: [], changed: [] },
});

const source = (id: string, line: number): ModelNode => ({ ...node(id), data: { ...node(id).data,
  shapeEvidence: 'inferred', error: undefined, sourceLocation: { file: 'model.py', line } } });

test('accepted source architecture is visible at once as source-only, never replaced by the prior capture', () => {
  const captured: ModelGraph = { nodes: [node('cap1:fc1'), node('cap1:relu')],
    edges: [{ id: 'cap1:e', source: 'cap1:fc1', target: 'cap1:relu' }] };
  const run = graphDebugPresentation({ ...payload(true), debug: undefined }, captured, []);
  const canvas = mergeProjection(run.graph, run.existing).nodes;

  // Accept adds fc2 and Sigmoid; source sync sends a source projection that
  // still carries the last complete capture as previous-run context.
  const accepted: ModelGraph = { nodes: [source('fc1', 4), source('relu', 5), source('fc2', 6), source('sigmoid', 7)],
    edges: [{ id: 'e1', source: 'fc1', target: 'relu' }, { id: 'e2', source: 'relu', target: 'fc2' },
      { id: 'e3', source: 'fc2', target: 'sigmoid' }] };
  const synced = payload(); synced.debug!.previousDocument = document(captured);
  const result = graphDebugPresentation(synced, accepted, canvas);
  expect(result.graph.nodes.map(n => [n.id, n.data.debugEvidence, n.data.shapeEvidence, n.data.sourceLocation?.line])).toEqual([
    ['fc1', 'source', 'inferred', 4], ['relu', 'source', 'inferred', 5],
    ['fc2', 'source', 'inferred', 6], ['sigmoid', 'source', 'inferred', 7]]);
  // Source wiring only: no dashed previous-run edges, no merged capture connections.
  expect(result.graph.edges).toEqual(accepted.edges);
  // Capture cards are not source cards; nothing is matched by runtime identity.
  expect(result.existing).toEqual([]);
  const shown = mergeProjection(result.graph, result.existing);
  expect(shown.nodes.map(n => n.id).sort()).toEqual(['fc1', 'fc2', 'relu', 'sigmoid']);

  // Further typing keeps the source cards where they are.
  const moved = shown.nodes.map(n => n.id === 'fc2' ? { ...n, position: { x: 900, y: 40 } } : n);
  const typed = graphDebugPresentation(synced, accepted, moved);
  expect(mergeProjection(typed.graph, typed.existing).nodes.find(n => n.id === 'fc2')!.position).toEqual({ x: 900, y: 40 });

  // An explicit measured capture then replaces the source view with current evidence.
  const after: ModelGraph = { nodes: ['fc1', 'relu', 'fc2', 'sigmoid'].map(id => node(`cap2:${id}`)),
    edges: [{ id: 'cap2:e3', source: 'cap2:fc2', target: 'cap2:sigmoid' }] };
  const measured = payload(true);
  measured.debug = { nodePairs: [{ beforeId: 'cap1:fc1', afterId: 'cap2:fc1' }, { beforeId: 'cap1:relu', afterId: 'cap2:relu' }], changed: [] };
  const next = graphDebugPresentation(measured, after, shown.nodes);
  expect(next.graph.nodes.map(n => [n.id, n.data.debugEvidence])).toEqual([
    ['cap2:fc1', 'current'], ['cap2:relu', 'current'], ['cap2:fc2', 'current'], ['cap2:sigmoid', 'current']]);
  expect(next.graph.edges).toEqual(after.edges);
});

test('an empty source projection still shows the prior capture as explicitly historical', () => {
  const result = graphDebugPresentation(payload(), { nodes: [], edges: [] }, before.nodes);
  expect(result.graph.nodes).toHaveLength(2);
  for (const n of result.graph.nodes) {
    expect(n.data).toMatchObject({ debugEvidence: 'previous', shapeEvidence: 'stale', sourceEditCapabilities: [] });
    expect(n.data.sourceLocation).toBeUndefined(); expect(n.data.error).toBeUndefined();
  }
  expect(result.graph.edges[0].data?.debugPrevious).toBe(true);
  expect(result.graph.edges[0].label).toBe('Previous run');
});

test('partial capture keeps measured current facts and cannot invent a connection through unmatched ports', () => {
  const p = payload(true); p.debug!.nodePairs = [{ beforeId: 'old:a', afterId: 'new:a' }];
  p.debug!.changed = [{ nodeId: 'new:a', details: ['width: 8 → 4'] }];
  const old = { ...before.nodes[0], type: 'moduleNode', data: { ...before.nodes[0].data, expanded: true } };
  const current = { ...node('new:a'), type: 'moduleNode' };
  const result = graphDebugPresentation(p, { nodes: [current], edges: [] }, [old, before.nodes[1]]);
  expect(result.graph.nodes.map(n => [n.id, n.data.debugEvidence])).toEqual([
    ['new:a', 'current'], ['previous-run:old:b', 'previous']]);
  expect(result.graph.nodes[0].data.debugChanges).toEqual(['width: 8 → 4']);
  expect(result.graph.edges).toEqual([]);
  expect(mergeProjection(result.graph, result.existing).nodes[0].data.expanded).toBe(true);
});

test('recovery drops historical nodes and clears failure and change markers', () => {
  const p = payload(true); p.debug = { nodePairs: [{ beforeId: 'old:a', afterId: 'new:a' }], changed: [] };
  const current = node('new:a'); delete current.data.error;
  const result = graphDebugPresentation(p, { nodes: [current], edges: [] }, before.nodes);
  expect(result.graph.nodes).toHaveLength(1);
  expect(result.graph.nodes[0].data.error).toBeUndefined(); expect(result.graph.nodes[0].data.debugChanges).toBeUndefined();
});

test('checking a partial run retains correspondence through the source-only interval', () => {
  const p = payload(true); p.greyed = true;
  const previous = document({ ...before, nodes: before.nodes.map(node => ({ ...node, type: 'moduleNode' })) });
  p.debug!.previousDocument = previous;
  p.debug!.nodePairs = [{ beforeId: 'old:a', afterId: 'new:a' }];
  const current = { ...node('new:a'), type: 'moduleNode' };
  const expanded = { ...current, position: { x: 123, y: 456 }, data: { ...current.data, expanded: true } };
  const checking = graphDebugPresentation(p, { nodes: [current], edges: [] }, [expanded]);
  const checked = mergeProjection(checking.graph, checking.existing);
  const sourcePayload = payload(); sourcePayload.debug!.previousDocument = previous;
  const source = graphDebugPresentation(sourcePayload, { nodes: [], edges: [] }, checked.nodes);
  const sourceNodes = mergeProjection(source.graph, source.existing).nodes;
  const retained = sourceNodes.find(node => node.data.debugPreviousId === 'old:a')!;
  expect(retained.position).toEqual(expanded.position);
  expect(retained.data.expanded).toBe(true);
  expect(checked.nodes[0].data.debugPreviousId).toBe('old:a');
  const next = graphDebugPresentation({ ...p, greyed: false }, { nodes: [current], edges: [] }, sourceNodes);
  expect(mergeProjection(next.graph, next.existing).nodes[0].data.expanded).toBe(true);
});

test('graph export uses the current document, never historical display nodes', () => {
  const p = payload();
  const page = { ...EMPTY_LOCAL_BROWSER_PAGE_STATE, projection: p, syncState: 'source-only' as const,
    activeTarget: { workspaceFolder: '/workspace', cls: 'Net', file: 'model.py', documentVersion: 1, backendRevision: 1, sourceDigest: p.sourceDigest } };
  const presented = graphDebugPresentation(p, { nodes: [node('new:a')], edges: [] }, before.nodes).graph;
  const exported = exportCurrentGraphFile(presented, page, p);
  expect(exported.document.semantic.nodes.map(n => n.node_id)).toEqual(['new:a']);
  expect(exported.document.semantic.edges).toEqual([]);
  expect(JSON.stringify(exported)).not.toContain('previous-run:');
});
