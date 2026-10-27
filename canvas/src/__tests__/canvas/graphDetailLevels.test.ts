import type { ModelNode } from '../../types';
import { expandedDetailNodes, graphDetailLevel } from '../../services/graphDetailLevels';
import { modelGraphToDocumentV2, graphDocumentV2ToModelGraph } from '../../services/graphDocumentAdapter';

const node = (id: string, parentId?: string, kind: 'invocation' | 'operation' = 'invocation'): ModelNode => ({
  id, type: kind === 'invocation' ? 'moduleNode' : 'layerNode', position: { x: 0, y: 0 },
  data: { parentId, expanded: false, parameters: {},
    layerType: { id: kind === 'invocation' ? 'module' : 'captured_operation', name: id,
      displayName: id, category: 'misc', description: '', icon: 'Box', parameters: [] },
    captureRef: { version: 1, captureId: 'run', sourceId: 'source', semanticId: id, kind } },
});

const graph = () => [node('model'), node('block', 'model'), node('linear', 'block'),
  node('mm', 'linear', 'operation'), node('activation', 'block'), node('gelu', 'activation', 'operation'),
  node('residual', 'block', 'operation'), node('custom', 'model'), node('pow', 'custom', 'operation')];

test('Layers opens composition at unequal depths while keeping computational calls compact', () => {
  const nodes = graph();
  const before = JSON.stringify(nodes);
  const shown = expandedDetailNodes(nodes, 'layers');
  expect(shown.filter(n => n.data.expanded).map(n => n.id)).toEqual(['model', 'block']);
  expect(shown.map(n => [n.id, n.data.parentId, n.data.captureRef])).toEqual(nodes.map(n => [n.id, n.data.parentId, n.data.captureRef]));
  expect(JSON.stringify(nodes)).toBe(before);
});

test('All operations opens every module; a failed leaf remains open in Layers', () => {
  const nodes = graph();
  nodes.find(n => n.id === 'mm')!.data.error = 'incompatible dimensions';
  expect(expandedDetailNodes(nodes, 'operations').filter(n => n.data.expanded).map(n => n.id))
    .toEqual(['model', 'block', 'linear', 'activation', 'custom']);
  expect(expandedDetailNodes(nodes, 'layers').find(n => n.id === 'linear')!.data.expanded).toBe(true);
});

test('detail choice round-trips as view state without changing canonical semantics', () => {
  const nodes = graph();
  const context = { name: 'Model', revision: 1 };
  const before = modelGraphToDocumentV2({ nodes, edges: [] }, context);
  const after = modelGraphToDocumentV2({ nodes: expandedDetailNodes(nodes, 'operations'), edges: [] }, context);
  expect(after.semantic).toEqual(before.semantic);
  expect(graphDetailLevel(graphDocumentV2ToModelGraph(after).nodes)).toBe('operations');
  expect(graphDetailLevel(nodes)).toBe('layers');
});

test('source-only graphs keep their existing expansion policy', () => {
  const nodes = graph().map(n => ({ ...n, data: { ...n.data, captureRef: undefined } }));
  expect(expandedDetailNodes(nodes, 'layers')).toBe(nodes);
});
