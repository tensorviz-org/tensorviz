import type { CapturePort, ModelEdge, ModelGraph, ModelNode } from '../../types';
import { deriveOperationGroups, operationGroupCard, revealOperationGroup } from '../../services/operationGroups';
import { reconcileModuleView } from '../../services/canvasLayout';
import { modelGraphToDocumentV2 } from '../../services/graphDocumentAdapter';
import { expandedDetailNodes } from '../../services/graphDetailLevels';
import { graphToSvg } from '../../services/graphSvg';
import { planCanvasFocus } from '../../canvas/focusSelection';
import { revealGraphNode } from '../../viewer/graphFileView';
import { operationTensorRows } from '../../canvas/operationTensorData';
import { operationDisplayName } from '../../canvas/operationDisplayName';

jest.mock('tldraw', () => ({ atom: (_name: string, value: unknown) => ({ get: () => value, set: () => {} }) }));

const port = (nodeId: string, direction: 'in' | 'out', valueId: string): CapturePort => ({
  nodeId, direction, valueId, id: `${nodeId}:${direction}`, argumentKind: 'value',
  paths: [direction === 'in' ? ['arguments', 'args', 0] : ['output']], memberPortId: null,
  shape: direction === 'in' ? [2, 8] : [8, 2], dtype: 'torch.float32', device: 'cpu',
});
const op = (id: string, name: string, input: string, output: string): ModelNode => ({
  id, type: 'layerNode', position: { x: 0, y: 0 }, data: { parentId: 'root', parameters: { _shape_provenance: 'observed' },
    shapeEvidence: 'measured', layerType: { id: 'captured_operation', name, displayName: name, category: 'misc',
      icon: 'Box', description: '', parameters: [] }, capturePorts: [port(id, 'in', input), port(id, 'out', output)],
    captureRef: { version: 1, captureId: 'run', sourceId: 'source', semanticId: id, kind: 'operation' } },
});
const edge = (source: string, target: string, valueId: string): ModelEdge => ({
  id: `${source}-${target}`, source, target, sourceHandle: `${source}:out`, targetHandle: `${target}:in`,
  data: { captureEdge: { id: `${source}-${target}`, kind: 'argument', valueId,
    sourcePortId: `${source}:out`, targetPortId: `${target}:in`, displaySourcePortId: null, displayTargetPortId: null } },
});
const graph = (): ModelGraph => {
  const root = op('root', 'Model', 'x', 'y');
  root.type = 'moduleNode'; root.data.parentId = undefined; root.data.expanded = true;
  root.data.captureRef!.kind = 'invocation'; root.data.captureDetailLevel = 'layers'; root.data.capturePorts = [];
  return { nodes: [root, op('add', 'aten.add.Tensor', 'x', 'a'), op('permute', 'aten.permute.default', 'a', 'b'),
    op('view', 'aten.view.default', 'b', 'c'), op('bmm', 'aten.bmm.default', 'c', 'y')],
    edges: [edge('add', 'permute', 'a'), edge('permute', 'view', 'b'), edge('view', 'bmm', 'c')] };
};

test('folds a tensor-layout chain with exact boundary ports and preserves canonical records', () => {
  const original = graph(), before = JSON.stringify(original);
  const layers = reconcileModuleView(original.nodes, original.edges, { relayout: true });
  expect(layers.nodes.filter(n => n.hidden).map(n => n.id)).toEqual(['permute']);
  const summary = layers.nodes.find(n => n.id === 'view')!;
  expect(summary.data.operationGroup?.members).toEqual(['permute', 'view']);
  expect(operationGroupCard(summary.data).layerType.displayName).toBe('Tensor layout');
  expect(summary.data.captureDisplayPorts).toEqual(expect.arrayContaining([
    expect.objectContaining({ id: 'boundary:permute:in', memberPortId: 'permute:in', valueId: 'a' }),
    expect.objectContaining({ id: 'view:out', valueId: 'c' }),
  ]));
  const visible = layers.edges.filter(e => !e.hidden);
  expect(visible.map(e => [e.source, e.target])).toEqual(expect.arrayContaining([['add', 'view'], ['view', 'bmm']]));
  expect(visible).toHaveLength(2);
  expect(visible.find(e => e.source === 'add')?.data?.captureEdge).toEqual(original.edges[0].data!.captureEdge);
  const operations = reconcileModuleView(expandedDetailNodes(layers.nodes, 'operations'), layers.edges, { relayout: true });
  expect(operations.nodes.every(n => !n.hidden && !n.data.operationGroup)).toBe(true);
  expect(operations.edges.filter(e => !e.hidden).map(e => e.id)).toEqual(original.edges.map(e => e.id));
  const context = { name: 'Model', revision: 1 };
  expect(modelGraphToDocumentV2(layers, context).semantic).toEqual(modelGraphToDocumentV2(operations, context).semantic);
  expect(JSON.stringify(original)).toBe(before);
  expect(graphToSvg(layers.nodes, layers.edges)).toContain('Tensor layout');
});

test.each(['failure', 'stale', 'unknown', 'effect', 'branch', 'join', 'scope', 'opaque', 'diff'])(
  '%s evidence prevents folding the chain', reason => {
    const data = graph(), first = data.nodes[2], second = data.nodes[3];
    if (reason === 'failure') first.data.error = 'shape mismatch';
    if (reason === 'stale') first.data.shapeEvidence = 'stale';
    if (reason === 'unknown') first.data.layerType.name = 'custom.reshape';
    if (reason === 'effect') data.edges[1].data!.captureEdge!.kind = 'effect';
    if (reason === 'branch') data.edges.push(edge('permute', 'bmm', 'b'));
    if (reason === 'join') data.edges.push(edge('add', 'view', 'a'));
    if (reason === 'scope') second.data.parentId = 'other-module';
    if (reason === 'opaque') first.data.capturePorts![0].argumentKind = 'opaque';
    if (reason === 'diff') first.data.diffStatus = 'modified';
    expect(deriveOperationGroups(data.nodes, data.edges).every(n => !n.data.operationGroupHiddenBy)).toBe(true);
  });

test('local reveal, error/search focus and offline navigation expose original operations', () => {
  const data = graph(), folded = reconcileModuleView(data.nodes, data.edges);
  for (const revealed of [
    reconcileModuleView(revealOperationGroup(folded.nodes, 'view'), folded.edges),
    reconcileModuleView(planCanvasFocus(folded.nodes, folded.edges, ['permute'], [])!.nodes, folded.edges),
    revealGraphNode(folded, 'permute'),
  ]) expect(revealed.nodes.every(n => !n.hidden && !n.data.operationGroup)).toBe(true);
});

function slicedGraph(): ModelGraph {
  const data = graph();
  const first = data.nodes[2], last = data.nodes[3];
  for (const node of [first, last]) {
    node.data.layerType.name = 'aten.slice.Tensor';
    node.data.captureSourceExpression = 'binding:pos-slice';
  }
  first.data.capturePorts![0].shape = [1, 12, 32];
  first.data.captureSliceAxis = 0;
  last.data.captureSliceAxis = 1;
  first.data.capturePorts![1].shape = last.data.capturePorts![0].shape = [1, 12, 32];
  last.data.capturePorts![1].shape = [1, 8, 32];
  return data;
}

test('one source slice has one card with its original input and final output, retaining both operations', () => {
  const data = slicedGraph(), before = JSON.stringify(data);
  const layers = reconcileModuleView(data.nodes, data.edges);
  const shown = layers.nodes.find(node => node.id === 'view')!;
  expect(layers.nodes.find(node => node.id === 'permute')!.hidden).toBe(true);
  expect(shown.data.operationGroup?.kind).toBe('slice');
  expect(operationDisplayName(operationGroupCard(shown.data))).toBe('Slice');
  expect(operationTensorRows(shown.data)?.map(row => [row.label, row.value])).toEqual([
    ['IN', '(1, 12, 32)'], ['OUT', '(1, 8, 32)'],
  ]);
  expect(layers.edges.filter(edge => !edge.hidden).map(edge => [edge.source, edge.target]).sort())
    .toEqual([['add', 'view'], ['view', 'bmm']]);
  const operations = reconcileModuleView(expandedDetailNodes(layers.nodes, 'operations'), layers.edges);
  expect(operations.nodes.every(node => !node.hidden && !node.data.operationGroup)).toBe(true);
  const context = { name: 'Model', revision: 1 };
  expect(modelGraphToDocumentV2(layers, context).semantic).toEqual(modelGraphToDocumentV2(operations, context).semantic);
  expect(graphToSvg(layers.nodes, layers.edges)).not.toContain('Double-click to show operations');
  expect(JSON.stringify(data)).toBe(before);
});

test('the initial Layers view folds source slices before the user changes detail level', () => {
  const data = slicedGraph();
  delete data.nodes[0].data.captureDetailLevel;
  const view = reconcileModuleView(data.nodes, data.edges);
  expect(view.nodes.find(node => node.id === 'permute')!.hidden).toBe(true);
  expect(view.nodes.find(node => node.id === 'view')!.data.operationGroup?.kind).toBe('slice');
});

test.each(['different-expression', 'missing-expression', 'different-capture', 'repeated-axis', 'layout', 'branch', 'failure', 'stale'])(
  'does not fold slices with %s evidence', reason => {
    const data = slicedGraph(), first = data.nodes[2], last = data.nodes[3];
    if (reason === 'different-expression') last.data.captureSourceExpression = 'binding:another-slice';
    if (reason === 'missing-expression') delete first.data.captureSourceExpression;
    if (reason === 'different-capture') last.data.captureRef!.captureId = 'another-run';
    if (reason === 'repeated-axis') last.data.captureSliceAxis = 0;
    if (reason === 'layout') last.data.layerType.name = 'aten.view.default';
    if (reason === 'branch') data.edges.push(edge('permute', 'bmm', 'b'));
    if (reason === 'failure') first.data.error = 'failed';
    if (reason === 'stale') first.data.shapeEvidence = 'stale';
    expect(deriveOperationGroups(data.nodes, data.edges).every(node => !node.data.operationGroup)).toBe(true);
  });
