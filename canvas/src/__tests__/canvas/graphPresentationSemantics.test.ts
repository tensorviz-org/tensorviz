import { changedDimensions, residualMetadata, tensorEvidence, tensorShapes, coverageLabel } from '../../canvas/graphSemantics';
import { createGraphFile, parseGraphFile } from '../../services/graphFile';
import { graphDocumentV2ToModelGraph } from '../../services/graphDocumentAdapter';
import { reconcileModuleView } from '../../services/canvasLayout';
import { nestedGraph } from '../../test/fixtures/graphFileFixture';

it('does not claim a shape was measured merely because values exist', () => {
  expect(tensorEvidence({ parameters: {}, inputShape: [1, 8] })).toBe('inferred');
  expect(tensorEvidence({ parameters: { _shape_provenance: 'observed', _out_shape: '1x4' }, outputShape: [1, 4] })).toBe('measured');
  expect(tensorEvidence({ parameters: { _shape_provenance: 'observed' }, outputShape: [1, 4] })).toBe('unknown');
  expect(tensorEvidence({ parameters: { _shape_provenance: 'observed' }, outputShape: [1, 4], shapeEvidence: 'stale' })).toBe('stale');
  expect(tensorEvidence({ parameters: { _shape_provenance: 'unreached' }, outputShape: [1, 4] })).toBe('unknown');
  expect(tensorEvidence({ parameters: {} })).toBe('unknown');
});

it('JAX state shapes come from inferred provenance instead of cached measurements', () => {
  const data = { parameters: { _shape_provenance: 'inferred', _out_shape: [3, 8] },
    outputShape: [1, 4], shapeEvidence: 'measured' as const };
  expect(tensorShapes(data).outputShape).toEqual([3, 8]);
  expect(tensorEvidence(data)).toBe('inferred');
});

it.each(['not_observed', 'capture_limited', 'static_only', 'unresolved', 'failed'])(
  'canonical %s coverage cannot inherit a cached/inferred output shape', status => {
    const data = { parameters: { _shape_provenance: status, _out_shape: '2x4', _capture_coverage: { status } },
      outputShape: [2, 4], shapeEvidence: 'measured' as const };
    expect(tensorShapes(data).outputShape).toBeUndefined();
    expect(tensorEvidence(data)).toBe('unknown');
    expect(coverageLabel(data)).toBeTruthy();
  });

it('state values have measured shapes without claiming a module invocation', () => {
  const data = { parameters: { _shape_provenance: 'state', _out_shape: [],
    _capture_coverage: { status: 'state', stateValueIds: ['v0'], invocationIds: [] } } };
  expect(tensorShapes(data).outputShape).toEqual([]);
  expect(tensorEvidence(data)).toBe('measured');
  expect(coverageLabel(data)).toBe('State value observed');
});

it('emphasizes only known corresponding dimensions', () => {
  expect(changedDimensions([2, 16, 768], [2, 16, 3072])).toEqual([2]);
  expect(changedDimensions([2, -1], [2, 16])).toEqual([]);
  expect(changedDimensions([2, 3, 4], [2, 12])).toEqual([]);
});

it('retains roles and evidence in portable snapshots without exporting private data', () => {
  const graph = nestedGraph();
  graph.edges[0].data = { role: 'residual', roleEvidence: 'identity-bypass' };
  Object.assign(graph.nodes[3].data, { inputShape: [1, 8], outputShape: [1, 4] });
  graph.nodes[3].data.parameters._shape_provenance = 'observed';
  graph.nodes[3].data.parameters._in_shape = '1x8';
  graph.nodes[3].data.parameters._out_shape = '1x4';
  const file = parseGraphFile(JSON.stringify(createGraphFile(graph, { name: 'Residual', status: 'traced' })));
  const restored = graphDocumentV2ToModelGraph(file.document);
  expect(restored.edges[0].data).toMatchObject({ role: 'residual', roleEvidence: 'identity-bypass' });
  expect(restored.nodes[3].data.shapeEvidence).toBe('measured');
  expect(JSON.stringify(file)).not.toContain('_shape_provenance');
  expect(residualMetadata({ role: 'residual', source: 'PRIVATE' })).toEqual({});
  const stale = createGraphFile(graph, { name: 'Residual', status: 'stale' });
  expect(stale.document.semantic.nodes[3].metadata.shapeEvidence).toBe('stale');
});

it('keeps underlying edge identities when collapse combines ordinary and residual paths', () => {
  const graph = nestedGraph();
  graph.edges[0].data = { role: 'residual', roleEvidence: 'identity-bypass' };
  graph.edges.push({ id: 'ordinary', source: 'input', target: 'fc' });
  const collapsed = reconcileModuleView(graph.nodes, graph.edges);
  const combined = collapsed.edges.find(e => e.data?.synthetic && e.source === 'input');
  expect(combined?.data).toMatchObject({ role: 'mixed', originalEdgeIds: [graph.edges[0].id, 'ordinary'] });
  expect(collapsed.edges.find(e => e.id === graph.edges[0].id)?.data?.role).toBe('residual');
});

it('keeps comparison changes visible when a collapsed path includes an unchanged connection', () => {
  const graph = nestedGraph();
  graph.edges[0].data = { diffStatus: 'removed', role: 'residual', roleEvidence: 'identity-bypass' };
  graph.edges.push({ id: 'unchanged', source: 'input', target: 'fc' });
  const combined = reconcileModuleView(graph.nodes, graph.edges).edges.find(e => e.data?.synthetic && e.source === 'input');
  expect(combined?.data).toMatchObject({ role: 'mixed', diffStatus: 'modified' });
});
