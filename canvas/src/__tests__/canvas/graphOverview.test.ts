import { nestedGraph } from '../../test/fixtures/graphFileFixture';
import { graphNodeOrigin, initialOverviewNodes } from '../../services/graphOverview';
import { reconcileModuleView } from '../../services/canvasLayout';
import { mergeProjection } from '../../ide/applyProjection';

it('holds the same nested module header in page space through expansion', () => {
  const graph = nestedGraph();
  graph.nodes[1].data.expanded = true;
  const closed = reconcileModuleView(graph.nodes, graph.edges, { relayout: true });
  const before = graphNodeOrigin(closed.nodes, 'inner');
  const opened = reconcileModuleView(closed.nodes.map(n => n.id === 'inner' ? { ...n, data: { ...n.data, expanded: true } } : n), closed.edges,
    { relayout: true, anchorNodeId: 'inner' });
  expect(graphNodeOrigin(opened.nodes, 'inner')).toEqual(before);
  expect(opened.nodes.find(n => n.id === 'fc')?.hidden).toBe(false);
});

it('retains manually arranged child coordinates across collapse and reopen', () => {
  const graph = nestedGraph();
  for (const n of graph.nodes) if (n.type === 'moduleNode') { n.data.expanded = true; n.data.moduleLayoutState = 'custom'; }
  graph.nodes.find(n => n.id === 'fc')!.position = { x: 74, y: 105 };
  const closed = reconcileModuleView(graph.nodes.map(n => n.id === 'inner' ? { ...n, data: { ...n.data, expanded: false } } : n), graph.edges, { relayout: true, preserveChildPositions: true });
  const opened = reconcileModuleView(closed.nodes.map(n => n.id === 'inner' ? { ...n, data: { ...n.data, expanded: true } } : n), closed.edges, { relayout: true, preserveChildPositions: true });
  expect(opened.nodes.find(n => n.id === 'fc')!.position).toEqual({ x: 74, y: 105 });
});

it('uses existing hierarchy only on first open and honors restored/user openness', () => {
  const graph = nestedGraph();
  graph.nodes[1].data.expanded = true;
  for (let i = 0; i < 30; i++) graph.nodes.push({ ...graph.nodes[3], id: `leaf${i}`, data: { ...graph.nodes[3].data, parentId: 'block' } });
  expect(initialOverviewNodes(graph.nodes).find(n => n.id === 'block')?.data.expanded).toBe(false);
  expect(mergeProjection(graph, [], { viewRestored: true }).nodes[1].data.expanded).toBe(true);
  expect(mergeProjection(graph, graph.nodes).nodes[1].data.expanded).toBe(true);
  expect(initialOverviewNodes(graph.nodes)).toHaveLength(graph.nodes.length);
});

it('shows descendant failures while collapsed and clears them when repaired', () => {
  const graph = nestedGraph();
  graph.nodes[3].data.error = 'Mismatched dimensions';
  const failed = reconcileModuleView(graph.nodes, graph.edges);
  expect(failed.nodes.find(n => n.id === 'block')?.data).toMatchObject({ moduleDiagnosticStatus: 'error', moduleDiagnosticCount: 1 });
  const repaired = reconcileModuleView(failed.nodes.map(n => ({ ...n, data: { ...n.data, error: undefined } })), failed.edges);
  expect(repaired.nodes.find(n => n.id === 'block')?.data.moduleDiagnosticStatus).toBeUndefined();
});

it('counts one recorded failure once through nested modules and keeps unrelated warnings separate', () => {
  const graph = nestedGraph();
  const diagnostic = { id: 'failure', code: 'invalid_dtype', field: ['args', 1], actualDtype: 'torch.float32' } as any;
  for (const node of graph.nodes.filter(n => ['block', 'inner', 'fc'].includes(n.id))) {
    node.data.error = 'Call failed';
    node.data.captureRef = { version: 1, captureId: 'run', sourceId: 'source', kind: 'operation', semanticId: node.id };
    node.data.captureDiagnostic = diagnostic;
  }
  const failed = reconcileModuleView(graph.nodes, graph.edges);
  for (const id of ['block', 'inner']) {
    const data = failed.nodes.find(n => n.id === id)!.data;
    expect(data.moduleDiagnosticCount).toBe(1);
    expect(data.moduleDiagnostics?.[0]).toMatchObject({ nodeId: 'fc', severity: 'error' });
  }
  graph.nodes.find(n => n.id === 'inner')!.data.resolutionSeverity = 'warning';
  const mixed = reconcileModuleView(graph.nodes, graph.edges).nodes.find(n => n.id === 'block')!.data;
  expect(mixed.moduleDiagnostics?.map(issue => issue.severity)).toEqual(['error', 'warning']);
  graph.nodes.find(n => n.id === 'inner')!.data.captureDiagnostic = { ...diagnostic, id: 'different-failure' };
  expect(reconcileModuleView(graph.nodes, graph.edges).nodes.find(n => n.id === 'block')!.data.moduleDiagnosticCount).toBe(3);
});
