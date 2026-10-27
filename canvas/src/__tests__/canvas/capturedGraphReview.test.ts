import { capturedGraphReview } from '../../ide/capturedGraphReview';
import { buildGraphReview } from '../../ide/graphReview';
import { applyProjection } from '../../ide/projection';
import { graphDocumentV2ToModelGraph } from '../../services/graphDocumentAdapter';
import type { GraphReviewState, ProjectionPayload } from '../../ide/protocol.generated';
import type { GraphDocumentV2 } from '../../types/graphDocument';
import captureDocument from '../../test/fixtures/canvas/capture-document.json';
import reviewFixture from '../../test/fixtures/canvas/graphReview.json';

const setup = () => {
  const state: GraphReviewState = { ...JSON.parse(JSON.stringify(reviewFixture)), id: 'r', beforeLabel: 'HEAD', afterLabel: 'Working',
    stale: false, changedFiles: ['model.py'], liveContext: { workspaceFolder: '/ws', file: 'model.py', cls: 'MlpTiny' } };
  const capture: ProjectionPayload = { ...state.after, document: JSON.parse(JSON.stringify(captureDocument)), traced: true };
  return { state, capture, source: buildGraphReview(state) };
};

const conditionalLayers = () => {
  const { state, capture } = setup();
  state.after = JSON.parse(JSON.stringify(state.before));
  for (const side of ['before', 'after'] as const) {
    state[side].sourceDigest = side;
    state[side].dependencies = [{ file: 'model.py', digest: side }];
    (state[side].document as GraphDocumentV2).semantic.metadata.reviewClass = {
      defaults: side === 'after' ? { project: 'True' } : {}, digest: side,
      members: side === 'after' ? { output_projection: { digest: 'one' }, output_linear: { digest: 'two' } } : {},
    };
  }
  capture.sourceDigest = state.after.sourceDigest;
  capture.dependencies = state.after.dependencies;
  const document = capture.document as GraphDocumentV2;
  const root = document.semantic.nodes.find(node => node.metadata.captureRef?.kind === 'invocation')!;
  const extra = ['output_projection', 'output_linear'].map((name, i) => ({
    ...JSON.parse(JSON.stringify(root)), node_id: `captured:linear:${i}`, metadata: {
      ...root.metadata, captureLabel: `unrelated display label ${i}`,
      captureRef: { ...root.metadata.captureRef, semanticId: `call:linear:${i}` },
    },
  }));
  document.semantic.nodes.push(...extra);
  document.semantic.metadata.sourceCorrespondence = extra.map((node, i) => ({
    nodeId: ['output_projection', 'output_linear'][i], basis: 'registered_path',
    captureId: node.metadata.captureRef.captureId, invocationIds: [node.metadata.captureRef.semanticId],
  }));
  return { state, capture, extra };
};

test('highlights each added conditional layer, not just the constructor option on the root', () => {
  const { state, capture, extra } = conditionalLayers();
  const review = capturedGraphReview(state, buildGraphReview(state), capture)!;
  expect(review.counts).toMatchObject({ added: 2, modified: 1 });
  for (const [i, name] of Array.from(['output_projection', 'output_linear'].entries())) {
    const change = review.changes.find(change => change.label === name)!;
    expect(change).toBeDefined();
    expect(review.focusNodes?.get(change.id)).toBe(extra[i].node_id);
    expect(review.projections.diff.reviewGraph.nodes.find(node => node.id === extra[i].node_id)?.data)
      .toMatchObject({ diffStatus: 'added', review: { status: 'added' } });
  }
  expect(review.projections.diff.reviewGraph.edges).toEqual(graphDocumentV2ToModelGraph(capture.document as GraphDocumentV2).edges);
});

test('preserves connection-only source labels on matching captured calls without rewiring the capture', () => {
  const { state, capture, extra } = conditionalLayers();
  state.before = JSON.parse(JSON.stringify(state.after));
  const document = state.after.document as GraphDocumentV2;
  document.semantic.edges.find(edge => edge.target_node_id === 'head')!.source_node_id = 'proj';
  ((capture.document as GraphDocumentV2).semantic.metadata.sourceCorrespondence as { nodeId: string }[])[0].nodeId = 'head';
  const review = capturedGraphReview(state, buildGraphReview(state), capture)!;
  const node = review.projections.diff.reviewGraph.nodes.find(node => node.id === extra[0].node_id)!;
  expect(node.data).toMatchObject({ diffStatus: 'rewired', review: { status: 'rewired', inputChanges: { added: 1, removed: 1 },
    summary: 'Input connections changed (+1 / −1). Source and parameters unchanged.' } });
  expect(review.counts).toMatchObject({ modified: 0, rewired: 1 });
  expect(review.projections.diff.reviewGraph.edges).toEqual(graphDocumentV2ToModelGraph(capture.document as GraphDocumentV2).edges);
});

test('removal highlights the retained before-side layer and modifications highlight all its calls', () => {
  const { state, capture, extra } = conditionalLayers();
  const removed = { ...state, before: state.after, after: state.before };
  const review = capturedGraphReview(removed, buildGraphReview(removed), capture)!;
  expect(review.counts.removed).toBe(2);
  expect(review.projections.diff.reviewGraph.nodes.find(node => node.id === extra[0].node_id)?.data.diffStatus).toBe('removed');
  state.before = JSON.parse(JSON.stringify(state.after));
  (state.before.document as GraphDocumentV2).semantic.metadata.reviewClass.members.output_projection.digest = 'old-width';
  const document = capture.document as GraphDocumentV2;
  document.semantic.metadata.sourceCorrespondence[0].invocationIds.push(extra[1].metadata.captureRef.semanticId);
  const modified = capturedGraphReview(state, buildGraphReview(state), capture)!;
  expect(modified.counts).toMatchObject({ added: 0, modified: 1 });
  for (const node of extra) expect(modified.projections.diff.reviewGraph.nodes.find(item => item.id === node.node_id)?.data.diffStatus).toBe('modified');
});

test('does not guess a layer match from stale source, shared aliases, or an ambiguous declaration', () => {
  for (const reason of ['stale', 'shared', 'ambiguous']) {
    const { state, capture, extra } = conditionalLayers();
    if (reason === 'stale') capture.dependencies = [{ file: 'model.py', digest: 'other-version' }];
    if (reason === 'shared') (capture.document as GraphDocumentV2).semantic.metadata.sourceCorrespondence[0].basis = 'shared_instance';
    if (reason === 'ambiguous') (state.after.document as GraphDocumentV2).semantic.metadata.reviewClass.members.output_projection.ambiguous = true;
    const review = capturedGraphReview(state, buildGraphReview(state), capture)!;
    expect(review.projections.diff.reviewGraph.nodes.find(node => node.id === extra[0].node_id)?.data.diffStatus).toBeUndefined();
    expect(review.changes.find(change => change.label === 'output_projection')?.detail).toContain('Not located in this capture');
  }
});

test('retains all captured nodes, connections, values and arrangement through review and return', () => {
  const { state, capture, source } = setup();
  const graph = graphDocumentV2ToModelGraph(capture.document as GraphDocumentV2);
  const existing = graph.nodes.map((node, i) => ({ ...node, position: { x: 123 + i * 90, y: 450 },
    data: { ...node.data, expanded: true, captureDetailLevel: 'operations' as const } }));
  const review = capturedGraphReview(state, source, capture)!;
  expect(review).not.toBeNull();
  const merged = applyProjection(review.projections.diff, existing)!;
  expect(merged.nodes.map(node => node.id)).toEqual(graph.nodes.map(node => node.id));
  expect(merged.edges).toEqual(graph.edges);
  expect(merged.nodes.map(node => node.position)).toEqual(existing.map(node => node.position));
  expect(merged.nodes.map(node => node.data.parameters)).toEqual(graph.nodes.map(node => node.data.parameters));
  const sourceGraph = applyProjection(source.projections.diff, merged.nodes)!;
  const restored = applyProjection(review.projections.diff, sourceGraph.nodes)!;
  expect(restored.nodes.map(node => node.position)).toEqual(existing.map(node => node.position));
  const live = applyProjection(capture, restored.nodes)!;
  expect(live.nodes.map(node => node.position)).toEqual(existing.map(node => node.position));
  expect(live.nodes.some(node => node.data.review)).toBe(false);
});

test('maps registered calls only against exact source snapshots, not labels or topology', () => {
  const { state, capture, source } = setup();
  const document = capture.document as GraphDocumentV2;
  const call = document.semantic.nodes.find(node => (node.metadata.captureRef as any)?.kind === 'invocation')!;
  // Use the real typed correspondence channel; visual names are intentionally different.
  document.semantic.metadata.sourceCorrespondence = [{ nodeId: 'proj', basis: 'registered_path',
    captureId: (call.metadata.captureRef as any).captureId, invocationIds: [(call.metadata.captureRef as any).semanticId] }];
  capture.dependencies = [{ file: 'model.py', digest: capture.sourceDigest }];
  state.after.dependencies = capture.dependencies;
  const mapped = capturedGraphReview(state, source, capture)!;
  expect(mapped.focusNodes?.get('review:after:proj')).toBe(call.node_id);
  expect(mapped.nodes.get(call.node_id)?.summary).toBe(source.changes.find(change => change.id === 'review:after:proj')?.detail);
  capture.dependencies = [{ file: 'model.py', digest: 'stale-dependency' }];
  const stale = capturedGraphReview(state, source, capture)!;
  expect(stale.changes.find(change => change.id === 'review:after:proj')?.detail).toContain('Not located in this capture');
  expect(stale.projections.diff.reviewGraph.edges).toEqual(mapped.projections.diff.reviewGraph.edges);
});

test('never borrows a capture for another model or a source-only projection', () => {
  const { state, capture, source } = setup();
  expect(capturedGraphReview(state, source, { ...capture, cls: 'OtherModel' })).toBeNull();
  expect(capturedGraphReview(state, source, { ...capture, traced: false })).toBeNull();
  expect(capturedGraphReview({ ...state, liveContext: undefined }, source, capture)).toBeNull();
});
