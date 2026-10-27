import fixture from '../../test/fixtures/canvas/draftNestedAssignment.json';
import { buildGraphReview } from '../../ide/graphReview';
import { graphDebugPresentation } from '../../ide/graphDebugPresentation';
import { graphDocumentV2ToModelGraph } from '../../services/graphDocumentAdapter';
import type { GraphReviewState, ProjectionPayload } from '../../ide/protocol.generated';
import type { GraphDocumentV2 } from '../../types/graphDocument';

const key = '@assignment:layer3[0].relu';

test('an unchanged assignment adds no detached card to the next draft', () => {
  const state = { ...fixture, before: fixture.after, id: 'next-draft', beforeLabel: 'Current', afterLabel: 'Draft',
    changedFiles: ['vision_models.py'], stale: false, proposal: { id: 'next-draft' } } as GraphReviewState;
  const review = buildGraphReview(state);
  expect(review.changes).toEqual([]);
  expect(review.projections.diff.reviewGraph.nodes.some(node => node.data.sourceAssignment)).toBe(false);
});

test('newly ambiguous evidence is not hidden by an unchanged assignment digest or another explained change', () => {
  const state = JSON.parse(JSON.stringify({ ...fixture, before: fixture.after, id: 'overlap', beforeLabel: 'Current',
    afterLabel: 'Draft', changedFiles: ['vision_models.py'], stale: false, proposal: { id: 'overlap' } })) as GraphReviewState;
  const facts = (state.after.document as any).semantic.metadata.reviewClass;
  facts.digest = 'changed-ancestor';
  facts.assignments[key].ambiguous = true;
  facts.assignments['@assignment:layer4[0].relu'] = { ...facts.assignments[key], target: 'layer4[0].relu', ambiguous: false };
  const review = buildGraphReview(state);
  expect(review.counts).toMatchObject({ modified: 1, unresolved: 1 });
  expect(review.projections.diff.reviewGraph.nodes.some(node => node.id === `review:after:${key}`)).toBe(false);
});

test('without a capture the exact nested edit stays navigable without a fabricated location', () => {
  const state = { ...fixture, id: 'nested-resnet', beforeLabel: 'Current', afterLabel: 'Draft',
    changedFiles: ['vision_models.py'], stale: false, proposal: { id: 'nested-resnet' } } as GraphReviewState;
  const snapshot = JSON.stringify(state);
  const review = buildGraphReview(state);
  expect(review.counts).toMatchObject({ added: 0, removed: 0, modified: 1, unresolved: 0 });
  expect(review.changes).toEqual([{ id: `review:after:${key}`, label: 'layer3[0].relu → nn.GELU',
    detail: 'Assignment set in source · not run · Location not verified in graph' }]);
  expect(review.projections.diff.reviewGraph.nodes.some(node => node.data.sourceAssignment)).toBe(false);
  expect(review.projections.diff.spanmap[review.changes[0].id]).toMatchObject({ file: 'vision_models.py', start: [10, 8] });
  expect(JSON.stringify(state)).toBe(snapshot);
  const accepted = fixture.after as ProjectionPayload;
  const graph = graphDocumentV2ToModelGraph(accepted.document as GraphDocumentV2);
  const live = graphDebugPresentation(accepted, graph, []).graph;
  expect(live.nodes).toHaveLength(graph.nodes.length);
  expect(live.nodes.some(node => node.data.sourceAssignment)).toBe(false);
  expect(live.edges).toEqual(graph.edges);
});
