import { buildGraphReview } from '../../ide/graphReview';
import type { GraphReviewState } from '../../ide/protocol.generated';
import fixture from '../../test/fixtures/canvas/helperGraphReview.json';

const state = (): GraphReviewState => ({ ...JSON.parse(JSON.stringify(fixture)), id: 'helpers', beforeLabel: 'Index', afterLabel: 'Working changes', changedFiles: ['model.py'], stale: false });

test('two identical inline ReLU additions inside nested helpers are added with actual source links', () => {
  const result = buildGraphReview(state());
  const nodes = result.projections.diff.reviewGraph.nodes.filter(node => node.data.layerType.name === 'ReLU');
  expect(nodes).toHaveLength(2);
  expect(nodes.every(node => node.data.diffStatus === 'added')).toBe(true);
  expect(new Set(nodes.map(node => node.data.sourceLocation?.line)).size).toBe(2);
  expect(result.counts.added).toBe(2);
  expect(result.counts.unresolved).toBe(0);
  expect(result.projections.before.reviewGraph.nodes.some(node => node.data.layerType.name === 'ReLU')).toBe(false);
  expect(result.projections.after.reviewGraph.nodes.filter(node => node.data.layerType.name === 'ReLU')).toHaveLength(2);
  expect(result.edgeCounts.added).toBeGreaterThanOrEqual(4);
});

test('the same source evidence proves removal in the opposite comparison direction', () => {
  const review = state();
  const result = buildGraphReview({ ...review, before: review.after, after: review.before });
  expect(result.counts.removed).toBe(2);
  expect(result.counts.unresolved).toBe(0);
});

test('insertion proof is withheld for partial snapshots and changed anchors', () => {
  const partial = state(); partial.before.truncated = true;
  expect(buildGraphReview(partial).counts.added).toBe(0);
  const changed = state();
  const semantic = (changed.after.document as any).semantic;
  const add = semantic.nodes.find((node: any) => node.op === 'torch.add');
  add.params._raw_call = 'x + 2'; add.params.other = 2;
  expect(buildGraphReview(changed).counts.added).toBe(0);
});

test('indistinguishable repeated anchors do not prove which operation was added', () => {
  const review = state();
  review.before = JSON.parse(JSON.stringify(review.after));
  const semantic = (review.after.document as any).semantic;
  const first = semantic.nodes.find((node: any) => node.op === 'ReLU');
  const duplicate = JSON.parse(JSON.stringify(first));
  duplicate.node_id = 'ambiguous-extra';
  duplicate.params._review_order += 0.1;
  semantic.nodes.push(duplicate);
  (review.after.spanmap as any)[duplicate.node_id] = (review.after.spanmap as any)[first.node_id];
  const result = buildGraphReview(review);
  expect(result.counts.added).toBe(0);
  expect(result.counts.unresolved).toBeGreaterThan(0);
});

test('a proven empty helper scope can receive its first operations', () => {
  const review = state();
  const semantic = (review.before.document as any).semantic;
  semantic.nodes = semantic.nodes.filter((node: any) => !node.params?._review_scope);
  semantic.edges = [];
  const result = buildGraphReview(review);
  expect(result.counts.added).toBe(3);
  expect(result.counts.unresolved).toBe(0);
});

test('adding a reassignment does not lose correspondence for the unchanged original statement', () => {
  const review = state();
  const add = (review.before.document as any).semantic.nodes.find((node: any) => node.op === 'torch.add');
  add.params._review_binding = [...add.params._review_scope, 'x'];
  const result = buildGraphReview(review);
  expect(result.counts.added).toBe(2);
  expect(result.counts.unresolved).toBe(0);
});
