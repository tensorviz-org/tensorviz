import fixture from '../../test/fixtures/canvas/deepDeclarationReview.json';
import { buildGraphReview } from '../../ide/graphReview';
import { capturedGraphReview } from '../../ide/capturedGraphReview';
import type { GraphReviewState, ProjectionPayload } from '../../ide/protocol.generated';

const stateFor = (before: unknown, after: unknown): GraphReviewState => ({
  id: 'deep', beforeLabel: 'HEAD', afterLabel: 'Working', stale: false, changedFiles: ['model.py'],
  before, after, liveContext: { file: 'model.py', cls: 'Model' },
} as GraphReviewState);

for (const name of ['dynamic', 'literal', 'static'] as const) {
  test(`real ${name} source change opens each recorded child call`, () => {
    const data = fixture[name];
    const state = stateFor(data.before, data.after);
    const source = buildGraphReview(state);
    const review = capturedGraphReview(state, source, data.capture as unknown as ProjectionPayload)!;
    expect(review.changes).toHaveLength(2);
    expect(review.counts.modified).toBe(1);
    expect(review.counts.unresolved).toBe(0);
    expect(review.changes.every(change => !change.detail.includes('Not located'))).toBe(true);
    const targets = review.changes.map(change => review.focusNodes?.get(change.id));
    const focused = review.projections.diff.reviewGraph.nodes.filter(node => targets.includes(node.id));
    expect(focused.map(node => node.data.captureLabel)).toEqual(name === 'dynamic'
      ? ['blocks.0.proj', 'blocks.1.proj'] : ['blocks.0.proj.1', 'blocks.1.proj.1']);
    expect(focused.every(node => !!node.data.parentId && node.data.diffStatus === 'modified')).toBe(true);
  });
}

for (const changed of ['inserted', 'reordered'] as const) {
  test(`does not specialize literal children after ${changed} slots`, () => {
    const data = fixture.literal;
    const state = stateFor(data.before, data[changed]);
    const review = capturedGraphReview(state, buildGraphReview(state), data.capture as unknown as ProjectionPayload)!;
    const children = review.projections.diff.reviewGraph.nodes.filter(node => /\.proj\.\d+$/.test(node.data.captureLabel ?? ''));
    expect(children.some(node => !!node.data.diffStatus)).toBe(false);
    expect(review.changes.some(change => change.detail.includes('Not located'))).toBe(true);
  });
}

test('constructor syntax equivalence does not become a supplemental declaration change', () => {
  const data = fixture.equivalent;
  const review = buildGraphReview(stateFor(data.before, data.after));
  expect(review.counts.modified).toBe(0);
  expect(review.changes.some(change => change.id.startsWith('review:declaration:'))).toBe(false);
});

test('repeated lowered copies of one declaration do not duplicate captured navigation', () => {
  const data = fixture.expanded, state = stateFor(data.before, data.after);
  const source = buildGraphReview(state);
  const review = capturedGraphReview(state, source, data.capture as unknown as ProjectionPayload)!;
  expect(review.changes).toHaveLength(2);
  expect(review.counts).toEqual(source.counts);
  const targets = review.changes.map(change => review.focusNodes?.get(change.id));
  expect(new Set(targets).size).toBe(2);
  expect(review.projections.diff.reviewGraph.nodes.filter(node => targets.includes(node.id)).map(node => node.data.captureLabel))
    .toEqual(['first.proj.1', 'last.proj.1']);
});

for (const name of ['dynamic', 'literal'] as const) {
  test(`proven file renames do not turn ${name} declarations into added and removed layers`, () => {
    const data = fixture[name];
    const after = JSON.parse(JSON.stringify(data.before).split('model.py').join('renamed.py'));
    const state = { ...stateFor(data.before, after), renamedFiles: { 'model.py': 'renamed.py' } };
    expect(buildGraphReview(state).changes).toEqual([]);
  });

  test(`a ${name} edit accompanying a proven rename still locates the before-side captured calls`, () => {
    const data = fixture[name];
    const after = JSON.parse(JSON.stringify(data.after).split('model.py').join('renamed.py'));
    const state = { ...stateFor(data.before, after), renamedFiles: { 'model.py': 'renamed.py' } };
    const source = buildGraphReview(state);
    const review = capturedGraphReview(state, source, data.capture as unknown as ProjectionPayload)!;
    expect(review.counts.modified).toBe(1);
    expect(review.counts.added + review.counts.removed).toBe(0);
    expect(review.changes).toHaveLength(2);
    expect(review.changes.every(change => !change.detail.includes('Not located'))).toBe(true);
    for (const change of source.changes) {
      expect(source.projections.before.spanmap[change.id]?.file).toBe('model.py');
      expect(source.projections.after.spanmap[change.id]?.file).toBe('renamed.py');
    }
  });
}
