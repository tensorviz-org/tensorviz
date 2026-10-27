import { fireEvent, render, screen } from '@testing-library/react';
import { GraphReviewToolbar } from '../../ide/GraphReviewToolbar';
import { buildGraphReview } from '../../ide/graphReview';
import type { GraphReviewState } from '../../ide/protocol.generated';
import fixture from '../../test/fixtures/canvas/graphReview.json';

const state = (): GraphReviewState => ({ ...JSON.parse(JSON.stringify(fixture)), id: 'r', beforeLabel: 'base', afterLabel: 'head', changedFiles: ['model.py'], stale: false });
const empty = (value: GraphReviewState['before']) => ({ ...value, spanmap: {}, document: {
  ...value.document as any, semantic: { ...(value.document as any).semantic, nodes: [], edges: [], inputs: [], modules: [] },
} });

test('whole-model additions and deletions include operations without invented cross-revision matches', () => {
  const added = state(); added.before = empty(added.before); added.modelStatus = 'added';
  const result = buildGraphReview(added);
  expect(result.counts.unresolved).toBe(0);
  expect(result.counts.added).toBeGreaterThan(0);
  expect(result.counts.removed).toBe(0);
  const removed = { ...added, before: added.after, after: added.before, modelStatus: 'removed' as const };
  expect(buildGraphReview(removed).counts.removed).toBe(result.counts.added);
  expect(buildGraphReview({ ...added, modelStatus: 'unresolved' }).counts.added).toBe(0);
});

test('proven file renames retain identity while source links still name the original paths', () => {
  const review = state(); review.after = JSON.parse(JSON.stringify(review.before));
  for (const span of Object.values(review.after.spanmap) as any[]) span.file = 'renamed.py';
  review.modelStatus = 'renamed'; review.renamedFiles = { 'model.py': 'renamed.py' };
  expect(buildGraphReview(review).changes).toEqual([]);
  const graph = buildGraphReview(review).projections.before.reviewGraph;
  expect(graph.nodes.find(node => node.id === 'review:after:proj')?.data.sourceLocation?.file).toBe('model.py');
});

test('model switching and PR refresh are review actions, with visible coverage limits', () => {
  const review = { ...state(), title: 'team/repo #12', selectedModel: 'one', updated: true,
    models: [{ id: 'one', label: 'Encoder', file: 'encoder.py', status: 'modified' as const },
      { id: 'two', label: 'Decoder', file: 'decoder.py', status: 'added' as const }],
    notes: ['Shared external helper could not be analyzed.'], repositoryNotes: ['Unused dataset class could not be analyzed.'] };
  const onModel = jest.fn(), onRefresh = jest.fn();
  render(<GraphReviewToolbar state={review} presentation={buildGraphReview(review)} mode="diff" index={0}
    onMode={jest.fn()} onChange={jest.fn()} onClose={jest.fn()} onSource={jest.fn()} onModel={onModel} onRefresh={onRefresh} />);
  fireEvent.change(screen.getByLabelText('Affected model'), { target: { value: 'two' } });
  expect(onModel).toHaveBeenCalledWith('two');
  fireEvent.click(screen.getByText('Refresh to latest changes'));
  expect(onRefresh).toHaveBeenCalledTimes(1);
  const coverage = screen.getByRole('button', { name: 'Coverage limits (1)' });
  expect(coverage.getAttribute('aria-expanded')).toBe('false');
  expect(screen.queryByRole('region', { name: 'Coverage limits for this model' })).toBeNull();
  fireEvent.click(coverage);
  expect(screen.getByText('Shared external helper could not be analyzed.')).toBeTruthy();
  const repository = screen.getByRole('button', { name: 'Other repository notes (1)' });
  expect(repository.getAttribute('aria-expanded')).toBe('false');
  fireEvent.click(repository);
  expect(screen.queryByText('Shared external helper could not be analyzed.')).toBeNull();
  expect(screen.getByText('Unused dataset class could not be analyzed.')).toBeTruthy();
  fireEvent.keyDown(screen.getByRole('region', { name: 'Other repository notes' }), { key: 'Escape' });
  expect(repository.getAttribute('aria-expanded')).toBe('false');
  expect(document.activeElement).toBe(repository);
});

test('counts and navigates connection-only changes separately from modified source', () => {
  const review = state(); review.after = JSON.parse(JSON.stringify(review.before));
  (review.after.document as any).semantic.edges.find((edge: any) => edge.target_node_id === 'head').source_node_id = 'proj';
  render(<GraphReviewToolbar state={review} presentation={buildGraphReview(review)} mode="diff" index={0}
    onMode={jest.fn()} onChange={jest.fn()} onClose={jest.fn()} onSource={jest.fn()} onRefresh={jest.fn()} />);
  expect(screen.getByLabelText('0 modified')).toHaveTextContent('~0');
  expect(screen.getByText('1 inputs changed')).toBeVisible();
  expect(screen.getByText('Input connections changed (+1 / −1). Source and parameters unchanged.').parentElement).toHaveAttribute('data-status', 'rewired');
});


test('a syntax error is named in the stepper and is not a whole-file deletion', () => {
  const review = state();
  review.modelStatus = 'removed';
  review.notes = ['After: model.py::Net: could not parse model.py:15'];
  review.after = empty(review.after);
  const result = buildGraphReview(review);
  expect(result.counts.removed).toBe(0);
  expect([...result.nodes.values()].every(node => node.status !== 'removed')).toBe(true);
  render(<GraphReviewToolbar state={review} presentation={result} mode="diff" index={0}
    onMode={jest.fn()} onChange={jest.fn()} onClose={jest.fn()} onSource={jest.fn()} onRefresh={jest.fn()} />);
  expect(screen.getByRole('status')).toHaveTextContent('could not parse model.py:15');
});

test('source reviews never present the adapter default input shape as a recorded shape', () => {
  const review = buildGraphReview(state());
  for (const mode of ['before', 'diff', 'after'] as const) {
    const input = review.projections[mode].reviewGraph.nodes.find(node => node.type === 'inputNode')!;
    expect(input).toBeDefined();
    expect(input.data.parameters.shape).toBe('');
    expect(input.data.inputShape).toBeUndefined();
    expect(input.data.outputShape).toBeUndefined();
    expect(input.data.shapeEvidence).toBe('unknown');
  }
});
