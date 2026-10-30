import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DraftGraphToolbar, GraphReviewToolbar } from '../../ide/GraphReviewToolbar';
import { buildGraphReview } from '../../ide/graphReview';
import type { GraphReviewState } from '../../ide/protocol.generated';
// Exact backend projections of a ResNet subclass whose draft adds `fc2` after the
// inherited torchvision forward (before: Input→inherited→Output; after: …→fc2→Output).
import fixture from '../../test/fixtures/canvas/draftResNet.json';

const PLACEHOLDER = 'review:after:@inherited-forward';
const draft = (proposal: GraphReviewState['proposal'] = { id: 'proposal-1' }): GraphReviewState => ({
  ...JSON.parse(JSON.stringify(fixture)), id: 'draft-1', beforeLabel: 'Current', afterLabel: 'Draft',
  changedFiles: ['vision_models.py'], stale: false, ...(proposal ? { proposal } : {}),
});

describe('draft graph preview from ResNet source projections', () => {
  it('proves the added fc2 declaration despite the external-base warning', () => {
    const state = draft();
    expect(state.after.diagnostics).toEqual([expect.objectContaining({ code: 'EXTERNAL_PACKAGE', severity: 'warning' })]);
    const result = buildGraphReview(state);
    expect(result.nodes.get('review:after:fc2')).toMatchObject({ status: 'added', after: 'fc2', before: undefined });
    expect(result.counts).toMatchObject({ added: 1, removed: 0, modified: 0, rewired: 0, unresolved: 0 });
    // Declared in source, never claimed as an observed runtime layer.
    expect(result.changes).toEqual([{ id: 'review:after:fc2', label: expect.any(String), detail: 'Declared in source · not run' }]);
    expect(result.focusNodes).toBeUndefined();
  });

  it('keeps the inherited base forward neutral, with no rewire of Output from the unknown base', () => {
    const result = buildGraphReview(draft());
    expect(result.inheritedBase).toBe(true);
    expect(result.nodes.get(PLACEHOLDER)).toMatchObject({ status: 'unchanged', before: '@inherited-forward', after: '@inherited-forward' });
    expect(result.nodes.get('review:after:output')?.status).toBe('unchanged');
    expect(result.edgeCounts).toEqual({ added: 1, removed: 0 });
    const graph = result.projections.diff.reviewGraph;
    expect(graph.nodes.find(node => node.id === PLACEHOLDER)?.data.diffStatus).toBeUndefined();
    expect(graph.nodes.find(node => node.id === 'review:after:fc2')?.data.diffStatus).toBe('added');
    for (const edge of graph.edges.filter(edge => edge.source === PLACEHOLDER || edge.target === PLACEHOLDER)) {
      expect(edge.data?.diffStatus).toBeUndefined();
    }
    expect(graph.edges.find(edge => edge.source === 'review:after:fc2')?.data?.diffStatus).toBe('added');
  });

  it('previews the proposed graph without the superseded old path, shapes or invented base internals', () => {
    const result = buildGraphReview(draft());
    const graph = result.projections.diff.reviewGraph;
    expect(result.projections.diff.proposal).toBe(true);
    expect(graph.nodes.map(node => node.id).sort()).toEqual(['review:after:@inherited-forward', 'review:after:fc2',
      'review:after:input', 'review:after:output', 'review:root']);
    expect(graph.edges.map(edge => [edge.source, edge.target])).toEqual([
      ['review:after:input', PLACEHOLDER], [PLACEHOLDER, 'review:after:fc2'], ['review:after:fc2', 'review:after:output']]);
    for (const node of graph.nodes.filter(node => node.id !== 'review:root')) {
      expect(node.data.inputShape).toBeUndefined();
      expect(node.data.outputShape).toBeUndefined();
      expect(node.data.shapeEvidence).toBe('unknown');
    }
  });

  it('leaves Git reviews of the same source unchanged', () => {
    const result = buildGraphReview(draft(null as unknown as undefined));
    expect(result.projections.diff.proposal).toBeUndefined();
    expect(result.inheritedBase).toBeUndefined();
    expect(result.nodes.get('review:after:fc2')?.status).toBe('unresolved');
  });

  it('shows an added declaration the projector did not lower as a visible unconnected card', () => {
    const state = draft();
    const semantic = (state.after.document as any).semantic;
    semantic.nodes = semantic.nodes.filter((node: any) => node.node_id !== 'fc2');
    semantic.edges = semantic.edges.filter((edge: any) => edge.source_node_id !== 'fc2' && edge.target_node_id !== 'fc2');
    delete (state.after.spanmap as any).fc2;
    const result = buildGraphReview(state);
    expect(result.changes).toEqual([{ id: 'review:after:fc2', label: 'fc2', detail: 'Declared · not called in the projected forward' }]);
    // Focus lands on a real card, never silently on the class frame.
    expect(result.focusNodes?.get('review:after:fc2')).toBeUndefined();
    const graph = result.projections.diff.reviewGraph;
    expect(graph.nodes.find(node => node.id === 'review:after:fc2')?.data.diffStatus).toBe('added');
    expect(graph.edges.some(edge => edge.source === 'review:after:fc2' || edge.target === 'review:after:fc2')).toBe(false);
    expect(result.projections.diff.spanmap['review:after:fc2']).toMatchObject({ file: 'vision_models.py', start: [6, 8] });
  });
});

const toolbar = (state: GraphReviewState, callbacks: Partial<Record<'onAccept' | 'onDiscard' | 'onContinueEditing' | 'onSource', jest.Mock>> = {}) => {
  const props = { onAccept: jest.fn(() => true), onDiscard: jest.fn(() => true), onContinueEditing: jest.fn(), onSource: jest.fn(), ...callbacks };
  const view = render(<DraftGraphToolbar state={state} presentation={buildGraphReview(state)} index={0} onChange={jest.fn()} {...props} />);
  const rerender = (next: GraphReviewState) => view.rerender(<DraftGraphToolbar state={next} presentation={buildGraphReview(next)} index={0} onChange={jest.fn()} {...props} />);
  return { ...props, rerender };
};
const decision = () => screen.getByRole('group', { name: 'Draft decision' });

test('the draft toolbar is compact: one code action and one decision row, no Close, Refresh or revision modes', () => {
  const { onSource } = toolbar(draft());
  expect(screen.getByText('Draft · Not run')).toBeVisible();
  expect(screen.getByRole('button', { name: 'Accept' })).toHaveAttribute('title',
    'Save the accepted source and run the model with its saved inputs and device');
  expect(screen.getByText('1 / 1')).toBeVisible();
  expect(screen.getByLabelText('1 added')).toHaveTextContent('+1');
  expect(screen.getByText(/inherited base forward was not inspected/)).toBeVisible();
  for (const name of ['Close graph diff', 'Refresh graph diff', /^Before$/, /^After$/, /^Apply/]) {
    expect(screen.queryByRole('button', { name })).toBeNull();
  }
  // Text buttons in reading and tab order: the decision first, then the way back to the conversation.
  expect(within(decision()).getAllByRole('button').map(button => button.textContent)).toEqual(['Accept', 'Discard', 'Continue editing']);
  fireEvent.click(screen.getByRole('button', { name: 'Open proposed code diff' }));
  expect(onSource).toHaveBeenCalledTimes(1);
});

test('Accept and Discard are keyboard reachable and request one decision until the host answers', () => {
  const state = draft();
  const { onAccept, onDiscard, rerender } = toolbar(state);
  screen.getByRole('button', { name: 'Open proposed code diff' }).focus();
  userEvent.tab();
  expect(screen.getByRole('button', { name: 'Accept' })).toHaveFocus();
  userEvent.keyboard('{Enter}');
  expect(onAccept).toHaveBeenCalledTimes(1);
  // A double click or a quick Discard cannot race the pending Accept.
  expect(within(decision()).getByRole('button', { name: 'Accepting…' })).toBeDisabled();
  expect(within(decision()).getByRole('button', { name: 'Discard' })).toBeDisabled();
  expect(within(decision()).getByRole('button', { name: 'Continue editing' })).toBeEnabled();
  fireEvent.click(within(decision()).getByRole('button', { name: 'Accepting…' }));
  fireEvent.click(within(decision()).getByRole('button', { name: 'Discard' }));
  expect(onAccept).toHaveBeenCalledTimes(1);
  expect(onDiscard).not.toHaveBeenCalled();
  // A refusal arrives as a new state; the decision is offered again.
  rerender({ ...state });
  fireEvent.click(screen.getByRole('button', { name: 'Discard' }));
  expect(onDiscard).toHaveBeenCalledTimes(1);
  expect(screen.getByRole('button', { name: 'Discarding…' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Accept' })).toBeDisabled();
});

test('a decision that could not be sent stays available', () => {
  const { onAccept } = toolbar(draft(), { onAccept: jest.fn(() => false) });
  fireEvent.click(screen.getByRole('button', { name: 'Accept' }));
  fireEvent.click(screen.getByRole('button', { name: 'Accept' }));
  expect(onAccept).toHaveBeenCalledTimes(2);
  expect(screen.getByRole('button', { name: 'Discard' })).toBeEnabled();
});

test('a stale draft cannot be accepted, says why, and can still be discarded', () => {
  const { onAccept, onDiscard } = toolbar({ ...draft(), stale: true });
  const accept = screen.getByRole('button', { name: 'Accept' });
  expect(accept).toBeDisabled();
  expect(accept).toHaveAccessibleDescription(/Source changed since this draft, so it can’t be accepted/);
  expect(screen.getByRole('status')).toBeVisible();
  fireEvent.click(accept);
  expect(onAccept).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Discard' }));
  expect(onDiscard).toHaveBeenCalledTimes(1);
});

test('Continue editing hands off to the chat and is omitted without one', () => {
  const { onContinueEditing } = toolbar(draft());
  fireEvent.click(screen.getByRole('button', { name: 'Continue editing' }));
  expect(onContinueEditing).toHaveBeenCalledTimes(1);
  cleanup();
  const state = draft();
  render(<DraftGraphToolbar state={state} presentation={buildGraphReview(state)} index={0} onChange={jest.fn()} onSource={jest.fn()}
    onAccept={() => true} onDiscard={() => true} />);
  expect(screen.queryByRole('button', { name: 'Continue editing' })).toBeNull();
  expect(screen.getByRole('button', { name: 'Accept' })).toBeVisible();
});

test('a Git review keeps its own toolbar, with no draft decisions', () => {
  const state = draft(null as unknown as undefined);
  // Close, Refresh and code actions sit in the page header for Git review (localBrowserApp tests).
  render(<GraphReviewToolbar state={state} presentation={buildGraphReview(state)} mode="diff" index={0} onMode={jest.fn()} onChange={jest.fn()}
    onRefresh={jest.fn()} />);
  expect(screen.getByRole('group', { name: 'Graph revision' })).toBeVisible();
  expect(screen.queryByRole('group', { name: 'Draft decision' })).toBeNull();
  for (const name of ['Accept', 'Discard', 'Continue editing']) expect(screen.queryByRole('button', { name })).toBeNull();
});
