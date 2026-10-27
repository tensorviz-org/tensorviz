import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { CandidateReviewPanel } from '../../ide/CandidateReviewPanel';
import { LocalBrowserContext, type LocalBrowserContextValue } from '../../ide/localBrowserContext';
import type { BrowserCandidateState } from '../../ide/localBrowserCandidateClient';
import type { CandidateRecord, AppliedSourceObservation } from '../../ide/protocol.generated';
import corpus from '../../../../tests/fixtures/contracts/local-browser-protocol.json';
const applied = (): AppliedSourceObservation => JSON.parse(JSON.stringify(corpus)).valid
  .find((item: { messageId: string }) => item.messageId === 'applied-source-checked').result.applied;

function harness() {
  let state: BrowserCandidateState = { selectedId: 'proposal', busy: false, reviews: [{ id: 'proposal', viewId: '1',
    target: { workspaceFolder: '/work', file: 'model.py', cls: 'Model', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 },
    open: true, prepared: true, candidate: { staticCheck: 'passed', runtime: 'not_run', freshness: 'unverified',
      capture: 'unavailable', inputs: 'unavailable', state: 'unavailable', cleanup: 'not_run', semanticPostconditions: 'not_checked', diagnostics: [] }, records: [], detail: '' }] };
  const listeners = new Set<() => void>();
  const request = jest.fn(async (_id: string, action: any) => ({ candidates: state.reviews[0].records,
    page: action.query ? { ...action.query, node: { key: null, kind: 'object', count: 0 }, items: [], total: 0, nextOffset: null } : null, detail: 'Historical evidence.' }));
  const run = jest.fn(async () => ({ candidates: null, page: null, detail: '' }));
  const client = { snapshot: () => JSON.parse(JSON.stringify(state)), onChange(fn: () => void) { listeners.add(fn); return () => { listeners.delete(fn); }; },
    request, run, cancelRead: jest.fn(), dispose: jest.fn(), select: jest.fn() };
  render(<LocalBrowserContext.Provider value={{ sourceEdits: { candidates: client } } as unknown as LocalBrowserContextValue}>
    <CandidateReviewPanel embedded />
  </LocalBrowserContext.Provider>);
  return { run, request, update(fn: (value: BrowserCandidateState) => void) { act(() => { fn(state); listeners.forEach(fn => fn()); }); } };
}

test('execution is explicit and preparation/coverage/state/postconditions remain independent', async () => {
  const h = harness(); expect(h.run).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Run reviewed candidate' })); expect(h.run).toHaveBeenCalledWith('proposal');
  h.update(state => { state.reviews[0].candidate = { ...state.reviews[0].candidate!, runId: 'run2', runtime: 'running', cleanup: 'pending' }; });
  expect(screen.getByRole('button', { name: 'Run reviewed candidate' })).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Cancel candidate run' }));
  expect(h.request).toHaveBeenCalledWith('proposal', { kind: 'cancel', runId: 'run2' });
  h.update(state => { state.reviews[0].candidate = { ...state.reviews[0].candidate!, runtime: 'succeeded',
    capture: 'partial', inputs: 'same', state: 'different', cleanup: 'complete' }; });
  expect(screen.getByText('Runtime').nextElementSibling).toHaveTextContent('succeeded');
  expect(screen.getByText('Capture').nextElementSibling).toHaveTextContent('partial');
  expect(screen.getByText('Initial model state').nextElementSibling).toHaveTextContent('different');
  expect(screen.getByText('Requested semantic change').nextElementSibling).toHaveTextContent('not checked');
  expect(screen.queryByRole('button', { name: 'Apply source change' })).not.toBeInTheDocument();
});

test('historical candidate pages remain inspectable after the source review closes', async () => {
  HTMLDialogElement.prototype.showModal = jest.fn();
  const h = harness();
  h.update(state => {
    const item = state.reviews[0]; item.open = false;
    item.candidate = { ...item.candidate!, runId: 'run1', runtime: 'failed', freshness: 'stale', capture: 'partial' };
    item.records = [{ runId: 'run1', startedAt: 1, finishedAt: 2, review: 'historical', summary: item.candidate,
      executionId: 'exec', capture: { captureId: 'copied-capture', sourceId: 'copied-source' }, unavailable: null }];
  });
  expect(screen.getByRole('button', { name: 'Run reviewed candidate' })).toBeDisabled();
  expect(screen.getByText(/Historical candidate evidence/)).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Inspect candidate capture…' }));
  await waitFor(() => expect(h.request).toHaveBeenCalledWith('proposal', { kind: 'inspect', runId: 'run1', query: {
    captureId: 'copied-capture', sourceId: 'copied-source', path: [], offset: 0, limit: 25,
  } }));
});

test('a new run never displays the older successful capture as its own result', () => {
  const h = harness();
  h.update(state => {
    const item = state.reviews[0];
    item.records = [{ runId: 'old', startedAt: 1, finishedAt: 2, review: 'open', summary: { ...item.candidate!, runId: 'old', runtime: 'succeeded' },
      executionId: 'old', capture: { captureId: 'old', sourceId: 'old' }, unavailable: null } as CandidateRecord];
    item.candidate = { ...item.candidate!, runId: 'new', runtime: 'running', cleanup: 'pending' };
  });
  expect(screen.getByText('Runtime').nextElementSibling).toHaveTextContent('running');
  expect(screen.queryByRole('button', { name: 'Inspect candidate capture…' })).not.toBeInTheDocument();
  fireEvent.change(screen.getByRole('combobox', { name: 'Candidate run' }), { target: { value: 'old' } });
  expect(screen.getByText('Runtime').nextElementSibling).toHaveTextContent('succeeded');
});

test('lost receipts can be retried after Apply without granting a fresh execution action', () => {
  const h = harness(); h.update(state => { state.reviews[0].open = false; state.reviews[0].retryPending = true; });
  fireEvent.click(screen.getByRole('button', { name: 'Retry run receipt' }));
  expect(h.run).toHaveBeenCalledWith('proposal');
});

test('cleanup failure disables a new execution while leaving result inspection available', () => {
  const h = harness();
  h.update(state => { state.reviews[0].candidate = { ...state.reviews[0].candidate!, runtime: 'refused', cleanup: 'failed' }; });
  expect(screen.getByRole('button', { name: 'Run reviewed candidate' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Refresh results' })).toBeEnabled();
  expect(screen.getByText('Temporary-file cleanup').nextElementSibling).toHaveTextContent('failed');
});

test('a semantic pass explains its observed scope beside stale source and changed state', () => {
  const h = harness();
  h.update(state => { state.reviews[0].candidate = { ...state.reviews[0].candidate!, runtime: 'succeeded', cleanup: 'complete',
    freshness: 'stale', state: 'different', capture: 'complete', semanticPostconditions: 'verified',
    postconditions: { version: 1, executionId: 'execution', captureId: 'capture', recipe: 'binary_scalar_v1', scope: 'observed_expression',
      status: 'verified', reason: 'observed_requested_literal', bindingId: 'binding', observedOperations: 2 } }; });
  expect(screen.getByText('Requested semantic change').nextElementSibling).toHaveTextContent(
    'Requested literal observed in 2 captured occurrences. Other executions are unverified.');
  expect(screen.getByText('Source freshness at check').nextElementSibling).toHaveTextContent('stale');
  expect(screen.getByText('Initial model state').nextElementSibling).toHaveTextContent('different');
});

test('partial capture gives an explicit unverified semantic result despite successful runtime', () => {
  const h = harness();
  h.update(state => { state.reviews[0].candidate = { ...state.reviews[0].candidate!, runtime: 'succeeded', cleanup: 'complete',
    capture: 'partial', semanticPostconditions: 'unavailable', postconditions: { version: 1, executionId: 'execution', captureId: 'capture',
      recipe: 'binary_scalar_v1', scope: 'observed_expression', status: 'unavailable', reason: 'capture_incomplete', bindingId: null, observedOperations: 0 } }; });
  expect(screen.getByText('Runtime').nextElementSibling).toHaveTextContent('succeeded');
  expect(screen.getByText('Requested semantic change').nextElementSibling).toHaveTextContent('Unverified. Capture is incomplete.');
});

test.each(['functional_activation_v1', 'module_activation_v1', 'straight_line_tensor_v1', 'straight_line_unary_insert_v1', 'straight_line_unary_delete_v1'] as const)('scopes %s evidence to observed occurrences', recipe => {
  const h = harness();
  h.update(state => { state.reviews[0].candidate = { ...state.reviews[0].candidate!, runtime: 'succeeded', cleanup: 'complete',
    capture: 'complete', semanticPostconditions: 'verified', postconditions: { version: 1, executionId: 'execution', captureId: 'capture',
      recipe, scope: 'observed_expression', status: 'verified',
      reason: recipe === 'straight_line_tensor_v1' ? 'observed_requested_argument'
        : recipe === 'straight_line_unary_insert_v1' ? 'observed_requested_insertion'
          : recipe === 'straight_line_unary_delete_v1' ? 'observed_requested_deletion' : 'observed_requested_activation', bindingId: 'binding', observedOperations: 2 } }; });
  expect(screen.getByText('Requested semantic change').nextElementSibling).toHaveTextContent(
    `Requested ${recipe === 'straight_line_tensor_v1' ? 'argument' : recipe === 'straight_line_unary_insert_v1' ? 'insertion'
      : recipe === 'straight_line_unary_delete_v1' ? 'deletion' : 'activation'} observed in 2 captured occurrences. Other executions are unverified.`);
});

test('constructor entry evidence is separate from forward behavior and operation counts', () => {
  const h = harness();
  h.update(state => { state.reviews[0].candidate = { ...state.reviews[0].candidate!, runtime: 'succeeded', cleanup: 'complete',
    capture: 'complete', semanticPostconditions: 'verified', postconditions: { version: 1, executionId: 'execution', captureId: 'capture',
      recipe: 'constructor_scalar_v1', scope: 'observed_initializer_entries', status: 'verified', reason: 'observed_requested_constructor',
      bindingId: 'constructor', observedOperations: 0, observedInitializers: 3 } }; });
  expect(screen.getByText('Requested semantic change').nextElementSibling).toHaveTextContent(
    'Requested constructor value observed in 3 retained initializer entries. Other constructors and later behavior are unverified.');
});

test('configuration evidence separates observed constructor arguments from other references', () => {
  const h = harness();
  h.update(state => { state.reviews[0].candidate = { ...state.reviews[0].candidate!, runtime: 'succeeded', cleanup: 'complete',
    capture: 'complete', semanticPostconditions: 'verified', postconditions: { version: 1, executionId: 'execution', captureId: 'capture',
      recipe: 'configuration_scalar_v1', scope: 'observed_configuration_arguments', status: 'verified', reason: 'observed_requested_configuration',
      bindingId: 'constructor', observedOperations: 0, observedInitializers: 3,
      configurationUses: [{ bindingId: 'constructor', sourceArgumentPath: ['args', 0] }, { bindingId: 'constructor', sourceArgumentPath: ['args', 1] }] } }; });
  expect(screen.getByText('Requested semantic change').nextElementSibling).toHaveTextContent(
    'Requested configuration value observed at 2 direct constructor arguments across 3 retained initializer entries. Other references and later behavior are unverified.');
});

test('applied-source checks require a captured native Apply and never start candidate execution', () => {
  const h = harness(); expect(screen.queryByRole('button', { name: 'Check applied source' })).not.toBeInTheDocument();
  h.update(state => { const item = state.reviews[0]; item.open = false; item.outcome = 'applied'; });
  expect(screen.queryByRole('button', { name: 'Check applied source' })).not.toBeInTheDocument();
  h.update(state => { state.reviews[0].captured = true; });
  fireEvent.click(screen.getByRole('button', { name: 'Check applied source' }));
  expect(h.request).toHaveBeenCalledWith('proposal', { kind: 'reconcile' }); expect(h.run).not.toHaveBeenCalled();
  h.update(state => { state.reviews[0].outcome = 'cancelled'; });
  expect(screen.queryByRole('button', { name: 'Check applied source' })).not.toBeInTheDocument();
});

test('recorded applied effects explain partial dependencies and remain separate from historical candidate results', () => {
  const h = harness(); h.update(state => {
    const item = state.reviews[0]; item.captured = true; item.open = false; item.outcome = 'applied'; item.applied = applied();
    item.applied.captureFreshness = 'unavailable'; item.applied.reconciliation!.state = 'different';
  });
  expect(screen.getByText('Source at check').nextElementSibling).toHaveTextContent('Reviewed changes saved');
  expect(screen.getByText('Saved setup at check').nextElementSibling).toHaveTextContent('unavailable');
  expect(screen.getByText('Recorded requested change').nextElementSibling).toHaveTextContent('Requested literal observed');
  expect(screen.getByText('Tracked dependency scope').nextElementSibling).toHaveTextContent('untracked dependencies are unverified');
  expect(screen.getByText('State compared with original capture').nextElementSibling).toHaveTextContent('different');
  expect(screen.getByText('Runtime').nextElementSibling).toHaveTextContent('not run');
  h.update(state => { state.reviews[0].appliedStale = true; });
  expect(screen.getByText(/graph or source has changed since this check/)).toBeVisible();
});

test.each(['reviewed_unsaved', 'reverted'] as const)('%s offers honest guidance without a captured-effect pass', source => {
  const h = harness(); h.update(state => {
    const item = state.reviews[0]; item.captured = true; item.open = false; item.outcome = 'applied';
    item.applied = { ...applied(), source, captureFreshness: 'stale', reconciliation: null,
      unavailable: source === 'reviewed_unsaved' ? 'source_not_saved' : 'source_changed' };
  });
  expect(screen.getByText('Source at check').nextElementSibling).toHaveTextContent(source === 'reviewed_unsaved' ? 'unsaved buffer' : 'Original source restored');
  expect(screen.queryByText('Recorded requested change')).not.toBeInTheDocument(); expect(h.run).not.toHaveBeenCalled();
});
