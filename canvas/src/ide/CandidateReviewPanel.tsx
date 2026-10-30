import React from 'react';
import { LocalBrowserContext } from './localBrowserContext';
import type { BrowserCandidateState, LocalBrowserCandidateClient } from './localBrowserCandidateClient';
import { candidatePostconditionsText, type CaptureInspectionQuery, type AppliedSourceObservation, type CandidateCheckpointPage } from './protocol.generated';
import { CaptureDetails } from '../canvas/CaptureDetails';
import './candidateReview.css';

const EMPTY: BrowserCandidateState = { reviews: [], selectedId: null, busy: false };
const label = (value: string) => value.replace(/_/g, ' ');
const rootQuery = { path: [] };
const sourceLabels: Record<AppliedSourceObservation['source'], string> = {
  reviewed_unsaved: 'Reviewed changes in unsaved buffer', reviewed_saved: 'Reviewed changes saved',
  reverted: 'Original source restored', changed: 'Source changed again', unavailable: 'Source unavailable',
};
const unavailableLabels = {
  source_not_saved: 'Save in VS Code, then Validate the model to capture the applied source.',
  source_changed: 'The editor no longer matches the reviewed change.', source_unavailable: 'The source file could not be checked.',
  evidence_unavailable: 'The private source evidence is no longer retained or exceeded its limit.',
  capture_unavailable: 'No finished capture from after Apply is available. Save and Validate the model.',
  check_stale: 'Source, setup or selected capture changed during the check. Check again.',
  check_failed: 'The source check could not finish. Try again.',
};

function CheckpointDetails({ client, proposalId, runId, busy }: {
  client: LocalBrowserCandidateClient; proposalId: string; runId: string; busy: boolean;
}) {
  const [page, setPage] = React.useState<CandidateCheckpointPage | null>(null);
  const [detail, setDetail] = React.useState('');
  React.useEffect(() => () => client.cancelRead(), [client]);
  const read = async (offset: number) => {
    const result = await client.request(proposalId, { kind: 'checkpoint', runId, offset, limit: 20 });
    setPage(result.checkpoint ?? null); setDetail(result.checkpoint ? '' : result.detail);
  };
  return <div className="candidate-checkpoint-details">
    <div className="candidate-review-actions">
      <button disabled={busy} onClick={() => { void read(0); }}>{page ? 'First checkpoint page' : 'Inspect checkpoint load…'}</button>
      {page?.nextOffset != null && <button disabled={busy} onClick={() => { void read(page.nextOffset!); }}>Next checkpoint entries</button>}
    </div>
    {page && <>
      <p>Checkpoint entries from {Math.min(page.offset + 1, page.count)} of {page.count}.
        {page.nextOffset !== null ? ' More entries are available.' : ' End of retained entries.'}</p>
      <pre tabIndex={0} aria-label="Checkpoint-load evidence">{page.text}</pre>
    </>}
    {detail && <p role="status">{detail}</p>}
  </div>;
}

function AppliedSourceResult({ value, stale }: { value: AppliedSourceObservation; stale?: boolean }) {
  const result = value.reconciliation;
  const sourceSet = value.sourceSetComparison;
  return <div aria-label="Applied-source result">
    <p>Checked {new Date(value.checkedAt).toLocaleTimeString()}. {stale ? 'The graph or source has changed since this check.' : 'This describes the source and setup at check time.'}</p>
    <dl>
      <dt>Source at check</dt><dd>{sourceLabels[value.source]}</dd>
      <dt>Saved setup at check</dt><dd>{label(value.captureFreshness)}</dd>
      {value.receipt.files && <><dt>Reviewed files</dt><dd>{value.receipt.files.map(file => file.file).join(', ')}</dd></>}
      {result && <>
        <dt>Recorded source comparison</dt><dd>{label(result.reason)}</dd>
        <dt>Tracked dependency scope</dt><dd>{result.dependencyScope}{result.dependencyScope === 'partial' ? ' — untracked dependencies are unverified' : ''}</dd>
        <dt>Inputs compared with original capture</dt><dd>{result.inputs}</dd>
        <dt>State compared with original capture</dt><dd>{result.state}</dd>
        <dt>Recorded requested change</dt><dd>{candidatePostconditionsText({ postconditions: result.postconditions ?? undefined })}</dd>
      </>}
      {sourceSet && <>
        <dt>Recorded source comparison</dt><dd>{label(sourceSet.reason)}</dd>
        <dt>Tracked dependency scope</dt><dd>{sourceSet.dependencyScope}{sourceSet.dependencyScope === 'partial' ? ' — untracked dependencies are unverified' : ''}</dd>
        <dt>Inputs compared with original capture</dt><dd>{sourceSet.inputs}</dd>
        <dt>State compared with original capture</dt><dd>{sourceSet.state}</dd>
        {sourceSet.comparison && <>
          <dt>Captured computation</dt><dd>{sourceSet.comparison.equivalent === true ? 'Equivalent within the captured paths.'
            : sourceSet.comparison.equivalent === false ? 'Recorded computation differs.' : 'Equivalence is unresolved.'}</dd>
          <dt>Observed changes</dt><dd>{sourceSet.comparison.changedFieldCount} computation fields; {sourceSet.comparison.observedFieldCount} observation fields.</dd>
          <dt>Unresolved evidence</dt><dd>{sourceSet.comparison.unresolvedFieldCount} fields; {sourceSet.comparison.unmatchedRecordCount} unmatched records; {sourceSet.comparison.unresolvedConnectionCount} connections.</dd>
        </>}
        <dt>Requested behavior</dt><dd>Not checked. A source-set comparison does not establish task correctness or training quality.</dd>
      </>}
    </dl>
    {value.unavailable && <p>{unavailableLabels[value.unavailable]}</p>}
    {value.receipt.unavailable && <p>Retained evidence: {label(value.receipt.unavailable)}.</p>}
  </div>;
}

/** Review controls stay separate from the live canvas and native Apply authority. */
export function CandidateReviewPanel({ embedded = false }: { embedded?: boolean }) {
  const context = React.useContext(LocalBrowserContext), client = context?.sourceEdits?.candidates;
  const [state, setState] = React.useState(() => client?.snapshot() ?? EMPTY);
  const [selectedRun, setSelectedRun] = React.useState('');
  React.useEffect(() => {
    setState(client?.snapshot() ?? EMPTY);
    return client?.onChange(() => setState(client.snapshot()));
  }, [client]);
  const review = state.reviews.find(item => item.id === state.selectedId), proposalId = review?.id;
  const latest = review?.candidate;
  const record = review?.records.find(item => item.runId === (selectedRun || latest?.runId));
  const runId = record?.runId;
  const summary = selectedRun ? record?.summary : latest;
  React.useEffect(() => {
    if (client && proposalId && review?.prepared && latest?.runtime !== 'running') void client.request(proposalId, { kind: 'list' });
  }, [client, proposalId, review?.prepared, review?.open, latest?.runId, latest?.runtime]);
  const reader = React.useMemo(() => ({
    async inspect(query: CaptureInspectionQuery) {
      if (!client || !proposalId || !runId) return { page: null, detail: 'Select a retained candidate result.' };
      const result = await client.request(proposalId, { kind: 'inspect', runId, query });
      return { page: result.page, detail: result.detail };
    },
    cancel: () => client?.cancelRead(), dispose: () => client?.cancelRead(),
  }), [client, proposalId, runId]);
  if (!client || !review) return null;
  const running = latest?.runtime === 'running';
  const canRun = !state.busy && (review.retryPending || (review.open && review.prepared && !!latest && !running && latest.cleanup !== 'failed'));
  return <details className={`candidate-review ${embedded ? 'candidate-review-embedded' : 'candidate-review-floating'}`} open={embedded || undefined}>
    <summary>Candidate review · {summary ? label(summary.runtime) : review.open ? 'preparing' : 'closed'}</summary>
    <section aria-label="Candidate review controls">
      <label>Source review<select aria-label="Candidate source review" value={review.id} onChange={event => { client.select(event.target.value); setSelectedRun(''); }}>
        {state.reviews.map((item, i) => <option key={item.id} value={item.id}>{i + 1}. {item.target.cls} · {item.target.file}{item.open ? ' · open' : ' · historical'}</option>)}
      </select></label>
      <p>{review.open ? 'Review and Apply/Discard the full source diff in VS Code.' : 'Historical candidate evidence. This does not validate the live editor buffer.'}</p>
      {review.prepared && !latest && <p>This review has no retained execution recipe. Save and Validate the edited source to check it.</p>}
      <div className="candidate-review-actions">
        <button disabled={!canRun} onClick={() => { setSelectedRun(''); void client.run(review.id); }}>{review.retryPending ? 'Retry run receipt' : 'Run reviewed candidate'}</button>
        <button disabled={state.busy || !running || !latest?.runId} onClick={() => { if (latest?.runId) void client.request(review.id, { kind: 'cancel', runId: latest.runId }); }}>Cancel candidate run</button>
        <button disabled={state.busy} onClick={() => { void client.request(review.id, { kind: 'list' }); }}>Refresh results</button>
      </div>
      {review.records.length > 0 && <label>Run<select aria-label="Candidate run" value={selectedRun} onChange={event => setSelectedRun(event.target.value)}>
        <option value="">Latest candidate</option>
        {review.records.map((item, i) => <option key={item.runId} value={item.runId}>Run {i + 1} · {label(item.summary.runtime)} · {new Date(item.startedAt).toLocaleTimeString()}</option>)}
      </select></label>}
      {summary && <dl aria-live="polite">
        <dt>Static checks</dt><dd>{summary.staticCheck}</dd><dt>Runtime</dt><dd>{label(summary.runtime)}</dd>
        <dt>Source freshness at check</dt><dd>{summary.freshness}</dd><dt>Capture</dt><dd>{summary.capture}</dd>
        <dt>Actual inputs</dt><dd>{summary.inputs}</dd><dt>Initial model state</dt><dd>{summary.state}</dd>
        <dt>Requested semantic change</dt><dd>{candidatePostconditionsText(summary)}</dd>
        {summary.checkpoint && <><dt>Checkpoint load</dt><dd>{summary.checkpoint.loadStatus}. {summary.checkpoint.complete ? 'Complete' : 'Partial'} metadata.
          {' '}{summary.checkpoint.shapeDifferences} shape differences, {summary.checkpoint.dtypeDifferences} dtype differences;
          {' '}{summary.checkpoint.registeredOnly} registered-only and {summary.checkpoint.checkpointOnly} checkpoint-only entries.
          {summary.checkpoint.unknownPresence > 0 ? ` Presence unknown for ${summary.checkpoint.unknownPresence} entries.` : ''}</dd></>}
        <dt>Temporary-file cleanup</dt><dd>{label(summary.cleanup)}</dd>
      </dl>}
      {summary?.checkpoint && runId && <CheckpointDetails key={`${review.id}:${runId}`} client={client}
        proposalId={review.id} runId={runId} busy={state.busy} />}
      {record?.capture && !record.unavailable && <CaptureDetails key={record.runId} label="Reviewed candidate" buttonLabel="Inspect candidate capture…"
        reference={{ version: 1, ...record.capture, kind: 'input', semanticId: record.capture.captureId }} client={reader} stale initialQuery={rootQuery} />}
      {record?.unavailable && <p>Capture evidence: {label(record.unavailable)}.</p>}
      {summary?.diagnostics.length ? <p>Diagnostics: {summary.diagnostics.join(', ')}</p> : null}
      {review.captured && review.outcome === 'applied' && <section className="applied-source-check" aria-label="Applied source">
        <h3>Applied source</h3>
        <button disabled={state.busy} onClick={() => { void client.request(review.id, { kind: 'reconcile' }); }}>Check applied source</button>
        <p>Reads the editor and later capture. Save in VS Code and use Validate to create new runtime evidence.</p>
        {review.applied && <AppliedSourceResult value={review.applied} stale={review.appliedStale} />}
      </section>}
      {review.detail && <p role="status">{review.detail}</p>}
      <small>Runs the reviewed copy with its retained setup. A successful forward covers only that input; it does not apply the edit.</small>
    </section>
  </details>;
}
