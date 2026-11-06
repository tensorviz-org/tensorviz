/** Per-review candidate state. It never updates the live graph or attempt ledger. */
import { EventEmitter } from 'vscode';
import type { ExecutionObservation } from '@tensorviz/graph-host/executionObservation';
import type { CandidateRunResult } from '@tensorviz/graph-host/candidateExecution';
import { checkpointEvidenceText, checkpointReviewSummary } from '@tensorviz/graph-host/checkpointEvidence';
import type { CaptureSnapshot } from '@tensorviz/graph-host/semanticCapture';
import type { CandidateHistory } from './candidateHistory';
import { candidatePostconditionsText, type CandidateReviewSnapshot } from '@tensorviz/contracts/protocol';
export type { CandidateReviewSnapshot } from '@tensorviz/contracts/protocol';

const initial = (): CandidateReviewSnapshot => ({ staticCheck: 'passed', runtime: 'not_run', freshness: 'unverified',
  capture: 'unavailable', inputs: 'unavailable', state: 'unavailable', semanticPostconditions: 'not_checked', cleanup: 'not_run', diagnostics: [] });
const identity = (base: string | null, candidate?: string | null): CandidateReviewSnapshot['inputs'] =>
  !base || !candidate ? 'unavailable' : base === candidate ? 'same' : 'different';

export class CandidateReview {
  private value = initial();
  private readonly changes = new EventEmitter<CandidateReviewSnapshot>();
  readonly onDidChange = this.changes.event;
  private pending?: Promise<CandidateReviewSnapshot>;
  private controller?: AbortController;
  private result?: CandidateRunResult;
  private checkpointText = '';
  private closed = false;
  private readonly abort = () => this.cancel();
  private readonly inputDigest: string | null;
  private readonly stateDigest: string | null;

  constructor(base: CaptureSnapshot, private readonly parent: AbortSignal,
    private readonly current: () => boolean,
    private readonly execute: (signal: AbortSignal) => Promise<CandidateRunResult>,
    private readonly retention?: { history: CandidateHistory; requestId: string; baseline?: ExecutionObservation }) {
    this.inputDigest = base.inputDigest; this.stateDigest = base.stateDigest;
    parent.addEventListener('abort', this.abort, { once: true });
  }

  snapshot(): CandidateReviewSnapshot {
    if (this.value.freshness === 'current') {
      let current = false;
      try { current = this.current(); } catch { /* unavailable source context */ }
      if (!current) this.value.freshness = 'stale';
    }
    if (this.value.runId) this.retention?.history.update(this.retention.requestId, this.value.runId, this.value);
    return structuredClone(this.value);
  }

  // A new run cannot prove cleanup of a previous run's private workspace.
  get available(): boolean { return !this.closed && !this.parent.aborted && this.value.cleanup !== 'failed'; }

  run(): Promise<CandidateReviewSnapshot> {
    if (this.pending) return this.pending;
    if (!this.available) return Promise.resolve(this.snapshot());
    this.result = undefined;
    this.checkpointText = '';
    this.value = { ...initial(), runtime: 'running', cleanup: 'pending' };
    const runId = this.retention?.history.begin(this.retention.requestId, this.value);
    if (runId) this.value.runId = runId;
    const controller = new AbortController(); this.controller = controller;
    // Publish only after assigning pending: synchronous listeners cannot start a
    // second run while the first request is being announced.
    this.pending = Promise.resolve().then(async () => {
      let terminal: CandidateReviewSnapshot;
      try {
        const result = await this.execute(controller.signal);
        if (result.cleanupFailed) this.value = { ...initial(), runtime: 'refused', cleanup: 'failed' };
        else if (controller.signal.aborted) this.value = { ...initial(), runtime: 'cancelled', cleanup: 'complete' };
        else {
          this.result = result;
          const { outcome, evidence } = result, capture = evidence?.capture;
          this.checkpointText = evidence?.execution.plan?.execution?.checkpoint !== undefined ? checkpointEvidenceText(evidence.checkpoint) : '';
          this.value = { ...initial(), cleanup: 'complete',
            runtime: !outcome.ok && outcome.cancelled ? 'cancelled' : !outcome.ok && outcome.timedOut ? 'timed_out'
              : !evidence ? 'refused' : outcome.ok ? 'succeeded' : 'failed',
            freshness: evidence ? outcome.executionCurrent === true && this.current() ? 'current' : 'stale' : 'unverified',
            capture: capture?.status ?? 'unavailable', inputs: identity(this.inputDigest, capture?.inputDigest),
            state: identity(this.stateDigest, capture?.stateDigest),
            ...(evidence?.checkpoint ? { checkpoint: checkpointReviewSummary(evidence.checkpoint) } : {}),
            ...(evidence?.postconditions ? { postconditions: structuredClone(evidence.postconditions),
              semanticPostconditions: evidence.postconditions.status } : {}),
            diagnostics: [...new Set(capture?.diagnostics.map(item => item.code).filter(code => /^[a-z_]{1,80}(?![\s\S])/.test(code)))].slice(0, 16),
          };
        }
      } catch {
        this.result = undefined;
        this.checkpointText = '';
        this.value = { ...initial(), cleanup: 'unavailable', runtime: controller.signal.aborted ? 'cancelled' : 'refused' };
      }
      finally {
        if (runId) {
          this.value.runId = runId;
          this.retention?.history.complete(this.retention.requestId, runId, this.snapshot(), this.result?.evidence, this.retention.baseline);
        }
        terminal = this.snapshot();
        this.pending = undefined; this.controller = undefined;
        this.changes.fire(structuredClone(terminal));
      }
      return terminal;
    });
    this.changes.fire(this.snapshot());
    return this.pending;
  }

  cancel(): void { this.controller?.abort(); }

  resultText(): string {
    return candidateReviewText(this.snapshot()) + this.checkpointText;
  }

  async refresh(): Promise<CandidateReviewSnapshot> {
    const result = this.result;
    if (result && this.value.freshness === 'current') {
      let current = false;
      try { current = !!await result.outcome.recheckExecution?.(this.parent); } catch { /* stale evidence */ }
      if (result === this.result && !current) this.value.freshness = 'stale';
    }
    this.changes.fire(this.snapshot());
    return this.snapshot();
  }

  /** Apply/Discard waits for private execution cleanup, without requiring a pass. */
  async settle(): Promise<void> {
    this.closed = true; this.cancel();
    await this.pending;
  }

  async finish(): Promise<void> {
    await this.settle();
    this.parent.removeEventListener('abort', this.abort);
    this.result = undefined;
    this.checkpointText = '';
    this.snapshot();
    this.retention?.history.close(this.retention.requestId);
    this.changes.dispose();
  }
}

export function candidateReviewText(snapshot: CandidateReviewSnapshot): string {
  return `# Candidate validation\n\nStatic source checks: passed\nRuntime: ${snapshot.runtime.replace('_', ' ')}\n`
    + `Freshness: ${snapshot.freshness}\nCapture: ${snapshot.capture}\nActual inputs: ${snapshot.inputs}\n`
    + `Pre-call model state: ${snapshot.state}\nSemantic postconditions: ${candidatePostconditionsText(snapshot)}\n`
    + `Temporary-file cleanup: ${snapshot.cleanup}\n`
    + (snapshot.cleanup === 'failed' ? '\nPrivate temporary files may remain. No source change was applied.\n' : '')
    + (snapshot.diagnostics.length ? `\nDiagnostics: ${snapshot.diagnostics.join(', ')}\n` : '')
    + '\nThis result describes the reviewed source in a temporary workspace. It does not validate the live editor buffer.\n'
    + 'A successful forward covers this input only. Apply source change and Discard proposal remain separate actions.\n';
}
