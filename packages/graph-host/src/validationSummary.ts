import type { LocalBrowserTargetIdentity, LocalBrowserValidationSummary } from '@tensorviz/contracts/protocol';
import { AttemptLedger } from './attemptLedger';
import type { components } from '@tensorviz/contracts/semanticContracts.generated';

export type LocalBrowserValidationPhase = keyof typeof validationPhaseDetails;

export const validationDetail = (detail: string): string => detail.length > 512 ? detail.slice(0, 511) + '…' : detail;
export const validationPhaseDetails = {
  checking: 'Checking the saved model and selected environment.',
  tracing: 'Running the model with the selected Python environment.',
  projecting: 'Updating tensor shapes on the graph.',
};

/** A bounded view of the latest requested attempt, never a verdict on retained shapes.
 * Full execution records stay with their worker/run owner. */
export class ValidationSummaryStore {
  readonly ledger = new AttemptLedger();
  private latest?: LocalBrowserValidationSummary;
  private generation = 0;
  constructor(private readonly publish: (summary: LocalBrowserValidationSummary) => void) {}

  snapshot(target: LocalBrowserTargetIdentity): LocalBrowserValidationSummary | undefined {
    const latest = this.latest;
    if (!latest || latest.target.workspaceFolder !== target.workspaceFolder
      || latest.target.file !== target.file || latest.target.cls !== target.cls) return undefined;
    return { ...latest, target: { ...latest.target }, evidence: this.ledger.attribution(target) };
  }

  begin(runId: string, target: LocalBrowserTargetIdentity,
    origin: components['schemas']['RunAttempt']['origin'] = 'manual', request: unknown = null) {
    const generation = ++this.generation;
    const requested = { ...target };
    const attempt = this.ledger.begin(runId, requested, origin, request);
    const update = (status: LocalBrowserValidationSummary['status'], detail: string,
      measured?: { target: LocalBrowserTargetIdentity; revision: number }) => {
      if (status !== 'running') attempt.finish(status);
      if (generation !== this.generation) {
        // An older worker can finish after a newer busy refusal. Refresh evidence
        // attribution without replacing the latest requested outcome.
        const current = this.snapshot(requested);
        if (current) this.publish(current);
        return;
      }
      const capture = measured && (status === 'passed' || status === 'failed') ? measured : undefined;
      this.latest = { runId, target: { ...(capture?.target ?? requested) }, status, detail: validationDetail(detail),
        ...(capture ? { projectionRevision: capture.revision } : {}) };
      this.publish(this.snapshot(this.latest.target)!);
    };
    update('running', validationPhaseDetails.checking);
    return Object.assign(update, { attempt });
  }
}
