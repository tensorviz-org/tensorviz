import { LocalBrowserHostPublish } from '@tensorviz/graph-host/localBrowserHostTransport';
import { LocalBrowserSync, type LocalBrowserValidationPhase } from './localBrowserSync';
import { LocalBrowserValidateRequestMessage } from '@tensorviz/contracts/protocol';
import { lookupProfile, type ProfileLookupOptions } from '@tensorviz/graph-host/profileValidation';
import { ValidationSummaryStore, validationDetail, validationPhaseDetails } from '@tensorviz/graph-host/validationSummary';
import type { LocalBrowserTargetIdentity, LocalBrowserValidationOutcome, ProjectionPayload } from '@tensorviz/contracts/protocol';
import type { AttemptHandle } from '@tensorviz/graph-host/attemptLedger';

import { ExecutionReview } from '@tensorviz/graph-host/executionReview';

/** Own one manual request; the broker owns per-view reply routing. */
export class LocalBrowserValidation {
  private pending?: { id: string; controller: AbortController };
  private readonly seen = new Set<string>();
  constructor(private readonly sync: LocalBrowserSync, private readonly publish: (message: LocalBrowserHostPublish) => unknown,
    private readonly profiles?: ProfileLookupOptions, private readonly summaries?: ValidationSummaryStore, private readonly review?: ExecutionReview) {}

  /** Lets a transport preserve another surface's in-flight request on disconnect. */
  get activeRequestId(): string | undefined { return this.pending?.id; }

  cancel(id?: string): void {
    if (!id || this.pending?.id === id) this.pending?.controller.abort();
  }

  handle(message: LocalBrowserValidateRequestMessage): void {
    if (this.seen.has(message.requestId)) return;
    this.seen.add(message.requestId);
    if (this.seen.size > 256) this.seen.delete(this.seen.values().next().value!);
    // A review waiting for Run has not started the model, so the attempt and its
    // running summary begin with the invocation (at once when there is no review).
    let observe: ReturnType<ValidationSummaryStore['begin']> | undefined;
    const begin = () => observe ??= this.summaries?.begin(message.requestId, message.target, 'manual');
    if (!this.review) begin();
    const record = (outcome: { status: LocalBrowserValidationOutcome['status']; detail: string },
      measured?: { target: LocalBrowserTargetIdentity; revision: number }) => {
      if (observe || outcome.status !== 'cancelled') { begin()?.(outcome.status, outcome.detail, measured); return; }
      // Nothing ran: a dismissed review shows the last result again.
      const previous = this.summaries?.snapshot(message.target);
      if (previous) this.publish({ type: 'validationState', ...previous });
    };
    const reply = (status: 'refused', detail: string) => {
      record({ status, detail });
      this.publish({ type: 'validateResult', requestId: message.requestId, target: message.target, status, detail });
    };
    if (this.pending) {
      reply('refused', 'Another validation is running. Wait for it to finish and try again.');
      return;
    }
    const controller = new AbortController();
    this.pending = { id: message.requestId, controller };
    const cancelled = () => ({ status: 'cancelled' as const, detail: 'Validation cancelled.' });
    let measured: { target: LocalBrowserTargetIdentity; revision: number } | undefined;
    void this.run(message, controller.signal, (target, projection) => {
      measured = { target: { ...target }, revision: projection.revision };
    }, phase => { this.review?.progress(validationPhaseDetails[phase]); observe?.('running', validationPhaseDetails[phase]); }, () => begin()?.attempt).then(result => {
      const outcome = controller.signal.aborted ? cancelled() : { ...result, detail: validationDetail(result.detail) };
      record(outcome, measured);
      this.publish({ type: 'validateResult', target: message.target, requestId: message.requestId,
        ...outcome });
    }).catch((error: unknown) => {
      const outcome = controller.signal.aborted || error instanceof Error && error.name === 'AbortError' ? cancelled()
        : { status: 'refused' as const, detail: 'Validation could not complete. Reconnect and try again.' };
      record(outcome);
      this.publish({ type: 'validateResult', target: message.target, requestId: message.requestId, ...outcome });
    })
      .finally(() => { if (this.pending?.controller === controller) this.pending = undefined; });
  }

  /** Legacy view requests still require explicit setup. */
  private async run(message: LocalBrowserValidateRequestMessage, signal: AbortSignal,
    measured: (target: LocalBrowserTargetIdentity, projection: ProjectionPayload) => void,
    progress: (phase: LocalBrowserValidationPhase) => void, attempt: () => AttemptHandle | undefined) {
    const missing = () => ({ status: 'refused' as const, detail: 'Choose model setup and inputs with Open as Graph before capturing.' });
    if (!this.profiles) return missing();
    const lookup = await lookupProfile(this.profiles, message.target.workspaceFolder,
      { file: message.target.file, cls: message.target.cls }, signal);
    if (signal.aborted) return { status: 'cancelled' as const, detail: 'Validation cancelled.' };
    if (lookup.ok) {
      const execute = (runSignal: AbortSignal, device?: string) => this.sync.validate(message.target, runSignal, undefined, progress, measured,
        undefined, lookup.run.execution, undefined, lookup.run.timeoutMs, lookup.guard, attempt(), device,
        // Run fresh executes even when an identical result is stored, and replaces it.
        message.fresh ? 'bypass' : 'allow');
      return this.review ? this.review.run(message.target, signal, { kind: 'capture',
        execution: lookup.run.execution, profileName: lookup.profile, interpreter: lookup.guard.spec.interpreter.executable,
        ...(message.fresh ? { fresh: true } : {}) }, execute)
        : execute(signal);
    }
    return lookup.kind === 'none' ? missing()
      : { status: lookup.kind === 'cancelled' ? 'cancelled' as const : 'refused' as const, detail: lookup.detail };
  }
}
