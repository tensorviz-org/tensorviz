import { createHash, randomUUID } from 'node:crypto';
import type { components } from '@tensorviz/contracts/semanticContracts.generated';
import type { ExecutionSnapshot, ResolvedExecutionPlan, PreparationSnapshot, CapturePreparation } from './executionPlanCore';
import type { LocalBrowserTargetIdentity, LocalBrowserValidationOutcome, LocalBrowserValidationEvidence } from '@tensorviz/contracts/protocol';
import type { TraceOutcome, WorkerEnvelope } from './traceWorkerCore';
import { semanticCapture } from './semanticCapture';
import { CAPTURE_ENVELOPE_MAX_BYTES } from './captureBudgets';

type RunAttempt = components['schemas']['RunAttempt'];
type CaptureState = components['schemas']['CaptureState'];
type CaptureSnapshot = components['schemas']['CaptureSnapshot'];
const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value));
const modelKey = (target: LocalBrowserTargetIdentity) => JSON.stringify([target.workspaceFolder, target.file, target.cls]);

export interface CaptureRecord {
  id: string;
  attemptId: string;
  format: 'legacy' | 'semantic';
  status: CaptureSnapshot['status'];
  forwardStatus: 'passed' | 'failed';
  sourceId: string | null;
  planDigest: string | null;
  preparationDigest: string | null;
  retainedBytes: number;
  unavailable: 'retention_limit' | null;
  semanticRejection?: 'invalid_capture';
  /** Attempts that reused this capture without running (execution cache); the capture's
   *  own `attemptId`, `sourceId` and ids stay those of the attempt that produced it. */
  reusedBy?: string[];
}

export interface AttemptHandle {
  phase(phase: RunAttempt['phase']): void;
  resolved(snapshot: Pick<ExecutionSnapshot, 'plan' | 'applicabilityDigest'>): void;
  prepared(snapshot: Pick<PreparationSnapshot, 'preparation' | 'applicabilityDigest'>): void;
  captured(outcome: TraceOutcome): void;
  finish(status: LocalBrowserValidationOutcome['status']): void;
  snapshot(): RunAttempt;
}

/** Host-owned evidence for one model surface. Three independently retained slots,
 * not a history of unbounded worker payloads. Presentation never establishes capture
 * completeness. Legacy results remain partial until the semantic adapter replaces them. */
export class AttemptLedger {
  private key?: string;
  private scope = 0;
  private sequence = 0;
  private capturedSequence = 0;
  private latest?: AttemptHandle;
  private complete: string | null = null;
  private partial: string | null = null;
  private displayed: string | null = null;
  private planBody?: string;
  private preparationBody?: string;
  private planLabels: LocalBrowserValidationEvidence['setup'] = null;
  private readonly captures = new Map<string, { token: symbol; record: CaptureRecord; body?: string }>();
  private readonly envelopes = new WeakMap<WorkerEnvelope, string>();

  constructor(private readonly maxCaptureBytes = CAPTURE_ENVELOPE_MAX_BYTES) {}

  begin(id: string, target: LocalBrowserTargetIdentity, origin: RunAttempt['origin'], request: unknown): AttemptHandle {
    const key = modelKey(target);
    this.planBody = undefined;
    this.preparationBody = undefined;
    this.planLabels = null;
    if (this.key !== key) {
      this.key = key;
      this.scope++;
      this.complete = this.partial = this.displayed = null;
      this.captures.clear();
    }
    const scope = this.scope, sequence = ++this.sequence;
    const attempt: RunAttempt = { version: 1, id, origin,
      requestDigest: createHash('sha256').update(JSON.stringify({ target, request })).digest('hex'),
      sourceId: null, planDigest: null, phase: 'configuration', status: 'running', startedAt: Date.now(),
      finishedAt: null, captureIds: [], diagnostics: [] };
    let plan: ResolvedExecutionPlan | undefined;
    let preparation: CapturePreparation | undefined;
    let preparationDigest: string | null = null;
    let outcome: { cancelled: boolean; timedOut: boolean } | undefined;
    const open = () => attempt.finishedAt === null;
    const handle: AttemptHandle = {
      phase: phase => { if (open()) attempt.phase = phase; },
      resolved: snapshot => {
        if (!open() || plan || preparation) return;
        plan = clone(snapshot.plan);
        attempt.sourceId = plan.source.id;
        attempt.planDigest = snapshot.applicabilityDigest;
        const body = JSON.stringify(plan);
        if (this.latest === handle && Buffer.byteLength(body) <= Math.min(this.maxCaptureBytes, 16 * 1024 * 1024)) this.planBody = body;
        if (this.latest === handle && (plan.execution?.mode === 'eval' || plan.execution?.mode === 'train' || plan.execution?.mode === 'preserve')
          && typeof plan.execution.device === 'string') {
          this.planLabels = { origin: plan.origin, profileName: plan.profileName,
            mode: plan.execution.mode, device: plan.execution.device };
        }
      },
      prepared: snapshot => {
        if (!open() || plan || preparation) return;
        preparation = clone(snapshot.preparation);
        preparationDigest = snapshot.applicabilityDigest;
        attempt.sourceId = preparation.source.id;
        const body = JSON.stringify(preparation);
        if (this.latest === handle && Buffer.byteLength(body) <= Math.min(this.maxCaptureBytes, 16 * 1024 * 1024)) this.preparationBody = body;
      },
      captured: result => {
        if (!open() || outcome) return;
        outcome = { cancelled: !result.ok && !!result.cancelled, timedOut: !result.ok && !!result.timedOut };
        const diagnostic = result.envelope?.error?.diagnostic;
        if (diagnostic) { attempt.diagnostics = [clone(diagnostic)]; attempt.phase = diagnostic.phase; }
        else if (result.envelope?.result) attempt.phase = 'capture';
        if (!result.envelope?.result) return;
        const diagnostics = (result.envelope.result as { diagnostics?: { semantic_capture?: unknown } }).diagnostics;
        const payload = diagnostics?.semantic_capture;
        if (result.ok && result.reused) {
          // An earlier attempt's identical capture, reused by this one: validate against
          // THAT attempt and plan, never relabel it as this attempt's.
          const reused = result.reused;
          const validated = payload == null ? undefined : semanticCapture(payload, reused.attemptId, reused.plan, reused.planDigest);
          if (validated && validated.id === reused.captureId && reused.planDigest === attempt.planDigest) {
            attempt.captureIds.push(validated.id);
            attempt.reusedCaptureId = validated.id;
            this.envelopes.set(result.envelope, validated.id);
            if (this.scope !== scope || this.key !== key || sequence < this.capturedSequence) return;
            this.capturedSequence = sequence;
            const existing = this.captures.get(validated.id);
            if (existing) {
              existing.record.reusedBy = [...(existing.record.reusedBy ?? []), id].slice(-8);
            } else {
              const body = JSON.stringify({ envelope: result.envelope, plan: reused.plan });
              const size = Buffer.byteLength(body);
              this.captures.set(validated.id, { token: Symbol(), record: { id: validated.id, attemptId: reused.attemptId, format: 'semantic',
                status: validated.status, forwardStatus: 'passed', sourceId: reused.plan.source.id, planDigest: reused.planDigest,
                preparationDigest: null, retainedBytes: size <= this.maxCaptureBytes ? size : 0,
                unavailable: size > this.maxCaptureBytes ? 'retention_limit' : null, reusedBy: [id] },
                ...(size <= this.maxCaptureBytes ? { body } : {}) });
            }
            this.complete = validated.id;
            this.prune();
            return;
          }
          // Not valid as the reused capture: fall through and treat it like any other
          // payload (it then fails validation against this attempt and stays partial).
        }
        const validated = payload == null ? undefined : semanticCapture(payload, id, plan ?? preparation, attempt.planDigest ?? preparationDigest);
        const canonical = validated && !this.captures.has(validated.id) ? validated : undefined;
        // A legacy projection may display measurements from this same forward;
        // retaining canonical evidence does not certify the projection's fidelity.
        const captureId = canonical?.id ?? `capture:${randomUUID()}`;
        attempt.captureIds.push(captureId);
        this.envelopes.set(result.envelope, captureId);
        // A newer busy refusal does not erase the evidence of a still-running
        // attempt. Only a newer actual capture or a retarget supersedes it.
        if (this.scope !== scope || this.key !== key || sequence < this.capturedSequence) return;
        this.capturedSequence = sequence;
        const body = JSON.stringify({ envelope: result.envelope, plan, preparation });
        const size = Buffer.byteLength(body);
        const rootStatus = canonical?.invocations.find(call => call.id === canonical.rootInvocationId)?.status;
        this.captures.set(captureId, { token: Symbol(), record: { id: captureId, attemptId: id, format: canonical ? 'semantic' : 'legacy',
          status: canonical?.status ?? 'partial',
          forwardStatus: rootStatus === 'completed' ? 'passed' : rootStatus === 'failed' ? 'failed' : result.ok ? 'passed' : 'failed',
          sourceId: attempt.sourceId, planDigest: attempt.planDigest, preparationDigest,
          retainedBytes: size <= this.maxCaptureBytes ? size : 0, unavailable: size > this.maxCaptureBytes ? 'retention_limit' : null },
          ...(size <= this.maxCaptureBytes ? { body } : {}) });
        if (payload != null && !canonical) this.captures.get(captureId)!.record.semanticRejection = 'invalid_capture';
        if (canonical?.status === 'complete') this.complete = captureId;
        else this.partial = captureId;
        this.prune();
      },
      finish: status => {
        if (!open()) return;
        attempt.status = status === 'cancelled' || outcome?.cancelled ? 'cancelled'
          : outcome?.timedOut ? 'timed_out'
          : status === 'passed' ? 'succeeded' : status === 'refused' && !outcome ? 'needs_setup' : 'failed';
        attempt.finishedAt = Math.max(attempt.startedAt, Date.now());
        plan = undefined;
        preparation = undefined;
      },
      snapshot: () => clone(attempt),
    };
    this.latest = handle;
    return handle;
  }

  projected(target: LocalBrowserTargetIdentity, trace?: WorkerEnvelope): void {
    if (this.key !== modelKey(target)) return;
    const id = trace && this.envelopes.get(trace);
    this.displayed = id && this.captures.has(id) ? id : null;
    this.prune();
  }

  state(target: LocalBrowserTargetIdentity): CaptureState {
    if (this.key !== modelKey(target)) return { latestAttempt: null,
      lastCompleteCaptureId: null, lastPartialCaptureId: null, displayedCaptureId: null };
    return { latestAttempt: this.latest?.snapshot() ?? null, lastCompleteCaptureId: this.complete,
      lastPartialCaptureId: this.partial, displayedCaptureId: this.displayed };
  }

  capture(id: string): { record: CaptureRecord; envelope?: WorkerEnvelope; plan?: ResolvedExecutionPlan; preparation?: CapturePreparation } | undefined {
    const capture = this.captures.get(id);
    return capture ? { record: clone(capture.record), ...(capture.body ? JSON.parse(capture.body) : {}) } : undefined;
  }

  /** Cheap identity for immutable retained bytes. Eviction/reinsertion of the
   * same external capture id creates a different token; no payload is exposed. */
  captureIdentity(id: string): { token: symbol; record: CaptureRecord } | undefined {
    const capture = this.captures.get(id);
    return capture ? { token: capture.token, record: clone(capture.record) } : undefined;
  }

  executionPlan(target: LocalBrowserTargetIdentity): ResolvedExecutionPlan | undefined {
    return this.key === modelKey(target) && this.planBody ? JSON.parse(this.planBody) : undefined;
  }

  capturePreparation(target: LocalBrowserTargetIdentity): CapturePreparation | undefined {
    return this.key === modelKey(target) && this.preparationBody ? JSON.parse(this.preparationBody) : undefined;
  }

  attribution(target: LocalBrowserTargetIdentity): LocalBrowserValidationEvidence | undefined {
    const state = this.state(target), attempt = state.latestAttempt;
    if (!attempt) return undefined;
    const capture = state.displayedCaptureId ? this.captures.get(state.displayedCaptureId)?.record : undefined;
    return { origin: attempt.origin, attemptStatus: attempt.status, sourceId: attempt.sourceId, planDigest: attempt.planDigest,
      lastCompleteCaptureId: state.lastCompleteCaptureId, lastPartialCaptureId: state.lastPartialCaptureId,
      displayedCapture: capture ? { id: capture.id, attemptId: capture.attemptId, format: capture.format,
        status: capture.status, forwardStatus: capture.forwardStatus, sourceId: capture.sourceId,
        planDigest: capture.planDigest, unavailable: capture.unavailable } : null,
      setup: this.planLabels ? { ...this.planLabels } : null };
  }

  private prune(): void {
    const retained = new Set([this.complete, this.partial, this.displayed]);
    for (const id of this.captures.keys()) if (!retained.has(id)) this.captures.delete(id);
  }
}
