import { randomUUID } from 'node:crypto';
import { isGraphSetupAction, type ExecutionResources, type ExecutionReuse, type ExecutionReviewAction, type ExecutionReviewState, type LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';
import type { ExecutionSpec } from './executionSpec';

export function executionStages(kind: ExecutionReviewState['kind']): ExecutionReviewState['stages'] {
  if (kind === 'prepare') return [{ label: 'Model construction to infer inputs', calls: 0 }];
  if (kind === 'trace') return [{ label: 'Trace Python and infer shapes', calls: 1 }];
  return [{ label: 'Capture shapes and operations', calls: 1 }];
}
export interface ExecutionReviewPlan {
  kind: ExecutionReviewState['kind'];
  execution?: ExecutionSpec;
  interpreter: string;
  profileName?: string;
  /** The request that started this invocation, such as opening the model, already chose to run it. */
  approved?: boolean;
  /** Run fresh: this invocation never reuses an identical earlier result. */
  fresh?: boolean;
}
const abortError = () => Object.assign(new Error('Execution cancelled.'), { name: 'AbortError' });
const sameSource = (a: LocalBrowserTargetIdentity, b: LocalBrowserTargetIdentity) =>
  a.workspaceFolder === b.workspaceFolder && a.file === b.file && a.cls === b.cls
    && a.sourceDigest === b.sourceDigest && a.documentVersion === b.documentVersion;

/** One approval (Run, or the request that started it) authorizes one exact invocation; activity never acts as a policy. */
export class ExecutionReview {
  private value?: ExecutionReviewState;
  private pending?: { target: LocalBrowserTargetIdentity; controller: AbortController; approve: () => void; reject: (error: Error) => void };
  private refreshGeneration = 0;
  private readonly seen = new Set<string>();
  constructor(private readonly options: {
    current(target: LocalBrowserTargetIdentity): boolean;
    resources(interpreter: string, signal: AbortSignal): Promise<ExecutionResources>;
    publish(state: ExecutionReviewState): void;
  }) {}
  get target(): LocalBrowserTargetIdentity | undefined { return this.pending?.target; }
  get state(): ExecutionReviewState | undefined { return this.value; }
  private set(update: Partial<ExecutionReviewState>): void {
    if (!this.value) return;
    this.value = { ...this.value, ...update };
    this.options.publish(this.value);
  }
  progress(detail: string): void {
    if (this.value?.phase === 'running') this.set({ detail: detail.slice(0, 1024) });
  }
  private async refresh(signal = this.pending?.controller.signal): Promise<void> {
    const state = this.value, generation = ++this.refreshGeneration;
    if (!state || !signal) return;
    try {
      const resources = await this.options.resources(state.interpreter, signal);
      if (this.value?.id === state.id && generation === this.refreshGeneration && !signal.aborted) this.set({ resources });
    } catch { /* Unknown readings do not prevent an explicitly chosen run. */ }
  }
  action(target: LocalBrowserTargetIdentity, requestId: string, action: ExecutionReviewAction): boolean {
    const state = this.value, pending = this.pending;
    if (!isGraphSetupAction(action) || !state || !pending || action.reviewId !== state.id || this.seen.has(requestId) || !sameSource(target, pending.target)) return false;
    this.seen.add(requestId);
    if (this.seen.size > 256) this.seen.delete(this.seen.values().next().value!);
    if (action.action === 'cancel') {
      this.set({ phase: state.phase === 'review' ? 'cancelled' : 'cancelling', detail: state.phase === 'review' ? 'Cancelled before execution. Nothing was run.' : 'Stopping the worker and its child processes…' });
      pending.controller.abort(); pending.reject(abortError()); return true;
    }
    if (action.action === 'refresh') { void this.refresh(); return true; }
    if (state.phase !== 'review' || pending.controller.signal.aborted) return false;
    if (!this.options.current(pending.target)) {
      pending.reject(new Error('Source changed. Refresh the graph and review the updated invocation.'));
      return false;
    }
    const devices = new Set(['cpu', state.configuredDevice, state.device, ...(state.resources?.devices.map(d => d.id) ?? [])]);
    if (!devices.has(action.device) || state.kind === 'prepare' && action.device !== 'cpu') return false;
    if (action.action === 'select') { this.set({ device: action.device }); return true; }
    if (action.device !== state.device) return false;
    // Set phase before resolving: replayed Run or another surface cannot execute twice.
    this.set({ phase: 'running', detail: 'Starting the reviewed invocation…' });
    pending.approve(); return true;
  }
  cancel(detail?: string): void {
    const p = this.pending;
    if (!p) return;
    this.set({ phase: this.value?.phase === 'review' ? 'cancelled' : 'cancelling', detail: 'Stopping this invocation…' });
    p.controller.abort(detail); p.reject(abortError());
  }
  async run<T>(target: LocalBrowserTargetIdentity, signal: AbortSignal, plan: ExecutionReviewPlan,
    work: (signal: AbortSignal, device: string) => Promise<T>): Promise<T> {
    if (this.pending) throw new Error('Another execution review is active. Finish or cancel it first.');
    const controller = new AbortController();
    let approve!: () => void, reject!: (error: Error) => void;
    const approval = new Promise<void>((yes, no) => { approve = yes; reject = no; });
    this.pending = { target: { ...target }, controller, approve, reject };
    const kind = plan.kind === 'capture' && plan.execution?.framework === 'jax' ? 'trace' : plan.kind;
    this.value = { id: randomUUID(), phase: plan.approved ? 'running' : 'review', kind, configuredDevice: plan.kind === 'prepare' ? 'cpu' : plan.execution?.device ?? 'cpu', device: plan.kind === 'prepare' ? 'cpu' : plan.execution?.device ?? 'cpu',
      detail: plan.approved ? 'Starting this invocation…' : `Review where this invocation will run. Your model has not started.${plan.fresh ? ' This run executes even if an identical result is available.' : ''}`, stages: executionStages(kind),
      ...(plan.fresh ? { fresh: true as const } : {}),
      inputs: plan.kind === 'prepare' ? 'Input dimensions are not known yet.' : JSON.stringify(plan.execution?.inputs ?? {}).slice(0, 4096),
      mode: plan.execution?.mode ?? 'eval', gradients: plan.execution?.gradEnabled ?? false,
      ...(plan.profileName ? { profileName: plan.profileName } : {}),
      interpreter: plan.interpreter || 'Resolved by the selected model environment' };
    this.options.publish(this.value);
    const abort = () => { controller.abort(); reject(abortError()); this.set({ phase: this.value?.phase === 'review' ? 'cancelled' : 'cancelling', detail: 'Stopping this invocation…' }); };
    signal.addEventListener('abort', abort, { once: true });
    if (signal.aborted) abort();
    if (plan.approved) approve();
    // A run that starts at once can settle before the first reading; that reading
    // still names its host, so it ends only with the caller's operation.
    void this.refresh(plan.approved ? signal : controller.signal);
    try {
      await approval;
      controller.signal.throwIfAborted();
      if (!this.options.current(target)) throw new Error('Source changed. Refresh the graph and review the updated invocation.');
      const result = await work(controller.signal, this.value.device);
      const outcome = result as { status?: string; detail?: string; reused?: ExecutionReuse;
        outcome?: { status: string; detail: string; reused?: ExecutionReuse } } | undefined;
      const status = outcome?.outcome?.status ?? outcome?.status;
      const phase = controller.signal.aborted || status === 'cancelled' ? 'cancelled' : status && status !== 'passed' ? 'failed' : 'succeeded';
      const reused = phase === 'succeeded' ? outcome?.outcome?.reused ?? outcome?.reused : undefined;
      this.set({ phase, detail: (outcome?.outcome?.detail ?? outcome?.detail ?? 'Invocation finished.').slice(0, 1024),
        ...(reused ? { reused: { capturedAt: reused.capturedAt } } : {}) });
      return result;
    } catch (error) {
      this.set({ phase: controller.signal.aborted ? 'cancelled' : 'failed',
        detail: controller.signal.aborted ? typeof controller.signal.reason === 'string' ? controller.signal.reason.slice(0, 1024)
          : 'Execution stopped. No automatic retry is scheduled.' : error instanceof Error ? error.message.slice(0, 1024) : 'Execution could not complete. Review setup and try again.' });
      throw error;
    } finally {
      signal.removeEventListener('abort', abort);
      controller.abort();
      this.pending = undefined;
    }
  }
}
