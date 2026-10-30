/** Host-only observations pinned to a resolved execution. No source or tensor reads. */
import { isDeepStrictEqual } from 'node:util';
import type { ResolvedExecutionPlan } from './executionPlanCore';
import type { WorkerEnvelope } from './traceWorkerCore';
import type { CaptureSnapshot } from './semanticCapture';
import { object, type JsonObject } from './mcpGraph';

export interface ExecutionObservation {
  version: 1;
  plan: ResolvedExecutionPlan;
  planDigest: string;
  status: 'passed' | 'failed';
  setup: JsonObject;
  execution: JsonObject;
}
const digest = (value: unknown): value is string => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);

/** Revalidate at retention, including snapshots provided by an internal caller. */
export function validatedExecutionObservation(value: unknown, plan: ResolvedExecutionPlan,
  capture?: CaptureSnapshot): ExecutionObservation | undefined {
  const item = object(value), setup = object(item.setup), execution = object(item.execution);
  const model = plan.source.files.find(file => file.path === plan.source.modelFile && file.role === 'source');
  if (item.version !== 1 || !['passed', 'failed'].includes(String(item.status)) || !model ||
      !digest(item.planDigest) || !isDeepStrictEqual(item.plan, plan) || setup.version !== 1 || execution.version !== 1 ||
      !isDeepStrictEqual(setup.resolvedPlan, plan) || setup.planDigest !== item.planDigest || setup.sourceDigest !== model.digest) return;
  if (capture && (capture.attemptId !== plan.id || capture.source.id !== plan.source.id ||
      capture.planDigest !== item.planDigest || capture.inputDigest !== (object(setup.inputs).digest ?? null) ||
      capture.stateDigest !== (object(setup.state).digest ?? null))) return;
  const result = structuredClone(value) as ExecutionObservation;
  return result;
}

export function executionObservation(envelope: WorkerEnvelope, plan: ResolvedExecutionPlan,
  status: ExecutionObservation['status'], capture?: CaptureSnapshot): ExecutionObservation | undefined {
  const diagnostics = object(object(envelope.result).diagnostics), setup = object(diagnostics.execution);
  if (envelope.worker.planId !== plan.id || envelope.worker.planDigest !== setup.planDigest) return;
  return validatedExecutionObservation({ version: 1, plan, planDigest: setup.planDigest,
    status, setup, execution: diagnostics.run_evidence }, plan, capture);
}

/** A candidate workspace has an opaque source identity, never a live graph ID. */
export function observationRun(value: ExecutionObservation) {
  const source = value.plan.source, model = source.files.find(file => file.path === source.modelFile && file.role === 'source')!;
  return { runId: value.plan.id, status: value.status,
    execution: value.plan.execution, job: value.plan.job,
    graph: { target: { workspaceFolder: source.workspaceId, file: source.modelFile,
      cls: source.className, sourceDigest: model.digest } },
    evidence: { setup: value.setup, execution: value.execution } };
}
