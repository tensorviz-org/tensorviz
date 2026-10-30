/** Host-only candidate boundary. A candidate never becomes a live capture slot. */
import Ajv from 'ajv';
import { executionObservation, type ExecutionObservation } from './executionObservation';
import { createHash, randomUUID } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
import schema from '@tensorviz/contracts/semanticSchema.generated';
import { semanticCapture, type CaptureSnapshot } from './semanticCapture';
import type { CandidateRequest, CandidateExecution, RecipeCandidateRequest, SourceCandidateRequest } from './executionPlanCore';
import type { TraceOutcome, WorkerEnvelope } from './traceWorkerCore';
import { isCandidatePostconditions, type CandidatePostconditions } from '@tensorviz/contracts/protocol';
import { candidatePostconditionsMatch } from './candidatePostconditions';
import { readCheckpointEvidence, type CheckpointEvidence } from './checkpointEvidence';
import { executionSpecSchema } from './executionSpec';

export type RecipeCandidateInput = Omit<RecipeCandidateRequest, 'id'>;
export type SourceCandidateInput = Omit<SourceCandidateRequest, 'id'>;
export type CandidateInput = RecipeCandidateInput | SourceCandidateInput;
export interface CandidateEvidence {
  execution: CandidateExecution;
  capture?: CaptureSnapshot;
  baseCurrent: boolean;
  candidateCurrent: boolean;
  postconditions?: CandidatePostconditions;
  checkpoint?: CheckpointEvidence;
  observations?: ExecutionObservation;
}
/** Separate envelope prevents accidental use as AttemptLedger.captured input. */
export interface CandidateRunResult { outcome: TraceOutcome; evidence?: CandidateEvidence; cleanupFailed?: true }

const ajv = new Ajv({ strict: false });
ajv.addSchema(schema, 'semantic');
const validRequest = ajv.compile<CandidateRequest>({ oneOf: [
  { $ref: 'semantic#/definitions/CandidateRequest' }, { $ref: 'semantic#/definitions/SourceCandidateRequest' }] });
const validExecution = ajv.compile<CandidateExecution>({ oneOf: [
  { $ref: 'semantic#/definitions/CandidateExecution' }, { $ref: 'semantic#/definitions/SourceCandidateExecution' }] });
const digest = (text: string) => createHash('sha256').update(text, 'utf8').digest('hex');

/** A report belongs to this reviewed plan, including the copied binary resource.
 * A matching path or the original source digest is insufficient. */
export function candidateCheckpointEvidence(value: unknown, plan: CandidateExecution['plan']): CheckpointEvidence | undefined {
  const spec = executionSpecSchema.safeParse(plan.execution);
  if (!spec.success || !spec.data.checkpoint) return undefined;
  const source = plan.source.files.find(item => item.path === plan.source.modelFile && item.role === 'source');
  const resource = plan.source.files.find(item => item.path === spec.data.checkpoint!.file && item.role === 'native');
  if (!source || !resource) return undefined;
  const report = readCheckpointEvidence(value, { sourceDigest: source.digest, checkpoint: spec.data.checkpoint });
  return report?.checkpoint.digest === resource.digest ? report : undefined;
}

function reviewedChanges(request: CandidateRequest) {
  const sources = request.version === 1 ? [{ file: request.reviewedFile, source: request.reviewedSource,
    beforeDigest: request.plan.source.files.find(item => item.path === request.reviewedFile)?.digest }]
    : request.reviewedSources;
  const paths = new Set<string>();
  let bytes = 0;
  const changes = sources.map(item => {
    const file = request.plan.source.files.find(file => file.path === item.file && file.role === 'source');
    const afterDigest = digest(item.source), size = Buffer.byteLength(item.source);
    bytes += size;
    if (!file || file.digest !== item.beforeDigest || file.digest === afterDigest || paths.has(item.file)
      || request.version === 2 && (!item.file.endsWith('.py') || size > 262144 || bytes > 1048576
        || Buffer.from(item.source).toString('utf8') !== item.source)) throw new Error('candidate_source_invalid');
    paths.add(item.file);
    return { file: item.file, beforeDigest: file.digest, afterDigest };
  });
  // UTF-8 ordering matches Python Unicode ordering, including non-BMP paths.
  return changes.sort((a, b) => Buffer.compare(Buffer.from(a.file), Buffer.from(b.file)));
}

/** Snapshot before asynchronous work; callers cannot change a running request. */
export function candidateRequest(input: RecipeCandidateInput): RecipeCandidateRequest;
export function candidateRequest(input: SourceCandidateInput): SourceCandidateRequest;
export function candidateRequest(input: CandidateInput): CandidateRequest;
export function candidateRequest(input: CandidateInput): CandidateRequest {
  const text = JSON.stringify({ ...input, id: `candidate:${randomUUID()}` });
  if (Buffer.byteLength(text) > 24 * 1024 * 1024) throw new Error('candidate_request_limit');
  const request: unknown = JSON.parse(text);
  if (!validRequest(request) || request.plan.source.sourceMode !== 'saved'
    || !semanticCapture(request.capture, request.plan.id, request.plan, request.capture.planDigest)
    || request.version === 1 && (request.intent.target.captureId !== request.capture.id
      || request.intent.target.sourceId !== request.capture.source.id)) {
    throw new Error('candidate_request_invalid');
  }
  reviewedChanges(request);
  return request;
}

/** Pin worker evidence to this host run and exact reviewed bytes. Python's intent
 * digest is opaque here; the unique host run ID and complete source digest bind it. */
export function candidateEvidence(envelope: WorkerEnvelope, request: CandidateRequest): CandidateEvidence | undefined {
  try {
    const execution = envelope.worker.candidateExecution;
    if (!validExecution(execution) || execution.version !== request.version) return undefined;
    const { plan } = execution, changes = reviewedChanges(request);
    if (execution.id !== request.id || plan.id !== request.id || request.id === request.plan.id
      || execution.baseCaptureId !== request.capture.id || execution.baseAttemptId !== request.plan.id
      || execution.baseSourceId !== request.plan.source.id || execution.baseWorkspaceId !== request.plan.source.workspaceId
      || execution.basePlanDigest !== request.capture.planDigest) return undefined;
    if (execution.version === 1 ? !isDeepStrictEqual(changes, [{ file: execution.file,
      beforeDigest: execution.beforeDigest, afterDigest: execution.afterDigest }])
      : !isDeepStrictEqual(execution.changes, changes)) return undefined;
    const setup = ({ id: _id, source: _source, ...rest }: CandidateExecution['plan']) => rest;
    if (!isDeepStrictEqual(setup(plan), setup(request.plan))) return undefined;
    const source = { ...request.plan.source, id: `source:${request.id}`, workspaceId: request.id, generation: 0,
      modelId: digest(JSON.stringify([request.id, request.plan.source.modelFile, request.plan.source.className])),
      files: request.plan.source.files.map(item => ({ ...item, documentVersion: null,
        digest: changes.find(change => change.file === item.path)?.afterDigest ?? item.digest })) };
    if (source.id === request.plan.source.id || source.workspaceId === request.plan.source.workspaceId
      || !isDeepStrictEqual(plan.source, source)) return undefined;
    const diagnostics = (envelope.result as { diagnostics?: { execution?: {
      resolvedPlan?: unknown; planDigest?: string; planCurrent?: boolean; checkpointCompatibility?: unknown;
    }; semantic_capture?: unknown } } | null)?.diagnostics;
    let capture: CaptureSnapshot | undefined;
    if (envelope.result !== null) {
      const actual = diagnostics?.execution;
      if (!actual || !/^[a-f0-9]{64}$/.test(actual.planDigest ?? '') || !isDeepStrictEqual(actual.resolvedPlan, plan)
        || envelope.worker.planId !== request.id || envelope.worker.planDigest !== actual.planDigest) return undefined;
      if (diagnostics?.semantic_capture != null) {
        capture = semanticCapture(diagnostics.semantic_capture, request.id, plan, actual.planDigest!);
        if (!capture) return undefined;
      }
    }
    const failedPlanCurrent = envelope.worker.candidatePlanCurrent;
    if ((failedPlanCurrent !== undefined || envelope.worker.candidatePlanDigest !== undefined)
      && (envelope.result !== null || typeof failedPlanCurrent !== 'boolean' || envelope.worker.planId !== request.id
        || !/^[a-f0-9]{64}$/.test(envelope.worker.planDigest ?? '')
        || envelope.worker.candidatePlanDigest !== envelope.worker.planDigest)) return undefined;
    const postconditions = envelope.worker.candidatePostconditions;
    if (postconditions !== undefined && (request.version !== 1 || execution.version !== 1 || !isCandidatePostconditions(postconditions)
      || !candidatePostconditionsMatch(postconditions, request, execution, capture))) return undefined;
    const observed = diagnostics?.execution?.checkpointCompatibility, failed = envelope.worker.checkpointCompatibility;
    if (observed !== undefined && failed !== undefined && !isDeepStrictEqual(observed, failed)) return undefined;
    const rawCheckpoint = observed !== undefined ? observed : failed;
    const checkpoint = rawCheckpoint === undefined ? undefined : candidateCheckpointEvidence(rawCheckpoint, plan);
    if (rawCheckpoint !== undefined && (!checkpoint
      || checkpoint.loadStatus === 'failed' && (envelope.result !== null || envelope.error === null))) return undefined;
    const observations = executionObservation(envelope, plan, envelope.error ? 'failed' : 'passed', capture);
    return { execution, capture, ...(postconditions === undefined ? {} : { postconditions: structuredClone(postconditions) }),
      ...(observations ? { observations } : {}),
      ...(checkpoint === undefined ? {} : { checkpoint }),
      baseCurrent: envelope.worker.candidateBaseCurrent === true,
      candidateCurrent: envelope.result === null ? failedPlanCurrent === true : diagnostics?.execution?.planCurrent === true };
  } catch { return undefined; }
}
