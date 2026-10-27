/** Host-only candidate boundary. A candidate never becomes a live capture slot. */
import Ajv from 'ajv';
import { createHash, randomUUID } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
import schema from '@tensorviz/contracts/semanticSchema.generated';
import { semanticCapture, type CaptureSnapshot } from './semanticCapture';
import type { CandidateRequest, CandidateExecution } from './executionPlanCore';
import type { TraceOutcome, WorkerEnvelope } from './traceWorkerCore';
import { isCandidatePostconditions, type CandidatePostconditions } from '@tensorviz/contracts/protocol';
import { candidatePostconditionsMatch } from './candidatePostconditions';

export type CandidateInput = Omit<CandidateRequest, 'id'>;
export interface CandidateEvidence {
  execution: CandidateExecution;
  capture?: CaptureSnapshot;
  baseCurrent: boolean;
  candidateCurrent: boolean;
  postconditions?: CandidatePostconditions;
}
/** Separate envelope prevents accidental use as AttemptLedger.captured input. */
export interface CandidateRunResult { outcome: TraceOutcome; evidence?: CandidateEvidence; cleanupFailed?: true }

const ajv = new Ajv({ strict: false });
ajv.addSchema(schema, 'semantic');
const validRequest = ajv.compile<CandidateRequest>({ $ref: 'semantic#/definitions/CandidateRequest' });
const validExecution = ajv.compile<CandidateExecution>({ $ref: 'semantic#/definitions/CandidateExecution' });
const digest = (text: string) => createHash('sha256').update(text, 'utf8').digest('hex');

/** Snapshot before asynchronous work; callers cannot change a running request. */
export function candidateRequest(input: CandidateInput): CandidateRequest {
  const text = JSON.stringify({ ...input, id: `candidate:${randomUUID()}` });
  if (Buffer.byteLength(text) > 24 * 1024 * 1024) throw new Error('candidate_request_limit');
  const request: unknown = JSON.parse(text);
  if (!validRequest(request) || request.plan.source.sourceMode !== 'saved'
    || !semanticCapture(request.capture, request.plan.id, request.plan, request.capture.planDigest)
    || request.intent.target.captureId !== request.capture.id || request.intent.target.sourceId !== request.capture.source.id) {
    throw new Error('candidate_request_invalid');
  }
  const file = request.plan.source.files.find(item => item.path === request.reviewedFile && item.role === 'source');
  if (!file || file.digest === digest(request.reviewedSource)) throw new Error('candidate_source_unchanged');
  return request;
}

/** Pin worker evidence to this host run and exact reviewed bytes. Python's intent
 * digest is opaque here; the unique host run ID and complete source digest bind it. */
export function candidateEvidence(envelope: WorkerEnvelope, request: CandidateRequest): CandidateEvidence | undefined {
  try {
    const execution = envelope.worker.candidateExecution;
    if (!validExecution(execution)) return undefined;
    const { plan, ...identity } = execution;
    const file = request.plan.source.files.find(item => item.path === request.reviewedFile && item.role === 'source');
    if (!file || identity.id !== request.id || plan.id !== request.id || request.id === request.plan.id
      || identity.baseCaptureId !== request.capture.id || identity.baseAttemptId !== request.plan.id
      || identity.baseSourceId !== request.plan.source.id || identity.baseWorkspaceId !== request.plan.source.workspaceId
      || identity.basePlanDigest !== request.capture.planDigest || identity.file !== file.path
      || identity.beforeDigest !== file.digest || identity.afterDigest !== digest(request.reviewedSource)) return undefined;
    const setup = ({ id: _id, source: _source, ...rest }: CandidateExecution['plan']) => rest;
    if (!isDeepStrictEqual(setup(plan), setup(request.plan))) return undefined;
    const source = { ...request.plan.source, id: `source:${request.id}`, workspaceId: request.id, generation: 0,
      modelId: digest(JSON.stringify([request.id, request.plan.source.modelFile, request.plan.source.className])),
      files: request.plan.source.files.map(item => ({ ...item, documentVersion: null,
        digest: item.path === file.path ? identity.afterDigest : item.digest })) };
    if (source.id === request.plan.source.id || source.workspaceId === request.plan.source.workspaceId
      || !isDeepStrictEqual(plan.source, source)) return undefined;
    const diagnostics = (envelope.result as { diagnostics?: { execution?: {
      resolvedPlan?: unknown; planDigest?: string; planCurrent?: boolean;
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
    const postconditions = envelope.worker.candidatePostconditions;
    if (postconditions !== undefined && (!isCandidatePostconditions(postconditions)
      || !candidatePostconditionsMatch(postconditions, request, execution, capture))) return undefined;
    return { execution, capture, ...(postconditions === undefined ? {} : { postconditions: structuredClone(postconditions) }),
      baseCurrent: envelope.worker.candidateBaseCurrent === true,
      candidateCurrent: diagnostics?.execution?.planCurrent === true };
  } catch { return undefined; }
}
