/** Private transport for the Python-resolved invocation shared by browser and MCP. */
import { createHash, randomUUID } from 'node:crypto';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { readHostSnapshot } from './hostSnapshot';
import type { components } from '@tensorviz/contracts/semanticContracts.generated';
import type { RunSpec } from '@tensorviz/contracts/protocol';
import type { ExecutionSpec, ExecutionJob } from './executionSpec';
import { ownedTemporaryDirectory } from './ownedTemporary';

export type ResolvedExecutionPlan = components['schemas']['ResolvedExecutionPlan'];
export type CapturePreparation = components['schemas']['CapturePreparation'];
export type CandidateRequest = components['schemas']['CandidateRequest'];
export type CandidateExecution = components['schemas']['CandidateExecution'];
export interface ExecutionSnapshot { plan: ResolvedExecutionPlan; applicabilityDigest: string; guards: Record<string, unknown> }
export interface PreparationSnapshot { preparation: CapturePreparation; applicabilityDigest: string; guards: Record<string, unknown> }
const session = randomUUID();
export const EXECUTION_PLAN_ENV = 'TENSORVIZ_EXECUTION_PLAN';
export const CANDIDATE_REQUEST_ENV = 'TENSORVIZ_CANDIDATE_REQUEST';

export type DocumentStates = Record<string, { version: number; dirty: boolean }>;

export interface SnapshotRequest {
  documents: DocumentStates;
  workspace: string; file: string; cls: string;
  timeoutMs: number; generation: number; profileSpec?: RunSpec; attemptId?: string;
}

async function resolveSnapshot(python: string, backendDir: string, request: SnapshotRequest,
  setup: { execution: ExecutionSpec; job?: ExecutionJob } | { captureRequest: CapturePreparation['request'] },
  signal: AbortSignal): Promise<ExecutionSnapshot | PreparationSnapshot> {
  const documents = request.documents;
  const id = request.profileSpec?.identity;
  const result = await readHostSnapshot(python, backendDir, {
    root: request.workspace, documents, executable: python, environmentRevision: 0,
    session: id?.workspaceSessionId ?? session,
    workspaceId: id?.workspaceId ?? createHash('sha256').update(request.workspace).digest('hex'),
    runId: request.attemptId ?? randomUUID(), generation: request.generation,
    modelFile: path.relative(request.workspace, request.file).split(path.sep).join('/'), className: request.cls,
    ...setup, timeoutMs: request.timeoutMs,
    ...(id ? { profile: id.profile, expectedIdentity: id } : {}),
  }, signal) as ExecutionSnapshot | PreparationSnapshot;
  const identity = result && ('plan' in result ? result.plan : result.preparation);
  if (!identity || identity.version !== 1 || identity.interpreter.executable !== python
    || !/^[a-f0-9]{64}$/.test(result.applicabilityDigest) || !result.guards) throw new Error('execution_plan_failed');
  return result;
}

export async function resolveExecutionPlan(python: string, backendDir: string,
  request: SnapshotRequest & { execution: ExecutionSpec; job?: ExecutionJob }, signal: AbortSignal): Promise<ExecutionSnapshot> {
  const result = await resolveSnapshot(python, backendDir, request, { execution: request.execution, job: request.job }, signal);
  if (!('plan' in result)) throw new Error('execution_plan_failed');
  return result;
}

export async function resolveCapturePreparation(python: string, backendDir: string,
  request: SnapshotRequest & { captureRequest: CapturePreparation['request'] }, signal: AbortSignal): Promise<PreparationSnapshot> {
  const result = await resolveSnapshot(python, backendDir, request, { captureRequest: request.captureRequest }, signal);
  if (!('preparation' in result)) throw new Error('capture_preparation_failed');
  return result;
}

/** Re-resolve this retained plan, including its selected profile and native data.
 * Python owns applicability; host buffer dirtiness is never omitted on replay. */
export async function resolveRetainedPlan(python: string, backendDir: string, workspace: string,
  plan: ResolvedExecutionPlan, signal: AbortSignal, documents: DocumentStates): Promise<ExecutionSnapshot> {
  if (python !== plan.interpreter.executable) throw new Error('execution_environment_changed');
  const result = await readHostSnapshot(python, backendDir, { root: workspace,
    documents, replayPlan: plan }, signal) as ExecutionSnapshot;
  if (!result?.plan || result.plan.interpreter.executable !== python
    || !/^[a-f0-9]{64}$/.test(result.applicabilityDigest) || !result.guards) throw new Error('execution_plan_failed');
  return result;
}

/** Large source sets never enter argv or logs. The directory and file are private,
 * bounded and removed on every exit, including refusal, cancellation and spawn errors. */
export async function executionPlanFile(snapshot: ExecutionSnapshot | PreparationSnapshot): Promise<{ file: string; dispose(): Promise<void> }> {
  const body = JSON.stringify({ version: 1, ...('plan' in snapshot ? { plan: snapshot.plan } : { preparation: snapshot.preparation }),
    applicabilityDigest: snapshot.applicabilityDigest });
  return privateExecutionFile(body, 16 * 1024 * 1024);
}

/** The host owns the outer directory, including staged files left by a killed worker. */
export async function candidateRequestFile(request: CandidateRequest): Promise<{ file: string; dispose(): Promise<void> }> {
  return privateExecutionFile(JSON.stringify(request), 24 * 1024 * 1024);
}

async function privateExecutionFile(body: string, maxBytes: number): Promise<{ file: string; dispose(): Promise<void> }> {
  if (Buffer.byteLength(body) > maxBytes) throw new Error('execution_plan_limit');
  const temporary = await ownedTemporaryDirectory(await fs.realpath(os.tmpdir()), 'tensorviz-execution-');
  const { directory, dispose } = temporary;
  try {
    const file = path.join(directory, 'plan.json');
    await fs.writeFile(file, body, { flag: 'wx', mode: 0o600 });
    return { file, dispose };
  } catch (error) { await dispose(); throw error; }
}
