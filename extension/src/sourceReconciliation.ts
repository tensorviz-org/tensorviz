/** Private static reconciliation. The host must separately own an Apply receipt,
 * retain this evidence within a byte/TTL budget, and check current editor state. */
import Ajv from 'ajv';
import { createHash } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
import schema from '@tensorviz/contracts/semanticSchema.generated';
import type { components } from '@tensorviz/contracts/semanticContracts.generated';
import type { CaptureSnapshot } from '@tensorviz/graph-host/semanticCapture';
import { isCandidatePostconditions } from '@tensorviz/contracts/protocol';
import { sourcePostconditionsMatch } from '@tensorviz/graph-host/candidatePostconditions';

export type CaptureReconciliation = components['schemas']['CaptureReconciliation'];
export type AppliedSourceEvidence = Omit<components['schemas']['CaptureReconciliationRequest'], 'current'>;
const ajv = new Ajv({ strict: false });
ajv.addSchema(schema, 'semantic');
const structure = ajv.compile<CaptureReconciliation>({ $ref: 'semantic#/definitions/CaptureReconciliation' });
const digest = (text: string) => createHash('sha256').update(text).digest('hex');
const identity = (left: string | null, right: string | null) => left === null || right === null ? 'unavailable' : left === right ? 'same' : 'different';
export class SourceReconciliationError extends Error {}

/** These metadata comparisons deliberately mirror the Python trust boundary;
 * source transformations and AST semantics remain Python's authority. */
function sourceReason(input: AppliedSourceEvidence, current: CaptureSnapshot): CaptureReconciliation['reason'] {
  const before = input.base.source, after = current.source;
  if (before.workspaceSessionId !== after.workspaceSessionId || before.workspaceId !== after.workspaceId
    || before.modelId !== after.modelId || before.modelFile !== after.modelFile || before.className !== after.className) return 'different_model';
  if (input.base.id === current.id || input.base.attemptId === current.attemptId) return 'capture_not_new';
  if (before.sourceMode !== 'saved' || after.sourceMode !== 'saved') return 'source_unavailable';
  const edited = after.files.find(file => file.path === input.reviewedFile), afterDigest = digest(input.reviewedSource);
  if (!edited || edited.role !== 'source') return 'source_unavailable';
  if (edited.digest !== afterDigest) return edited.digest === digest(input.sources[input.reviewedFile]) ? 'source_reverted' : 'source_changed';
  const files = (source: CaptureSnapshot['source'], expected: boolean) => source.files.map(file =>
    [file.path, file.role, expected && file.path === input.reviewedFile ? afterDigest : file.digest])
    .sort((a, b) => a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0);
  if (!isDeepStrictEqual(files(before, true), files(after, false)) || before.configDigest !== after.configDigest) return 'dependencies_changed';
  if (before.environmentDigest !== after.environmentDigest) return 'environment_changed';
  return 'reviewed_source';
}

function matches(result: unknown, input: AppliedSourceEvidence, current: CaptureSnapshot): result is CaptureReconciliation {
  if (!structure(result) || result.baseCaptureId !== input.base.id || result.captureId !== current.id
    || result.sourceId !== current.source.id || result.reviewedDigest !== digest(input.reviewedSource)
    || result.dependencyScope !== (input.base.source.dependencyScope === 'complete' && current.source.dependencyScope === 'complete' ? 'complete' : 'partial')
    || result.reason !== sourceReason(input, current)) return false;
  const checked = result.reason === 'reviewed_source';
  if ((result.status === 'checked') !== checked) return false;
  if (!checked) return result.postconditions === null && result.inputs === 'unavailable' && result.state === 'unavailable';
  return result.inputs === identity(input.base.inputDigest, current.inputDigest)
    && result.state === identity(input.base.stateDigest, current.stateDigest)
    && result.postconditions !== null && isCandidatePostconditions(result.postconditions)
    && result.postconditions.executionId === current.attemptId && result.postconditions.captureId === current.id
    && sourcePostconditionsMatch(result.postconditions, { capture: input.base, intent: input.intent, reviewedFile: input.reviewedFile },
      result.reviewedDigest, current);
}

export async function reconcileAppliedCapture(evidence: AppliedSourceEvidence, capture: CaptureSnapshot, options: {
  origin: string; token: string; signal: AbortSignal; fetchImpl?: typeof fetch;
}): Promise<CaptureReconciliation> {
  const body = JSON.stringify({ ...evidence, current: capture });
  if (Buffer.byteLength(body) > 48 * 1024 * 1024) throw new SourceReconciliationError('Applied-source evidence exceeds the reconciliation input limit.');
  // Freeze both request and response comparison inputs against caller mutation.
  const { current, ...input } = JSON.parse(body) as AppliedSourceEvidence & { current: CaptureSnapshot };
  const controller = new AbortController();
  let rejectAbort!: (error: Error) => void;
  const cancelled = new Promise<never>((_resolve, reject) => { rejectAbort = reject; });
  const abort = () => { controller.abort(); rejectAbort(new SourceReconciliationError('Source reconciliation was cancelled or timed out.')); };
  options.signal.addEventListener('abort', abort, { once: true });
  const timer = setTimeout(abort, 10_000);
  const request = async () => {
    controller.signal.throwIfAborted();
    const response = await (options.fetchImpl ?? fetch)(`${options.origin}/api/ide/capture/reconcile`, {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'X-tensorViz-Token': options.token },
      body, signal: controller.signal, redirect: 'error',
    });
    if (!response.ok) { await response.body?.cancel(); throw new SourceReconciliationError('The local backend could not reconcile this applied source.'); }
    const reader = response.body?.getReader();
    if (!reader) throw new SourceReconciliationError('The local backend returned no source reconciliation.');
    const chunks: Uint8Array[] = []; let size = 0;
    const cancelRead = () => { void reader.cancel().catch(() => undefined); };
    controller.signal.addEventListener('abort', cancelRead, { once: true });
    try {
      for (;;) {
        controller.signal.throwIfAborted();
        const next = await reader.read();
        if (next.done) break;
        size += next.value.byteLength;
        if (size > 64 * 1024) throw new SourceReconciliationError('Source reconciliation exceeds the response limit.');
        chunks.push(next.value);
      }
    } finally { controller.signal.removeEventListener('abort', cancelRead); await reader.cancel(); }
    const result: unknown = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(Buffer.concat(chunks)));
    if (!matches(result, input, current)) throw new SourceReconciliationError('The local backend returned mismatched source reconciliation evidence.');
    controller.signal.throwIfAborted();
    return result;
  };
  try {
    if (options.signal.aborted) abort();
    return await Promise.race([request(), cancelled]);
  } catch (error) {
    if (error instanceof SourceReconciliationError) throw error;
    throw new SourceReconciliationError('The local backend could not prepare source reconciliation evidence.');
  } finally { clearTimeout(timer); options.signal.removeEventListener('abort', abort); }
}
