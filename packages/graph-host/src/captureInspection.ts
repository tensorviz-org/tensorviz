import { z } from 'zod';
import { executionObservation } from './executionObservation';
import type { AttemptLedger } from './attemptLedger';
import type { CaptureSnapshot } from './semanticCapture';
import type { ResolvedExecutionPlan } from './executionPlanCore';
import type { CaptureInspectionPage, CaptureInspectionQuery, LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';
import { GraphToolError, object } from './mcpGraph';
import { CaptureReadError, inspectCapture as readCapture } from '@tensorviz/contracts/captureReader';
import { ComputationIndex, computationQuerySchema, computationFindSchema } from './computationQuery';
import { analyzeComputationChange, computationChangeSchema } from './computationChange';

const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value));
type IndexedCapture = { capture: CaptureSnapshot; index: ComputationIndex; bytes: number; usedAt: number };
const INDEX_IDLE_MS = 5 * 60_000;

export const captureQuerySchema = z.object({
  captureId: z.string().min(1).max(128), sourceId: z.string().min(1).max(128),
  entity: z.object({ kind: z.enum(['operation', 'invocation', 'value', 'input', 'output']), id: z.string().min(1).max(128) }).strict().optional(),
  path: z.array(z.union([z.string().max(128), z.number().int().min(0).max(1_000_000)])).max(128).default([]),
  offset: z.number().int().min(0).max(1_000_000).default(0),
  limit: z.number().int().min(1).max(200).default(50),
}).strict();

/** Preserve MCP error codes while sharing all reading logic with the viewer. */
export function inspectCapture(capture: CaptureSnapshot, requested: CaptureInspectionQuery): CaptureInspectionPage {
  const query = captureQuerySchema.safeParse(requested);
  if (!query.success) throw new GraphToolError('INVALID_QUERY', 'Use a bounded capture path and page size.');
  try { return readCapture(capture, query.data); }
  catch (error) {
    if (error instanceof CaptureReadError) throw new GraphToolError(error.code, error.message);
    throw error;
  }
}

/** Both browser and MCP read the same host-owned retained slots. No disk reads,
 * imports, execution, or write capability are involved in inspection. */
export class CaptureInspector {
  private readonly indexes = new Map<symbol, IndexedCapture>();
  constructor(private readonly ledger: AttemptLedger, private readonly maxIndexInputBytes = 8 * 1024 * 1024) {}

  list(target: LocalBrowserTargetIdentity) {
    const state = this.ledger.state(target);
    const ids = [...new Set([state.lastCompleteCaptureId, state.lastPartialCaptureId, state.displayedCaptureId])].filter((id): id is string => id !== null);
    return { ...state, captures: ids.flatMap(id => {
      const record = this.ledger.captureIdentity(id)?.record;
      return record ? [record] : [];
    }) };
  }

  inspect(target: LocalBrowserTargetIdentity, query: CaptureInspectionQuery): CaptureInspectionPage {
    return inspectCapture(this.retained(target, query.captureId, query.sourceId), query);
  }

  queryComputation(target: LocalBrowserTargetIdentity, requested: unknown) {
    const parsed = computationQuerySchema.safeParse(requested);
    if (!parsed.success) throw new GraphToolError('INVALID_QUERY', 'Use bounded canonical anchors, traversal depth and pagination.');
    const { capture, index } = this.indexed(target, parsed.data.captureId, parsed.data.sourceId);
    return clone({ page: index.query(parsed.data), source: capture.source });
  }

  findComputation(target: LocalBrowserTargetIdentity, requested: unknown) {
    const parsed = computationFindSchema.safeParse(requested);
    if (!parsed.success) throw new GraphToolError('INVALID_QUERY', 'Use bounded literal name text, entity kinds and pagination.');
    const { index } = this.indexed(target, parsed.data.captureId, parsed.data.sourceId);
    return clone(index.find(parsed.data));
  }

  analyzeComputation(target: LocalBrowserTargetIdentity, requested: unknown) {
    const parsed = computationChangeSchema.safeParse(requested);
    if (!parsed.success) throw new GraphToolError('INVALID_QUERY', 'Use bounded exact value identities and concrete nonnegative shapes.');
    const { capture, index } = this.indexed(target, parsed.data.captureId, parsed.data.sourceId);
    return clone(analyzeComputationChange(capture, parsed.data, index));
  }

  private indexed(target: LocalBrowserTargetIdentity, captureId: string, sourceId: string): IndexedCapture {
    const now = Date.now();
    for (const [token, entry] of this.indexes) {
      if (now - entry.usedAt >= INDEX_IDLE_MS || this.ledger.captureIdentity(entry.capture.id)?.token !== token) {
        this.indexes.delete(token);
      }
    }
    const retained = this.retention(target, captureId, sourceId);
    const cached = this.indexes.get(retained.token);
    if (cached) {
      cached.usedAt = now;
      this.indexes.delete(retained.token); this.indexes.set(retained.token, cached);
      return cached;
    }
    const capture = this.retained(target, captureId, sourceId);
    const entry = { capture, index: new ComputationIndex(capture), bytes: retained.record.retainedBytes, usedAt: now };
    if (entry.bytes <= this.maxIndexInputBytes) {
      // This caps serialized inputs and entry count, not JavaScript heap bytes.
      while (this.indexes.size && (this.indexes.size >= 2
        || [...this.indexes.values()].reduce((sum, item) => sum + item.bytes, entry.bytes) > this.maxIndexInputBytes)) {
        this.indexes.delete(this.indexes.keys().next().value!);
      }
      this.indexes.set(retained.token, entry);
    }
    return entry;
  }

  /** A later failed/busy/no-capture attempt cannot borrow an older successful
   * capture for applied-source validation. Partial captures remain inspectable. */
  latest(target: LocalBrowserTargetIdentity) {
    const state = this.list(target), attempt = state.latestAttempt;
    if (!attempt || attempt.finishedAt === null || attempt.captureIds.length !== 1) return undefined;
    const record = state.captures.find(item => item.id === attempt.captureIds[0]);
    if (!record || record.format !== 'semantic' || record.unavailable || !record.sourceId) return undefined;
    try {
      return { capture: this.retained(target, record.id, record.sourceId),
        plan: this.retainedPlan(target, record.id, record.sourceId), startedAt: attempt.startedAt };
    } catch { return undefined; }
  }

  /** Replay belongs to this retained capture, never to a newer attempt's setup. */
  retainedPlan(target: LocalBrowserTargetIdentity, captureId: string, sourceId: string): ResolvedExecutionPlan | undefined {
    const capture = this.retained(target, captureId, sourceId);
    const retained = this.ledger.capture(captureId);
    const plan = retained?.plan;
    if (!plan || plan.id !== capture.attemptId || plan.source.id !== sourceId
      || capture.planDigest === null || retained?.record.planDigest !== capture.planDigest) return undefined;
    return plan;
  }

  /** Baseline observations belong to the exact retained capture and saved plan. */
  retainedObservations(target: LocalBrowserTargetIdentity, captureId: string, sourceId: string) {
    const plan = this.retainedPlan(target, captureId, sourceId);
    const retained = this.ledger.capture(captureId);
    if (!plan || !retained?.envelope) return undefined;
    return executionObservation(retained.envelope, plan,
      retained.envelope.error ? 'failed' : retained.record.forwardStatus, this.retained(target, captureId, sourceId));
  }

  private retention(target: LocalBrowserTargetIdentity, captureId: string, sourceId: string) {
    const state = this.ledger.state(target);
    if (![state.lastCompleteCaptureId, state.lastPartialCaptureId, state.displayedCaptureId].includes(captureId)) {
      throw new GraphToolError('UNKNOWN_CAPTURE', 'This capture is no longer retained for the selected model.');
    }
    const retained = this.ledger.captureIdentity(captureId);
    if (!retained || retained.record.unavailable) throw new GraphToolError('CAPTURE_UNAVAILABLE', 'The capture payload exceeded the host retention limit.');
    if (retained.record.format !== 'semantic') throw new GraphToolError('LEGACY_CAPTURE', 'This legacy run has no canonical argument/state records.');
    if (retained.record.sourceId !== sourceId) throw new GraphToolError('SOURCE_MISMATCH', 'Use the exact capture source identity.');
    return retained;
  }

  /** Async source readers must recheck ownership/eviction after their reads. */
  assertRetained(target: LocalBrowserTargetIdentity, captureId: string, sourceId: string): void {
    this.retention(target, captureId, sourceId);
  }

  /** Export shares exactly the same ownership and eviction checks as inspection. */
  retained(target: LocalBrowserTargetIdentity, captureId: string, sourceId: string): CaptureSnapshot {
    this.assertRetained(target, captureId, sourceId);
    const retained = this.ledger.capture(captureId);
    const capture = object(object(retained?.envelope?.result).diagnostics).semantic_capture as CaptureSnapshot | undefined;
    if (!capture) throw new GraphToolError('CAPTURE_UNAVAILABLE', 'The retained capture payload is unavailable.');
    if (capture.source.id !== sourceId) throw new GraphToolError('SOURCE_MISMATCH', 'Use the exact capture source identity.');
    return capture;
  }
}
