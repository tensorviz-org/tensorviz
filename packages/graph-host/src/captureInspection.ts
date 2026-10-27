import { z } from 'zod';
import type { AttemptLedger } from './attemptLedger';
import type { CaptureSnapshot } from './semanticCapture';
import type { ResolvedExecutionPlan } from './executionPlanCore';
import type { CaptureInspectionPage, CaptureInspectionQuery, LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';
import { GraphToolError, object } from './mcpGraph';
import { CaptureReadError, inspectCapture as readCapture } from '@tensorviz/contracts/captureReader';

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
  constructor(private readonly ledger: AttemptLedger) {}

  list(target: LocalBrowserTargetIdentity) {
    const state = this.ledger.state(target);
    const ids = [...new Set([state.lastCompleteCaptureId, state.lastPartialCaptureId, state.displayedCaptureId])].filter((id): id is string => id !== null);
    return { ...state, captures: ids.flatMap(id => {
      const record = this.ledger.capture(id)?.record;
      return record ? [record] : [];
    }) };
  }

  inspect(target: LocalBrowserTargetIdentity, query: CaptureInspectionQuery): CaptureInspectionPage {
    return inspectCapture(this.retained(target, query.captureId, query.sourceId), query);
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

  /** Export shares exactly the same ownership and eviction checks as inspection. */
  retained(target: LocalBrowserTargetIdentity, captureId: string, sourceId: string): CaptureSnapshot {
    const state = this.ledger.state(target);
    if (![state.lastCompleteCaptureId, state.lastPartialCaptureId, state.displayedCaptureId].includes(captureId)) {
      throw new GraphToolError('UNKNOWN_CAPTURE', 'This capture is no longer retained for the selected model.');
    }
    const retained = this.ledger.capture(captureId);
    if (!retained || retained.record.unavailable) throw new GraphToolError('CAPTURE_UNAVAILABLE', 'The capture payload exceeded the host retention limit.');
    if (retained.record.format !== 'semantic') throw new GraphToolError('LEGACY_CAPTURE', 'This legacy run has no canonical argument/state records.');
    const capture = object(object(retained.envelope?.result).diagnostics).semantic_capture as CaptureSnapshot | undefined;
    if (!capture) throw new GraphToolError('CAPTURE_UNAVAILABLE', 'The retained capture payload is unavailable.');
    if (capture.source.id !== sourceId) throw new GraphToolError('SOURCE_MISMATCH', 'Use the exact capture source identity.');
    return capture;
  }
}
