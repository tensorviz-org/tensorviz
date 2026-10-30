/** Host-owned candidate evidence, independent of the live attempt ledger. */
import { randomUUID } from 'node:crypto';
import { z } from 'zod';
import { validatedExecutionObservation, observationRun, type ExecutionObservation } from '@tensorviz/graph-host/executionObservation';
import { inspectEvidence } from '@tensorviz/graph-host/mcpRunEvidence';
import { compareRetainedRuns } from '@tensorviz/graph-host/mcpRunComparison';
import { candidateCheckpointEvidence, type CandidateEvidence } from '@tensorviz/graph-host/candidateExecution';
import { checkpointEvidencePage, checkpointEvidenceText, checkpointQuerySchema, type CheckpointEvidence } from '@tensorviz/graph-host/checkpointEvidence';
import type { CandidateReviewSnapshot } from './candidateReview';
import { semanticCapture } from '@tensorviz/graph-host/semanticCapture';
import { inspectCapture } from '@tensorviz/graph-host/captureInspection';
import type { CaptureInspectionQuery, CandidateRecord, CandidateCheckpointPage } from '@tensorviz/contracts/protocol';
export type { CandidateRecord } from '@tensorviz/contracts/protocol';
import { GraphToolError } from '@tensorviz/graph-host/mcpGraph';

export const CANDIDATE_HISTORY_LIMITS = { records: 32, captureBytes: 16 * 1024 * 1024,
  totalBytes: 32 * 1024 * 1024, ttlMs: 30 * 60_000 };

export const candidateObservationQuerySchema = z.object({
  section: z.enum(['summary', 'setup', 'outputs']).default('summary'),
  offset: z.number().int().min(0).default(0), limit: z.number().int().min(1).max(100).default(50),
  atol: z.number().finite().min(0).default(1e-6), rtol: z.number().finite().min(0).default(1e-5),
}).strict();
export type CandidateObservationQuery = z.input<typeof candidateObservationQuerySchema>;
interface ObservationPair { candidate: ExecutionObservation; baseline?: ExecutionObservation;
  baselineUnavailable: 'not_retained' | 'identity_mismatch' | null }

interface Retained { requestId: string; record: CandidateRecord; json?: string; checkpoint?: string;
  checkpointUnavailable?: 'invalid_checkpoint' | 'evicted'; observations?: string;
  observationsUnavailable?: 'invalid_observations' | 'too_large' | 'evicted'; bytes: number }

/** Callers must resolve transport ownership before using a source request ID.
 * The store retains JSON and small summaries, never worker envelopes/closures. */
export class CandidateHistory {
  private readonly records = new Map<string, Retained>();
  constructor(private readonly limits = CANDIDATE_HISTORY_LIMITS, private readonly now = Date.now) {}

  begin(requestId: string, summary: CandidateReviewSnapshot): string {
    this.prune();
    while (this.records.size >= this.limits.records) this.records.delete(this.records.keys().next().value!);
    const runId = randomUUID();
    this.records.set(runId, { requestId, bytes: 0, record: { runId, startedAt: this.now(), finishedAt: null,
      review: 'open', summary: structuredClone({ ...summary, runId }), executionId: null,
      capture: null, unavailable: 'not_captured' } });
    return runId;
  }

  update(requestId: string, runId: string, summary: CandidateReviewSnapshot): void {
    const item = this.find(requestId, runId);
    if (item) item.record.summary = structuredClone({ ...summary, runId });
  }

  complete(requestId: string, runId: string, summary: CandidateReviewSnapshot, evidence?: CandidateEvidence, baseline?: ExecutionObservation): void {
    const item = this.find(requestId, runId);
    // Revocation/expiry is permanent for this run, even if its worker completes later.
    if (!item || item.record.finishedAt !== null) return;
    this.update(requestId, runId, summary);
    item.record.finishedAt = this.now();
    item.record.executionId = evidence?.execution.id ?? null;
    if (evidence?.capture) try {
      const json = JSON.stringify(evidence.capture), size = Buffer.byteLength(json);
      if (size > this.limits.captureBytes) item.record.unavailable = 'too_large';
      else {
        const capture = semanticCapture(JSON.parse(json), evidence.execution.id,
          evidence.execution.plan, evidence.capture.planDigest);
        if (!capture) item.record.unavailable = 'invalid_capture';
        else {
          item.json = json; item.bytes = size;
          item.record.capture = { captureId: capture.id, sourceId: capture.source.id };
          item.record.unavailable = null;
        }
      }
    } catch { item.record.unavailable = 'invalid_capture'; }
    if (evidence?.checkpoint) try {
      const report = candidateCheckpointEvidence(evidence.checkpoint, evidence.execution.plan);
      if (!report) item.checkpointUnavailable = 'invalid_checkpoint';
      else {
        item.checkpoint = JSON.stringify(report);
        item.bytes += Buffer.byteLength(item.checkpoint);
      }
    } catch { item.checkpointUnavailable = 'invalid_checkpoint'; }
    if (evidence?.observations) try {
      const candidate = validatedExecutionObservation(evidence.observations, evidence.execution.plan, evidence.capture);
      if (!candidate) item.observationsUnavailable = 'invalid_observations';
      else {
        // The host outcome, rather than a partial worker observation, owns success.
        candidate.status = summary.runtime === 'succeeded' && candidate.status === 'passed' ? 'passed' : 'failed';
        const prior = baseline && validatedExecutionObservation(baseline, baseline.plan);
        const base = prior && prior.plan.id === evidence.execution.baseAttemptId &&
          prior.plan.source.id === evidence.execution.baseSourceId && prior.plan.source.workspaceId === evidence.execution.baseWorkspaceId &&
          prior.planDigest === evidence.execution.basePlanDigest &&
          prior.plan.source.modelFile === candidate.plan.source.modelFile && prior.plan.source.className === candidate.plan.source.className
          ? prior : undefined;
        const pair: ObservationPair = { candidate, ...(base ? { baseline: base } : {}),
          baselineUnavailable: base ? null : baseline ? 'identity_mismatch' : 'not_retained' };
        const json = JSON.stringify(pair), size = Buffer.byteLength(json);
        if (size > this.limits.captureBytes) item.observationsUnavailable = 'too_large';
        else { item.observations = json; item.bytes += size; }
      }
    } catch { item.observationsUnavailable = 'invalid_observations'; }
    this.prune();
  }

  close(requestId: string): void {
    this.prune();
    for (const item of this.records.values()) if (item.requestId === requestId) item.record.review = 'historical';
  }

  list(requestId: string): CandidateRecord[] {
    this.prune();
    return [...this.records.values()].filter(item => item.requestId === requestId).map(item => structuredClone(item.record));
  }

  get(requestId: string, runId: string): CandidateRecord { return structuredClone(this.require(requestId, runId).record); }

  inspect(requestId: string, runId: string, query: CaptureInspectionQuery) {
    const item = this.require(requestId, runId);
    if (!item.json) throw new GraphToolError('CAPTURE_UNAVAILABLE', 'This candidate capture is unavailable. Read its retained summary for the reason.');
    return inspectCapture(JSON.parse(item.json), query);
  }

  inspectObservations(requestId: string, runId: string, query: CandidateObservationQuery = {}) {
    const item = this.require(requestId, runId), parsed = candidateObservationQuerySchema.safeParse(query);
    if (!parsed.success) throw new GraphToolError('INVALID_QUERY', 'Use a supported observation section and bounded page/tolerances.');
    if (item.record.finishedAt === null) throw new GraphToolError('RUN_IN_PROGRESS', 'Wait for the candidate to finish.');
    if (!item.observations) throw new GraphToolError('OBSERVATIONS_UNAVAILABLE',
      `Candidate observations are unavailable (${item.observationsUnavailable ?? 'not_recorded'}). Inspection never reruns code.`);
    const pair = JSON.parse(item.observations) as ObservationPair;
    const { section, offset, limit, atol, rtol } = parsed.data;
    const candidate = observationRun(pair.candidate), baseline = pair.baseline && observationRun(pair.baseline);
    const inspected = inspectEvidence({ ...candidate.graph, execution: candidate.execution, job: candidate.job },
      candidate.evidence, section, offset, limit);
    const result = { runId, immutable: true, scope: 'exact_reviewed_candidate',
      candidateExecutionId: candidate.runId, baselineExecutionId: baseline?.runId ?? null,
      baselineUnavailable: pair.baselineUnavailable, evidence: inspected,
      ...(section === 'outputs' ? { comparison: baseline
        ? compareRetainedRuns(baseline, candidate, section, offset, limit, atol, rtol) : null } : {}),
      detail: 'Historical observations for the reviewed source and its saved conditions. Numerical differences, comparable conditions and runtime success are separate facts; none establishes training quality or current live-source validity.' };
    if (Buffer.byteLength(JSON.stringify(result)) > 128 * 1024) throw new GraphToolError('RESULT_TOO_LARGE', 'Use a smaller observation page.');
    return result;
  }

  inspectCheckpoint(requestId: string, runId: string, query: { offset?: number; limit?: number } = {}) {
    const { report, offset, limit } = this.checkpointReport(requestId, runId, query);
    return checkpointEvidencePage(report, offset, limit);
  }

  inspectCheckpointText(requestId: string, runId: string, query: { offset: number; limit: number }): CandidateCheckpointPage {
    const { report, offset, limit } = this.checkpointReport(requestId, runId, query);
    const page = checkpointEvidencePage(report, offset, limit), text = checkpointEvidenceText(report, page.items!);
    if (Buffer.byteLength(text) > 131072) throw new GraphToolError('RESULT_TOO_LARGE', 'Use a smaller checkpoint page.');
    return { runId, offset, count: report.entries.length, nextOffset: page.nextOffset!, text };
  }

  private checkpointReport(requestId: string, runId: string, query: { offset?: number; limit?: number }) {
    const item = this.require(requestId, runId), parsed = checkpointQuerySchema.safeParse(query);
    if (!parsed.success) throw new GraphToolError('INVALID_QUERY', 'Use a bounded checkpoint offset and page size.');
    if (!item.checkpoint) throw new GraphToolError('CHECKPOINT_UNAVAILABLE',
      `No checkpoint-load report is retained for this candidate (${item.checkpointUnavailable ?? 'not_recorded'}). Missing evidence is not a successful load.`);
    return { report: JSON.parse(item.checkpoint) as CheckpointEvidence, ...parsed.data };
  }

  forget(requestId: string): void {
    for (const [id, item] of this.records) if (item.requestId === requestId) this.records.delete(id);
  }
  clear(): void { this.records.clear(); }

  private find(requestId: string, runId: string): Retained | undefined {
    this.prune();
    const item = this.records.get(runId);
    return item?.requestId === requestId ? item : undefined;
  }
  private require(requestId: string, runId: string): Retained {
    const item = this.find(requestId, runId);
    if (!item) throw new GraphToolError('UNKNOWN_CANDIDATE', 'This candidate run is no longer retained for this proposal.');
    return item;
  }
  private prune(): void {
    for (const [id, item] of this.records) {
      if (this.now() - (item.record.finishedAt ?? item.record.startedAt) >= this.limits.ttlMs) this.records.delete(id);
    }
    let bytes = [...this.records.values()].reduce((sum, item) => sum + item.bytes, 0);
    for (const item of this.records.values()) {
      if (bytes <= this.limits.totalBytes) break;
      if (!item.json && !item.checkpoint && !item.observations) continue;
      bytes -= item.bytes; item.bytes = 0;
      if (item.json) { item.json = undefined; item.record.unavailable = 'evicted'; }
      if (item.checkpoint) { item.checkpoint = undefined; item.checkpointUnavailable = 'evicted'; }
      if (item.observations) { item.observations = undefined; item.observationsUnavailable = 'evicted'; }
    }
  }
}
