/** Host-owned candidate evidence, independent of the live attempt ledger. */
import { randomUUID } from 'node:crypto';
import type { CandidateEvidence } from '@tensorviz/graph-host/candidateExecution';
import type { CandidateReviewSnapshot } from './candidateReview';
import { semanticCapture } from '@tensorviz/graph-host/semanticCapture';
import { inspectCapture } from '@tensorviz/graph-host/captureInspection';
import type { CaptureInspectionQuery, CandidateRecord } from '@tensorviz/contracts/protocol';
export type { CandidateRecord } from '@tensorviz/contracts/protocol';
import { GraphToolError } from '@tensorviz/graph-host/mcpGraph';

export const CANDIDATE_HISTORY_LIMITS = { records: 32, captureBytes: 16 * 1024 * 1024,
  totalBytes: 32 * 1024 * 1024, ttlMs: 30 * 60_000 };

interface Retained { requestId: string; record: CandidateRecord; json?: string; bytes: number }

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

  complete(requestId: string, runId: string, summary: CandidateReviewSnapshot, evidence?: CandidateEvidence): void {
    const item = this.find(requestId, runId);
    // Revocation/expiry is permanent for this run, even if its worker completes later.
    if (!item || item.record.finishedAt !== null) return;
    this.update(requestId, runId, summary);
    item.record.finishedAt = this.now();
    item.record.executionId = evidence?.execution.id ?? null;
    if (!evidence?.capture) return;
    try {
      const json = JSON.stringify(evidence.capture), size = Buffer.byteLength(json);
      if (size > this.limits.captureBytes) { item.record.unavailable = 'too_large'; return; }
      const capture = semanticCapture(JSON.parse(json), evidence.execution.id,
        evidence.execution.plan, evidence.capture.planDigest);
      if (!capture) { item.record.unavailable = 'invalid_capture'; return; }
      item.json = json; item.bytes = size;
      item.record.capture = { captureId: capture.id, sourceId: capture.source.id };
      item.record.unavailable = null;
      this.prune();
    } catch { item.record.unavailable = 'invalid_capture'; }
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
      if (!item.json) continue;
      bytes -= item.bytes; item.bytes = 0; item.json = undefined; item.record.unavailable = 'evicted';
    }
  }
}
