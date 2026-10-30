/** Private evidence of a confirmed native source write, separate from candidates. */
import { randomUUID } from 'node:crypto';
import type { LocalBrowserTargetIdentity, AppliedSourceReceipt } from '@tensorviz/contracts/protocol';
import { isAppliedSourceFiles } from '@tensorviz/contracts/protocol';
export type { AppliedSourceReceipt } from '@tensorviz/contracts/protocol';
import type { AppliedSourceEvidence } from './sourceReconciliation';
import type { AppliedSourceSetEvidence } from './sourceSetReconciliation';
import { sha256 } from './sourceChanges';

export const APPLIED_SOURCE_LIMITS = { records: 32, evidenceBytes: 24 * 1024 * 1024,
  totalBytes: 32 * 1024 * 1024, ttlMs: 30 * 60_000 };

interface Entry {
  id: string; createdAt: number; target: LocalBrowserTargetIdentity;
  receipt: Omit<AppliedSourceReceipt, 'appliedAt'> & { appliedAt: number | null };
  json?: string; bytes: number;
}

/** Reserve before review; confirm only after SourceWriter acknowledges Apply.
 * Ownership lives in the browser/MCP adapter. An old reservation cannot recreate
 * forgotten, expired, evicted or disposed evidence when Apply completes late. */
export class AppliedSourceReceipts {
  private readonly entries = new Map<string, Entry>();
  private disposed = false;
  constructor(private readonly limits = APPLIED_SOURCE_LIMITS, private readonly now = Date.now) {}

  reserve(requestId: string, target: LocalBrowserTargetIdentity, evidence: AppliedSourceEvidence | AppliedSourceSetEvidence): string | undefined {
    this.prune();
    if (this.disposed || this.entries.has(requestId)) return undefined;
    if (evidence.version === 2 && (!isAppliedSourceFiles(evidence.files) || evidence.files.some(file =>
      !evidence.base.source.files.some(source => source.path === file.file && source.role === 'source' && source.digest === file.beforeDigest)))) return undefined;
    const json = JSON.stringify(evidence), bytes = Buffer.byteLength(json);
    while (this.entries.size >= this.limits.records) this.entries.delete(this.entries.keys().next().value!);
    const id = randomUUID(), tooLarge = bytes > this.limits.evidenceBytes;
    this.entries.set(requestId, { id, createdAt: this.now(), target: structuredClone(target),
      receipt: { appliedAt: null, ...(evidence.version === 2 ? { ...evidence.files[0], files: structuredClone(evidence.files) }
        : { file: evidence.reviewedFile, beforeDigest: sha256(evidence.sources[evidence.reviewedFile]), afterDigest: sha256(evidence.reviewedSource) }),
        unavailable: tooLarge ? 'too_large' : null },
      json: tooLarge ? undefined : json, bytes: tooLarge ? 0 : bytes });
    this.prune();
    return id;
  }

  confirm(requestId: string, reservation: string): void {
    this.prune();
    const entry = this.entries.get(requestId);
    if (entry?.id === reservation && entry.receipt.appliedAt === null) entry.receipt.appliedAt = this.now();
  }

  discard(requestId: string, reservation: string): void {
    const entry = this.entries.get(requestId);
    if (entry?.id === reservation && entry.receipt.appliedAt === null) this.entries.delete(requestId);
  }

  summary(requestId: string): AppliedSourceReceipt | null {
    this.prune();
    const entry = this.entries.get(requestId);
    return entry && entry.receipt.appliedAt !== null ? structuredClone(entry.receipt) as AppliedSourceReceipt : null;
  }

  /** Host-only input for reconciliation; source bodies never enter public summaries. */
  read(requestId: string): { reservation: string; receipt: AppliedSourceReceipt; target: LocalBrowserTargetIdentity; evidence?: AppliedSourceEvidence | AppliedSourceSetEvidence } | undefined {
    const receipt = this.summary(requestId), entry = this.entries.get(requestId);
    if (!receipt || !entry) return undefined;
    return { reservation: entry.id, receipt, target: structuredClone(entry.target), ...(entry.json ? { evidence: JSON.parse(entry.json) } : {}) };
  }

  current(requestId: string, reservation: string): boolean {
    this.prune();
    const entry = this.entries.get(requestId);
    return entry?.id === reservation && entry.receipt.appliedAt !== null;
  }

  forget(requestId: string): void { this.entries.delete(requestId); }
  dispose(): void { this.disposed = true; this.entries.clear(); }

  private prune(): void {
    for (const [id, entry] of this.entries) {
      if (this.now() - (entry.receipt.appliedAt ?? entry.createdAt) >= this.limits.ttlMs) this.entries.delete(id);
    }
    let bytes = [...this.entries.values()].reduce((sum, entry) => sum + entry.bytes, 0);
    for (const entry of this.entries.values()) {
      if (bytes <= this.limits.totalBytes) break;
      if (!entry.json) continue;
      bytes -= entry.bytes; entry.bytes = 0; entry.json = undefined; entry.receipt.unavailable = 'evicted';
    }
  }
}
