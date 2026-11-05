/** Delivery and inactivity limits sit outside the host's generation deadlines.
 * These describe a missing explanation reply, never model execution or GPU work.
 */
export const EXPLANATION_ACK_TIMEOUT_MS = 10000;
export const NODE_EXPLANATION_IDLE_TIMEOUT_MS = 60000;
export const MODEL_EXPLANATION_IDLE_TIMEOUT_MS = 75000;

type Status = 'queued' | 'loading' | 'ready' | 'unavailable' | 'cancelled';
type ExpiryReason = 'timeout' | 'provider';
interface Entry {
  requestId: string;
  status: Status | 'expired';
  progress?: number;
  phases?: ReadonlySet<string>;
  timer?: ReturnType<typeof setTimeout>;
  expire?: (reason: ExpiryReason) => void;
}
interface Pending {
  timeoutMs: number;
  expire(reason: ExpiryReason): void;
}

/** One clock per node/pass. Retired identities prevent late replies reopening
 * settled work; repeated terminal replies may still add source documentation.
 */
export class ExplanationWatchdog {
  private readonly entries = new Map<string, Entry>();
  private readonly retired = new Set<string>();
  private identity(key: string, requestId: string): string { return JSON.stringify([key, requestId]); }
  private retire(key: string, entry: Entry): void {
    clearTimeout(entry.timer);
    this.retired.add(this.identity(key, entry.requestId));
    while (this.retired.size > 512) this.retired.delete(this.retired.values().next().value!);
  }
  accepts(key: string, requestId: string, status: Status): boolean {
    if (this.retired.has(this.identity(key, requestId))) return false;
    const previous = this.entries.get(key);
    if (!previous || previous.requestId !== requestId) return true;
    if (previous.status === 'expired') return false;
    if (previous.status === 'ready' || previous.status === 'unavailable' || previous.status === 'cancelled') return previous.status === status;
    return previous.status !== 'loading' || status !== 'queued';
  }
  expired(key: string, requestId: string): boolean {
    const entry = this.entries.get(key);
    return this.retired.has(this.identity(key, requestId)) || (entry?.requestId === requestId && entry.status === 'expired');
  }
  observe(key: string, requestId: string, status: Status, pending?: Pending, progress?: number, phase?: string): boolean {
    if (!this.accepts(key, requestId, status)) return false;
    const previous = this.entries.get(key);
    if (previous?.requestId === requestId && progress !== undefined && previous.progress !== undefined) {
      if (progress < previous.progress) return false;
      // Replayed progress is not activity and must not keep a lost run alive.
      if (status === 'loading' && previous.status === status && progress === previous.progress
        && (previous.phases?.has(phase ?? '') || (previous.phases?.size ?? 0) >= 32)) return true;
    }
    if (previous && previous.requestId !== requestId) this.retire(key, previous);
    else clearTimeout(previous?.timer);
    const phases = new Set(previous?.requestId === requestId ? previous.phases : undefined);
    if (progress !== undefined && phases.size < 32) phases.add(phase ?? '');
    const entry: Entry = { requestId, status, progress, phases };
    this.entries.delete(key); this.entries.set(key, entry);
    if ((status === 'queued' || status === 'loading') && pending) {
      entry.expire = pending.expire;
      entry.timer = setTimeout(() => {
        if (this.entries.get(key) !== entry) return;
        entry.status = 'expired'; entry.timer = undefined;
        pending.expire('timeout');
      }, pending.timeoutMs);
    }
    while (this.entries.size > 256) {
      // A large number of selected nodes must not evict the model's live clock.
      const oldest = [...this.entries.keys()].find(key => key !== 'model')!;
      this.retire(oldest, this.entries.get(oldest)!); this.entries.delete(oldest);
    }
    return true;
  }
  /** A deliberate retry may use the prior identity until the host issues a new one. */
  retry(key: string, requestId: string, pending: Pending): void {
    const previous = this.entries.get(key);
    if (previous) { clearTimeout(previous.timer); this.entries.delete(key); }
    this.retired.delete(this.identity(key, requestId));
    this.observe(key, requestId, 'queued', pending);
  }
  interrupt(): void {
    for (const entry of this.entries.values()) {
      if (entry.status !== 'queued' && entry.status !== 'loading') continue;
      clearTimeout(entry.timer); entry.timer = undefined; entry.status = 'expired';
      entry.expire?.('provider');
    }
  }
  clear(): void {
    for (const entry of this.entries.values()) clearTimeout(entry.timer);
    this.entries.clear(); this.retired.clear();
  }
}
