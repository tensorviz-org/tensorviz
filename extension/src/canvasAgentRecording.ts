import { createHash, randomUUID } from 'node:crypto';
import { createWriteStream, existsSync, mkdirSync, type WriteStream } from 'node:fs';
import * as path from 'node:path';

/** Explicit local QA recording. No provider wire traffic, credentials or upload. */
export class CanvasAgentRecording {
  readonly file: string;
  private readonly stream: WriteStream;
  private readonly started = process.hrtime.bigint();
  private readonly timer: NodeJS.Timeout;
  private sequence = 0;
  private bytes = 0;
  private stopped = false;
  readonly closed: Promise<void>;

  constructor(directory: string, metadata: Record<string, unknown>, private readonly report: (message: string) => void,
    private readonly limits = { eventBytes: 256 * 1024, totalBytes: 64 * 1024 * 1024, queueBytes: 4 * 1024 * 1024 }) {
    if (!path.isAbsolute(directory)) throw new Error('The recording directory must be absolute.');
    if (existsSync(path.join(directory, 'STOP'))) throw new Error('Choose a new recording directory; this session was stopped.');
    mkdirSync(directory, { recursive: true, mode: 0o700 });
    this.file = path.join(directory, `${Date.now()}-${process.pid}-${randomUUID()}.jsonl`);
    this.stream = createWriteStream(this.file, { flags: 'wx', mode: 0o600 });
    this.closed = new Promise(resolve => this.stream.once('close', resolve));
    this.stream.on('error', error => {
      this.stopped = true; clearInterval(this.timer);
      this.notify(`Agent recording failed: ${error.message}. The recording is incomplete.`);
    });
    let ticks = 0;
    this.timer = setInterval(() => {
      if (existsSync(path.join(directory, 'STOP'))) this.stop('requested');
      else if (++ticks % 10 === 0) this.record({ kind: 'recording_heartbeat' });
    }, 1000);
    this.timer.unref();
    this.record({ kind: 'recording_start', format: 1, pid: process.pid, metadata });
  }

  record(event: Record<string, unknown>): void {
    if (this.stopped) return;
    try {
      const payload = JSON.stringify(event);
      const payloadBytes = Buffer.byteLength(payload);
      const recorded = payloadBytes > this.limits.eventBytes
        ? { kind: 'omitted_event', originalKind: event.kind, turnId: event.turnId, callId: event.callId,
          bytes: payloadBytes, sha256: createHash('sha256').update(payload).digest('hex'), reason: 'event_byte_limit' }
        : event;
      const line = this.line(recorded);
      if (this.bytes + Buffer.byteLength(line) + 1024 > this.limits.totalBytes) { this.stop('total_byte_limit'); return; }
      if (this.stream.writableLength + Buffer.byteLength(line) > this.limits.queueBytes) { this.stop('write_queue_limit'); return; }
      this.bytes += Buffer.byteLength(line);
      this.sequence++;
      this.stream.write(line);
    } catch { this.stop('serialization_failure'); }
  }

  private line(event: Record<string, unknown>): string {
    return JSON.stringify({ ...event, sequence: this.sequence + 1, at: new Date().toISOString(),
      elapsedMs: Number(process.hrtime.bigint() - this.started) / 1e6 }) + '\n';
  }

  stop(reason = 'extension_shutdown'): void {
    if (this.stopped) return;
    this.stopped = true; clearInterval(this.timer);
    this.stream.end(this.line({ kind: 'recording_end', reason, recordedBytes: this.bytes }));
    this.notify(`Agent recording stopped (${reason}): ${this.file}`);
  }

  private notify(message: string): void { try { this.report(message); } catch { /* Observation cannot interrupt work. */ } }

  dispose(): void { this.stop(); }
}
