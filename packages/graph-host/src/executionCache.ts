/** Whole-result execution cache, layer 0 (shadow slice).
 *
 * A worker run whose inputs are exactly identical to an earlier accepted run produces the
 * same envelope, apart from a small, measured set of per-attempt fields. This module owns
 * the three pieces that claim needs: the key (`resultKey`), the rules for which runs may
 * be looked up and stored (`requestIneligible`, `outcomeUnstorable`), and a bounded
 * in-memory store (`ExecutionResultStore`). `normaliseEnvelope` / `firstDifference`
 * compare a stored envelope with a fresh one.
 *
 * Modes: `off` (nothing but reading the setting), `shadow` (the worker always runs; a
 * stored entry is only compared with the fresh result and logged) and `on` (a key hit
 * whose entry passes `servableEntry` is returned instead of launching the worker; a
 * miss runs and is stored exactly as in shadow). */
import { createHash } from 'node:crypto';
import * as os from 'node:os';
import { isDeepStrictEqual } from 'node:util';
import { backendDigest } from './backendIdentity';
import type { ExecutionSnapshot, ResolvedExecutionPlan } from './executionPlanCore';
import { semanticCapture } from './semanticCapture';
import type { TraceOutcome, TraceRequest, WorkerEnvelope } from './traceWorkerCore';

export type ExecutionCacheMode = 'off' | 'shadow' | 'on';
export interface ExecutionCacheSettings { mode: ExecutionCacheMode; maxSizeMB?: number }

/** Bump when the key's meaning changes, so no entry outlives a change of rules. */
export const RESULT_KEY_VERSION = 1;
export const DEFAULT_MAX_ENTRIES = 32;
export const DEFAULT_MAX_SIZE_MB = 96;
/** Same bound as the attempt ledger's retained capture. */
export const MAX_ENTRY_BYTES = 16 * 1024 * 1024;

/** Environment variables that are per-run transport, not run inputs: the private plan and
 *  candidate files have a fresh path every run (and their content is the plan, already
 *  in `applicability`). Kept in sync with traceWorkerCore's PER_RUN_ENV. */
export const PER_RUN_ENV_KEYS = ['TENSORVIZ_EXECUTION_PLAN', 'TENSORVIZ_CANDIDATE_REQUEST'];

/**
 * Every component of the key, each one a separate field so a test (and a future miss
 * report) can name which one changed.
 *
 *  - `applicability`: the resolved plan's `applicabilityDigest` — every workspace `.py`,
 *    config/profile, fixtures and checkpoints, the execution spec (inputs, seed, device,
 *    mode), timeout, capture limits, interpreter and its Python/torch/package versions.
 *  - `argv`: the worker's command line (depth, tier, kind, capture flags, entry point,
 *    file, class, workspace). It carries no per-run path.
 *  - `backend`: content digest of the backend that produces the envelope (F1), plus the
 *    worker protocol version.
 *  - `env`: sha256 over every variable the worker is spawned with (sorted), excluding the
 *    per-run plan/candidate paths. Thread counts (`OMP_NUM_THREADS`...), `PYTHONPATH`,
 *    `PYTORCH_*`, HF caches and anything the model reads via `os.environ` are all in it.
 *  - `machine`: CPU count, CPU model, architecture and platform — the defaults torch
 *    derives thread counts and kernels from.
 *  - `interpreter`: the executable that runs the worker.
 */
export interface ResultKeyParts {
  version: number;
  applicability: string;
  argv: string[];
  backend: { digest: string; protocol: number };
  env: string;
  machine: { platform: string; arch: string; cpus: number; cpuModel: string };
  interpreter: string;
}

/** Facts about this machine that change torch's default numerics. */
export function machineFacts(cpus: os.CpuInfo[] = os.cpus()): ResultKeyParts['machine'] {
  return { platform: process.platform, arch: process.arch, cpus: cpus.length, cpuModel: cpus[0]?.model ?? '' };
}

/** sha256 over the worker's environment, minus per-run transport paths. Values are hashed,
 *  never stored. */
export function envDigest(env: NodeJS.ProcessEnv): string {
  const entries = Object.entries(env)
    .filter((entry): entry is [string, string] => typeof entry[1] === 'string' && !PER_RUN_ENV_KEYS.includes(entry[0]))
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  return createHash('sha256').update(JSON.stringify(entries)).digest('hex');
}

export function resultKeyParts(input: { applicabilityDigest: string; args: string[]; backendDir: string; protocol: number;
  env: NodeJS.ProcessEnv; interpreter: string; machine?: ResultKeyParts['machine']; backend?: string }): ResultKeyParts {
  return {
    version: RESULT_KEY_VERSION,
    applicability: input.applicabilityDigest,
    argv: [...input.args],
    backend: { digest: input.backend ?? backendDigest(input.backendDir), protocol: input.protocol },
    env: envDigest(input.env),
    machine: input.machine ?? machineFacts(),
    interpreter: input.interpreter,
  };
}

/** Canonical JSON (sorted object keys) so key order never changes the digest. */
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map(k => `${JSON.stringify(k)}:${canonical((value as Record<string, unknown>)[k])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

export function resultKey(parts: ResultKeyParts): string {
  return createHash('sha256').update(`tensorviz.execution-cache\0${canonical(parts)}`).digest('hex');
}

/**
 * Why this request may not use the cache, or undefined when it may.
 *
 * Callers opt in with `reuse: 'allow'` (Canvas opening and Validate runs); MCP runs and
 * candidate runs do not. Everything that is not a plain, seeded, deterministic CPU forward
 * of a resolved plan is excluded here as well, so a caller that opts in by mistake still
 * cannot cache an excluded run.
 */
export function requestIneligible(request: TraceRequest, snapshot: ExecutionSnapshot | undefined, candidate: boolean): string | undefined {
  if (request.reuse !== 'allow' && request.reuse !== 'bypass') return 'not_requested';
  if (candidate) return 'candidate';
  if (request.prepareInputs || request.prepareConfigDefaults) return 'input_preparation';
  if (!request.execution || !snapshot) return 'no_resolved_plan';
  if (request.job || snapshot.plan.job) return 'diagnostic_job';
  if (request.kind && request.kind !== 'model') return 'kind';
  if (request.steps !== undefined || request.callable) return 'kind';
  if (request.tier && request.tier !== 'real') return 'tier';
  const execution = snapshot.plan.execution as { seed?: unknown; deterministic?: unknown; device?: unknown; framework?: unknown };
  if (execution.framework === 'jax') return 'jax';
  if (execution.seed === null || execution.seed === undefined) return 'no_seed';
  if (execution.deterministic !== true) return 'nondeterministic';
  if (execution.device !== 'cpu') return 'device';
  return undefined;
}

interface ReuseDiagnostics { storable?: unknown; reasons?: unknown; untrackedReads?: unknown }

/** Why this finished run may not be stored, or undefined when it may. */
export function outcomeUnstorable(outcome: TraceOutcome, snapshot: ExecutionSnapshot): string | undefined {
  if (!outcome.ok) return outcome.cancelled ? 'cancelled' : outcome.timedOut ? 'timed_out' : 'failed';
  if (outcome.exitCode !== 0) return 'exit_code';
  if (outcome.executionCurrent !== true) return 'not_current';
  const envelope = outcome.envelope;
  if (envelope.error) return 'worker_error';
  const diagnostics = (envelope.result as { diagnostics?: {
    execution?: { planDigest?: unknown; planCurrent?: unknown };
    semantic_capture?: unknown; semantic_capture_error?: unknown; reuse?: ReuseDiagnostics | null;
  } } | null)?.diagnostics;
  if (!diagnostics) return 'no_result';
  if (diagnostics.execution?.planDigest !== snapshot.applicabilityDigest || diagnostics.execution?.planCurrent !== true) return 'not_current';
  if (!diagnostics.semantic_capture || diagnostics.semantic_capture_error) return 'partial';
  const reuse = diagnostics.reuse;
  if (!reuse || typeof reuse !== 'object') return 'no_reuse_report';
  if (reuse.storable !== true) {
    const reasons = Array.isArray(reuse.reasons) ? reuse.reasons.filter(r => typeof r === 'string') : [];
    return `worker:${reasons.join('+') || 'unknown'}`;
  }
  return undefined;
}

/** What an entry needs, beyond the envelope, to be served: the identity of the attempt that
 *  produced it. `plan` is the host-resolved plan of that attempt (its `id` is the attempt
 *  id), which the reused capture is validated against — never the current attempt's. */
export interface StoredRunMeta {
  attemptId: string;
  captureId: string;
  plan: ResolvedExecutionPlan;
  planDigest: string;
  /** Epoch ms when the original run finished. */
  capturedAt: number;
}

interface StoredEntry { text: string; meta?: string; bytes: number }

/**
 * Bounded least-recently-used store of accepted envelopes, as JSON text (one compact copy
 * per entry, re-parsed on read so a caller can never mutate a stored result).
 */
export class ExecutionResultStore {
  private readonly entries = new Map<string, StoredEntry>();
  private totalBytes = 0;
  private maxBytes: number;

  constructor(private readonly limits: { maxEntries?: number; maxBytes?: number; maxEntryBytes?: number } = {}) {
    this.maxBytes = limits.maxBytes ?? DEFAULT_MAX_SIZE_MB * 1024 * 1024;
  }

  get size(): number { return this.entries.size; }
  get bytes(): number { return this.totalBytes; }

  /** Change the byte cap (the `maxSizeMB` setting), evicting down to it. */
  resize(maxBytes: number): void {
    this.maxBytes = maxBytes;
    this.evict();
  }

  has(key: string): boolean { return this.entries.has(key); }

  /** The stored envelope for `key` (a fresh copy), marking it most recently used. */
  get(key: string): WorkerEnvelope | undefined {
    const text = this.text(key);
    return text === undefined ? undefined : JSON.parse(text) as WorkerEnvelope;
  }

  /** The stored JSON text for `key` without parsing it, marking it most recently used. */
  text(key: string): string | undefined {
    return this.entry(key)?.text;
  }

  /** Text and (unparsed) serving metadata for `key`, marking it most recently used. */
  entry(key: string): { text: string; meta?: string } | undefined {
    const entry = this.entries.get(key);
    if (entry === undefined) return undefined;
    this.entries.delete(key);
    this.entries.set(key, entry);
    return { text: entry.text, ...(entry.meta !== undefined ? { meta: entry.meta } : {}) };
  }

  /** Store or refresh; false (and nothing stored) when the entry exceeds a cap on its own. */
  put(key: string, envelope: WorkerEnvelope, meta?: StoredRunMeta): boolean {
    return this.putText(key, JSON.stringify(envelope), meta);
  }

  /** `put` for an envelope already serialised (the worker's own stdout). */
  putText(key: string, text: string, meta?: StoredRunMeta): boolean {
    const metaText = meta === undefined ? undefined : JSON.stringify(meta);
    const bytes = Buffer.byteLength(text) + (metaText === undefined ? 0 : Buffer.byteLength(metaText));
    this.delete(key);
    if (bytes > (this.limits.maxEntryBytes ?? MAX_ENTRY_BYTES) || bytes > this.maxBytes) return false;
    this.entries.set(key, { text, ...(metaText !== undefined ? { meta: metaText } : {}), bytes });
    this.totalBytes += bytes;
    this.evict();
    return this.entries.has(key);
  }

  delete(key: string): void {
    const entry = this.entries.get(key);
    if (entry === undefined) return;
    this.entries.delete(key);
    this.totalBytes -= entry.bytes;
  }

  clear(): void { this.entries.clear(); this.totalBytes = 0; }

  private evict(): void {
    const maxEntries = this.limits.maxEntries ?? DEFAULT_MAX_ENTRIES;
    for (const key of this.entries.keys()) {
      if (this.entries.size <= maxEntries && this.totalBytes <= this.maxBytes) break;
      this.delete(key);
    }
  }
}

/** A reused capture, as `TraceOutcome.reused` reports it: the attempt that produced it. */
export interface ReusedCapture extends StoredRunMeta {}

type Capture = NonNullable<ReturnType<typeof semanticCapture>>;

/**
 * Serving metadata for a finished run that `outcomeUnstorable` accepted, or undefined when
 * the capture would not be servable (so the entry is stored for shadow comparison only and
 * a later `on` lookup runs fresh).
 */
export function storedRunMeta(outcome: TraceOutcome, snapshot: ExecutionSnapshot, now = Date.now()): StoredRunMeta | undefined {
  if (!outcome.ok) return undefined;
  const capture = completeCapture(outcome.envelope, snapshot.plan.id, snapshot.plan, snapshot.applicabilityDigest);
  return capture ? { attemptId: snapshot.plan.id, captureId: capture.id, plan: snapshot.plan,
    planDigest: snapshot.applicabilityDigest, capturedAt: now } : undefined;
}

function completeCapture(envelope: WorkerEnvelope, attemptId: string, plan: ResolvedExecutionPlan, digest: string): Capture | undefined {
  const payload = (envelope.result as { diagnostics?: { semantic_capture?: unknown } } | null)?.diagnostics?.semantic_capture;
  const capture = payload == null ? undefined : semanticCapture(payload, attemptId, plan, digest);
  if (!capture || capture.status !== 'complete') return undefined;
  const root = capture.invocations.find(call => call.id === capture.rootInvocationId);
  if (root?.status !== 'completed' || capture.invocations.some(call => call.status === 'failed')) return undefined;
  return capture;
}

/**
 * `source.files[*].documentVersion` is the editor's version counter for each open file. The
 * applicability digest (the key) leaves it out because the file's content digest already
 * covers it, so an identical run can be stored under one version and requested under
 * another (undo, or a file opened after the run). A served result must carry the versions
 * of the run that asked for it: navigation refuses a node whose file version is not the
 * open document's. Rewrites the stored plan's versions, and the envelope's two copies, from
 * the current plan, in place on `envelope`. Returns the rewritten plan, or undefined if the
 * file lists differ in anything but `documentVersion` (never expected under an equal key).
 */
function withCurrentDocumentVersions(envelope: WorkerEnvelope, stored: ResolvedExecutionPlan,
  current: ResolvedExecutionPlan): ResolvedExecutionPlan | undefined {
  const versions = current.source.files;
  const matches = (files: unknown): files is Array<Record<string, unknown>> => Array.isArray(files)
    && files.length === versions.length && files.every((file, index) => {
      if (!file || typeof file !== 'object') return false;
      const { documentVersion: _left, ...rest } = file as Record<string, unknown>;
      const { documentVersion: _right, ...want } = versions[index];
      return isDeepStrictEqual(rest, want);
    });
  const diagnostics = (envelope.result as { diagnostics?: Record<string, unknown> } | null)?.diagnostics;
  const sources = [
    (diagnostics?.execution as { resolvedPlan?: { source?: { files?: unknown } } } | undefined)?.resolvedPlan?.source,
    (diagnostics?.semantic_capture as { source?: { files?: unknown } } | undefined)?.source,
  ].filter((source): source is { files?: unknown } => !!source && typeof source === 'object');
  if (!matches(stored.source.files) || !sources.every(source => matches(source.files))) return undefined;
  const rewrite = <T extends object>(files: T[]): T[] =>
    files.map((file, index) => ({ ...file, documentVersion: versions[index].documentVersion }));
  for (const source of sources) source.files = rewrite(source.files as object[]);
  return { ...stored, source: { ...stored.source, files: rewrite(stored.source.files) } };
}

/**
 * Re-validate a stored entry before serving it — the store is never trusted. Returns the
 * envelope (a fresh parse) and the reused identity, or the reason it may not be served
 * (the caller deletes the entry and runs fresh). Checks, in order: the entry and metadata
 * parse and fit the size cap; the protocol version; the entry was produced under exactly
 * this applicability digest; the worker certified the plan current and the run storable
 * (`outcomeUnstorable` rules on the stored envelope); the semantic capture validates against
 * the ORIGINAL attempt and plan, is complete, its root call completed and no call failed.
 */
export function servableEntry(entry: { text: string; meta?: string } | undefined, snapshot: ExecutionSnapshot, protocol: number):
  { ok: true; envelope: WorkerEnvelope; reused: ReusedCapture } | { ok: false; reason: string } {
  try {
    if (!entry) return { ok: false, reason: 'missing' };
    if (entry.meta === undefined) return { ok: false, reason: 'no_serving_metadata' };
    if (Buffer.byteLength(entry.text) + Buffer.byteLength(entry.meta) > MAX_ENTRY_BYTES) return { ok: false, reason: 'oversized' };
    const meta = JSON.parse(entry.meta) as StoredRunMeta;
    const envelope = JSON.parse(entry.text) as WorkerEnvelope;
    if (!meta || typeof meta !== 'object' || typeof meta.attemptId !== 'string' || typeof meta.captureId !== 'string'
      || !meta.plan || typeof meta.plan !== 'object' || meta.plan.id !== meta.attemptId
      || !Number.isFinite(meta.capturedAt)) return { ok: false, reason: 'corrupt_metadata' };
    if (!envelope || typeof envelope !== 'object' || envelope.protocol_version !== protocol) return { ok: false, reason: 'protocol' };
    if (meta.planDigest !== snapshot.applicabilityDigest) return { ok: false, reason: 'plan_digest' };
    const unstorable = outcomeUnstorable({ ok: true, envelope, exitCode: 0, executionCurrent: true }, snapshot);
    if (unstorable) return { ok: false, reason: `stored_${unstorable}` };
    const capture = completeCapture(envelope, meta.attemptId, meta.plan, meta.planDigest);
    if (!capture || capture.id !== meta.captureId) return { ok: false, reason: 'capture' };
    const plan = withCurrentDocumentVersions(envelope, meta.plan, snapshot.plan);
    if (!plan) return { ok: false, reason: 'source_files' };
    const current = completeCapture(envelope, meta.attemptId, plan, meta.planDigest);
    if (!current || current.id !== meta.captureId) return { ok: false, reason: 'capture' };
    return { ok: true, envelope, reused: { attemptId: meta.attemptId, captureId: meta.captureId, plan,
      planDigest: meta.planDigest, capturedAt: meta.capturedAt } };
  } catch {
    return { ok: false, reason: 'corrupt' };
  }
}

/**
 * The fields that legitimately differ between two runs of an identical plan, measured by
 * running all 69 corpus models twice with different attempt ids (see
 * `packages/graph-host/test/executionCache.test.ts`, "allowed differences"):
 *
 *  1. timing: `worker.elapsed_ms`, `result.diagnostics.worker.elapsed_ms`,
 *     `result.diagnostics.timings_ms.build_ms`, `result.diagnostics.timings_ms.trace_ms`;
 *  2. `result.diagnostics.semantic_capture.id` (uuid4 per capture);
 *  3. run identity, as VALUES wherever they appear: the plan id (`worker.planId`), the
 *     source snapshot id, the attempt id, every source binding id (hashed from the
 *     snapshot id) and the two `source.generation` counters.
 *
 * Identity values are replaced by role placeholders (`<binding:3>` is the 4th binding),
 * by exact string equality only, so a value that is not one of these tokens — or a token
 * appearing somewhere new — still shows up as a difference.
 */
export const ALLOWED_TIMING_PATHS = [
  'worker.elapsed_ms',
  'result.diagnostics.worker.elapsed_ms',
  'result.diagnostics.timings_ms.build_ms',
  'result.diagnostics.timings_ms.trace_ms',
] as const;
export const ALLOWED_GENERATION_PATHS = [
  'result.diagnostics.semantic_capture.source.generation',
  'result.diagnostics.execution.resolvedPlan.source.generation',
] as const;

/** Lists whose entries' `documentVersion` may differ: the editor's version counter, which the
 *  key leaves out (see `withCurrentDocumentVersions`). Measured in the installed-VS Code
 *  shadow soak: it changes on undo to an identical file and when a file is opened after a run. */
export const ALLOWED_DOCUMENT_VERSION_LISTS = [
  'result.diagnostics.execution.resolvedPlan.source.files',
  'result.diagnostics.semantic_capture.source.files',
] as const;

type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

function at(root: Json, dotted: string): { parent: { [key: string]: Json }; key: string } | undefined {
  const parts = dotted.split('.');
  let node: Json = root;
  for (const part of parts.slice(0, -1)) {
    if (!node || typeof node !== 'object' || Array.isArray(node)) return undefined;
    node = node[part];
  }
  return node && typeof node === 'object' && !Array.isArray(node) ? { parent: node, key: parts[parts.length - 1] } : undefined;
}

export function normaliseEnvelope(envelope: unknown): unknown {
  const copy = JSON.parse(JSON.stringify(envelope)) as Json;
  for (const dotted of [...ALLOWED_TIMING_PATHS, ...ALLOWED_GENERATION_PATHS]) {
    const slot = at(copy, dotted);
    if (slot && slot.key in slot.parent) slot.parent[slot.key] = `<${dotted.split('.').pop()}>`;
  }
  for (const dotted of ALLOWED_DOCUMENT_VERSION_LISTS) {
    const slot = at(copy, dotted), files = slot?.parent[slot.key];
    if (Array.isArray(files)) for (const file of files) {
      if (file && typeof file === 'object' && !Array.isArray(file) && 'documentVersion' in file) file.documentVersion = '<documentVersion>';
    }
  }
  const tokens = new Map<string, string>();
  const capture = at(copy, 'result.diagnostics.semantic_capture.id');
  const record = (value: Json | undefined, role: string): void => {
    if (typeof value === 'string' && value && !tokens.has(value)) tokens.set(value, role);
  };
  if (capture) {
    const semantic = capture.parent;
    record(semantic.id, '<capture>');
    record(semantic.attemptId, '<attempt>');
    const source = semantic.source;
    if (source && typeof source === 'object' && !Array.isArray(source)) record(source.id, '<source>');
    const bindings = semantic.sourceBindings;
    if (Array.isArray(bindings)) bindings.forEach((binding, index) => {
      if (binding && typeof binding === 'object' && !Array.isArray(binding)) record(binding.id, `<binding:${index}>`);
    });
  }
  const worker = at(copy, 'worker.planId');
  if (worker) record(worker.parent.planId, '<plan>');
  const plan = at(copy, 'result.diagnostics.execution.resolvedPlan.id');
  if (plan) {
    record(plan.parent.id, '<plan>');
    const source = plan.parent.source;
    if (source && typeof source === 'object' && !Array.isArray(source)) record(source.id, '<source>');
  }
  const replace = (node: Json): Json => {
    if (typeof node === 'string') return tokens.get(node) ?? node;
    if (Array.isArray(node)) return node.map(replace);
    if (node && typeof node === 'object') return Object.fromEntries(Object.entries(node).map(([k, v]) => [k, replace(v)]));
    return node;
  };
  return replace(copy);
}

function identityTokens(envelope: unknown): Map<string, string> {
  // The same tokens `normaliseEnvelope` substitutes, collected without copying.
  const tokens = new Map<string, string>();
  const record = (value: unknown, role: string): void => {
    if (typeof value === 'string' && value && !tokens.has(value)) tokens.set(value, role);
  };
  const object = (value: unknown): Record<string, unknown> | undefined =>
    value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : undefined;
  const root = object(envelope), result = object(root?.result), diagnostics = object(result?.diagnostics);
  const semantic = object(diagnostics?.semantic_capture);
  if (semantic && 'id' in semantic) {
    record(semantic.id, '<capture>');
    record(semantic.attemptId, '<attempt>');
    record(object(semantic.source)?.id, '<source>');
    if (Array.isArray(semantic.sourceBindings)) semantic.sourceBindings.forEach((binding, index) => record(object(binding)?.id, `<binding:${index}>`));
  }
  const worker = object(root?.worker);
  if (worker && 'planId' in worker) record(worker.planId, '<plan>');
  const plan = object(object(diagnostics?.execution)?.resolvedPlan);
  if (plan && 'id' in plan) { record(plan.id, '<plan>'); record(object(plan.source)?.id, '<source>'); }
  return tokens;
}

type Trie = { [key: string]: Trie | true };
const ALLOWED: Trie = {};
for (const dotted of [...ALLOWED_TIMING_PATHS, ...ALLOWED_GENERATION_PATHS]) {
  const parts = dotted.split('.');
  let node = ALLOWED;
  parts.slice(0, -1).forEach(part => { node = (node[part] ??= {}) as Trie; });
  node[parts[parts.length - 1]] = true;
}
// `[]` is the allowance for every element of an array.
for (const dotted of ALLOWED_DOCUMENT_VERSION_LISTS) {
  let node = ALLOWED;
  dotted.split('.').forEach(part => { node = (node[part] ??= {}) as Trie; });
  node['[]'] = { documentVersion: true };
}

/**
 * Shadow comparison: the first path at which a stored and a fresh envelope differ once
 * the allowed differences are normalised, or undefined if they are identical.
 *
 * Same answer as `firstDifference(normaliseEnvelope(a), normaliseEnvelope(b))` (tested),
 * but one walk over both values with no copies: envelopes reach 8 MB.
 */
export function shadowDifference(stored: unknown, fresh: unknown): string | undefined {
  const left = identityTokens(stored), right = identityTokens(fresh);
  const kind = (v: unknown) => (v === null ? 'null' : Array.isArray(v) ? 'array' : typeof v);
  const walk = (a: unknown, b: unknown, here: string, allowed: Trie | undefined): string | undefined => {
    if (typeof a === 'string' && typeof b === 'string') return (left.get(a) ?? a) === (right.get(b) ?? b) ? undefined : here || '.';
    if (a === b) return undefined;
    if (kind(a) !== kind(b)) return here || '.';
    if (Array.isArray(a)) {
      const other = b as unknown[];
      if (a.length !== other.length) return `${here}[length]`;
      for (let i = 0; i < a.length; i++) {
        const element = allowed?.['[]'];
        const found = walk(a[i], other[i], `${here}[${i}]`, element === true ? undefined : element);
        if (found) return found;
      }
      return undefined;
    }
    if (a && typeof a === 'object') {
      const x = a as Record<string, unknown>, y = b as Record<string, unknown>;
      const keys = Object.keys(x);
      const same = keys.length === Object.keys(y).length && keys.every(k => k in y);
      for (const key of same ? keys.sort() : [...new Set([...keys, ...Object.keys(y)])].sort()) {
        if (!(key in x) || !(key in y)) return `${here}.${key}`;
        const next = allowed?.[key];
        if (next === true) continue;
        const found = walk(x[key], y[key], `${here}.${key}`, next);
        if (found) return found;
      }
      return undefined;
    }
    return here || '.';
  };
  return walk(stored, fresh, '', ALLOWED);
}

/** The first path at which two JSON values differ (depth first, keys sorted), or undefined. */
export function firstDifference(a: unknown, b: unknown, here = ''): string | undefined {
  if (a === b) return undefined;
  const kind = (v: unknown) => (v === null ? 'null' : Array.isArray(v) ? 'array' : typeof v);
  if (kind(a) !== kind(b)) return here || '.';
  if (Array.isArray(a)) {
    const other = b as unknown[];
    if (a.length !== other.length) return `${here}[length]`;
    for (let i = 0; i < a.length; i++) {
      const found = firstDifference(a[i], other[i], `${here}[${i}]`);
      if (found) return found;
    }
    return undefined;
  }
  if (a && typeof a === 'object') {
    const left = a as Record<string, unknown>, right = b as Record<string, unknown>;
    for (const key of [...new Set([...Object.keys(left), ...Object.keys(right)])].sort()) {
      if (!(key in left) || !(key in right)) return `${here}.${key}`;
      const found = firstDifference(left[key], right[key], `${here}.${key}`);
      if (found) return found;
    }
    return undefined;
  }
  return here || '.';
}

/** Shadow counters, per TraceWorkerCore (per window). */
export interface ExecutionCacheStats {
  lookups: number; hits: number; identical: number; different: number; misses: number;
  stored: number; notStorable: number; ineligible: number; hitNotStorable: number;
  /** `on` mode: results returned without launching the worker, and stored entries that
   *  failed `servableEntry` (deleted; the run went fresh). */
  served: number; serveRejected: number;
}
export function emptyStats(): ExecutionCacheStats {
  return { lookups: 0, hits: 0, identical: 0, different: 0, misses: 0, stored: 0, notStorable: 0, ineligible: 0, hitNotStorable: 0,
    served: 0, serveRejected: 0 };
}

/** The user-visible label for a reused result. Local time of the original run; the date as
 *  well when it was not today. */
export function reusedResultLabel(capturedAt: number, now = Date.now()): string {
  const then = new Date(capturedAt), today = new Date(now);
  const time = then.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const when = then.toDateString() === today.toDateString() ? time : `${then.toLocaleDateString()} ${time}`;
  return `Reused identical result from ${when} (no Python ran).`;
}
