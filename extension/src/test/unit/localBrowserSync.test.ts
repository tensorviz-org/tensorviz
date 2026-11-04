/** Step-5 "synchronize the source and the runtime-enriched graph" unit tests.
 *
 * What they pin, in product language:
 *  - opening a target shows the AST graph at once and labels it source-only;
 *    a save in a trusted workspace with traceOnSave measures the model and the label
 *    becomes runtime-traced;
 *  - typing re-projects from the buffer (never re-running the model for a
 *    keystroke); a broken buffer greys the last valid graph and names the
 *    line; fixing the buffer un-stales;
 *  - a trace that never starts leaves the source tier; a failed merge visibly
 *    stales retained evidence instead of claiming it represents the latest run;
 *  - a retarget discards the old class's in-flight answer;
 *  - the syncs never writes a workspace file: viewing a graph is read-only.
 *
 * Step 6 rides the same rig: the cursor entering a node's span focuses it on
 * the page (once per entry, innermost wins), a view's `revealNode` moves the
 * editor cursor only for the exact state the page is showing — and its own
 * selection event cannot echo straight back.
 */

import * as assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import * as path from 'node:path';
import { mock } from 'node:test';

import { LocalBrowserSync, LocalBrowserTraceSupport } from '../../localBrowserSync';
import { isLocalBrowserMessage, LocalBrowserTargetIdentity, ProjectionPayload } from '@tensorviz/contracts/protocol';
import { TraceOutcome, TraceRequest, WorkerEnvelope } from '../../worker';
import { executionSpecSchema } from '@tensorviz/graph-host/executionSpec';
import type { ResolvedExecutionPlan } from '@tensorviz/graph-host/executionPlanCore';
import { ValidationSummaryStore } from '@tensorviz/graph-host/validationSummary';
import { automaticFixture } from '../../../../tests/fixtures/semantic.cjs';
import type { ProfileLookupOptions } from '@tensorviz/graph-host/profileValidation';
import {
  Uri,
  commands,
  documentChanges,
  editorState,
  openStubEditor,
  reset,
  resetEditors,
  saveChanges,
  selectionChanges,
  state,
  stubSourceIO,
} from '../vscodeStub';

const ROOT = path.join(path.sep, 'ws');
const MODEL = path.join(ROOT, 'models', 'net.py');
const REL = 'models/net.py';
/** The dependency the multi-file tests register next to the root. */
const BLOCKS = path.join(ROOT, 'models', 'blocks.py');
const BLOCKS_REL = 'models/blocks.py';
/** A file the resolver never consults: edits to it must not pay a projection. */
const UNRELATED = path.join(ROOT, 'models', 'unrelated.py');
const UNRELATED_REL = 'models/unrelated.py';

const explicitProfile = { kind: 'model', model: 'models.net:Net', cwd: '.', tracked_files: [],
  limits: { timeout_ms: 60000, max_events: 10000, max_tensors: 10000, max_record_bytes: 4194304 },
  mode: 'eval', device: 'cpu', grad_enabled: false, init: { args: [], kwargs: {} },
  inputs: { args: [{ kind: 'tensor', shape: [1, 8], dtype: 'float32', requires_grad: false,
    generator: { kind: 'full', value: 1 } }], kwargs: {} } };
const explicitProfiles: ProfileLookupOptions = {
  snapshot: async (_root, selected) => (selected ? { guards: {}, resolution: { status: 'resolved', spec: {
    profile: explicitProfile, identity: { profile: selected }, files: [], interpreter: { executable: '/python' }, dependencyScope: 'partial',
  } } } : { profiles: ['example'], profileDefinitions: { example: explicitProfile }, guards: {}, resolution: { status: 'needs_setup', requirements: [
    { profile: null, field: ['profiles'], code: 'ambiguous', phase: 'static', reason: 'Choose.', remedy: 'Choose.' },
  ] } }) as never,
  choose: async () => undefined,
};
const SAVED = [
  'import torch.nn as nn',
  '',
  '',
  'class Net(nn.Module):',
  '    def forward(self, x):',
  '        return x',
  '',
].join('\n');
/** Same file mid-edit: the class statement has lost its colon. Line 4 breaks. */
const BROKEN = SAVED.replace('class Net(nn.Module):', 'class Net(nn.Module)');
/** Same file, valid, one keystroke later. */
const TYPED = `${SAVED}\nclass Added(nn.Module):\n    pass\n`;
const BLOCK_SOURCE = [
  'import torch.nn as nn',
  'class Block(nn.Module):',
  '    def forward(self, x):',
  '        return x',
  '',
].join('\n');
const BLOCK_EDITED = BLOCK_SOURCE.replace('return x', 'return x + 1');

const ENVELOPE: WorkerEnvelope = {
  protocol_version: 1,
  result: { graph: { nodes: [], edges: [] } },
  spans: {},
  error: null,
  worker: { python: 'test', torch: 'fake', elapsed_ms: 1 },
};

/** Two spans over SAVED, both valid for navigation: `net` covers the class
 *  (lines 4-6), `fwd` the forward body (lines 5-6) — nested, so a cursor walk
 *  observes innermost-wins. */
const SPANS: Record<string, unknown> = {
  net: { file: REL, start: [4, 0], end: [6, 8], kind: 'decl', digest: 'd1' },
  fwd: { file: REL, start: [5, 4], end: [6, 8], kind: 'call', digest: 'd2' },
};

const DEPENDENCY_SPANS: Record<string, unknown> = {
  'left.block': { file: BLOCKS_REL, start: [2, 0], end: [4, 16], kind: 'decl' },
  'right.block': { file: BLOCKS_REL, start: [2, 0], end: [4, 16], kind: 'decl' },
};

function baseIdentity(version: number, source: string): LocalBrowserTargetIdentity {
  return {
    workspaceFolder: ROOT,
    file: REL,
    cls: 'Net',
    documentVersion: version,
    sourceDigest: createHash('sha256').update(source, 'utf8').digest('hex'),
    backendRevision: 0,
  };
}

/** Let the loop's microtasks and the rig's zeroed timers run out. */
async function settle(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 0));
  await new Promise((resolve) => setTimeout(resolve, 0));
  await new Promise((resolve) => setTimeout(resolve, 0));
}

type Published = Record<string, unknown>;

interface SyncRig {
  sync: LocalBrowserSync;
  published: Published[];
  /** Every projection POST, oldest first: /parse (AST tier) or /project (traced). */
  fetches: Array<{ url: string; body: Record<string, unknown>; signal?: AbortSignal | null }>;
  runs: TraceRequest[];
  envelopes: Array<{ folder: string; file: string; cls: string; reused?: true }>;
  logs: string[];
  releaseFirstParse(): void;
  releaseFirstRun(): void;
}

const rigs: SyncRig[] = [];

/** The multifile request body the sync now POSTs. `source`/`cls` still exist in
 *  old tests' vocabulary via the helpers below. */
interface MultifileBody extends Record<string, unknown> {
  protocol_version: number;
  root: { file: string; cls: string; source_digest: string; document_version?: number };
  sources: Array<{ file: string; source: string; digest: string; document_version?: number }>;
  truncated: boolean;
  trace_result?: unknown;
}

/** The one endpoint every projection goes through (P0 step 4). */
function isProjection(url: string): boolean {
  return url.endsWith('/api/ide/multifile/project');
}

/** The root entry's text — the buffer the request is ABOUT. */
function rootSourceOf(body: Record<string, unknown>): string {
  const multifile = body as MultifileBody;
  return multifile.sources.find((entry) => entry.file === multifile.root.file)?.source ?? '';
}

function makeSync(opts: {
  summaries?: ValidationSummaryStore;
  profiles?: ProfileLookupOptions;
  debounceMs?: number;
  trace?: { trusted?: boolean; onSave?: boolean; fail?: boolean; deferFirstRun?: boolean; outcome?: TraceOutcome;
    tier?: 'real' | 'fake' | 'auto'; respond?: (request: TraceRequest) => TraceOutcome };
  deferFirstParse?: boolean;
  rejectProject?: boolean;
  diskSources?: Record<string, string>;
  spans?: Record<string, unknown>;
  mutateResponse?: (response: Record<string, unknown>, body: MultifileBody) => void;
  fetchResponse?: (body: MultifileBody, signal: AbortSignal) => Promise<Response | undefined>;
}): SyncRig {
  const published: Published[] = [];
  const fetches: SyncRig['fetches'] = [];
  const runs: TraceRequest[] = [];
  const envelopes: SyncRig['envelopes'] = [];
  const logs: string[] = [];
  let revision = 0;
  let gate: Promise<void> | undefined;
  let releaseFirstParse = (): void => {
    gate = undefined;
  };
  if (opts.deferFirstParse) {
    gate = new Promise<void>((resolve) => {
      releaseFirstParse = () => {
        gate = undefined;
        resolve();
      };
    });
  }
  let runGate: Promise<void> | undefined;
  let releaseFirstRun = (): void => {
    runGate = undefined;
  };
  if (opts.trace?.deferFirstRun) {
    runGate = new Promise<void>((resolve) => {
      releaseFirstRun = () => {
        runGate = undefined;
        resolve();
      };
    });
  }
  const fetchImpl = (async (input: unknown, init?: RequestInit) => {
    const url = String(input);
    const body = JSON.parse(String(init?.body)) as MultifileBody;
    fetches.push({ url, body, signal: init?.signal });
    if (gate) {
      await gate;
    }
    const override = await opts.fetchResponse?.(body, init!.signal as AbortSignal);
    if (override) return override;
    const rootText = rootSourceOf(body);
    // The broken buffer does not parse: same shape the backend 400 has.
    if (/class Net\(nn\.Module\)\s*\n/.test(rootText)) {
      return new Response(
        JSON.stringify({
          detail: { error: 'syntax_error', message: 'expected colon', line: 4, column: 21 },
        }),
        { status: 400 },
      );
    }
    if (body.trace_result !== undefined && opts.rejectProject) {
      return new Response(
        JSON.stringify({ detail: { error: 'trace_rejected', message: 'protocol_version 99' } }),
        { status: 400 },
      );
    }
    revision += 1;
    // The fake resolver "consults" every supplied source except the control
    // file: that asymmetry is what lets a test prove unrelated edits pay no
    // request.
    const dependencies = body.sources
      .filter((entry) => entry.file !== UNRELATED_REL)
      .map((entry) => ({ file: entry.file, digest: entry.digest }));
    const spanmap = Object.fromEntries(
      Object.entries(opts.spans ?? {}).map(([nodeId, raw]) => {
        if (typeof raw !== 'object' || raw === null) {
          return [nodeId, raw];
        }
        const span = raw as Record<string, unknown>;
        const source = body.sources.find((entry) => entry.file === span.file);
        return [
          nodeId,
          Object.prototype.hasOwnProperty.call(span, 'file_digest') || !source
            ? span
            : { ...span, file_digest: source.digest },
        ];
      }),
    );
    const responseBody: Record<string, unknown> = {
        cls: body.root.cls,
        document: { view: {}, semantic: { nodes: [] } },
        spanmap,
        source_digest: body.root.source_digest,
        projection_digest: createHash('sha256')
          .update(dependencies.map((entry) => `${entry.file}\0${entry.digest}`).join('\n'))
          .digest('hex'),
        graph_id: 'g-test',
        revision,
        traced: body.trace_result !== undefined,
        trace_error: null,
        protocol_version: 1,
        dependencies,
        diagnostics: [],
        truncated: body.truncated,
    };
    opts.mutateResponse?.(responseBody, body);
    return new Response(
      JSON.stringify(responseBody),
      { status: 200 },
    );
  }) as typeof fetch;

  let trace: LocalBrowserTraceSupport | undefined;
  if (opts.trace) {
    const plan = opts.trace;
    trace = {
      worker: {
        async run(request: TraceRequest): Promise<TraceOutcome> {
          runs.push(request);
          if (runGate) {
            await runGate;
          }
          if (request.beforeLaunch && !await request.beforeLaunch()) {
            return { ok: false, reason: 'Saved execution changed before launch.', detail: 'Nothing was run.' };
          }
          if (plan.fail) {
            return { ok: false, reason: 'no interpreter', detail: 'select one' };
          }
          if (plan.respond) return plan.respond(request);
          return plan.outcome ?? { ok: true, envelope: ENVELOPE, exitCode: 0 };
        },
        cancel(): void {},
      },
      trusted: async () => plan.trusted !== false,
      settings: () => ({ onSave: plan.onSave !== false, tier: plan.tier ?? 'real' }),
      onEnvelope: (_envelope, target, run) => {
        envelopes.push(run?.reused ? { ...target, reused: true } : target);
      },
    };
  }

  const sync = new LocalBrowserSync({
    channel: { appendLine: line => logs.push(line) },
    origin: () => 'http://127.0.0.1:8631',
    token: () => 'test-token',
    publish: (message) => {
      published.push(message as Published);
    },
    trace,
    fetchImpl,
    debounceMs: opts.debounceMs ?? 0,
    saveCoalesceMs: 0,
    summaries: opts.summaries,
    profiles: opts.profiles ?? explicitProfiles,
    // The stub has no filesystem: the source set enumerates/reads the stubbed
    // editor documents — the same dirty-buffer-wins rule the production
    // vscodeSourceIO applies.
    sourceIO: (folder) => {
      const io = stubSourceIO(folder);
      return { ...io, read: async file => opts.diskSources?.[file] ?? io.read(file) };
    },
  });
  // Lifecycle tests supply an explicit tensor unless the test supplies an execution profile.
  const validate = sync.validate.bind(sync);
  sync.validate = (target, signal, inputs = [{ shape: [1, 8], dtype: 'float32', fill: 'ones' }], ...rest) =>
    validate(target, signal, inputs, ...rest);
  const rig: SyncRig = { sync, published, fetches, runs, envelopes, logs, releaseFirstParse, releaseFirstRun };
  rigs.push(rig);
  return rig;
}

/** Anchor on the model file: an open buffer at SAVED, version recorded. */
function anchor(cls = 'Net'): LocalBrowserTargetIdentity {
  const editor = openStubEditor(MODEL, SAVED);
  void editor;
  const version = editorState.versions.get(MODEL) ?? 1;
  return baseIdentity(version, SAVED);
}

function modelUri() {
  return Uri.file(MODEL) as unknown as Parameters<LocalBrowserSync['setTarget']>[0];
}

function shapeUri(fsPath: string) {
  return Uri.file(fsPath) as unknown as Parameters<LocalBrowserSync['shapesFor']>[0];
}

/** Retype the buffer and fire the change, the way the editor does. */
function type(text: string): void {
  const editor = openStubEditor(MODEL, text);
  documentChanges.fire({ document: editor.document });
}

function editFile(fsPath: string, text: string): void {
  const editor = openStubEditor(fsPath, text);
  documentChanges.fire({ document: editor.document });
}

/** The editor the suite's model file is open in. */
function modelEditor() {
  const editor = editorState.visibleTextEditors.find((e) => e.document.uri.fsPath === MODEL);
  assert.ok(editor, 'the model file is not open');
  return editor;
}

/** Move the caret onto `line1` (1-based) in the model editor and fire the
 *  event the way vscode does. */
function moveCursor(line1: number): void {
  const editor = modelEditor();
  editor.selection = { active: { line: line1 - 1, character: 0 } };
  selectionChanges.fire({ textEditor: editor, selections: [editor.selection] });
}

/** The identity of the projection that landed — exactly what the page shows. */
function liveTarget(rig: SyncRig): LocalBrowserTargetIdentity {
  const projection = rig.published.find((m) => m.type === 'projection');
  assert.ok(projection, 'no projection was published');
  return projection.target as LocalBrowserTargetIdentity;
}

/** The focusNode messages the page has been told, in order. */
function focusPublishes(
  rig: SyncRig,
): Array<{ target: LocalBrowserTargetIdentity; nodeId: string }> {
  return rig.published
    .filter((m) => m.type === 'focusNode')
    .map((m) => ({ target: m.target as LocalBrowserTargetIdentity, nodeId: m.nodeId as string }));
}

suite('LocalBrowserSync', () => {
  const files = require('../../executionFiles') as typeof import('../../executionFiles');
  const verifyImportedSources = files.verifyImportedSources;
  setup(() => {
    // Worker/source evidence is covered by executionFiles and real-worker integration tests.
    files.verifyImportedSources = async () => {};
    reset();
    resetEditors();
    state.workspaceFolders = [{ uri: { fsPath: ROOT } }];
  });

  teardown(() => {
    files.verifyImportedSources = verifyImportedSources;
    for (const rig of rigs.splice(0)) {
      rig.sync.dispose();
    }
  });

  test('normal capture never searches when neither a profile nor inputs are supplied', async () => {
    const rig = makeSync({ trace: {} });
    rig.sync.setTarget(modelUri(), anchor()); await settle();
    const result = await LocalBrowserSync.prototype.validate.call(rig.sync, rig.sync.currentTarget()!, new AbortController().signal);
    assert.equal(result.status, 'refused');
    assert.match(result.detail, /No candidate shapes/);
    assert.equal(rig.runs.length, 0);
  });

  test('same-path desktop and foreign-SSH events cannot refresh or capture the remote model', async () => {
    const rig = makeSync({ trace: { onSave: true } });
    rig.sync.setTarget(modelUri(), anchor()); await settle();
    const before = rig.published.length;
    for (const scheme of ['vscode-local', 'vscode-remote', 'git']) {
      const document = { ...modelEditor().document,
        uri: { ...Uri.file(MODEL), scheme, authority: 'ssh-remote+other' } };
      documentChanges.fire({ document, contentChanges: [{ text: 'foreign edit' }] } as never);
      saveChanges.fire(document as never);
      selectionChanges.fire({ textEditor: { ...modelEditor(), document },
        selections: [{ active: { line: 4, character: 0 } }] } as never);
    }
    await settle();
    assert.equal(rig.runs.length, 0, 'foreign save events must not execute this model');
    assert.equal(rig.published.length, before, 'foreign buffers must not retarget or stale the graph');
    saveChanges.fire(modelEditor().document); await settle();
    assert.equal(rig.runs.length, 1, 'the transformed file URI still captures on save');
  });

  test('explicit capture can recover from unsupported static projection', async () => {
    const rig = makeSync({ trace: {}, fetchResponse: async body => body.trace_result ? undefined
      : new Response(JSON.stringify({ detail: 'Unsupported static lowering' }), { status: 422 }) });
    rig.sync.setTarget(modelUri(), anchor()); await settle();
    assert.equal(rig.sync.snapshotState()?.projection, null);
    const execution = executionSpecSchema.parse({ inputs: { args: [{ shape: [1, 8], dtype: 'float32', fill: 'ones' }] } });
    const result = await rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal,
      undefined, undefined, undefined, undefined, execution);
    assert.equal(result.status, 'passed', result.detail);
    assert.equal(rig.runs.length, 1);
    assert.equal(rig.sync.snapshotState()?.traced, true);
  });

  test('automatic timeout persists independently of its earlier measured capture', async () => {
    const summaries = new ValidationSummaryStore(() => {});
    const trace: { outcome?: TraceOutcome } = {};
    const rig = makeSync({ trace, summaries });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    saveChanges.fire(modelEditor().document);
    await settle();
    const target = rig.sync.currentTarget()!;
    const first = summaries.ledger.state(target);
    assert.equal(first.latestAttempt!.origin, 'automatic');
    assert.equal(rig.runs[0].attemptId, first.latestAttempt!.id);
    assert.equal(first.latestAttempt!.status, 'succeeded');
    trace.outcome = { ok: false, timedOut: true, reason: 'timeout', detail: 'private' };
    saveChanges.fire(modelEditor().document);
    await settle();
    const second = summaries.ledger.state(target);
    assert.equal(second.latestAttempt!.status, 'timed_out');
    assert.equal(rig.runs[1].attemptId, second.latestAttempt!.id);
    assert.notEqual(rig.runs[0].attemptId, rig.runs[1].attemptId);
    assert.deepEqual(second.latestAttempt!.captureIds, []);
    assert.equal(second.displayedCaptureId, first.displayedCaptureId);
    assert.equal(summaries.snapshot(target)!.status, 'failed');
    assert.equal(summaries.snapshot(target)!.projectionRevision, undefined);
    assert.ok(!JSON.stringify(summaries.snapshot(target)).includes('private'));
  });

  test('trace-on-save retains the prepared canonical call in reconnect attribution', async () => {
    const summaries = new ValidationSummaryStore(() => {});
    const trace: { deferFirstRun: boolean; outcome?: TraceOutcome } = { deferFirstRun: true };
    const rig = makeSync({ trace, summaries });
    rig.sync.setTarget(modelUri(), anchor()); await settle();
    saveChanges.fire(modelEditor().document); await settle();
    const request = rig.runs[0];
    assert.equal(request.workspace, ROOT);
    const { capture, preparation } = automaticFixture(request.attemptId);
    request.onCapturePrepared!({ preparation, applicabilityDigest: capture.preparationDigest });
    const envelope = structuredClone(ENVELOPE);
    (envelope.result as any).diagnostics = { semantic_capture: capture };
    trace.outcome = { ok: true, envelope, exitCode: 0, executionCurrent: true };
    rig.releaseFirstRun(); await settle();
    const target = rig.sync.currentTarget()!, evidence = summaries.snapshot(target)!.evidence!;
    assert.equal(evidence.origin, 'automatic');
    assert.equal(evidence.sourceId, capture.source.id);
    assert.equal(evidence.planDigest, null);
    assert.equal(evidence.setup, null);
    assert.equal(evidence.lastCompleteCaptureId, capture.id);
    assert.equal(evidence.displayedCapture?.id, capture.id);
    assert.equal(evidence.displayedCapture?.format, 'semantic');
  });

  test('automatic tracing refuses invalid explicit settings and dirty saved dependencies', async () => {
    for (const invalidConfig of [false, true]) {
      resetEditors();
      const summaries = new ValidationSummaryStore(() => {});
      const profiles: ProfileLookupOptions | undefined = invalidConfig ? {
        snapshot: async () => { throw new Error('private configuration details'); }, choose: async () => undefined,
      } : undefined;
      const rig = makeSync({ trace: {}, summaries, profiles });
      openStubEditor(BLOCKS, BLOCK_SOURCE);
      rig.sync.setTarget(modelUri(), anchor());
      await settle();
      if (!invalidConfig) editorState.dirtyDocuments.add(BLOCKS);
      saveChanges.fire(modelEditor().document);
      await settle();
      assert.equal(rig.runs.length, 0);
      const target = rig.sync.currentTarget()!;
      assert.equal(summaries.snapshot(target)!.status, 'refused');
      assert.equal(summaries.ledger.state(target).latestAttempt!.status, 'needs_setup');
    }
  });

  test('automatic tracing honors a single saved profile and its timeout instead of the shape guess', async () => {
    const profile = { cwd: '.', tracked_files: [], limits: { timeout_ms: 9000, max_events: 10, max_tensors: 10, max_record_bytes: 1024 },
      kind: 'model', model: 'models.net:Net', mode: 'eval', device: 'cpu', grad_enabled: false, seed: 3,
      init: { args: [], kwargs: {} }, inputs: { args: [{ kind: 'tensor', shape: [3, 7], dtype: 'float32', requires_grad: false,
        generator: { kind: 'full', value: 1 } }], kwargs: {} } };
    const profiles: ProfileLookupOptions = {
      snapshot: async (_root, selected) => (selected ? { guards: {}, resolution: { status: 'resolved', spec: {
        profile, identity: { profile: selected }, files: [], interpreter: { executable: '/python' }, dependencyScope: 'partial',
      } } }
        : { profiles: ['one'], profileDefinitions: { one: profile }, guards: {}, resolution: { status: 'needs_setup', requirements: [
          { profile: null, field: ['profiles'], code: 'ambiguous', phase: 'static', reason: 'Choose a profile.', remedy: 'Choose.' },
        ] } }) as never,
      choose: async () => { throw new Error('a unique profile must not ask'); },
    };
    const rig = makeSync({ trace: {}, profiles });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    saveChanges.fire(modelEditor().document);
    await settle();
    assert.equal(rig.runs.length, 1);
    assert.equal(rig.runs[0].timeoutMs, 9000);
    assert.equal(rig.runs[0].execution?.seed, 3);
    assert.deepEqual(rig.runs[0].execution?.inputs, { binding: profile.inputs });
    assert.equal(rig.runs[0].input, undefined);
    assert.equal(rig.runs[0].overrides, undefined);
  });

  test('a cancelled automatic profile lookup cannot launch after its replacement completes', async () => {
    const summaries = new ValidationSummaryStore(() => {});
    let release!: () => void;
    let lookups = 0;
    const profiles: ProfileLookupOptions = {
      hasConfig: async () => { if (++lookups === 1) await new Promise<void>(resolve => { release = resolve; }); return false; },
      snapshot: async () => { throw new Error('absent config must not resolve'); }, choose: async () => undefined,
    };
    const rig = makeSync({ trace: {}, summaries, profiles });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    saveChanges.fire(modelEditor().document);
    await settle();
    saveChanges.fire(modelEditor().document);
    await settle();
    assert.equal(rig.runs.length, 0);
    const current = summaries.snapshot(rig.sync.currentTarget()!)!;
    assert.equal(current.status, 'refused');
    release(); await settle();
    assert.equal(rig.runs.length, 0);
    assert.deepEqual(summaries.snapshot(rig.sync.currentTarget()!), current);
  });

  test('a delayed cancel from a superseded manual request cannot cancel the newer automatic trace', async () => {
    const rig = makeSync({ trace: { deferFirstRun: true } });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    const controller = new AbortController();
    const pending = rig.sync.validate(rig.sync.currentTarget()!, controller.signal);
    await settle();
    type(TYPED); saveChanges.fire(modelEditor().document);
    await settle();
    controller.abort();
    rig.releaseFirstRun();
    assert.equal((await pending).status, 'cancelled');
    await settle();
    assert.equal(rig.runs.length, 2);
    assert.equal(rig.sync.snapshotState()?.traced, true);
  });

  test('an automatic saved refresh consumes the typing debounce instead of invalidating its own launch', async () => {
    const rig = makeSync({ trace: { deferFirstRun: true }, debounceMs: 40 });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    type(TYPED); saveChanges.fire(modelEditor().document);
    await settle();
    assert.equal(rig.runs.length, 1);
    await new Promise(resolve => setTimeout(resolve, 60));
    rig.releaseFirstRun();
    await settle();
    assert.equal(rig.sync.snapshotState()?.traced, true);
    assert.equal(rig.fetches.filter(fetch => !fetch.body.trace_result).length, 2,
      'initial open and the saved refresh are the only source projections');
  });

  test('manual validation runs a saved target with traceOnSave off and merges the trace', async () => {
    const rig = makeSync({ trace: { onSave: false } });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    const result = await rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal);
    assert.equal(result.status, 'passed');
    assert.equal(rig.runs.length, 1);
    assert.equal(rig.envelopes.length, 1);
    assert.equal(rig.sync.snapshotState()?.traced, true);
    assert.equal(editorState.appliedEdits.length, 0, 'validation never writes source');
  });

  test('traceTier auto: first run real, a large model then runs fake, a fake-tensor fallback returns it to real', async () => {
    const withParameters = (parameterCount: number, extra: Partial<TraceOutcome> = {}): TraceOutcome => ({ ok: true, exitCode: 0, ...extra,
      envelope: { ...ENVELOPE, result: { ...(ENVELOPE.result as object), diagnostics: { semantic_capture: { rootInvocationId: 'invocation:0',
        invocations: [{ id: 'invocation:0', parameterCount }] } } } } } as TraceOutcome);
    let respond = (_request: TraceRequest) => withParameters(300_000_000);
    const rig = makeSync({ trace: { onSave: false, tier: 'auto', respond: request => respond(request) } });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    const validate = () => rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal);
    await validate();
    await validate();
    respond = () => withParameters(300_000_000, { fakeTensorFallback: { reason: 'data_dependent', errorType: 'E', operation: '', phase: 'forward' } });
    await validate();
    await validate();
    // 1: unknown size -> real. 2: learned 300M parameters -> fake. 3: fake again; it falls back to
    // real (the worker re-runs it). 4: remembered as needing values -> real directly.
    assert.deepEqual(rig.runs.map(run => run.tier), ['real', 'fake', 'fake', 'real']);
    const small = makeSync({ trace: { onSave: false, tier: 'auto', respond: () => withParameters(1_000) } });
    small.sync.setTarget(modelUri(), anchor());
    await settle();
    await small.sync.validate(small.sync.currentTarget()!, new AbortController().signal);
    await small.sync.validate(small.sync.currentTarget()!, new AbortController().signal);
    assert.deepEqual(small.runs.map(run => run.tier), ['real', 'real'], 'a small model stays real');
  });

  test('a reused identical result is projected like a fresh run and labelled; no worker run is recorded', async () => {
    const plan = { source: { id: 'source:original' } } as unknown as ResolvedExecutionPlan;
    const capturedAt = new Date(2026, 9, 25, 14, 2).getTime();
    const rig = makeSync({ trace: { onSave: false, outcome: { ok: true, envelope: ENVELOPE, exitCode: 0, executionCurrent: true,
      reused: { attemptId: 'original', captureId: 'capture:original', plan, planDigest: 'a'.repeat(64), capturedAt } } } });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    const execution = executionSpecSchema.parse({ inputs: { args: [{ shape: [1, 8], dtype: 'float32', fill: 'ones' }] } });
    const result = await rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal,
      undefined, undefined, undefined, undefined, execution, undefined, undefined, undefined, undefined, undefined, 'allow');
    assert.equal(result.status, 'passed', result.detail);
    assert.deepEqual(result.reused, { capturedAt });
    assert.match(result.detail, /^Reused identical result from [^(]*:02[^(]* \(model not run again\)\. Validation passed/);
    assert.equal(rig.runs[0].reuse, 'allow');
    assert.ok(rig.fetches.some(fetch => fetch.body.trace_result !== undefined), 'the reused envelope is projected');
    assert.equal(rig.sync.snapshotState()?.traced, true);
    assert.deepEqual(rig.envelopes.map(entry => entry.reused), [true], 'the runs list is told this was not a new run');
    // Run fresh reaches the worker as bypass.
    await rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal,
      undefined, undefined, undefined, undefined, execution, undefined, undefined, undefined, undefined, undefined, 'bypass');
    assert.equal(rig.runs[1].reuse, 'bypass');
  });

  test('a bounded profile can finish with measured evidence after both former outer deadlines', async () => {
    const rig = makeSync({ trace: { deferFirstRun: true } });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    mock.timers.enable();
    try {
      let measured = false;
      const pending = rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal,
        undefined, undefined, () => { measured = true; }, undefined, undefined, undefined, 300_000);
      for (let i = 0; i < 30; i++) await Promise.resolve();
      assert.equal(rig.runs[0]?.timeoutMs, 300_000);
      mock.timers.tick(240_000);
      assert.equal(measured, false, 'time alone is never execution evidence');
      rig.releaseFirstRun();
      assert.equal((await pending).status, 'passed');
      assert.equal(measured, true);
      assert.equal(rig.sync.snapshotState()?.traced, true);
      assert.equal(rig.envelopes.length, 1);
      mock.timers.tick(100_000);
      assert.equal(rig.sync.tracePending(), false, 'completed requests leave no deadline behind');
    } finally { mock.timers.reset(); }
  });

  test('a worker deadline is a failed timeout without a measured graph or receipt', async () => {
    const rig = makeSync({ trace: { outcome: { ok: false, timedOut: true,
      reason: 'The trace timed out.', detail: 'private-source-or-secret' } } });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    let measured = false;
    const result = await rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal,
      undefined, undefined, () => { measured = true; });
    assert.equal(rig.runs[0].timeoutMs, 180_000, 'automatic forward validation remains bounded');
    assert.equal(result.status, 'failed');
    assert.match(result.detail, /timed out/i);
    assert.ok(!result.detail.includes('private-source-or-secret'));
    assert.equal(measured, false);
    assert.equal(rig.sync.snapshotState()?.traced, false);
    assert.equal(rig.envelopes.length, 0);
  });

  test('automatic preparation survives the request lifecycle and a setup change during projection leaves stale evidence', async () => {
    let checked = false, measured = false;
    const rig = makeSync({ trace: { outcome: { ok: true, envelope: ENVELOPE, exitCode: 0,
      executionCurrent: true, recheckExecution: async () => { checked = true; return false; } } } });
    rig.sync.setTarget(modelUri(), anchor()); await settle();
    const result = await rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal,
      undefined, undefined, () => { measured = true; });
    assert.equal(rig.runs[0].workspace, ROOT, 'automatic preparation is bound to the host workspace');
    assert.equal(checked, true);
    assert.equal(result.status, 'refused');
    assert.match(result.detail, /configuration.*environment changed/);
    assert.equal(rig.sync.snapshotState()?.syncState, 'stale');
    assert.equal(measured, false);
    assert.equal(rig.envelopes.length, 0);
  });

  test('editing a runtime import absent from the AST dependencies invalidates its measured graph', async () => {
    const envelope = structuredClone(ENVELOPE);
    (envelope.result as any).diagnostics = { automatic_capture: { importedSources: [{
      file: UNRELATED_REL, digest: createHash('sha256').update(BLOCK_SOURCE).digest('hex'),
      realPath: UNRELATED, bytes: Buffer.byteLength(BLOCK_SOURCE),
    }] } };
    openStubEditor(UNRELATED, BLOCK_SOURCE);
    const diskSources = { [UNRELATED_REL]: BLOCK_SOURCE };
    const rig = makeSync({ diskSources, trace: { onSave: false, outcome: { ok: true, envelope, exitCode: 0 } } });
    rig.sync.setTarget(modelUri(), anchor()); await settle();
    assert.equal((await rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal)).status, 'passed');
    assert.equal(rig.sync.snapshotState()?.traced, true);
    assert.equal(await rig.sync.sourcesCurrent(), true, 'unchanged captures remain reusable');
    assert.equal(rig.sync.sourceEditContext(rig.sync.currentTarget()!)?.dependencies.get(UNRELATED_REL),
      createHash('sha256').update(BLOCK_SOURCE).digest('hex'), 'verified runtime imports must be available to agent source reads and coordinated proposals');
    diskSources[UNRELATED_REL] = BLOCK_EDITED;
    assert.equal(await rig.sync.sourcesCurrent(), false, 'external edits cannot hide behind an unchanged editor buffer');
    diskSources[UNRELATED_REL] = BLOCK_SOURCE;
    const calls = rig.fetches.length;
    editFile(UNRELATED, BLOCK_EDITED); await settle();
    assert.equal(rig.fetches.length, calls + 1, 'runtime imports must participate in source synchronization');
    assert.equal(rig.sync.snapshotState()?.traced, false, 'previous measurements cannot validate a changed imported layer');
    assert.equal(rig.sync.sourceEditContext(rig.sync.currentTarget()!)?.dependencies.has(UNRELATED_REL), false,
      'a new source-only projection cannot carry forward old runtime source authority');
    assert.equal(rig.runs.length, 1, 'typing does not execute the model');
    assert.equal(await rig.sync.sourcesCurrent(), false, 'a new AST projection cannot make the old capture current');
  });

  test('reuse checks detect saved dependency edits even without an editor change event', async () => {
    const diskSources = { [BLOCKS_REL]: BLOCK_SOURCE };
    const rig = makeSync({ diskSources });
    openStubEditor(BLOCKS, BLOCK_SOURCE);
    rig.sync.setTarget(modelUri(), anchor()); await settle();
    assert.equal(await rig.sync.sourcesCurrent(), true);
    editorState.visibleTextEditors.splice(editorState.visibleTextEditors.findIndex(editor => editor.document.uri.fsPath === BLOCKS), 1);
    editorState.documents.delete(BLOCKS);
    diskSources[BLOCKS_REL] = BLOCK_EDITED;
    assert.equal(await rig.sync.sourcesCurrent(), false);
  });

  test('an imported helper changing during projection cannot leave the published measurements current', async () => {
    const files = require('../../executionFiles') as typeof import('../../executionFiles');
    const original = files.verifyImportedSources;
    let checks = 0;
    files.verifyImportedSources = async () => {
      if (++checks === 2) throw new files.ExecutionFileError('Imported Python source changed during execution. Save and rerun.');
    };
    const envelope = structuredClone(ENVELOPE);
    (envelope.result as any).diagnostics = { automatic_capture: { importedSources: [] } };
    const rig = makeSync({ trace: { outcome: { ok: true, envelope, exitCode: 0, executionCurrent: true } } });
    try {
      rig.sync.setTarget(modelUri(), anchor()); await settle();
      const result = await rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal);
      assert.equal(checks, 2);
      assert.equal(result.status, 'refused');
      assert.equal(rig.sync.snapshotState()?.syncState, 'stale');
      assert.equal(rig.envelopes.length, 0);
    } finally { files.verifyImportedSources = original; }
  });

  for (const stop of ['cancel', 'retarget', 'edit'] as const) {
    test(`${stop} after the old cutoff still rejects a late long-profile measurement`, async () => {
      const rig = makeSync({ trace: { deferFirstRun: true } });
      rig.sync.setTarget(modelUri(), anchor());
      await settle();
      const controller = new AbortController();
      mock.timers.enable();
      try {
        let measured = false;
        const pending = rig.sync.validate(rig.sync.currentTarget()!, controller.signal,
          undefined, undefined, () => { measured = true; }, undefined, undefined, undefined, 300_000);
        for (let i = 0; i < 30; i++) await Promise.resolve();
        assert.equal(rig.runs.length, 1);
        mock.timers.tick(240_000);
        if (stop === 'cancel') controller.abort();
        else if (stop === 'retarget') rig.sync.setTarget(modelUri(), { ...anchor(), cls: 'OtherModel' });
        else type(TYPED);
        rig.releaseFirstRun();
        assert.equal((await pending).status, 'cancelled');
        assert.equal(measured, false);
        assert.equal(rig.envelopes.length, 0);
        assert.ok(!rig.published.some(message => message.type === 'projection' && message.traced));
      } finally { rig.sync.dispose(); mock.timers.reset(); }
    });
  }

  test('MCP typed validation reaches the same worker and measured projection path', async () => {
    const rig = makeSync({ trace: { onSave: false } });
    rig.sync.setTarget(modelUri(), anchor()); await settle();
    const inputs = [{ shape: [1, 8], dtype: 'float32' as const, fill: 'ones' as const }];
    const phases: string[] = [];
    const result = await rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal, inputs, phase => phases.push(phase));
    assert.equal(result.status, 'passed'); assert.deepEqual(rig.runs[0].tensorInputs, inputs);
    assert.equal(rig.runs[0].input, undefined); assert.deepEqual(phases, ['tracing', 'projecting']);
  });


  test('explicit execution uses the selected workspace and captures worker evidence before projection', async () => {
    const rig = makeSync({ trace: { onSave: false } });
    rig.sync.setTarget(modelUri(), anchor()); await settle();
    const execution = executionSpecSchema.parse({ inputs: { kwargs: { x: { shape: [1, 8], dtype: 'float32', fill: 'ones' } } }, mode: 'train' });
    const files = require('../../executionFiles') as typeof import('../../executionFiles');
    const original = files.verifyImportedSources;
    files.verifyImportedSources = async () => {};
    const captured: TraceOutcome[] = [];
    let result;
    try { result = await rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal,
      undefined, undefined, undefined, outcome => captured.push(outcome), execution);
    } finally { files.verifyImportedSources = original; }
    assert.equal(result.status, 'passed'); assert.equal(captured.length, 1);
    assert.deepEqual(rig.runs[0].execution, execution);
    assert.equal(rig.runs[0].workspace, ROOT); assert.equal(rig.runs[0].tier, 'real');
    assert.equal(rig.runs[0].tensorInputs, undefined); assert.equal(rig.runs[0].input, undefined);
  });

  test('reviewed CPU choice uses an inline plan while retaining saved identity and interpreter checks', async () => {
    const rig = makeSync({ trace: { onSave: false } });
    rig.sync.setTarget(modelUri(), anchor()); await settle();
    const execution = executionSpecSchema.parse({ device: 'cuda:0', inputs: { args: [] } });
    const files = require('../../executionFiles') as typeof import('../../executionFiles');
    const original = files.verifyImportedSources;
    files.verifyImportedSources = async () => {};
    let checks = 0;
    const guard = { spec: { interpreter: { executable: '/selected/python' } } as never, current: async () => { checks++; return true; } };
    try {
      const result = await rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal,
        undefined, undefined, undefined, undefined, execution, undefined, undefined, guard, undefined, 'cpu');
      assert.equal(result.status, 'passed'); assert.ok(checks >= 3, 'device selection must not bypass saved setup freshness');
      assert.equal(rig.runs[0].execution!.device, 'cpu'); assert.equal(execution.device, 'cuda:0');
      assert.equal(rig.runs[0].expectedInterpreter, '/selected/python');
      assert.equal(rig.runs[0].profileSpec, undefined, 'an override must not claim to be the unchanged saved profile');
    } finally { files.verifyImportedSources = original; }
  });

  for (const phase of ['before launch', 'after capture', 'after merge']) {
    test(`saved profile changes ${phase} cannot certify the graph`, async () => {
      const rig = makeSync({ trace: { onSave: false } });
      rig.sync.setTarget(modelUri(), anchor()); await settle();
      const execution = executionSpecSchema.parse({ inputs: { args: [] } });
      const files = require('../../executionFiles') as typeof import('../../executionFiles');
      const original = files.verifyImportedSources;
      files.verifyImportedSources = async () => {};
      let checks = 0, measured = false;
      const failAt = phase === 'before launch' ? 2 : phase === 'after capture' ? 3 : 4;
      const guard = { spec: { interpreter: { executable: '/selected/python' } } as never,
        current: async () => ++checks < failAt };
      try {
        const result = await rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal,
          undefined, undefined, () => { measured = true; }, undefined, execution, undefined, undefined, guard);
        assert.equal(result.status, 'refused');
        assert.equal(measured, false);
        assert.equal(rig.envelopes.length, 0);
        assert.equal(rig.runs[0].expectedInterpreter, '/selected/python');
        if (phase === 'after merge') {
          assert.equal(rig.sync.snapshotState()?.syncState, 'stale');
          assert.equal(rig.sync.snapshotState()?.greyed, true);
        } else assert.ok(!rig.published.some(message => message.type === 'projection' && message.traced));
      } finally { files.verifyImportedSources = original; rig.sync.dispose(); }
    });
  }

  test('explicit execution refuses dirty workspace helpers before running user code', async () => {
    const rig = makeSync({ trace: { onSave: false } });
    rig.sync.setTarget(modelUri(), anchor()); await settle();
    const helper = openStubEditor(UNRELATED, 'WIDTH = 4');
    Object.defineProperty(helper.document, 'isDirty', { value: true });
    const execution = executionSpecSchema.parse({ inputs: { args: [] } });
    const result = await rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal,
      undefined, undefined, undefined, undefined, execution);
    assert.equal(result.status, 'refused'); assert.match(result.detail, /Save changed Python/);
    assert.equal(rig.runs.length, 0);
  });

  test('manual validation refuses an unsaved root or dependency instead of running older disk bytes', async () => {
    for (const file of [REL, BLOCKS_REL]) {
      resetEditors();
      openStubEditor(BLOCKS, BLOCK_SOURCE);
      const rig = makeSync({ trace: { onSave: false }, diskSources: { [file]: '# older saved source' } });
      rig.sync.setTarget(modelUri(), anchor());
      await settle();
      const result = await rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal);
      assert.equal(result.status, 'refused');
      assert.match(result.detail, /Save/);
      assert.equal(rig.runs.length, 0);
    }
  });

  test('manual validation refuses stale identities and untrusted workspaces', async () => {
    const rig = makeSync({ trace: { trusted: false } });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    assert.equal((await rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal)).status, 'refused');
    assert.equal((await rig.sync.validate({ ...rig.sync.currentTarget()!, sourceDigest: '0'.repeat(64) }, new AbortController().signal)).status, 'refused');
    assert.equal(rig.runs.length, 0);
  });

  test('manual failures and failed trace merges cannot claim validation passed', async () => {
    for (const opts of [
      { trace: { fail: true } },
      { trace: {}, rejectProject: true },
    ]) {
      const rig = makeSync(opts);
      rig.sync.setTarget(modelUri(), anchor());
      await settle();
      const result = await rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal);
      assert.notEqual(result.status, 'passed');
      assert.ok(!result.detail.includes('private-source-or-secret'));
      assert.equal(rig.envelopes.length, 0);
    }
  });

  test('the real validation loop distinguishes retained capture from a failed projection attempt', async () => {
    const summaries = new ValidationSummaryStore(() => {});
    let reject = false;
    const rig = makeSync({ trace: {}, summaries, mutateResponse: response => {
      if (reject) response.document = null;
    } });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    const run = async (id: string) => {
      const target = rig.sync.currentTarget()!;
      const summary = summaries.begin(id, target);
      const outcome = await rig.sync.validate(target, new AbortController().signal, undefined, undefined,
        undefined, undefined, undefined, undefined, undefined, undefined, summary.attempt);
      summary(outcome.status, outcome.detail);
      assert.equal(rig.runs.at(-1)!.attemptId, id, 'the worker receives the same host attempt identity');
      return summaries.ledger.state(target);
    };
    const first = await run('first');
    assert.equal(first.latestAttempt!.status, 'succeeded');
    assert.ok(first.displayedCaptureId);
    reject = true;
    const second = await run('second');
    assert.equal(second.latestAttempt!.status, 'failed');
    assert.equal(second.latestAttempt!.phase, 'projection');
    assert.notEqual(second.lastPartialCaptureId, first.lastPartialCaptureId);
    assert.equal(second.displayedCaptureId, first.displayedCaptureId);
  });

  test('a worker cancellation remains intentional even when the caller did not abort', async () => {
    const rig = makeSync({ trace: { outcome: { ok: false, cancelled: true, reason: 'superseded', detail: '' } } });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    assert.equal((await rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal)).status, 'cancelled');
  });

  test('manual forward failure publishes partial shapes and a safe mapped error without claiming success', async () => {
    const rig = makeSync({ trace: { outcome: { ok: false, reason: 'trace failed', detail: '', exitCode: 4,
      envelope: { ...ENVELOPE, error: { type: 'RuntimeError', message: 'private-source-or-secret', frames: [] } } } },
      mutateResponse: (response, body) => {
        if (!body.trace_result) return;
        response.document = { semantic: { nodes: [{ node_id: 'Net.fc', params: { _in_shape: '1×4', _shape_provenance: 'unreached' } }] }, view: {} };
        response.trace_error = { message: 'private-source-or-secret', node_id: 'Net.fc', file: '/ws/models/net.py', line: 5 };
      } });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    const result = await rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal);
    assert.equal(result.status, 'failed');
    assert.equal(rig.envelopes.length, 1, 'a partial forward is still a recorded run');
    const snapshot = rig.sync.snapshotState()!;
    assert.equal(snapshot.traced, true);
    const projected = snapshot.projection as ProjectionPayload;
    assert.equal(projected.traceError?.nodeId, 'Net.fc');
    assert.equal(projected.traceError?.file, REL);
    assert.ok(!JSON.stringify(rig.published).includes('private-source-or-secret'));
    assert.ok(!result.detail.includes('private-source-or-secret'));
    assert.equal((projected.document as any).semantic.nodes[0].params._in_shape, '1×4');
  });

  test('cancelling while a manual trace projection is merging cannot publish it', async () => {
    const controller = new AbortController();
    const rig = makeSync({ trace: {}, mutateResponse: (_response, body) => {
      if (body.trace_result) controller.abort();
    } });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    const result = await rig.sync.validate(rig.sync.currentTarget()!, controller.signal);
    assert.equal(result.status, 'cancelled');
    assert.equal(rig.sync.snapshotState()?.traced, false);
    assert.equal(rig.envelopes.length, 0);
  });

  test('manual validation cancels on typing and never publishes the old trace', async () => {
    const rig = makeSync({ trace: { onSave: false, deferFirstRun: true } });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    const pending = rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal);
    await settle();
    type(TYPED);
    rig.releaseFirstRun();
    assert.equal((await pending).status, 'cancelled');
    await settle();
    assert.equal(rig.envelopes.length, 0);
    assert.equal(rig.sync.snapshotState()?.traced, false);
  });

  test('manual cancellation, retargeting and repeated clicks never start overlapping traces', async () => {
    const rig = makeSync({ trace: { deferFirstRun: true } });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    const controller = new AbortController();
    const pending = rig.sync.validate(rig.sync.currentTarget()!, controller.signal);
    await settle();
    assert.equal((await rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal)).status, 'refused');
    saveChanges.fire(modelEditor().document);
    await settle();
    assert.equal(rig.runs.length, 1, 'save must not cancel or overlap the explicit run');
    controller.abort();
    rig.sync.setTarget(modelUri(), { ...anchor(), cls: 'OtherModel' });
    rig.releaseFirstRun();
    assert.equal((await pending).status, 'cancelled');
    assert.equal(rig.envelopes.length, 0);
  });

  test('saving an edit that cancelled manual validation still honors trace-on-save', async () => {
    const rig = makeSync({ trace: { onSave: true, deferFirstRun: true } });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    const pending = rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal);
    await settle();
    type(TYPED);
    saveChanges.fire(modelEditor().document);
    rig.releaseFirstRun();
    assert.equal((await pending).status, 'cancelled');
    await settle();
    assert.equal(rig.runs.length, 2, 'the saved replacement source must still get its opted-in trace');
    assert.equal(rig.sync.snapshotState()?.traced, true);
  });

  test('manual validation detects an external disk change during execution', async () => {
    const diskSources = { [REL]: SAVED };
    const rig = makeSync({ trace: { deferFirstRun: true }, diskSources });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    const pending = rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal);
    await settle();
    diskSources[REL] = '# changed on disk';
    rig.releaseFirstRun();
    assert.equal((await pending).status, 'refused');
    assert.equal(rig.sync.snapshotState()?.traced, false);
  });

  test('opening projects the buffer at once and labels it source-only', async () => {
    const rig = makeSync({});
    rig.sync.setTarget(modelUri(), anchor());
    await settle();

    assert.deepEqual(
      rig.published.map((m) => m.type),
      ['projection', 'syncState'],
    );
    const projection = rig.published[0] as {
      target: LocalBrowserTargetIdentity;
      projection: { traced: boolean; greyed: boolean; sourceDigest: string };
      traced: boolean;
      greyed: boolean;
    };
    assert.equal(projection.target.backendRevision, 1);
    assert.equal(projection.target.cls, 'Net');
    assert.equal(projection.target.file, REL);
    assert.equal(projection.traced, false);
    assert.equal(projection.greyed, false);
    assert.equal(
      projection.projection.sourceDigest,
      createHash('sha256').update(SAVED, 'utf8').digest('hex'),
      'the graph is the digest of the buffer that was POSTed',
    );
    assert.deepEqual(rig.published[1], {
      type: 'syncState',
      target: projection.target,
      state: 'source-only',
    });
    assert.equal(rig.fetches.length, 1);
    assert.ok(
      isProjection(rig.fetches[0].url),
      'the projection is the multi-file endpoint, both tiers',
    );
    assert.equal(
      rootSourceOf(rig.fetches[0].body),
      SAVED,
      'the backend reads the buffer, not the disk',
    );
    assert.equal(rig.fetches[0].body.trace_result, undefined);
  });

  test('opening and retargeting never execute a model even with trace-on-save enabled', async () => {
    const rig = makeSync({ trace: {} });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    rig.sync.setTarget(modelUri(), { ...anchor(), cls: 'OtherModel' });
    await settle();
    assert.equal(rig.runs.length, 0, 'Open as Graph is a static, non-executing operation');
    assert.equal(rig.envelopes.length, 0, 'opening must not record a workspace run');
    assert.equal(rig.sync.tracePending(), false);
    assert.ok(rig.published.filter(message => message.type === 'syncState').every(message => message.state === 'source-only'));
  });

  test('retarget cancels the old target’s queued refresh and save-triggered execution', async () => {
    const rig = makeSync({ trace: {} });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    type(TYPED);
    saveChanges.fire(modelEditor().document);
    rig.sync.setTarget(modelUri(), { ...anchor(), cls: 'OtherModel' });
    await settle();
    assert.equal(rig.runs.length, 0, 'an old save must not execute the newly opened model');
    assert.equal(rig.fetches.length, 2, 'the old refresh timer must not reproject the new target twice');
    assert.equal(rig.sync.tracePending(), false);
  });

  test('an opted-in save measures the trusted workspace after the coalescing window', async () => {
    const rig = makeSync({ trace: {} });
    rig.sync.setTarget(modelUri(), anchor());
    saveChanges.fire(modelEditor().document);
    await settle();

    assert.deepEqual(
      rig.published.map((m) => m.type),
      ['projection', 'syncState', 'projection', 'syncState'],
    );
    assert.equal((rig.published[1] as { state: string }).state, 'source-only');
    assert.equal((rig.published[3] as { state: string }).state, 'runtime-traced');
    assert.equal((rig.published[2] as { traced: boolean }).traced, true);
    assert.equal(rig.runs.length, 1, 'the model ran once');
    assert.equal(rig.runs[0].cls, 'Net');
    assert.deepEqual(rig.envelopes, [{ folder: ROOT, file: REL, cls: 'Net' }]);
    const project = rig.fetches.find(
      (fetch) => isProjection(fetch.url) && fetch.body.trace_result !== undefined,
    );
    assert.ok(project, 'the merged tier is the projection carrying the trace');
    assert.ok(project.body.trace_result, 'the worker envelope rode the request');
  });

  test('a measured save carries observed shapes to forward calls in their own files', async () => {
    openStubEditor(BLOCKS, BLOCK_SOURCE);
    const rig = makeSync({
      trace: {},
      spans: {
        ...SPANS,
        // An executed call in an imported file belongs beside that call.
        'blocks.fc': { file: BLOCKS_REL, start: [3, 8], end: [3, 40], kind: 'call' },
        guessed: { file: REL, start: [6, 8], end: [6, 8], kind: 'decl' },
      },
      mutateResponse: (response, body) => {
        if (!body.trace_result) {
          return;
        }
        response.document = {
          view: {},
          semantic: {
            nodes: [
              { node_id: 'net', params: { _out_shape: '2x8', _shape_provenance: 'observed' } },
              // Only the call gets a hint; the declaration above does not.
              { node_id: 'fwd', params: { _out_shape: '2x8', _shape_provenance: 'observed' } },
              { node_id: 'blocks.fc', params: { _out_shape: '2x4', _shape_provenance: 'observed' } },
              // Read off the source, not measured. Shown in the same grey it would be a
              // lie in the same font.
              { node_id: 'guessed', params: { _out_shape: '2x99', _shape_provenance: 'static' } },
            ],
          },
        };
      },
    });
    let refreshes = 0;
    rig.sync.onDidChangeShapes(() => {
      refreshes += 1;
    });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    assert.deepEqual(
      rig.sync.shapesFor(shapeUri(MODEL)),
      [],
      'the source tier measured nothing, so it must claim nothing',
    );

    saveChanges.fire(modelEditor().document);
    await settle();

    assert.deepEqual(rig.sync.shapesFor(shapeUri(MODEL)), [
      { nodeId: 'fwd', line: 6, column: 8, shape: '2x8', provenance: 'observed' },
    ]);
    assert.deepEqual(rig.sync.shapesFor(shapeUri(BLOCKS)), [
      { nodeId: 'blocks.fc', line: 3, column: 40, shape: '2x4', provenance: 'observed' },
    ]);
    assert.deepEqual(
      rig.sync.shapesFor(shapeUri(UNRELATED)),
      [],
      'a file this graph never reached has no measured shape to show',
    );
    // Without this the provider is never asked again and the editor keeps showing the
    // source tier's nothing.
    assert.ok(refreshes > 0, 'the inlay hints were never told the shapes changed');
  });

  test('tracePending is armed on save and terminal once the trace lands', async () => {
    const sourceOnly = makeSync({});
    sourceOnly.sync.setTarget(modelUri(), anchor());
    assert.equal(
      sourceOnly.sync.tracePending(),
      false,
      'no trace support means nothing can ever be pending',
    );
    await settle();
    assert.equal(sourceOnly.sync.tracePending(), false);

    const rig = makeSync({ trace: {} });
    assert.equal(rig.sync.tracePending(), false, 'nothing to trace before the anchor');
    rig.sync.setTarget(modelUri(), anchor());
    saveChanges.fire(modelEditor().document);
    assert.equal(rig.sync.tracePending(), true, 'the save-coalesced trace is armed on save');
    await settle();
    assert.equal(rig.sync.tracePending(), false, 'the landed trace is terminal');
    assert.equal(rig.runs.length, 1, 'the model ran once');
  });

  test('tracePending stays set while the trace is running and clears when it lands', async () => {
    const rig = makeSync({ trace: { deferFirstRun: true } });
    rig.sync.setTarget(modelUri(), anchor());
    saveChanges.fire(modelEditor().document);
    await settle();
    assert.equal(rig.runs.length, 1, 'the worker is running the trace');
    assert.equal(rig.sync.tracePending(), true, 'a trace in flight is still pending');
    rig.releaseFirstRun();
    await settle();
    assert.equal(rig.sync.tracePending(), false);
  });

  test('tracePending clears even when the trace cannot run', async () => {
    const rig = makeSync({ trace: { fail: true } });
    rig.sync.setTarget(modelUri(), anchor());
    saveChanges.fire(modelEditor().document);
    assert.equal(rig.sync.tracePending(), true, 'a failing trace is still armed first');
    await settle();
    assert.equal(rig.sync.tracePending(), false, 'a failed trace is equally terminal');
    const states = rig.published
      .filter((message) => message.type === 'syncState')
      .map((message) => message.state);
    assert.ok(
      states.length > 0 && states.every((state) => state === 'source-only'),
      'the page stays honestly source-only',
    );
  });

  test('typing re-projects from the new buffer, runs nothing, and writes nothing', async () => {
    const rig = makeSync({ trace: {} });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    const afterOpen = rig.published.length;

    type(TYPED);
    await settle();

    assert.deepEqual(
      rig.published.slice(afterOpen).map((m) => m.type),
      ['projection', 'syncState'],
    );
    assert.equal(rig.runs.length, 0, 'neither opening nor typing may run the model');
    const latestParse = rig.fetches[rig.fetches.length - 1];
    assert.ok(isProjection(latestParse.url));
    assert.equal(rootSourceOf(latestParse.body), TYPED);
    assert.equal((rig.published[afterOpen + 1] as { state: string }).state, 'source-only');
    // The whole point of the slice: looking at a graph never edits the file.
    assert.equal(
      editorState.appliedEdits.length,
      0,
      'viewing a graph must never write to the workspace',
    );
  });

  test('a dirty dependency refreshes from its live buffer; an unrelated edit does nothing', async () => {
    openStubEditor(BLOCKS, BLOCK_SOURCE);
    openStubEditor(UNRELATED, 'value = 1\n');
    const rig = makeSync({});
    rig.sync.setTarget(modelUri(), anchor());
    await settle();

    const initial = rig.fetches[0].body as MultifileBody;
    const initialBlock = initial.sources.find((entry) => entry.file === BLOCKS_REL);
    assert.ok(initialBlock);
    assert.equal(initialBlock.source, BLOCK_SOURCE);
    assert.equal(initialBlock.document_version, 1);

    editFile(BLOCKS, BLOCK_EDITED);
    await settle();

    assert.equal(rig.fetches.length, 2, 'the resolved dependency edit reprojections once');
    const refreshed = rig.fetches[1].body as MultifileBody;
    const dirtyBlock = refreshed.sources.find((entry) => entry.file === BLOCKS_REL);
    assert.ok(dirtyBlock);
    assert.equal(dirtyBlock.source, BLOCK_EDITED, 'the unsaved editor text wins over disk');
    assert.equal(dirtyBlock.digest, createHash('sha256').update(BLOCK_EDITED).digest('hex'));
    assert.equal(dirtyBlock.document_version, 2);

    editFile(UNRELATED, 'value = 2\n');
    await settle();
    assert.equal(rig.fetches.length, 2, 'an unconsulted file edit must not pay a projection');
  });

  test('a newer edit aborts the obsolete HTTP projection before its late answer', async () => {
    const rig = makeSync({ deferFirstParse: true });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    assert.equal(rig.fetches.length, 1);
    assert.equal(rig.fetches[0].signal?.aborted, false);

    type(TYPED);
    await settle();

    assert.equal(rig.fetches[0].signal?.aborted, true, 'the old request is physically aborted');
    assert.equal(rig.fetches.length, 2, 'the current buffer has its own request');
    rig.releaseFirstParse();
    await settle();
    const projections = rig.published.filter((message) => message.type === 'projection');
    assert.equal(projections.length, 1, 'the aborted answer never reaches the page');
    assert.equal((projections[0].target as LocalBrowserTargetIdentity).sourceDigest,
      createHash('sha256').update(TYPED).digest('hex'));
  });

  test('a buffer that stops parsing greys the last graph and names the line', async () => {
    const rig = makeSync({ trace: {} });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();

    type(BROKEN);
    await settle();

    const tail = rig.published.slice(-2);
    assert.deepEqual(tail.map((m) => m.type), ['staleState', 'syncState']);
    const stale = tail[0] as {
      lastValidProjection: { greyed: boolean; sourceDigest: string };
      reason: string;
      target: unknown;
    };
    assert.equal(stale.lastValidProjection.greyed, true, 'the kept graph is visibly dimmed');
    assert.equal(
      stale.lastValidProjection.sourceDigest,
      createHash('sha256').update(SAVED, 'utf8').digest('hex'),
      'the kept graph is the last one that parsed — the digest of the OLD text',
    );
    assert.match(stale.reason, /models\/net\.py:4/);
    assert.deepEqual(tail[1], {
      type: 'syncState',
      target: stale.target,
      state: 'stale',
      reason: stale.reason,
    });

    // A view that reconnects now restores the same honest state.
    const snapshot = rig.sync.snapshotState();
    assert.ok(snapshot && snapshot.syncState === 'stale');
    if (snapshot?.syncState === 'stale') {
      assert.equal(snapshot.reason, stale.reason);
      assert.equal(snapshot.greyed, true);
    }
  });

  test('fixing the buffer un-stales: a fresh projection lands and the label returns', async () => {
    const rig = makeSync({ trace: {} });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    type(BROKEN);
    await settle();
    assert.equal((rig.published[rig.published.length - 1] as { state: string }).state, 'stale');

    type(TYPED);
    await settle();

    const tail = rig.published.slice(-2);
    assert.deepEqual(tail.map((m) => m.type), ['projection', 'syncState']);
    assert.equal((tail[0] as { greyed: boolean }).greyed, false);
    assert.equal((tail[1] as { state: string }).state, 'source-only');
    const snapshot = rig.sync.snapshotState();
    assert.equal(snapshot?.syncState, 'source-only');
  });

  test('an untrusted workspace is the source tier and the model never runs', async () => {
    const rig = makeSync({ trace: { trusted: false } });
    rig.sync.setTarget(modelUri(), anchor());
    saveChanges.fire(modelEditor().document);
    await settle();

    assert.deepEqual(
      rig.published.map((m) => m.type),
      ['projection', 'syncState'],
    );
    assert.equal(rig.runs.length, 0, 'untrusted means no run, not a quiet one');
    assert.equal((rig.published[1] as { state: string }).state, 'source-only');
  });

  test('traceOnSave off keeps saves on the source tier too', async () => {
    const rig = makeSync({ trace: { onSave: false } });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    assert.equal(rig.runs.length, 0);

    saveChanges.fire(openStubEditor(MODEL, SAVED).document);
    await settle();
    assert.equal(rig.runs.length, 0, 'off means off, even on an explicit save');
  });

  test('a trace that cannot run keeps the source graph, honestly labelled', async () => {
    const rig = makeSync({ trace: { fail: true } });
    rig.sync.setTarget(modelUri(), anchor());
    saveChanges.fire(modelEditor().document);
    await settle();

    assert.deepEqual(
      rig.published.map((m) => m.type),
      ['projection', 'syncState'],
    );
    assert.equal(rig.runs.length, 1, 'honest fallback still names the attempt');
    assert.ok(
      !rig.fetches.some(
        (fetch) => isProjection(fetch.url) && fetch.body.trace_result !== undefined,
      ),
      'a trace with no envelope is never offered for merging',
    );
    assert.equal((rig.published[1] as { state: string }).state, 'source-only');
  });

  test('a trace the backend refuses keeps the graph visibly stale without claiming a merge', async () => {
    const rig = makeSync({ trace: {}, rejectProject: true });
    rig.sync.setTarget(modelUri(), anchor());
    saveChanges.fire(modelEditor().document);
    await settle();

    assert.ok(rig.runs.length === 1, 'the model ran');
    assert.ok(
      rig.fetches.some(
        (fetch) => isProjection(fetch.url) && fetch.body.trace_result !== undefined,
      ),
      'the envelope was offered',
    );
    assert.deepEqual(
      rig.published.map((m) => m.type),
      ['projection', 'syncState', 'staleState', 'syncState'],
      'the refusal publishes nothing that claims runtime truth',
    );
    assert.equal(rig.sync.snapshotState()?.syncState, 'stale');
    assert.equal(rig.sync.snapshotState()?.traced, false);
  });

  const failures: Array<{ name: string; response(): Promise<Response | undefined>; reason: RegExp }> = [
    { name: 'network', response: async () => { throw new Error('private-source-or-secret'); }, reason: /could not refresh/i },
    { name: 'server', response: async () => new Response('private-source-or-secret', { status: 500 }), reason: /HTTP 500/ },
    { name: 'invalid JSON', response: async () => new Response('private-source-or-secret'), reason: /invalid response/i },
    { name: 'malformed projection', response: async () => Response.json({ dependencies: [], document: null }), reason: /invalid response/i },
    { name: 'missing class', response: async () => Response.json({ detail: 'private-source-or-secret' }, { status: 404 }), reason: /class.*no longer/i },
    { name: 'unsupported source', response: async () => Response.json({ detail: 'private-source-or-secret' }, { status: 422 }), reason: /supported.*module/i },
    { name: 'syntax', response: async () => Response.json({ detail: { error: 'syntax_error', line: 4, message: 'private-source-or-secret' } }, { status: 400 }), reason: /models\/net.py:4.*does not parse/i },
  ];

  for (const failure of failures) {
    for (const retained of [false, true]) {
      test(`${failure.name} failure is actionable ${retained ? 'on a retained graph' : 'on first load'} and recovers`, async () => {
        let fail = !retained;
        const rig = makeSync({ trace: {}, fetchResponse: async () => fail ? failure.response() : undefined });
        rig.sync.setTarget(modelUri(), anchor());
        await settle();
        const before = rig.sync.snapshotState();
        if (retained) {
          fail = true;
          type(TYPED);
          await settle();
        }
        const snapshot = rig.sync.snapshotState();
        assert.equal(snapshot?.syncState, 'stale', 'reload must retain the failure, including before any graph exists');
        assert.ok(snapshot?.syncState === 'stale');
        assert.match(snapshot.reason, failure.reason);
        assert.match(snapshot.reason, /VS Code|Open as Graph/);
        assert.equal(snapshot.greyed, true);
        assert.equal(isLocalBrowserMessage({ type: 'snapshot', role: 'host', protocolVersion: 1,
          workspaceSessionId: 'test', messageId: 'snapshot', hostSequence: 1, state: snapshot }), true);
        assert.deepEqual(snapshot.projection, retained ? { ...(before!.projection as ProjectionPayload), greyed: true } : null);
        assert.equal(rig.sync.sourceEditContext(rig.sync.currentTarget()!), undefined);
        assert.equal((await rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal)).status, 'refused');
        assert.equal(rig.runs.length, 0);
        assert.ok(!JSON.stringify([rig.published, rig.logs]).includes('private-source-or-secret'));
        fail = false;
        type(TYPED);
        await settle();
        assert.equal(rig.sync.snapshotState()?.syncState, 'source-only');
        assert.equal(rig.sync.snapshotState()?.greyed, false);
        assert.ok(rig.sync.sourceEditContext(rig.sync.currentTarget()!));
      });
    }
  }

  test('malformed graph and identity responses cannot replace a valid projection', async () => {
    let mutate: ((response: Record<string, unknown>) => void) | undefined;
    const rig = makeSync({ mutateResponse: response => mutate?.(response) });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    const before = rig.sync.snapshotState()!.projection as ProjectionPayload;
    for (const invalid of [
      { document: null }, { document: { semantic: { nodes: null }, view: {} } },
      { document: { semantic: { nodes: [null] }, view: {} } }, { traced: true },
      { spanmap: null }, { revision: '2' }, { traced: 'false' }, { cls: 'Other' },
      { source_digest: 'wrong' }, { diagnostics: [null] }, { dependencies: [null] },
    ]) {
      mutate = response => Object.assign(response, invalid);
      type(TYPED);
      await settle();
      assert.equal(rig.sync.snapshotState()?.syncState, 'stale');
      assert.deepEqual(rig.sync.snapshotState()?.projection, { ...before, greyed: true });
    }
  });

  test('a deadline surfaces even when fetch ignores abort and its late success cannot recover the graph', async function () {
    this.timeout(15_000);
    let release!: (response: Response | undefined) => void;
    let fail = true;
    const rig = makeSync({ fetchResponse: async () => fail ? new Promise(resolve => { release = resolve; }) : undefined });
    rig.sync.setTarget(modelUri(), anchor());
    await new Promise(resolve => setTimeout(resolve, 10_050));
    assert.equal(rig.fetches[0].signal?.aborted, true);
    const snapshot = rig.sync.snapshotState();
    assert.ok(snapshot?.syncState === 'stale');
    assert.match(snapshot.reason, /timed out.*Open as Graph/);
    release(undefined);
    await settle();
    assert.deepEqual(rig.sync.snapshotState(), snapshot, 'late success cannot erase the timeout');
    fail = false;
    type(TYPED);
    await settle();
    assert.equal(rig.sync.snapshotState()?.syncState, 'source-only');
  });

  for (const supersede of ['edit', 'retarget', 'dispose'] as const) {
    test(`late network failure after ${supersede} cannot poison current state`, async () => {
      let reject!: (error: Error) => void;
      let first = true;
      const rig = makeSync({ fetchResponse: async () => {
        if (!first) return undefined;
        first = false;
        return new Promise((_resolve, rejectRequest) => { reject = rejectRequest; });
      } });
      rig.sync.setTarget(modelUri(), anchor());
      await settle();
      if (supersede === 'edit') type(TYPED);
      else if (supersede === 'retarget') rig.sync.setTarget(modelUri(), { ...anchor(), cls: 'Other' });
      else rig.sync.dispose();
      await settle();
      const messages = [...rig.published];
      reject(new Error('private-source-or-secret'));
      await settle();
      assert.equal(rig.fetches[0].signal?.aborted, true);
      assert.deepEqual(rig.published, messages);
      assert.ok(!rig.published.some(message => message.type === 'staleState' || message.state === 'stale'));
    });
  }

  test('a failed manual merge cannot leave the previous trace looking current and emits no measured receipt', async () => {
    let fail = false;
    const rig = makeSync({ trace: {}, spans: SPANS,
      mutateResponse: response => {
        response.document = { view: {}, semantic: { nodes: [{ node_id: 'fwd',
          params: { _out_shape: '2x8', _shape_provenance: 'observed' } }] } };
        response.diagnostics = [{ code: 'UNRESOLVED_DYNAMIC', severity: 'warning', file: REL,
          line: 4, import_text: '', detail: 'Dynamic module remains opaque.', node: 'net' }];
      },
      fetchResponse: async body => body.trace_result && fail
        ? new Response('private-source-or-secret', { status: 500 }) : undefined });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    assert.equal((await rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal)).status, 'passed');
    const before = rig.sync.snapshotState()!.projection as ProjectionPayload;
    assert.equal(rig.sync.shapesFor(modelUri()).length, 1);
    fail = true;
    let measured = false;
    const result = await rig.sync.validate(rig.sync.currentTarget()!, new AbortController().signal, undefined, undefined, () => { measured = true; });
    assert.notEqual(result.status, 'passed');
    assert.match(result.detail, /trace.*could not refresh/i);
    assert.equal(measured, false);
    assert.equal(rig.sync.snapshotState()?.syncState, 'stale');
    assert.deepEqual(rig.sync.snapshotState()?.projection, { ...before, greyed: true });
    assert.equal(rig.sync.shapesFor(modelUri()).length, 0);
  });

  test('a cancelled manual merge failure does not mark an unchanged graph stale', async () => {
    const controller = new AbortController();
    const rig = makeSync({ trace: {}, fetchResponse: async body => {
      if (!body.trace_result) return undefined;
      controller.abort();
      throw new Error('private-source-or-secret');
    } });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    const before = rig.sync.snapshotState();
    assert.equal((await rig.sync.validate(rig.sync.currentTarget()!, controller.signal)).status, 'cancelled');
    assert.deepEqual(rig.sync.snapshotState(), before);
  });

  test('a retarget discards the old class’s in-flight answer', async () => {
    const rig = makeSync({ deferFirstParse: true });
    rig.sync.setTarget(modelUri(), anchor());
    const gate = baseIdentity(1, SAVED);
    rig.sync.setTarget(modelUri(), { ...gate, cls: 'Gate' });
    rig.releaseFirstParse();
    await settle();

    assert.ok(
      !rig.published.some((m) => (m.target as { cls?: string } | undefined)?.cls === 'Net'),
      'the old class’s late answer must not land over the new target',
    );
    assert.ok(
      rig.published.some((m) => (m.target as { cls?: string } | undefined)?.cls === 'Gate'),
    );
  });

  test('the snapshot is undefined until a projection exists', () => {
    const rig = makeSync({ deferFirstParse: true });
    rig.sync.setTarget(modelUri(), anchor());

    assert.equal(rig.sync.snapshotState(), undefined);
  });

  test('the cursor enters a node once and quietly; leaving and re-entering speaks again', async () => {
    const rig = makeSync({ spans: SPANS });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    const live = liveTarget(rig);

    moveCursor(3); // a blank line above the class: no span, no message
    assert.deepEqual(focusPublishes(rig), []);

    moveCursor(4); // the class statement: only `net` covers it
    moveCursor(6); // inside forward: `fwd`, the innermost
    moveCursor(6); // a keystroke within the same span: quiet
    assert.deepEqual(
      focusPublishes(rig).map((m) => m.nodeId),
      ['net', 'fwd'],
    );

    moveCursor(3); // leaving span-land: silent
    moveCursor(6); // and back in: fires again
    assert.deepEqual(
      focusPublishes(rig).map((m) => m.nodeId),
      ['net', 'fwd', 'fwd'],
    );
    // Every focus names the identity of the graph the page is showing.
    for (const message of focusPublishes(rig)) {
      assert.deepEqual(message.target, live);
    }
  });

  test('a reveal the page asked for moves the cursor — and its own selection event cannot echo back', async () => {
    const rig = makeSync({ spans: SPANS });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    const live = liveTarget(rig);

    const result = await rig.sync.revealNode(live, 'fwd');

    assert.deepEqual(result, { ok: true });
    // `fwd` starts at span line 5, column 4: 0-based for the editor.
    const active = modelEditor().selection.active;
    assert.equal(active.line, 4);
    assert.equal(active.character, 4);

    // The selection event the reveal itself caused: no focusNode bounces back
    // at the page that asked.
    selectionChanges.fire({
      textEditor: modelEditor(),
      selections: [modelEditor().selection],
    });
    assert.deepEqual(focusPublishes(rig), []);
  });

  test('a reveal first makes room beside a Canvas that fills the window', async () => {
    const rig = makeSync({ spans: SPANS });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    // Where the cursor was when the side bar was restored: before the reveal moved it.
    const restored: number[] = [];
    const registration = commands.registerCommand('workbench.action.restoreAuxiliaryBar',
      () => { restored.push(modelEditor().selection.active.line); });
    try {
      assert.deepEqual(await rig.sync.revealNode(liveTarget(rig), 'fwd'), { ok: true });
    } finally { registration.dispose(); }
    assert.deepEqual(restored, [0]);
    assert.equal(modelEditor().selection.active.line, 4);
  });

  test('dependency navigation opens the real file and shared definitions focus every instance', async () => {
    const blockEditor = openStubEditor(BLOCKS, BLOCK_SOURCE);
    const rig = makeSync({ spans: DEPENDENCY_SPANS });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    const live = liveTarget(rig);

    assert.deepEqual(await rig.sync.revealNode(live, 'right.block'), { ok: true });
    assert.equal(blockEditor.selection.active.line, 1);

    blockEditor.selection = { active: { line: 0, character: 0 } };
    selectionChanges.fire({ textEditor: blockEditor, selections: [blockEditor.selection] });
    blockEditor.selection = { active: { line: 2, character: 0 } };
    selectionChanges.fire({ textEditor: blockEditor, selections: [blockEditor.selection] });
    const focus = rig.published.find((message) => message.type === 'focusNode') as
      | { nodeId: string; nodeIds: string[] }
      | undefined;
    assert.ok(focus);
    assert.equal(focus.nodeId, 'left.block');
    assert.deepEqual(focus.nodeIds, ['left.block', 'right.block']);
  });

  test('dependency navigation refuses malformed paths, digest disagreement, and changed buffers', async () => {
    openStubEditor(BLOCKS, BLOCK_SOURCE);
    const spans = {
      ...DEPENDENCY_SPANS,
      traversal: { file: '../secret.py', start: [1, 0], end: [1, 1], file_digest: 'x' },
      absolute: { file: '/tmp/secret.py', start: [1, 0], end: [1, 1], file_digest: 'x' },
      outsideSet: { file: 'models/not-projected.py', start: [1, 0], end: [1, 1], file_digest: 'x' },
      wrongDigest: { file: REL, start: [4, 0], end: [4, 1], file_digest: '0'.repeat(64) },
      missingDigest: { file: REL, start: [4, 0], end: [4, 1], file_digest: null },
    };
    const rig = makeSync({ spans });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    const live = liveTarget(rig);

    for (const nodeId of ['traversal', 'absolute', 'outsideSet', 'wrongDigest', 'missingDigest']) {
      const result = await rig.sync.revealNode(live, nodeId);
      assert.equal(result.ok, false, `${nodeId} must be refused`);
      if (!result.ok) assert.equal(result.errorCode, 'INVALID_MESSAGE');
    }

    openStubEditor(BLOCKS, BLOCK_EDITED); // dirty after projection; no change event yet
    assert.deepEqual(await rig.sync.revealNode(live, 'left.block'), {
      ok: false,
      errorCode: 'STALE_TARGET',
      detail: `${BLOCKS_REL} changed after this projection; wait for the refresh and click again`,
    });
  });

  test('two workspace syncs ignore each other’s dependency and root edits', async () => {
    const otherRoot = path.join(path.sep, 'ws-two');
    const otherRel = 'models/net.py';
    const otherModel = path.join(otherRoot, otherRel);
    openStubEditor(MODEL, SAVED);
    openStubEditor(otherModel, SAVED);
    const first = makeSync({});
    const second = makeSync({});
    first.sync.setTarget(modelUri(), baseIdentity(1, SAVED));
    second.sync.setTarget(Uri.file(otherModel) as never, {
      workspaceFolder: otherRoot,
      file: otherRel,
      cls: 'Net',
      documentVersion: 1,
      sourceDigest: createHash('sha256').update(SAVED).digest('hex'),
      backendRevision: 0,
    });
    await settle();
    assert.equal(first.fetches.length, 1);
    assert.equal(second.fetches.length, 1);

    editFile(MODEL, TYPED);
    await settle();
    assert.equal(first.fetches.length, 2);
    assert.equal(second.fetches.length, 1, 'the other window remains untouched');
  });

  test('a reveal is refused unless the page names the state it is really showing', async () => {
    const rig = makeSync({ spans: SPANS });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    const live = liveTarget(rig);
    const editor = modelEditor();

    // A projection behind the live one: same class, older revision.
    assert.deepEqual(await rig.sync.revealNode({ ...live, backendRevision: 0 }, 'fwd'), {
      ok: false,
      errorCode: 'STALE_TARGET',
      detail:
        'the page is not showing this window’s current target state; ' +
        'wait for the refresh and click again',
    });
    // A page aimed at another class entirely: refused the same way.
    assert.deepEqual(await rig.sync.revealNode({ ...live, cls: 'Gate' }, 'fwd'), {
      ok: false,
      errorCode: 'STALE_TARGET',
      detail:
        'the page is not showing this window’s current target state; ' +
        'wait for the refresh and click again',
    });
    // A node the graph on the page does not have.
    assert.deepEqual(await rig.sync.revealNode(live, 'ghost'), {
      ok: false,
      errorCode: 'INVALID_MESSAGE',
      detail: 'no node named ghost in the graph the page is showing',
    });

    // None of those moved the editor.
    assert.equal(editor.selection.active.line, 0);
    assert.equal(editor.selection.active.character, 0);
  });

  test('before the first projection there is nothing to reveal and nothing to focus', async () => {
    const rig = makeSync({ deferFirstParse: true, spans: SPANS });
    rig.sync.setTarget(modelUri(), anchor());
    const base = rig.sync.currentTarget();
    assert.ok(base);

    assert.deepEqual(await rig.sync.revealNode(base, 'fwd'), {
      ok: false,
      errorCode: 'INVALID_MESSAGE',
      detail: 'no projection exists for this target yet',
    });
    moveCursor(6);
    assert.deepEqual(focusPublishes(rig), []);

    // Once the graph exists, both directions switch on.
    rig.releaseFirstParse();
    await settle();
    moveCursor(6);
    assert.deepEqual(
      focusPublishes(rig).map((m) => m.nodeId),
      ['fwd'],
    );
  });

  test('a retarget re-arms cursor focus for the new class’s graph', async () => {
    const rig = makeSync({ spans: SPANS });
    rig.sync.setTarget(modelUri(), anchor());
    await settle();
    moveCursor(6);
    assert.deepEqual(
      focusPublishes(rig).map((m) => m.nodeId),
      ['fwd'],
    );

    // The buffer gains a second class; the command re-anchors on it in place.
    type(TYPED);
    await settle();
    rig.sync.setTarget(modelUri(), {
      ...baseIdentity(editorState.versions.get(MODEL) ?? 1, TYPED),
      cls: 'Added',
    });
    await settle();

    // Same line, new graph: the entry memory from the OLD class must not
    // swallow this one — it names the new class's identity.
    moveCursor(6);
    const publishes = focusPublishes(rig);
    assert.deepEqual(
      publishes.map((m) => m.nodeId),
      ['fwd', 'fwd'],
    );
    assert.equal(publishes[1].target.cls, 'Added');
  });
});
