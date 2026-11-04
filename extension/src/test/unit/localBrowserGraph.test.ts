import { agentGraphContext } from '../../canvasAgentContext';
/** Step-4 "Open as Graph -> external browser" unit tests.
 *
 * What they pin, in product language:
 *  - the class the user MEANS comes from the buffer the editor holds, not the
 *    disk: an unsaved class resolves, and the buffer text is what the backend
 *    scans (so the two can never silently diverge);
 *  - resolution is innermost-under-cursor, then the only class, then a Quick
 *    Pick — and Escape is a quiet no-op;
 *  - the target identity is honest: containing workspace folder, live document
 *    version, and the sha256 of the buffer text;
 *  - one window, one session: a retarget reuses it (no second tab), an ended
 *    one is reminted, and a view that asks is answered snapshot-then-target;
 *  - remote windows and non-file documents are refused with nothing started.
 *
 * Step 6 rides the same rig: a view's `revealNode` moves the editor cursor
 * only for the exact state its page is showing, and is acked; anything older,
 * another class, or an unknown node is refused by ack without moving anything.
 * The reverse direction is pinned here too: moving the editor cursor focuses
 * the node on the page, and a reveal the host just answered cannot echo back.
 *
 */

import * as assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import * as vscode from 'vscode';
import * as path from 'node:path';
import { readFileSync } from 'node:fs';
import { mkdtemp, mkdir, writeFile, rm, realpath } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { canonicalFixture } from '../../../../tests/fixtures/semantic.cjs';
import { mock } from 'node:test';
import { GraphToolError, McpGraphService } from '@tensorviz/graph-host/mcpGraph';
import { GraphMcpServer } from '../../mcpServer';
import { registerMcpValidationTools } from '@tensorviz/graph-host/mcpValidation';
import { getModelContext } from '@tensorviz/graph-host/modelContext';
import { registerMcpProposalTools } from '../../mcpProposals';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import type { LocalBrowserSourceEdits } from '../../localBrowserSourceEdits';
import type { LocalBrowserMessage } from '@tensorviz/contracts/protocol';
import { LOCAL_BROWSER_PROTOCOL_VERSION } from '@tensorviz/contracts/protocol';
import { SidebarCanvas } from '../../sidebarCanvas';
import { resetWebviews, StubWebviewView, webviewState } from '../vscodeStub';
import type { TraceOutcome } from '../../worker';
import type { CodeArtifact } from '../../sourceChanges';
import * as sourceReview from '../../sourceReview';
import { ValidationSetup } from '../../validationSetup';
import { ExecutionReview } from '@tensorviz/graph-host/executionReview';
import { rememberEnvironment, savedEnvironment, selectedExecutionEnvironment } from '../../projectEnvironment';

import {
  BufferClass,
  LocalBrowserGraph,
  LocalBrowserGraphOptions,
  emptyLocalBrowserSnapshotState,
} from '../../localBrowserGraph';
import { MintLocalBrowserSessionOptions } from '@tensorviz/graph-host/localBrowserHostTransport';
import {
  editorState,
  openStubEditor,
  reset,
  resetEditors,
  selectionChanges,
  state,
  stubSourceIO,
  documentChanges,
  saveChanges,
  configFileChanges,
  Uri,
  infoMessages,
  languages,
  window,
} from '../vscodeStub';

const ROOT = path.join(path.sep, 'ws');
const RESID = path.join(ROOT, 'models', 'resid.py');

const SAVED = [
  'import torch.nn as nn',
  '',
  '',
  'class Net(nn.Module):',
  '    def forward(self, x):',
  '        return x',
  '',
].join('\n');
/** The saved text with one class typed in and never written anywhere. Lines:
 *  8 and 9 are blank, `class Unsaved` is line 10, its body line 11. */
const EDITED = `${SAVED}\n\nclass Unsaved(nn.Module):\n    pass\n`;

const NET: BufferClass = { name: 'Net', line: 4, end_line: 6, bases: ['nn.Module'] };
const UNSAVED: BufferClass = { name: 'Unsaved', line: 10, end_line: 11, bases: ['nn.Module'] };
const GATE: BufferClass = { name: 'Gate', line: 13, end_line: 15, bases: ['nn.Module'] };
const OUTER: BufferClass = { name: 'Outer', line: 12, end_line: 40, bases: ['nn.Module'] };
const INNER: BufferClass = { name: 'Inner', line: 20, end_line: 24, bases: ['nn.Module'] };

/** Two spans over SAVED, valid enough for navigation: `net` covers the class
 *  (lines 4-6), `fwd` the forward body (lines 5-6). */
const SPANS: Record<string, unknown> = {
  net: { file: 'models/resid.py', start: [4, 0], end: [6, 8], kind: 'decl', digest: 'd1' },
  fwd: { file: 'models/resid.py', start: [5, 4], end: [6, 8], kind: 'call', digest: 'd2' },
};

/** The resid editor, for cursor assertions. */
function residEditor() {
  const editor = editorState.visibleTextEditors.find((e) => e.document.uri.fsPath === RESID);
  assert.ok(editor, 'resid.py is not open');
  return editor;
}

/** The identity of the projection that landed — what the page being asked to
 *  navigate is showing. */
function liveTarget(published: Array<Record<string, unknown>>): Record<string, unknown> {
  const projection = published.find((m) => m.type === 'projection');
  assert.ok(projection, 'no projection was published');
  return projection.target as Record<string, unknown>;
}

interface FetchCall {
  url: string;
  token: string | undefined;
  body: unknown;
}

type PlannedAnswer = { classes: BufferClass[] } | { syntaxError: { line: number } };

/** One projection POST the sync made — since the multi-file task BOTH tiers
 *  are /api/ide/multifile/project, the traced one just carries trace_result. */
interface ProjectionCall {
  url: string;
  token: string | undefined;
  body: {
    root: { file: string; cls: string; source_digest: string };
    sources: Array<{ file: string; source: string; digest: string }>;
    truncated: boolean;
    trace_result?: unknown;
  };
}

/** A fetch that knows /api/ide/buffer-classes and /api/ide/multifile/project —
 *  the whole network the command and the sync together may
 *  touch. Any other request, or a buffer scan a test did not plan for, throws
 *  — the command's network is under test, so it must not be free to wander.
 *  The projection answer is built from the request, so the identity messages
 *  name the digest of exactly the buffer text that was POSTed. */
function makeFetch(
  planned: PlannedAnswer[],
  opts: {
    deferFirstParse?: boolean;
    spans?: Record<string, unknown>;
    document?: unknown;
  } = {},
): {
  fetchImpl: typeof fetch;
  calls: FetchCall[];
  projectionCalls: ProjectionCall[];
  releaseFirstParse(): void;
} {
  const calls: FetchCall[] = [];
  const projectionCalls: ProjectionCall[] = [];
  let revision = 0;
  let gate: Promise<void> | undefined;
  let released = (): void => {
    gate = undefined;
  };
  if (opts.deferFirstParse) {
    gate = new Promise<void>((resolve) => {
      released = () => {
        gate = undefined;
        resolve();
      };
    });
  }
  const fetchImpl = (async (input: unknown, init?: RequestInit) => {
    const url = String(input);
    const token = init?.headers
      ? (init.headers as Record<string, string>)['X-tensorViz-Token']
      : undefined;
    if (url.endsWith('/project/index')) {
      calls.push({ url, token, body: JSON.parse(String(init?.body)) });
      return new Response(JSON.stringify({ status: 'ok', digest: 'a'.repeat(64),
        index: { experiments: [], unreached: [], roots: [], truncated: false } }));
    }
    if (url.endsWith('/api/ide/multifile/project')) {
      const body = JSON.parse(String(init?.body)) as ProjectionCall['body'];
      projectionCalls.push({ url, token, body });
      if (gate) {
        await gate;
      }
      revision += 1;
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
      return new Response(
        JSON.stringify({
          cls: body.root.cls,
          document: opts.document ?? { view: {}, semantic: { nodes: [] } },
          spanmap,
          source_digest: body.root.source_digest,
          projection_digest: 'p-test',
          graph_id: 'g-test',
          revision,
          traced: body.trace_result !== undefined,
          trace_error: null,
          protocol_version: 1,
          dependencies: body.sources.map((entry) => ({
            file: entry.file,
            digest: entry.digest,
          })),
          diagnostics: [],
          truncated: body.truncated,
        }),
        { status: 200 },
      );
    }
    if (url.endsWith('/local/sessions/ws-test/view-links')) {
      return new Response(JSON.stringify({ workspaceSessionId: 'ws-test', launchOrigin: 'http://127.0.0.1:8631',
        handoffId: 'handoff-1234567890', viewCredential: 'single-use-1234567890', expiresAt: new Date(Date.now() + 120000).toISOString() }));
    }
    if (!url.endsWith('/api/ide/buffer-classes')) {
      throw new Error(`unexpected fetch: ${url}`);
    }
    const plan = planned[calls.length];
    if (!plan) {
      throw new Error(`unplanned buffer-classes call #${calls.length + 1}`);
    }
    calls.push({ url, token, body: JSON.parse(String(init?.body)) as FetchCall['body'] });
    if ('syntaxError' in plan) {
      return new Response(
        JSON.stringify({
          detail: {
            error: 'syntax_error',
            message: 'expected colon',
            line: plan.syntaxError.line,
            column: 0,
          },
        }),
        { status: 400 },
      );
    }
    return new Response(JSON.stringify({ classes: plan.classes }), { status: 200 });
  }) as typeof fetch;
      return {
        fetchImpl,
        calls,
        projectionCalls,
        releaseFirstParse: released,
      };
}

interface FakeTransport {
  readonly status: { phase: string; epoch: number; error: unknown };
  published: Array<Record<string, unknown>>;
  setPhase(phase: string): void;
  fire(message: unknown): void;
  isDisposed(): boolean;
}

/** The transport the command drives: phase is the only state the reuse rule
 *  reads, and published/fire are the only effects it has. */
function makeTransport(connectError?: Error): FakeTransport {
  const published: Array<Record<string, unknown>> = [];
  const listeners: Array<(message: unknown) => void> = [];
  const statusRef = { phase: 'connected', epoch: 1, error: null };
  const statusListeners: Array<(status: typeof statusRef) => void> = [];
  let disposed = false;
  const impl = {
    published,
    get status() {
      return statusRef;
    },
    protocolViolations: 0,
    async connect(): Promise<void> {
      impl.setPhase('connecting');
      if (connectError) {
        throw connectError;
      }
      impl.setPhase('connected');
    },
    publish(message: unknown) {
      published.push(message as Record<string, unknown>);
      return message;
    },
    onMessage(listener: (message: unknown) => void) {
      listeners.push(listener);
      return () => {};
    },
    onStatusChange(listener: (status: typeof statusRef) => void) {
      statusListeners.push(listener);
      return () => {};
    },
    dispose() {
      disposed = true;
    },
    setPhase(phase: string) {
      statusRef.phase = phase;
      statusListeners.forEach(listener => listener(statusRef));
    },
    fire(message: unknown) {
      for (const listener of listeners.slice()) {
        listener(message);
      }
    },
    isDisposed() {
      return disposed;
    },
  };
  return impl as unknown as FakeTransport;
}

interface GraphRig {
  graph: LocalBrowserGraph;
  fetchCalls: FetchCall[];
  projectionCalls: ProjectionCall[];
  mintCalls: MintLocalBrowserSessionOptions[];
  opened: string[];
  transports: FakeTransport[];
  ensureCalls(): number;
  releaseFirstParse(): void;
}

/** Every rig the suite has made; teardown disposes it so a disposed sync from
 *  one test never answers a later test's editor events. */
const made: GraphRig[] = [];

/** The collaborators the command needs, faked at the seams. Nothing reaches a
 *  network or a browser: mint records its arguments, openExternal records URLs. */
function makeGraph(
  planned: PlannedAnswer[],
  overrides: Partial<LocalBrowserGraphOptions> & {
    deferFirstParse?: boolean;
    spans?: Record<string, unknown>;
    document?: unknown;
  } = {},
  connectError?: Error,
): GraphRig {
  const { deferFirstParse, spans, document, ...graphOverrides } = overrides;
  const fetch = makeFetch(planned, { deferFirstParse, spans, document });
  const mintCalls: MintLocalBrowserSessionOptions[] = [];
  const opened: string[] = [];
  const transports: FakeTransport[] = [];
  let ensureCalls = 0;
  const graph = new LocalBrowserGraph({
    channel: { appendLine: () => {} },
    projectSourceIO: folder => ({ ...stubSourceIO(folder), readIgnore: async () => undefined }),
    backend: { origin: () => 'http://127.0.0.1:8631', launchToken: () => 'test-token' },
    launchOrigin: () => undefined,
    ensureBackend: async () => {
      ensureCalls += 1;
      return true;
    },
    remoteName: () => undefined,
    openExternal: async (url: string) => {
      opened.push(url);
      return true;
    },
    mintSession: (async (options: MintLocalBrowserSessionOptions) => {
      mintCalls.push(options);
      return {
        workspaceSessionId: 'ws-test',
        launchOrigin: options.launchOrigin,
        expiresAt: '2026-01-01T00:00:00.000Z',
        hostCookieKey: 'host-key',
        viewCredential: 'view-credential',
        pageUrl: `${options.launchOrigin}/#session=ws-test&credential=view-credential`,
      };
    }) as never,
    createTransport: (() => {
      const transport = makeTransport(connectError);
      transports.push(transport);
      return transport;
    }) as never,
    fetchImpl: fetch.fetchImpl,
    // The multi-file source set enumerates/reads the stubbed editor documents:
    // there is no filesystem in a unit test.
    sourceIO: (folder) => stubSourceIO(folder),
    // The unit rig has no trace worker: the page these tests drive is the
    // source tier. Zero delays keep the sync's timers one macrotask short.
    debounceMs: 0,
    saveCoalesceMs: 0,
    ...graphOverrides,
  });
  const rig: GraphRig = {
    graph,
    fetchCalls: fetch.calls,
    projectionCalls: fetch.projectionCalls,
    mintCalls,
    opened,
    transports,
    ensureCalls: () => ensureCalls,
    releaseFirstParse: fetch.releaseFirstParse,
  };
  made.push(rig);
  return rig;
}

/** Let the projection loop's microtasks and the rig's zeroed timers run out. */
async function settle(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 0));
  await new Promise((resolve) => setTimeout(resolve, 0));
  await new Promise((resolve) => setTimeout(resolve, 0));
}

/** Exercise the same explicit Run action as the native execution panel. */
function approveExecution(graph: LocalBrowserGraph): void {
  const opening = graph.opening?.state;
  if (opening?.execution?.phase === 'review') graph.handleCanvasMessage({ type: 'graphSetupRequest', openingId: opening.id,
    target: opening.target, requestId: `run-${opening.execution.id}`, action: { kind: 'execution', reviewId: opening.execution.id,
      action: 'run', device: opening.execution.device } } as never);
}

suite('LocalBrowserGraph', () => {
  for (const scenario of ['saved', 'first-save-fails', 'second-save-fails', 'next-file-changed', 'earlier-file-changed'] as const)
    test(`accepted source set saves only reviewed bytes before one opening: ${scenario}`, async () => {
      const rig = makeGraph([], { profiles: {} as never,
        trace: { worker: { run: async () => { throw new Error('Opening seam owns this test'); }, cancel() {} },
          trusted: async () => true, settings: () => ({ onSave: true, tier: 'real' }) } });
      rig.graph.opening!.open = async target => { rig.graph.opening!.sourceOnly(target); };
      openStubEditor(RESID, SAVED);
      await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
      const target = { ...rig.graph.target! }, internals = rig.graph as any;
      const dep = path.join(ROOT, 'models', 'dep.py'), other = path.join(ROOT, 'unrelated.py');
      const reviewed = new Map([[RESID, SAVED + '# accepted model\n'], [dep, 'WIDTH = 16\n']]);
      for (const [file, text] of reviewed) { openStubEditor(file, text); editorState.dirtyDocuments.add(file); }
      openStubEditor(other, '# unrelated unsaved work\n'); editorState.dirtyDocuments.add(other);
      const opened: any[] = [], saved: string[] = [];
      rig.graph.opening!.open = async next => { opened.push(next); rig.graph.opening!.sourceOnly(next); };
      const vscode = require('../vscodeStub'), original = vscode.workspace.openTextDocument;
      const read = mock.method(vscode.workspace, 'openTextDocument', async (uri: any) => {
        const document = await original(uri);
        return Object.assign(document, { save: async () => {
          saved.push(uri.fsPath);
          if (scenario === 'first-save-fails' && saved.length === 1 || scenario === 'second-save-fails' && saved.length === 2) return false;
          if (scenario === 'next-file-changed' && saved.length === 1) editorState.documents.set(dep, 'WIDTH = 99\n');
          if (scenario === 'earlier-file-changed' && saved.length === 2) editorState.documents.set(RESID, SAVED + '# unreviewed save edit\n');
          editorState.dirtyDocuments.delete(uri.fsPath); saveChanges.fire(document); return true;
        } });
      });
      try {
        const detail = await internals.sourceEdits.options.accepted({ target, file: target.file,
          files: [...reviewed.keys()].map(file => path.relative(ROOT, file).split(path.sep).join('/')),
          current: () => [...reviewed].every(([file, text]) => editorState.documents.get(file) === text) });
        await settle();
        assert.deepEqual(saved, ['first-save-fails', 'next-file-changed'].includes(scenario) ? [RESID] : [RESID, dep]);
        assert.equal(opened.length, scenario === 'saved' ? 1 : 0, 'partial saves or changes to any reviewed file cannot start execution');
        assert.match(detail, scenario === 'saved' ? /Saved the accepted change/ : /Automatic capture did not start/);
        assert.equal(editorState.documents.get(other), '# unrelated unsaved work\n');
        assert.ok(editorState.dirtyDocuments.has(other), 'an accepted source set never saves unrelated work');
        if (opened.length) assert.equal(opened[0].file, target.file);
      } finally { read.mock.restore(); }
    });

  for (const scenario of ['saved', 'dependency-edit', 'save-false', 'save-throw', 'format-on-save', 'retarget', 'lost-trust', 'dependency-change', 'other-dirty', 'dispose'] as const)
    test(`automatic capture after acceptance: ${scenario}`, async () => {
      const rig = makeGraph([], { profiles: {} as never,
        trace: { worker: { run: async () => { throw new Error('Opening seam owns this test'); }, cancel() {} },
          trusted: async () => true, settings: () => ({ onSave: true, tier: 'real' }) } });
      rig.graph.opening!.open = async target => { rig.graph.opening!.sourceOnly(target); };
      openStubEditor(RESID, SAVED);
      const dep = path.join(ROOT, 'models', 'dep.py');
      openStubEditor(dep, 'x = 1\n');
      await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas', 'automatic'); await settle();
      const target = { ...rig.graph.target! }, internals = rig.graph as any;
      internals.saveRunEnabled = true;
      internals.changedSinceRun = true;
      const edited = scenario === 'dependency-edit' ? dep : RESID;
      const text = editorState.documents.get(edited)! + '# accepted\n';
      openStubEditor(edited, text); editorState.dirtyDocuments.add(edited);
      if (scenario === 'other-dirty') editorState.dirtyDocuments.add(dep);
      let sourceCurrent = true;
      const opened: any[] = [], saved: string[] = [];
      rig.graph.opening!.open = async next => { opened.push(next); rig.graph.opening!.sourceOnly(next); };
      const vscode = require('../vscodeStub'), original = vscode.workspace.openTextDocument;
      const read = mock.method(vscode.workspace, 'openTextDocument', async (uri: any) => {
        const document = await original(uri);
        return Object.assign(document, { save: async () => {
          saved.push(uri.fsPath);
          if (scenario === 'save-throw') throw new Error('write failed');
          if (scenario === 'save-false') return false;
          if (scenario === 'format-on-save') editorState.documents.set(edited, text + '# save participant\n');
          if (scenario === 'dependency-change') sourceCurrent = false;
          if (scenario === 'retarget') internals.sync.clearTarget();
          if (scenario === 'lost-trust') state.isTrusted = false;
          if (scenario === 'dispose') rig.graph.dispose();
          editorState.dirtyDocuments.delete(uri.fsPath);
          saveChanges.fire(document);
          return true;
        } });
      });
      try {
        const accepted = internals.sourceEdits.options.accepted;
        const invocation = () => accepted({ target, file: path.relative(ROOT, edited).split(path.sep).join('/'),
          current: () => sourceCurrent && state.isTrusted && editorState.documents.get(edited) === text });
        if (scenario === 'save-throw') await assert.rejects(invocation(), /write failed/);
        else {
          const detail = await invocation();
          await settle();
          const runs = ['saved', 'dependency-edit', 'other-dirty'].includes(scenario);
          assert.equal(opened.length, runs ? 1 : 0);
          if (runs) {
            assert.match(detail, /Saved the accepted change/);
            assert.equal(opened[0].file, target.file, 'dependency edits still run the originating model');
            assert.equal(opened[0].sourceDigest, createHash('sha256').update(editorState.documents.get(RESID)!).digest('hex'));
            assert.equal(internals.openingApproved(), true, 'human acceptance approves a previously automatic opening');
          } else assert.match(detail, /Automatic capture did not start/);
          if (scenario === 'other-dirty') {
            assert.ok(editorState.dirtyDocuments.has(dep));
            const lookup = await (rig.graph.opening as any).dependencies.lookup(opened[0], new AbortController().signal);
            assert.equal(lookup.kind, 'blocked', 'another dirty dependency must still block normal opening');
          }
        }
        assert.deepEqual(saved, [edited], 'acceptance saves only the reviewed file');
      } finally { read.mock.restore(); }
    });

  for (const actor of ['human', 'agent', 'accepted', 'saved'] as const) test(`opening as ${actor} captures once on the saved device without a Run click`, async () => {
    const rig = makeGraph([], { profiles: {} as never,
      trace: { worker: { run: async () => { throw new Error('The sync seam owns this test'); }, cancel() {} },
        trusted: async () => true, settings: () => ({ onSave: false, tier: 'real' }) } });
    rig.graph.opening!.open = async target => { rig.graph.opening!.sourceOnly(target); };
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas', actor === 'accepted' || actor === 'saved' ? 'human' : actor); await settle();
    const target = rig.graph.opening!.state!.target, internals = rig.graph as any;
    if (actor === 'accepted') await internals.sourceEdits.options.accepted({ target, file: target.file, current: () => true });
    if (actor === 'saved') { internals.openingActor = 'automatic'; internals.freshOpeningId = rig.graph.opening!.state!.id; }
    const calls: Array<{ job: unknown; device: unknown; cache: unknown }> = [];
    const ready = mock.method(internals.sync, 'readyTarget', async () => target);
    const validate = mock.method(internals.sync, 'validate', async (_t: unknown, _s: unknown, _ti: unknown, _p: unknown, measured: Function, _c: unknown,
      _e: unknown, job: unknown, _timeout: unknown, _guard: unknown, _attempt: unknown, device: unknown, cache: unknown) => {
      calls.push({ job, device, cache }); measured({ ...target, backendRevision: 2 }, { revision: 2 });
      return { status: 'passed', detail: 'Captured.' };
    });
    try {
      const result = await (rig.graph.opening as any).dependencies.capture(target, { profile: 'saved', run: { execution: { device: 'cpu' } },
        guard: { spec: { interpreter: { executable: '/env/python' } } } }, new AbortController().signal, () => {});
      assert.deepEqual(calls, [{ job: undefined, device: 'cpu', cache: actor === 'human' ? 'allow' : undefined }],
        'one capture on the saved device; accepting changes and agent requests must bypass cache reuse');
      assert.equal(result.hasCapture, true);
      const review = internals.executionReview as ExecutionReview;
      assert.equal(review.state!.kind, 'capture'); assert.equal(review.state!.phase, 'succeeded');
      // This run settled before the host reading's 150 ms sample; the strip must still name the host.
      for (let i = 0; i < 100 && !rig.graph.opening!.state!.execution?.resources; i++) await new Promise(resolve => setTimeout(resolve, 20));
      assert.ok(rig.graph.opening!.state!.execution?.resources?.host, 'the late host reading reaches the opening the strip renders');
      assert.equal(rig.graph.opening!.state!.execution?.phase, 'succeeded');
    } finally { ready.mock.restore(); validate.mock.restore(); }
  });
  test('an editor auto-opened graph waits for Run and captures once, refusing a timing request', async () => {
    const rig = makeGraph([], { profiles: {} as never,
      trace: { worker: { run: async () => { throw new Error('The sync seam owns this test'); }, cancel() {} },
        trusted: async () => true, settings: () => ({ onSave: false, tier: 'real' }) } });
    rig.graph.opening!.open = async target => { rig.graph.opening!.sourceOnly(target); };
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas', 'automatic'); await settle();
    const target = rig.graph.opening!.state!.target, internals = rig.graph as any;
    const jobs: unknown[] = [], targets: unknown[] = [];
    const ready = mock.method(internals.sync, 'readyTarget', async () => target);
    const validate = mock.method(internals.sync, 'validate', async (_t: unknown, _s: unknown, _ti: unknown, _p: unknown, measured: Function, _c: unknown, _e: unknown, job: unknown) => {
      jobs.push(job); targets.push(_t); measured({ ...target, backendRevision: jobs.length + 1 }, { revision: jobs.length + 1 });
      return { status: 'passed', detail: 'Captured.' };
    });
    const capture = (rig.graph.opening as any).dependencies.capture(target, { profile: 'saved', run: { execution: { device: 'cpu' } },
      guard: { spec: { interpreter: { executable: '/env/python' } } } }, new AbortController().signal, () => {});
    try {
      await settle(); assert.equal(jobs.length, 0);
      assert.equal(internals.validationSummaries.snapshot(target), undefined, 'a capture waiting for Run is not reported as running');
      const review = internals.executionReview as ExecutionReview;
      assert.equal(review.action(target, 'timing', { kind: 'execution', reviewId: review.state!.id, device: 'cpu', action: 'configure',
        timing: { warmup: 0, profileIterations: 2, iterations: 4 } } as never), false);
      assert.equal(jobs.length, 0); approveExecution(rig.graph);
      const result = await capture;
      assert.deepEqual(jobs, [undefined]);
      assert.deepEqual(targets, [target]);
      assert.equal(result.hasCapture, true); assert.equal('validationPassed' in result.outcome, false);
    } finally { ready.mock.restore(); validate.mock.restore(); }
  });
  test('execution Edit resolves only the reviewed profile on the current host target without approving work', async () => {
    const rig = makeGraph([], { profiles: {} as never,
      trace: { worker: { run: async () => { throw new Error('Navigation must not start a worker'); }, cancel() {} },
        trusted: async () => true, settings: () => ({ onSave: false, tier: 'real' }) } });
    rig.graph.opening!.open = async target => { rig.graph.opening!.sourceOnly(target); };
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    const opening = rig.graph.opening!.state!, review = (rig.graph as unknown as { executionReview: ExecutionReview }).executionReview;
    let executions = 0;
    const run = review.run(opening.target, new AbortController().signal, { kind: 'capture', interpreter: '/env/python', profileName: 'reviewed.v2' }, async () => { executions++; });
    const reveals: unknown[][] = [];
    const diagnostics = Object.getOwnPropertyDescriptor(languages, 'createDiagnosticCollection');
    Object.defineProperty(languages, 'createDiagnosticCollection', { configurable: true, value: () => ({ delete() {}, dispose() {} }) });
    const stub = mock.method(ValidationSetup.prototype, 'reveal', async (target: any, field: unknown[]) => {
      assert.equal(target.root, ROOT); assert.equal(target.current(), true); reveals.push(field);
    });
    const message = { type: 'graphSetupRequest', openingId: opening.id, requestId: 'edit', target: opening.target,
      action: { kind: 'edit_execution', reviewId: review.state!.id, field: ['inputs', 'args', 0, 'shape'] } };
    try {
      rig.graph.handleCanvasMessage({ ...message, openingId: 'old' } as never);
      rig.graph.handleCanvasMessage({ ...message, action: { ...message.action, reviewId: 'old' } } as never);
      rig.graph.handleCanvasMessage({ ...message, target: { ...message.target, cls: 'Other' } } as never);
      rig.graph.handleCanvasMessage({ ...message, action: { ...message.action, field: ['profile'] } } as never);
      await settle(); assert.equal(reveals.length, 0);
      rig.graph.handleCanvasMessage(message as never); await settle();
      assert.deepEqual(reveals, [['profiles', 'reviewed.v2', 'inputs', 'args', 0, 'shape']]);
      assert.equal(executions, 0); assert.equal(review.state!.phase, 'review');
      const stopped = assert.rejects(run);
      saveChanges.fire({ uri: Uri.file(path.join(ROOT, 'tensorviz.toml')) } as never);
      await stopped; await settle();
      assert.equal(executions, 0); assert.equal(review.state!.phase, 'cancelled');
      assert.match(review.state!.detail, /Saved setup changed.*Run again/);
      openStubEditor(RESID, SAVED + '\n# changed');
      rig.graph.handleCanvasMessage(message as never); await settle(); assert.equal(reveals.length, 1);
    } finally {
      if (diagnostics) Object.defineProperty(languages, 'createDiagnosticCollection', diagnostics);
      else delete (languages as any).createDiagnosticCollection;
      stub.mock.restore(); const stopped = assert.rejects(run); review.cancel(); await stopped;
    }
  });
  test('agent context includes verified selected facts without weakening stale-revision checks', async () => {
    let service!: McpGraphService;
    const rig = makeGraph([], { agent: { run: async () => {}, clear() {}, reference: model => service.reference(model),
      selection: (reference, ids) => service.inspectNodes(reference, ids) } });
    service = new McpGraphService('agent-context', () => rig.graph.mcpModel);
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    (rig.graph.mcpModel!.projection.document as any).semantic = { nodes: [{ node_id: 'fc', op: 'Linear', params: { out_features: 8 } }], edges: [] };
    const prepare = (rig.graph as any).agent.options.prepare;
    const prepared = await prepare(rig.graph.target!, ['fc'], undefined, new AbortController().signal);
    const attached = JSON.parse(prepared.context.split('\n')[1]);
    const ref = service.reference(rig.graph.mcpModel!);
    assert.equal(attached.modelId, ref.modelId); assert.equal(attached.revision, ref.revision);
    assert.deepEqual(attached.selectionInspection, JSON.parse(JSON.stringify(service.inspectNodes(ref, ['fc']))));
    assert.equal(attached.selectionInspection.nodes[0].parameters.out_features, 8);
    assert.match(prepared.context, /Use the attached modelId and revision directly/);
    assert.doesNotMatch(prepared.context, /Call list_models to obtain/);
    assert.doesNotThrow(() => service.require(ref));
    rig.graph.mcpModel!.projection.projectionDigest = 'new-projection';
    assert.throws(() => service.require(ref), /revision changed/);
    (rig.graph.mcpModel!.projection.document as any).semantic.nodes[0].params.out_features = 16;
    const refreshed = JSON.parse((await prepare(rig.graph.target!, ['fc'], undefined, new AbortController().signal)).context.split('\n')[1]);
    assert.notEqual(refreshed.revision, ref.revision);
    assert.equal(refreshed.selectionInspection.nodes[0].parameters.out_features, 16, 'a new Send reads fresh facts, not the previous prompt');
    assert.doesNotThrow(() => service.require(refreshed));
    await assert.rejects(prepare({ ...rig.graph.target!, sourceDigest: 'f'.repeat(64) }, [], undefined, new AbortController().signal), /changed/);
    openStubEditor(RESID, SAVED + '\n# changed');
    await assert.rejects(prepare(rig.graph.target!, ['fc'], undefined, new AbortController().signal), /changed/);
  });
  test('Canvas attaches the shared current packet before source and reduces graph preload only when available', async () => {
    let service!: McpGraphService;
    const rig = makeGraph([], { agent: { run: async () => {}, clear() {}, reference: model => service.reference(model),
      graph: (reference, maxBytes) => agentGraphContext(service, reference, maxBytes),
      modelContext: reference => getModelContext(service, rig.graph.captureInspector, reference) } });
    service = new McpGraphService('context', () => rig.graph.mcpModel);
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    const target = rig.graph.target!, { capture, plan } = canonicalFixture();
    capture.source.modelFile = target.file; capture.source.className = target.cls;
    capture.source.files = [{ path: target.file, digest: target.sourceDigest, documentVersion: target.documentVersion, role: 'source' }];
    plan.source = structuredClone(capture.source);
    const ledger = (rig.graph as any).validationSummaries.ledger;
    const handle = ledger.begin(capture.attemptId, target, 'manual', null);
    handle.resolved({ plan, applicabilityDigest: capture.planDigest });
    handle.captured({ ok: true, exitCode: 0, envelope: { protocol_version: 1,
      result: { diagnostics: { semantic_capture: capture } }, spans: {}, error: null, worker: {} } });
    handle.finish('passed');
    (rig.graph.mcpModel!.projection.document as any).semantic = { edges: [],
      nodes: Array.from({ length: 100 }, (_, i) => ({ node_id: `node-${i}`, op: 'Linear', params: { description: '界'.repeat(80) } })) };
    const prepare = (rig.graph as any).agent.options.prepare;
    const attached = JSON.parse((await prepare(target, [], undefined, new AbortController().signal)).context.split('\n')[1]);
    assert.equal(Object.keys(attached)[0], 'modelContext');
    assert.equal(attached.modelContext.available, true);
    assert.deepEqual(attached.modelContext, getModelContext(service, rig.graph.captureInspector, service.reference(rig.graph.mcpModel!)));
    assert.ok(Buffer.byteLength(JSON.stringify(attached.modelContext)) <= 4000);
    assert.ok(Buffer.byteLength(JSON.stringify(attached.graphInspection)) <= 8 * 1024);
    assert.equal(attached.graphInspection.nextNodeOffset, attached.graphInspection.nodes.length);
    assert.equal(attached.source, SAVED);
    ledger.begin('newer-no-capture', target, 'manual', null).finish('refused');
    const missing = JSON.parse((await prepare(target, [], undefined, new AbortController().signal)).context.split('\n')[1]);
    assert.equal(missing.modelContext.available, false);
    assert.equal(missing.modelContext.reason, 'no_latest_capture');
    assert.equal(missing.graphInspection.nodes.length, 100, 'normal graph context remains available without measured facts');
  });

  test('empty and oversized selections use the existing inspection tool path', async () => {
    let service!: McpGraphService, inspections = 0;
    const rig = makeGraph([], { agent: { run: async () => {}, clear() {}, reference: model => service.reference(model),
      selection: (reference, ids) => { inspections++; return service.inspectNodes(reference, ids); } } });
    service = new McpGraphService('agent-context', () => rig.graph.mcpModel);
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    const semantic = { nodes: [{ node_id: 'fc', op: 'Linear', params: { description: '界'.repeat(12_000) } }], edges: [] };
    (rig.graph.mcpModel!.projection.document as any).semantic = semantic;
    const prepare = (rig.graph as any).agent.options.prepare;
    const context = async (ids: string[]) => JSON.parse((await prepare(rig.graph.target!, ids, undefined, new AbortController().signal)).context.split('\n')[1]);
    assert.equal((await context([])).selectionInspection, undefined);
    assert.equal(inspections, 0);
    assert.equal((await context(['fc'])).selectionInspection, undefined, 'the budget is bytes, including multibyte text');
    assert.equal(inspections, 1);
    semantic.nodes[0].params.description = 'small';
    assert.ok((await context(['fc'])).selectionInspection, 'only the oversized attachment is omitted');
  });
  test('empty selections still attach graph facts and the exact requesting Canvas for direct highlighting', async () => {
    let service!: McpGraphService;
    const rig = makeGraph([], { agent: { run: async () => {}, clear() {}, reference: model => service.reference(model),
      graph: reference => agentGraphContext(service, reference) } });
    service = new McpGraphService('context', () => rig.graph.mcpModel);
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    const semantic = { nodes: [{ node_id: 'fc', op: 'Linear', params: {}, metadata: { outputShape: [2, 8] } }], edges: [] };
    (rig.graph.mcpModel!.projection.document as any).semantic = semantic;
    const target = rig.graph.target!, signal = new AbortController().signal;
    for (const viewId of ['requesting', 'other']) {
      rig.graph.mcpViews.connection(viewId, true);
      rig.graph.mcpViews.handle({ type: 'viewSelection', protocolVersion: 1, workspaceSessionId: 'workspace',
        messageId: 'selection', viewId, target, nodeIds: [], edgeIds: [] });
    }
    const prepare = (rig.graph as any).agent.options.prepare;
    const attached = JSON.parse((await prepare(target, [], undefined, signal, undefined, 'requesting')).context.split('\n')[1]);
    assert.equal(attached.source, SAVED); assert.equal(attached.sourceTruncated, false);
    assert.equal(attached.target.sourceDigest, target.sourceDigest);
    assert.equal(attached.graphInspection.nodes[0].nodeId, 'fc');
    assert.deepEqual(attached.graphInspection.nodes[0].shapes.output, [2, 8]);
    assert.equal(attached.graphInspection.contextLimited, false);
    assert.equal(attached.activeView.viewId, 'requesting');
    semantic.nodes[0].metadata.outputShape = [2, 16];
    rig.graph.mcpModel!.projection.projectionDigest = 'changed';
    const refreshed = JSON.parse((await prepare(target, [], undefined, signal, undefined, 'other')).context.split('\n')[1]);
    assert.notEqual(refreshed.graphInspection.revision, attached.graphInspection.revision);
    assert.deepEqual(refreshed.graphInspection.nodes[0].shapes.output, [2, 16]);
    assert.equal(refreshed.activeView.viewId, 'other');
    rig.graph.mcpViews.connection('requesting', false);
    const hidden = JSON.parse((await prepare(target, [], undefined, signal, undefined, 'requesting')).context.split('\n')[1]);
    assert.equal(hidden.activeView, undefined, 'do not substitute a different Canvas after the sender disconnects');
  });
  test('chat opening prepares the connected view with the same checks as Send and never grants a draft revision', async () => {
    let service!: McpGraphService;
    const rig = makeGraph([], { agent: { run: async () => {}, clear() {}, reference: model => service.reference(model),
      selection: (reference, ids) => service.inspectNodes(reference, ids) } });
    service = new McpGraphService('preparation', () => rig.graph.mcpModel);
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    (rig.graph.mcpModel!.projection.document as any).semantic = { nodes: [{ node_id: 'fc', op: 'Linear', params: { out_features: 8 } }], edges: [] };
    const options = (rig.graph as any).agent.options, target = rig.graph.target!, signal = new AbortController().signal;
    assert.equal(await options.prepareWarm('absent', signal), undefined);
    rig.graph.mcpViews.connection('warm-view', true);
    const selection = { type: 'viewSelection' as const, protocolVersion: 1 as const, workspaceSessionId: 'workspace', messageId: 'selection',
      viewId: 'warm-view', target, nodeIds: ['fc'], edgeIds: [] };
    rig.graph.mcpViews.handle(selection);
    assert.deepEqual(await options.prepareWarm('warm-view', signal), await options.prepare(target, ['fc'], undefined, signal, undefined, 'warm-view'));
    rig.graph.mcpViews.handle({ ...selection, target: { ...target, sourceDigest: 'f'.repeat(64) } });
    assert.equal(await options.prepareWarm('warm-view', signal), undefined);
    rig.graph.mcpViews.handle({ ...selection, tooLarge: true });
    assert.equal(await options.prepareWarm('warm-view', signal), undefined);
    rig.graph.mcpViews.handle(selection);
    Object.defineProperty((rig.graph as any).sourceDrafts, 'current', { get: () => ({ id: 'draft' }), configurable: true });
    assert.equal(await options.prepareWarm('warm-view', signal), undefined);
    Object.defineProperty((rig.graph as any).sourceDrafts, 'current', { get: () => undefined });
    openStubEditor(RESID, SAVED + '\n# changed');
    assert.equal(await options.prepareWarm('warm-view', signal), undefined);
  });
  test('Use Agent starts the attached opening without an MCP guide or an existing graph projection', async () => {
    const runs: any[] = [];
    const rig = makeGraph([], { profiles: {} as never,
      trace: { worker: { run: async () => { throw new Error('No capture requested by the button itself.'); }, cancel() {} },
        trusted: async () => true, settings: () => ({ onSave: false, tier: 'real' }) },
      agent: { run: async input => { runs.push(input); }, clear() {} } });
    rig.graph.opening!.open = async target => {
      rig.graph.opening!.sourceOnly(target);
      Object.assign(rig.graph.opening!.state!, { phase: 'needs_setup', requirements: ['init.num_heads: required'] });
    };
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    Object.defineProperty(rig.graph, 'mcpModel', { get: () => undefined });
    const opening = rig.graph.opening!.state!;
    const message = { type: 'graphSetupRequest', openingId: opening.id, requestId: 'use-agent', target: opening.target, action: { kind: 'agent_help', draft: { profile: { kind: 'model', init: { args: [], kwargs: { num_heads: 12 } } } } } };
    rig.graph.handleCanvasMessage({ ...message, openingId: 'stale' } as never); await settle(); assert.equal(runs.length, 0);
    rig.graph.handleCanvasMessage(message as never); await settle();
    assert.equal(runs.length, 1); assert.equal(runs[0].intent, 'open_graph');
    assert.match(runs[0].context, /init.num_heads: required/); assert.match(runs[0].context, /class Net/);
    assert.match(runs[0].context, /"num_heads":12/);
    assert.match(runs[0].context, /"workflow":/); assert.match(runs[0].context, /"nextStep":\{"kind":"inputs"/);
    assert.equal(runs[0].cwd, ROOT);
    rig.graph.handleCanvasMessage(message as never); await settle(); assert.equal(runs.length, 1, 'a replayed setup action cannot start another paid turn');
    const complete = (rig.graph as any).agent.options.openingComplete;
    const currentOpening = rig.graph.opening!.state!;
    assert.equal(complete(opening.target), false);
    Object.assign(currentOpening, { phase: 'failed', hasCapture: true }); assert.equal(complete(opening.target), false, 'a partial capture is not success');
    Object.assign(currentOpening, { phase: 'capturing' }); assert.equal(complete(opening.target), false, 'a running capture is not a completed graph');
    Object.assign(currentOpening, { phase: 'resolving' }); assert.equal(complete(opening.target), false, 'preparing unknown inputs is not completed setup');
    const execution = (rig.graph as any).executionReview;
    execution.pending = { target: opening.target }; execution.value = { phase: 'review', kind: 'capture' };
    Object.assign(currentOpening, { phase: 'awaiting_execution' });
    assert.equal(complete(opening.target), false, 'waiting for Run is an explicit blocker, not a completed graph');
    execution.value.kind = 'prepare'; assert.equal(complete(opening.target), false, 'preparing unknown inputs is not completed setup');
    execution.pending = undefined; execution.value = undefined;
    Object.assign(currentOpening, { phase: 'ready' }); assert.equal(complete(opening.target), true);
    assert.equal(complete({ ...opening.target, cls: 'Other' }), false);
    openStubEditor(RESID, SAVED + '\n# changed');
    assert.equal(complete(opening.target), false, 'measurements for previous source cannot complete this request');
  });

  test('same-turn setup progress waits for host updates and rejects aborted, replaced or changed-source openings', async () => {
    const rig = makeGraph([], { profiles: {} as never,
      trace: { worker: { run: async () => { throw new Error('No execution.'); }, cancel() {} },
        trusted: async () => true, settings: () => ({ onSave: false, tier: 'real' }) },
      agent: { run: async () => {}, clear() {} } });
    rig.graph.opening!.open = async target => {
      rig.graph.opening!.sourceOnly(target); Object.assign(rig.graph.opening!.state!, { phase: 'resolving' });
    };
    openStubEditor(RESID, SAVED); await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    const opening = rig.graph.opening!, target = opening.state!.target;
    const progress = (rig.graph as any).agent.options.openingProgress;
    const signal = new AbortController().signal;
    assert.equal((await progress(target, signal)).id, opening.state!.id);
    let settled = false;
    const reading = progress(target, signal, 1000); void reading.then(() => { settled = true; });
    await new Promise(resolve => setTimeout(resolve, 20)); assert.equal(settled, false);
    opening.cancel('Stopped setup.'); assert.equal((await reading).phase, 'cancelled');
    Object.assign(opening.state!, { phase: 'resolving' });
    const replaced = progress(target, signal, 1000);
    opening.sourceOnly(target); assert.equal(await replaced, undefined, 'same-source replacement still has a new opening identity');
    Object.assign(opening.state!, { phase: 'resolving' });
    const controller = new AbortController(), aborted = progress(target, controller.signal, 1000);
    controller.abort(); await assert.rejects(aborted, /abort/i);
    openStubEditor(RESID, SAVED + '\n# changed');
    assert.equal(await progress(target, signal), undefined);
  });

  test('Use Agent accepts an unchanged buffer whose editor version restarted', async () => {
    const runs: any[] = [];
    const rig = makeGraph([], { profiles: {} as never,
      trace: { worker: { run: async () => { throw new Error('No capture requested by the button itself.'); }, cancel() {} },
        trusted: async () => true, settings: () => ({ onSave: false, tier: 'real' }) },
      agent: { run: async input => { runs.push(input); }, clear() {} } });
    rig.graph.opening!.open = async target => {
      rig.graph.opening!.sourceOnly(target);
      Object.assign(rig.graph.opening!.state!, { phase: 'needs_setup', requirements: ['init.num_heads: required'] });
    };
    openStubEditor(RESID, SAVED); openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    Object.defineProperty(rig.graph, 'mcpModel', { get: () => undefined });
    const opening = rig.graph.opening!.state!;
    assert.equal(opening.target.documentVersion, 2);
    // VS Code reloaded the unchanged buffer: version 1, no change event, same digest.
    editorState.versions.set(RESID, 1);
    rig.graph.handleCanvasMessage({ type: 'graphSetupRequest', openingId: opening.id, requestId: 'use-agent',
      target: opening.target, action: { kind: 'agent_help' } } as never);
    await settle();
    assert.equal(runs.length, 1, (rig.graph as any).agent.state.activity);
    assert.equal(runs[0].intent, 'open_graph');
  });

  test('setup source links accept only evidence of the current opening inside its workspace', async () => {
    const rig = makeGraph([], { profiles: {} as never,
      trace: { worker: { run: async () => { throw new Error('No execution.'); }, cancel() {} },
        trusted: async () => true, settings: () => ({ onSave: false, tier: 'real' }) } });
    rig.graph.opening!.open = async target => {
      rig.graph.opening!.sourceOnly(target);
      Object.assign(rig.graph.opening!.state!, { phase: 'needs_setup', discovery: {
        model: 'models.resid:Net', sourceDigest: target.sourceDigest,
        init: [], inputs: [{ name: 'x', kind: 'keyword', required: true, source: { file: 'models/resid.py', line: 5 } }],
        configs: [], factories: [], examples: [], parents: [], constraints: [], truncated: false } });
    };
    openStubEditor(RESID, SAVED); await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    const opening = rig.graph.opening!.state!, reveals: any[] = [];
    const show = mock.method(window as any, 'showTextDocument', async (doc: any, options: any) => { reveals.push({ doc, options }); return residEditor(); });
    const send = async (id: string, action: any, openingId = opening.id) => {
      rig.graph.handleCanvasMessage({ type: 'graphSetupRequest', openingId, requestId: id, target: opening.target, action } as never); await settle();
    };
    try {
      const valid = { kind: 'reveal_setup_source', file: 'models/resid.py', line: 5 };
      await send('old', valid, 'obsolete');
      await send('unlisted', { ...valid, line: 6 });
      await send('outside', { ...valid, file: '../private.py' });
      assert.equal(reveals.length, 0);
      await send('declared', valid);
      assert.equal(reveals.length, 1); assert.equal(reveals[0].doc.uri.fsPath, RESID);
      assert.equal(reveals[0].options.selection.start.line, 4);
      openStubEditor(RESID, SAVED + '# changed');
      await send('changed', valid); assert.equal(reveals.length, 1);
    } finally { show.mock.restore(); }
  });

  test('agent context includes verified selected facts without weakening stale-revision checks', async () => {
    let service!: McpGraphService;
    const rig = makeGraph([], { agent: { run: async () => {}, clear() {}, reference: model => service.reference(model),
      selection: (reference, ids) => service.inspectNodes(reference, ids) } });
    service = new McpGraphService('agent-context', () => rig.graph.mcpModel);
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    (rig.graph.mcpModel!.projection.document as any).semantic = { nodes: [{ node_id: 'fc', op: 'Linear', params: { out_features: 8 } }], edges: [] };
    const prepare = (rig.graph as any).agent.options.prepare;
    const prepared = await prepare(rig.graph.target!, ['fc'], undefined, new AbortController().signal);
    const attached = JSON.parse(prepared.context.split('\n')[1]);
    const ref = service.reference(rig.graph.mcpModel!);
    assert.equal(attached.modelId, ref.modelId); assert.equal(attached.revision, ref.revision);
    assert.deepEqual(attached.selectionInspection, JSON.parse(JSON.stringify(service.inspectNodes(ref, ['fc']))));
    assert.deepEqual(JSON.parse(JSON.stringify(prepared.selectionEvidence)), attached.selectionInspection,
      'retained evidence must be exactly the verified selection sent to the agent');
    assert.equal(attached.selectionInspection.nodes[0].parameters.out_features, 8);
    assert.match(prepared.context, /Use the attached modelId and revision directly/);
    assert.doesNotMatch(prepared.context, /Call list_models to obtain/);
    assert.doesNotThrow(() => service.require(ref));
    rig.graph.mcpModel!.projection.projectionDigest = 'new-projection';
    assert.throws(() => service.require(ref), /revision changed/);
    (rig.graph.mcpModel!.projection.document as any).semantic.nodes[0].params.out_features = 16;
    const refreshed = JSON.parse((await prepare(rig.graph.target!, ['fc'], undefined, new AbortController().signal)).context.split('\n')[1]);
    assert.notEqual(refreshed.revision, ref.revision);
    assert.equal(refreshed.selectionInspection.nodes[0].parameters.out_features, 16, 'a new Send reads fresh facts, not the previous prompt');
    assert.doesNotThrow(() => service.require(refreshed));
    await assert.rejects(prepare({ ...rig.graph.target!, sourceDigest: 'f'.repeat(64) }, [], undefined, new AbortController().signal), /changed/);
    openStubEditor(RESID, SAVED + '\n# changed');
    await assert.rejects(prepare(rig.graph.target!, ['fc'], undefined, new AbortController().signal), /changed/);
  });
  test('empty and oversized selections use the existing inspection tool path', async () => {
    let service!: McpGraphService, inspections = 0;
    const rig = makeGraph([], { agent: { run: async () => {}, clear() {}, reference: model => service.reference(model),
      selection: (reference, ids) => { inspections++; return service.inspectNodes(reference, ids); } } });
    service = new McpGraphService('agent-context', () => rig.graph.mcpModel);
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    const semantic = { nodes: [{ node_id: 'fc', op: 'Linear', params: { description: '界'.repeat(12_000) } }], edges: [] };
    (rig.graph.mcpModel!.projection.document as any).semantic = semantic;
    const prepare = (rig.graph as any).agent.options.prepare;
    const context = async (ids: string[]) => JSON.parse((await prepare(rig.graph.target!, ids, undefined, new AbortController().signal)).context.split('\n')[1]);
    assert.equal((await context([])).selectionInspection, undefined);
    assert.equal(inspections, 0);
    assert.equal((await context(['fc'])).selectionInspection, undefined, 'the budget is bytes, including multibyte text');
    assert.equal(inspections, 1);
    semantic.nodes[0].params.description = 'small';
    assert.ok((await context(['fc'])).selectionInspection, 'only the oversized attachment is omitted');
  });
  test('chat opening prepares the connected view with the same checks as Send and never grants a draft revision', async () => {
    let service!: McpGraphService;
    const rig = makeGraph([], { agent: { run: async () => {}, clear() {}, reference: model => service.reference(model),
      selection: (reference, ids) => service.inspectNodes(reference, ids) } });
    service = new McpGraphService('preparation', () => rig.graph.mcpModel);
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    (rig.graph.mcpModel!.projection.document as any).semantic = { nodes: [{ node_id: 'fc', op: 'Linear', params: { out_features: 8 } }], edges: [] };
    const options = (rig.graph as any).agent.options, target = rig.graph.target!, signal = new AbortController().signal;
    assert.equal(await options.prepareWarm('absent', signal), undefined);
    rig.graph.mcpViews.connection('warm-view', true);
    const selection = { type: 'viewSelection' as const, protocolVersion: 1 as const, workspaceSessionId: 'workspace', messageId: 'selection',
      viewId: 'warm-view', target, nodeIds: ['fc'], edgeIds: [] };
    rig.graph.mcpViews.handle(selection);
    assert.deepEqual(await options.prepareWarm('warm-view', signal), await options.prepare(target, ['fc'], undefined, signal, undefined, 'warm-view'));
    rig.graph.mcpViews.handle({ ...selection, nodeIds: ['previous:call'] });
    assert.equal(await options.prepareWarm('warm-view', signal), undefined, 'historical call selections cannot start preparation');
    rig.graph.mcpViews.handle({ ...selection, target: { ...target, sourceDigest: 'f'.repeat(64) } });
    assert.equal(await options.prepareWarm('warm-view', signal), undefined);
    rig.graph.mcpViews.handle({ ...selection, tooLarge: true });
    assert.equal(await options.prepareWarm('warm-view', signal), undefined);
    rig.graph.mcpViews.handle(selection);
    Object.defineProperty((rig.graph as any).sourceDrafts, 'current', { get: () => ({ id: 'draft' }), configurable: true });
    assert.equal(await options.prepareWarm('warm-view', signal), undefined);
    Object.defineProperty((rig.graph as any).sourceDrafts, 'current', { get: () => undefined });
    openStubEditor(RESID, SAVED + '\n# changed');
    assert.equal(await options.prepareWarm('warm-view', signal), undefined);
  });
  test('only a current draft follow-up gets pending source and draft-node context', async () => {
    const rig = makeGraph([], { agent: { run: async () => {}, clear() {}, reference: () => ({ modelId: 'context', revision: 'a'.repeat(64) }),
      modelContext: () => { throw new Error('Base measurements must not describe a proposed draft'); },
      selection: () => { throw new Error('Draft or historical selections must not be attached as live facts'); } } });
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    const target = rig.graph.target!, signal = new AbortController().signal;
    let draft: any = { id: 'draft-1', proposal: { id: 'draft-1' }, stale: false,
      before: { document: { semantic: { nodes: [] } }, spanmap: {} },
      after: { document: { semantic: { nodes: [{ node_id: 'fc2' }] } }, spanmap: {} } };
    Object.defineProperty((rig.graph as any).sourceDrafts, 'current', { get: () => draft });
    const grants: unknown[][] = [];
    mock.method(rig.graph.mcpProposals, 'beginRevision', (...args: unknown[]) => {
      grants.push(args); return { draftId: 'draft-1', files: [{ file: target.file, baseDigest: target.sourceDigest }] };
    });
    const prepare = (rig.graph as any).agent.options.prepare;
    await assert.rejects(prepare(target, [], undefined, signal), /Refresh the graph/);
    await assert.rejects(prepare(target, [], 'old-id', signal), /Refresh the graph/);
    await assert.rejects(prepare(target, ['review:after:invented'], 'draft-1', signal), /no longer in this graph/);
    const prepared = await prepare(target, ['review:after:fc2'], 'draft-1', signal);
    assert.equal(JSON.parse(prepared.context.split('\n')[1]).selectionInspection, undefined);
    assert.deepEqual(JSON.parse(prepared.context.split('\n')[1]).pendingDraft, { draftId: 'draft-1',
      files: [{ file: target.file, baseDigest: target.sourceDigest }] });
    assert.match(prepared.context, /propose_source_set with EVERY revised pending file/);
    assert.doesNotMatch(prepared.context, /propose_code_edit/);
    assert.deepEqual(grants, [['draft-1', signal]]);
    draft = { ...draft, proposal: { ...draft.proposal, canRevise: false } };
    const inspection = await prepare(target, ['review:after:fc2'], 'draft-1', signal);
    const inspected = JSON.parse(inspection.context.split('\n')[1]);
    assert.equal(inspected.pendingDraft, undefined);
    assert.deepEqual(inspected.reviewedDraft, { id: 'draft-1', canRevise: false });
    assert.match(inspection.context, /inspection only/);
    assert.doesNotMatch(inspection.context, /propose_source_set/);
    assert.equal(grants.length, 1, 'a typed draft cannot grant workspace-code revision authority');
    const sourceDocument = await vscode.workspace.openTextDocument(vscode.Uri.file(RESID));
    const read = mock.method(vscode.workspace, 'openTextDocument', async () => {
      draft = { ...draft, id: 'replacement' }; return sourceDocument;
    });
    await assert.rejects(prepare(target, [], 'draft-1', signal), /draft changed or closed/i);
    read.mock.restore();
    draft = { ...draft, id: 'draft-1' };
    draft = { ...draft, stale: true };
    await assert.rejects(prepare(target, [], 'draft-1', signal), /Refresh the graph/);
    assert.equal(grants.length, 1, 'stale or invented context cannot authorize replacing a native review');
  });

  test('a draft revision accepts host-owned captured context and rejects invented call IDs', async () => {
    const rig = makeGraph([], { agent: { run: async () => {}, clear() {} } });
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    const target = rig.graph.target!, signal = new AbortController().signal;
    const model = rig.graph.mcpModel!;
    (model.projection.document as any).semantic = { nodes: [{ node_id: 'captured:relu:1' }], metadata: { reviewClass: { structureDigest: 'same', dependencyScope: 'partial', dependencyDigests: {} } } };
    model.projection.debug = { nodePairs: [], changed: [], previousDocument: { semantic: {
      nodes: [{ node_id: 'retained:relu:2' }], metadata: { capture: { id: 'run', status: 'complete' }, reviewClass: { structureDigest: 'same', dependencyScope: 'partial', dependencyDigests: {} } },
    } } };
    const draft = { id: 'mapped', stale: false, before: { document: { semantic: { nodes: [] } } },
      after: { document: { semantic: { nodes: [] } } } };
    Object.defineProperty((rig.graph as any).sourceDrafts, 'current', { get: () => draft });
    mock.method(rig.graph.mcpProposals, 'beginRevision', () => ({ draftId: 'mapped', files: [{ file: target.file, baseDigest: target.sourceDigest }] }));
    const prepare = (rig.graph as any).agent.options.prepare;
    await prepare(target, ['captured:relu:1', 'retained:relu:2'], 'mapped', signal);
    await assert.rejects(prepare(target, ['captured:invented'], 'mapped', signal), /no longer in this graph/);
    (model.projection.debug.previousDocument as any).semantic.metadata.reviewClass.structureDigest = 'stale';
    await assert.rejects(prepare(target, ['retained:relu:2'], 'mapped', signal), /no longer in this graph/);
  });

  test('after Accept an attested previous call remains usable for the next agent message', async () => {
    const rig = makeGraph([], { agent: { run: async () => {}, clear() {}, reference: () => ({ modelId: 'context', revision: 'a'.repeat(64) }),
      modelContext: () => { throw new Error('Current measurements must not describe historical selections'); },
      selection: () => { throw new Error('Draft or historical selections must not be attached as live facts'); } } });
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    const target = rig.graph.target!, signal = new AbortController().signal;
    const model = rig.graph.mcpModel!;
    const evidence = { structureDigest: 'same', dependencyScope: 'partial', dependencyDigests: { 'block.py': 'unchanged' } };
    (model.projection.document as any).semantic = { nodes: [{ node_id: 'current:source' }], metadata: { reviewClass: { ...evidence } } };
    model.projection.debug = { nodePairs: [], changed: [], previousDocument: { semantic: {
      nodes: [{ node_id: 'previous:relu:2', op: 'torch.nn.GELU', metadata: { captureLabel: 'blocks.0.relu · call 2/3' } }, { node_id: 'current:source' }], metadata: { capture: { id: 'run', status: 'complete' }, reviewClass: evidence },
    } } };
    const prepare = (rig.graph as any).agent.options.prepare;
    const prepared = await prepare(target, ['current:source', 'previous:relu:2'], undefined, signal);
    assert.equal(JSON.parse(prepared.context.split('\n')[1]).selectionInspection, undefined);
    assert.match(prepared.context, /"historicalNodeIds":\["previous:relu:2"\]/);
    assert.match(prepared.context, /do not prove the current source has run/);
    assert.match(prepared.context, /blocks.0.relu · call 2\/3/);
    assert.match(prepared.context, /torch.nn.GELU/);
    await assert.rejects(prepare(target, ['previous:invented'], undefined, signal), /no longer in this graph/);
    await assert.rejects(prepare(target, ['previous:relu:2', 'previous:relu:2'], undefined, signal), /no longer in this graph/);
    for (const changed of [
      { ...evidence, structureDigest: 'different' },
      { ...evidence, dependencyDigests: { 'block.py': 'changed' } },
      { ...evidence, dependencyDigests: {} },
    ]) {
      (model.projection.document as any).semantic.metadata.reviewClass = changed;
      await assert.rejects(prepare(target, ['previous:relu:2'], undefined, signal), /no longer in this graph/);
    }
    const metadata = (model.projection.debug.previousDocument as any).semantic.metadata;
    (model.projection.document as any).semantic.metadata.reviewClass = evidence;
    for (const capture of [{ id: 'run', status: 'partial' }, { id: 'run', status: 'failed' }, undefined]) {
      metadata.capture = capture;
      await assert.rejects(prepare(target, ['previous:relu:2'], undefined, signal), /no longer in this graph/);
    }
    metadata.capture = { id: 'run', status: 'complete' };
    metadata.reviewClass = { ...evidence, dependencyDigests: {} };
    (model.projection.document as any).semantic.metadata.reviewClass = metadata.reviewClass;
    await prepare(target, ['previous:relu:2'], undefined, signal);
    (model.projection.document as any).semantic.metadata.reviewClass = evidence;
    (model.projection.debug.previousDocument as any).semantic.metadata.reviewClass = { structureDigest: 'same' };
    await assert.rejects(prepare(target, ['previous:relu:2'], undefined, signal), /no longer in this graph/);
  });

  test('closing a source draft restores an overlapping Git review instead of stranding the canvas', async () => {
    const digest = (text: string) => createHash('sha256').update(text).digest('hex');
    const rig = makeGraph([]);
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    const posts: any[] = [];
    rig.graph.attachCanvas(message => { posts.push(message); return true; });
    const previous = { id: 'existing-git-review' };
    (rig.graph as any).graphReview = { active: true, current: previous, dispose() {} };
    const target = rig.graph.target!;
    const after = SAVED + '# proposed change\n';
    const handle = (rig.graph as any).sourceDrafts.start({ id: 'draft-1', target,
      artifacts: [{ path: target.file, before: SAVED, after, before_digest: digest(SAVED), after_digest: digest(after), edits: [] }],
      sources: { entries: [{ file: target.file, source: SAVED, digest: digest(SAVED) }], truncated: false },
    }, new AbortController().signal, () => true);
    handle.show(async () => {}, () => {}); await settle();
    assert.equal(posts.filter(message => message.type === 'graphReviewState').at(-1).state.id, 'draft-1');
    handle.dispose();
    assert.equal(posts.filter(message => message.type === 'graphReviewState').at(-1).state, previous);
    assert.equal(rig.graph.reviewState, previous);
    const hidden = (rig.graph as any).sourceDrafts.start({ id: 'draft-2', target,
      artifacts: [{ path: target.file, before: SAVED, after, before_digest: digest(SAVED), after_digest: digest(after), edits: [] }],
      sources: { entries: [{ file: target.file, source: SAVED, digest: digest(SAVED) }], truncated: false },
    }, new AbortController().signal, () => true);
    const updated = { id: 'git-review-finished-opening' };
    (rig.graph as any).graphReview.current = updated;
    hidden.dispose(); // Native review never reached show(). Its suppressed Git update still needs replay.
    assert.equal(posts.filter(message => message.type === 'graphReviewState').at(-1).state, updated);
  });

  test('a displayed source draft clears code focus, as starting a Git review does', async () => {
    // Review hides the Code focus control, so live-graph decorations must not outlast the switch.
    const digest = (text: string) => createHash('sha256').update(text).digest('hex');
    const rig = makeGraph([]);
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    rig.graph.attachCanvas(() => true);
    const cleared = mock.method((rig.graph as any).codeFocus, 'clear');
    const target = rig.graph.target!;
    const after = SAVED + '# proposed change\n';
    const handle = (rig.graph as any).sourceDrafts.start({ id: 'draft-1', target,
      artifacts: [{ path: target.file, before: SAVED, after, before_digest: digest(SAVED), after_digest: digest(after), edits: [] }],
      sources: { entries: [{ file: target.file, source: SAVED, digest: digest(SAVED) }], truncated: false },
    }, new AbortController().signal, () => true);
    await settle();
    assert.equal(cleared.mock.callCount(), 0, 'an unrevealed draft shows nothing and keeps focus');
    handle.show(async () => {}); await settle();
    assert.ok(cleared.mock.callCount() > 0);
    handle.dispose();
  });

  for (const surface of ['browser', 'canvas'] as const) test(`${surface} opens a nested Python project with its own source and profile root`, async () => {
    const workspace = await mkdtemp(path.join(tmpdir(), 'tensorviz-nested-'));
    const project = path.join(workspace, 'examples', 'research');
    await mkdir(path.join(project, 'models'), { recursive: true });
    await writeFile(path.join(project, 'tensorviz.toml'), 'version = 1\n');
    state.workspaceFolders = [{ uri: { fsPath: workspace } }];
    const file = path.join(project, 'models', 'resid.py');
    openStubEditor(file, SAVED);
    openStubEditor(path.join(project, 'models', 'block.py'), 'class Block: pass\n');
    for (let i = 0; i < 300; i++) openStubEditor(path.join(workspace, 'aaa', `noise_${i}.py`), 'x = 1\n');
    const rig = makeGraph([]);
    try {
      await rig.graph.open({ uri: file, cls: 'Net' }, surface); await settle();
      assert.equal(rig.graph.target?.workspaceFolder, project, 'setup and execution must use the same root as opening this project alone');
      assert.equal(rig.graph.target?.file, 'models/resid.py', 'the parent directory is not a Python module prefix');
      const body = rig.projectionCalls.at(-1)!.body;
      assert.deepEqual(body.sources.map(entry => entry.file), ['models/resid.py', 'models/block.py']);
      assert.equal(body.truncated, false, 'unrelated projects must not consume this model source budget');
    } finally {
      rig.graph.dispose();
      await rm(workspace, { recursive: true, force: true });
    }
  });

  for (const surface of ['browser', 'canvas'] as const)
  for (const outcome of ['success', 'source-change', 'cancel', 'eviction', 'reconnect'] as const) test(`${surface} portable export: ${outcome} while preparation is in flight`, async () => {
    const rig = makeGraph([]);
    openStubEditor(RESID, SAVED);
    const native: any[] = [];
    rig.graph.attachCanvas(message => { native.push(message); });
    await rig.graph.open({ uri: RESID, cls: 'Net' }, surface); await settle();
    const send = (message: any) => surface === 'canvas' ? rig.graph.handleCanvasMessage(message) : rig.transports[0].fire(message);
    const published = () => surface === 'canvas' ? native : rig.transports[0].published;
    const target = { ...rig.graph.target! }, { capture } = canonicalFixture();
    const reference = { captureId: capture.id, sourceId: capture.source.id };
    const initial = rig.graph.captureInspector.list(target);
    let displayed: string | null = capture.id;
    rig.graph.captureInspector.list = () => ({ ...initial, displayedCaptureId: displayed });
    rig.graph.captureInspector.retained = (actual, id, source) => {
      assert.deepEqual(actual, target); assert.equal(id, capture.id); assert.equal(source, capture.source.id); return capture;
    };
    const options = (rig.graph as unknown as { options: LocalBrowserGraphOptions }).options;
    const originalFetch = options.fetchImpl!;
    let release!: (response: Response) => void, signal!: AbortSignal;
    let preparations = 0;
    options.fetchImpl = (url, init) => {
      if (!String(url).endsWith('/capture/portable')) return originalFetch(url, init);
      preparations++;
      assert.equal((init?.headers as Record<string, string>)['X-tensorViz-Token'], 'test-token');
      assert.deepEqual(JSON.parse(init!.body as string), capture);
      signal = init!.signal!;
      return new Promise<Response>(resolve => { release = resolve; });
    };
    send({ type: 'captureExportRequest', target: { ...target, backendRevision: 999 }, requestId: 'stale', reference });
    assert.equal(preparations, 0);
    assert.equal(published().at(-1)!.capture, null);
    send({ type: 'captureExportRequest', target, requestId: 'export', reference });
    assert.equal(preparations, 1);
    // An unrelated cancellation must not abort this request.
    send({ type: 'captureExportCancel', target, requestId: 'foreign' });
    assert.equal(signal.aborted, false);
    if (outcome === 'source-change') {
      openStubEditor(RESID, SAVED + '\n# changed source\n');
      await rig.graph.open({ uri: RESID, cls: 'Net' }); await settle();
    } else if (outcome === 'cancel') {
      send({ type: 'captureExportCancel', target, requestId: 'export' });
      assert.equal(signal.aborted, true);
    } else if (outcome === 'eviction') displayed = null;
    else if (outcome === 'reconnect') {
      if (surface === 'canvas') {
        await rig.graph.openSelectedInBrowser();
        rig.transports[0].setPhase('reconnecting');
        assert.equal(signal.aborted, false, 'browser lifecycle cannot cancel native export');
      } else (rig.transports[0].status as { epoch: number }).epoch++;
    }
    const portable = JSON.parse(readFileSync(path.join(__dirname, '../../../../tests/fixtures/contracts/portable-capture.json'), 'utf8'));
    release(new Response(JSON.stringify(portable))); await settle();
    const result = published().find(message => message.type === 'captureExportResult' && message.requestId === 'export');
    if (outcome === 'cancel' || (outcome === 'reconnect' && surface === 'browser')) assert.equal(result, undefined);
    else {
      assert.ok(result);
      assert.deepEqual(result.capture, outcome === 'success' || (outcome === 'reconnect' && surface === 'canvas') ? portable : null);
      assert.deepEqual(result.reference, reference);
    }
  });

  for (const surface of ['browser', 'canvas'] as const) test(`${surface} captured-detail reads refuse a different model`, async () => {
    const rig = makeGraph([]);
    openStubEditor(RESID, SAVED);
    const native: any[] = [];
    rig.graph.attachCanvas(message => { native.push(message); });
    await rig.graph.open({ uri: RESID, cls: 'Net' }, surface);
    await settle();
    const send = (message: any) => surface === 'canvas' ? rig.graph.handleCanvasMessage(message) : rig.transports[0].fire(message);
    const published = () => surface === 'canvas' ? native : rig.transports[0].published;
    const target = rig.graph.target!;
    const query = { captureId: 'capture:1', sourceId: 'source:1', path: ['operations'], offset: 0, limit: 20 };
    const page = { captureId: query.captureId, sourceId: query.sourceId, path: query.path,
      node: { key: null, kind: 'array' as const, count: 0 }, items: [], total: 0, nextOffset: null };
    let inspected = 0;
    rig.graph.captureInspector.inspect = (active, requested) => {
      inspected++; assert.deepEqual(active, target); assert.deepEqual(requested, query); return page;
    };
    // Retained evidence can be read even when the requesting view's source is old.
    const oldTarget = { ...target, documentVersion: target.documentVersion - 1 };
    send({ type: 'captureInspectRequest', target: oldTarget, requestId: 'read-one', query });
    assert.deepEqual(published().at(-1), { type: 'captureInspectResult', target: oldTarget,
      requestId: 'read-one', page, detail: 'Retained capture evidence.' });
    send({ type: 'captureInspectRequest', target: { ...target, cls: 'Other' }, requestId: 'foreign', query });
    assert.equal(inspected, 1);
    assert.equal(published().at(-1)!.page, null);
  });

  setup(() => {
    reset();
    resetEditors();
    state.workspaceFolders = [{ uri: { fsPath: ROOT } }];
  });

  teardown(() => {
    for (const rig of made.splice(0)) {
      rig.graph.dispose();
    }
  });

  test('native canvas shares the current projection controller without minting a browser session or executing', async () => {
    let executions = 0;
    const rig = makeGraph([], { trace: {
      worker: { cancel() {}, trace: async () => { executions++; throw new Error('opening must be static'); } } as never,
      trusted: async () => true, settings: () => ({ onSave: false, tier: 'real' }),
    } });
    openStubEditor(RESID, SAVED);
    const messages: any[] = [];
    const attachment = rig.graph.attachCanvas(message => { messages.push(message); });
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    assert.equal(rig.mintCalls.length, 0);
    assert.equal(rig.opened.length, 0);
    assert.equal(executions, 0);
    assert.equal(messages[0].type, 'activeTarget', 'the real renderer refuses projections until the target is announced');
    assert.equal(messages[0].target.cls, 'Net');
    assert.equal(messages.filter(message => message.type === 'projection').length, 1);
    const selected = rig.graph.target;
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    assert.deepEqual(rig.graph.target, selected);
    assert.equal(rig.projectionCalls.length, 1, 'reselecting the same target preserves measurements and review');
    attachment.dispose();
    const restored: any[] = [];
    rig.graph.attachCanvas(message => { restored.push(message); });
    rig.graph.replayCanvas();
    assert.deepEqual(restored[0].state.activeTarget, selected);
    assert.equal(rig.projectionCalls.length, 1, 'view recreation replays instead of parsing/executing');
  });

  test('TOML watcher follows only the active root and never starts capture', async () => {
    let exists = false, reads = 0, openings = 0;
    const rig = makeGraph([], { profiles: { hasConfig: async () => { reads++; return exists; } } as never,
      trace: { worker: { cancel() {} } as never, trusted: async () => true, settings: () => ({ onSave: false, tier: 'real' }) } });
    rig.graph.opening!.open = async target => { openings++; rig.graph.opening!.sourceOnly(target); };
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    assert.equal(rig.graph.opening!.state!.configFileExists, false);
    const root = rig.graph.opening!.state!.target.workspaceFolder;
    const before = reads;
    configFileChanges.fire({ kind: 'create', uri: Uri.file('/other/tensorviz.toml') });
    await settle(); assert.equal(reads, before);
    exists = true;
    configFileChanges.fire({ kind: 'create', uri: Uri.file(path.join(root, 'tensorviz.toml')) });
    await settle(); assert.equal(rig.graph.opening!.state!.configFileExists, true);
    exists = false;
    configFileChanges.fire({ kind: 'delete', uri: Uri.file(path.join(root, 'tensorviz.toml')) });
    await settle(); assert.equal(rig.graph.opening!.state!.configFileExists, false);
    assert.equal(openings, 1, 'file events update evidence without reopening or executing');
    rig.graph.dispose();
    const disposed = reads;
    configFileChanges.fire({ kind: 'change', uri: Uri.file(path.join(root, 'tensorviz.toml')) });
    await settle(); assert.equal(reads, disposed);
  });

  test('declining model libraries blocks execution with the install reason and a retry path', async () => {
    let profileReads = 0;
    const rig = makeGraph([], { profiles: { snapshot: async () => { profileReads++; throw new Error('must not execute'); } } as never,
      prepareDependencies: async () => { throw new GraphToolError('INSTALL_DECLINED', 'Dependency installation was not approved.'); },
      trace: { worker: { cancel() {} } as never, trusted: async () => true, settings: () => ({ onSave: false, tier: 'real' }) } });
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    assert.equal(profileReads, 0);
    assert.equal(rig.graph.opening!.state?.phase, 'failed');
    assert.equal(rig.graph.opening!.state?.failure?.kind, 'environment');
    assert.match(rig.graph.opening!.state!.detail, /not approved.*Refresh/);
  });

  test('inline environment recovery pauses native opening before profile imports and bypasses the old automatic installer', async () => {
    let profiles = 0, oldInstalls = 0, repairs = 0;
    const environment = { reason: 'missing_libraries' as const, detail: 'Needs SciPy.', requirements: ['scipy'], selectedId: 'project',
      candidates: [{ id: 'project', interpreter: '/project/python', source: 'Saved', status: 'libraries_present' as const, torch: '2.6.0' }] };
    const rig = makeGraph([], { profiles: { snapshot: async () => { profiles++; throw new Error('must wait for environment'); } } as never,
      environment: async () => environment,
      recoverEnvironment: async (_target, actual, action) => { repairs++; assert.deepEqual(actual, environment); assert.equal(action.kind, 'install_dependencies'); return false; },
      prepareDependencies: async () => { oldInstalls++; return true; },
      trace: { worker: { cancel() {} } as never, trusted: async () => true, settings: () => ({ onSave: false, tier: 'real' }) } });
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    const opening = rig.graph.opening!.state!;
    assert.equal(opening.phase, 'needs_setup'); assert.deepEqual(opening.environment, environment);
    assert.equal(profiles, 0); assert.equal(oldInstalls, 0);
    await rig.graph.opening!.action(opening.id, 'repair', opening.target, { kind: 'install_dependencies' });
    assert.equal(repairs, 1); assert.equal(rig.graph.opening!.state!.id, opening.id);
    assert.equal(rig.graph.opening!.state!.phase, 'needs_setup'); assert.equal(profiles, 0);
  });

  for (const cancelled of [false, true]) test(`sidebar creation reaches recovery with a failed source projection${cancelled ? ' after cancellation' : ''}`, async () => {
    resetWebviews();
    let creations = 0;
    const diagnostics: string[] = [];
    let finish!: (accepted: boolean) => void;
    const pending = new Promise<boolean>(resolve => { finish = resolve; });
    const environment = { reason: 'missing_libraries' as const, canCreate: true as const,
      detail: 'This model needs a source dependency.', candidates: [] };
    const base = makeFetch([]);
    const rig = makeGraph([], {
      channel: { appendLine: detail => diagnostics.push(detail) },
      fetchImpl: ((input, init) => String(input).endsWith('/api/ide/multifile/project')
        ? Promise.resolve(new Response(JSON.stringify({ detail: { error: 'syntax_error', message: 'No PyTorch model classes found', line: 4 } }), { status: 400 }))
        : base.fetchImpl(input, init)) as typeof fetch,
      profiles: { snapshot: async () => { throw new Error('model imports must wait for environment setup'); } } as never,
      environment: async () => environment,
      recoverEnvironment: async (_target, actual, requested, signal, progress) => {
        assert.deepEqual(actual, environment); assert.equal(requested.kind, 'create_environment'); assert.equal(signal.aborted, false);
        creations++; progress('Creating a separate Python environment for this model…'); return pending;
      },
      trace: { worker: { cancel() {} } as never, trusted: async () => true, settings: () => ({ onSave: false, tier: 'real' }) },
    });
    const canvas = new SidebarCanvas({ prepare: async () => ({ appUrl: 'http://127.0.0.1:8631/', appOrigin: 'http://127.0.0.1:8631', port: 8631, workspaceDir: ROOT }),
      receive: message => rig.graph.handleCanvasMessage(message), replay: () => rig.graph.replayCanvas(),
      viewConnection: (id, connected) => rig.graph.setCanvasView(id, connected), channel: { appendLine() {} } as never });
    const view = new StubWebviewView();
    const registration = canvas.register(); webviewState.views.push(view);
    canvas.resolveWebviewView(view as never);
    const attached = rig.graph.attachCanvas(message => canvas.post(message));
    try {
      await canvas.show(); view.receive({ type: 'canvasConnect', protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION });
      openStubEditor(RESID, SAVED);
      await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
      const opening = rig.graph.opening!.state!;
      assert.equal(opening.phase, 'needs_setup');
      assert.ok(view.messages.some((message: any) => message.type === 'canvasMessage' && message.message.type === 'syncState'
        && message.message.state === 'stale'));
      if (cancelled) rig.graph.opening!.cancel('Agent stopped.');
      const session = view.messages.find((message: any) => message.type === 'canvasConnected') as any;
      const request = { type: 'canvasMessage', message: { type: 'graphSetupRequest', protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
        workspaceSessionId: session.sessionId, messageId: 'create-message', requestId: 'create-request',
        openingId: opening.id, target: opening.target, action: { kind: 'create_environment' } } };
      view.receive(request); await settle();
      assert.equal(creations, 1, 'the visible action must reach creation, even without a source graph');
      assert.equal(rig.graph.opening!.state!.id, opening.id); assert.equal(rig.graph.opening!.state!.phase, 'resolving');
      assert.ok(view.messages.some((message: any) => message.type === 'canvasMessage' && message.message.type === 'graphOpening'
        && message.message.state.detail === 'Creating a separate Python environment for this model…'), 'progress reaches the same sidebar');
      view.receive({ ...request, message: { ...request.message, messageId: 'duplicate-click', requestId: 'duplicate-click' } }); await settle();
      assert.equal(creations, 1, 'a pending creation cannot be started twice');
      finish(false); await settle(); assert.equal(rig.graph.opening!.state!.phase, 'needs_setup');
      assert.ok(diagnostics.includes(`[model environment] Create requested (phase=${cancelled ? 'cancelled' : 'needs_setup'}, currentOpening=true).`));
      assert.ok(diagnostics.includes('[model environment] Create action finished (accepted=false, phase=needs_setup).'));
    } finally { finish(false); attached.dispose(); canvas.dispose(); registration.dispose(); resetWebviews(); }
  });

  for (const suggestion of ['python-extension', 'managed']) test(`capture on save waits for resolution of a ${suggestion} suggestion`, async () => {
    const root = await realpath(await mkdtemp(path.join(tmpdir(), 'tensorviz-save-environment-')));
    const file = path.join(root, 'models', 'resid.py'), python = path.join(root, 'python');
    await mkdir(path.dirname(file)); await writeFile(file, SAVED);
    state.workspaceFolders = [{ uri: Uri.file(root) }];
    const choices = new Map<string, unknown>();
    const context = { workspaceState: { get: (key: string) => choices.get(key),
      update: async (key: string, value: unknown) => { choices.set(key, value); } } } as unknown as vscode.ExtensionContext;
    if (suggestion === 'python-extension') state.extensions.set('ms-python.python', { activate: async () => ({ environments: {
      getActiveEnvironmentPath: () => ({ path: python }), resolveEnvironment: async () => ({ executable: { uri: Uri.file(python) } }),
    } }) });
    const resolve = () => selectedExecutionEnvironment(context, vscode.Uri.file(root), async () => python);
    const profile = { kind: 'model', model: 'models.resid:Net', mode: 'eval', device: 'cpu', grad_enabled: false,
      init: { args: [], kwargs: {} }, inputs: { args: [{ kind: 'tensor', shape: [1, 8], dtype: 'float32',
        requires_grad: false, generator: { kind: 'full', value: 1 } }], kwargs: {} }, limits: { timeout_ms: 1000 } };
    let prepared = 0, runs = 0;
    let source = SAVED;
    const rig = makeGraph([], {
      environment: async () => ({ reason: savedEnvironment(context, vscode.Uri.file(file)) ? 'ready' : 'selection_required', candidates: [], detail: 'Choose Python.' }),
      profiles: { hasConfig: async () => true, choose: async () => undefined,
        snapshot: async (_root, name) => {
          const selected = await resolve();
          if (!selected.interpreter) throw new Error('No execution interpreter is selected.');
          prepared++;
          return (name ? { guards: {}, resolution: { status: 'resolved', spec: { profile, identity: { profile: name },
            files: [], interpreter: { executable: selected.interpreter }, dependencyScope: 'partial' } } }
            : { profiles: ['one'], profileDefinitions: { one: profile }, guards: {}, resolution: { status: 'needs_setup', requirements: [
              { profile: null, field: ['profiles'], code: 'ambiguous', phase: 'static', reason: 'Choose.', remedy: 'Choose.' },
            ] } }) as never;
        },
      },
      trace: { trusted: async () => true, settings: () => ({ onSave: true, tier: 'real' }), worker: {
        run: async request => {
          runs++; assert.equal(request.expectedInterpreter, python);
          return { ok: true, exitCode: 0, envelope: { protocol_version: 1, result: { graph: { nodes: [], edges: [] },
            diagnostics: { execution: { importedSources: [{ file: 'models/resid.py', realPath: file,
              bytes: Buffer.byteLength(source), digest: createHash('sha256').update(source).digest('hex') }] } } },
            spans: {}, error: null, worker: { python: 'test', torch: 'test', elapsed_ms: 1 } } };
        }, cancel() {},
      } },
    });
    rig.graph.attachCanvas(() => {});
    const save = async () => {
      const previous = rig.graph.opening!.state!.id;
      source += '\n# saved edit\n';
      const editor = openStubEditor(file, source);
      documentChanges.fire({ document: editor.document });
      await writeFile(file, source);
      saveChanges.fire(editor.document);
      for (let i = 0; i < 200 && (rig.graph.opening!.state!.id === previous
        || ['resolving', 'capturing'].includes(rig.graph.opening!.state!.phase)); i++) await settle();
      return rig.graph.opening!.state!;
    };
    openStubEditor(file, SAVED);
    try {
      await rig.graph.open({ uri: file, cls: 'Net' }, 'canvas'); await settle();
      const opening = rig.graph.opening!.state!;
      assert.equal(opening.environment!.reason, 'selection_required');
      assert.equal((await save()).phase, 'needs_setup');
      assert.equal(prepared, 0, 'an unresolved suggestion must not prepare a Python profile');
      assert.equal(runs, 0, 'saving cannot bypass unresolved opening setup');
      assert.notEqual(rig.graph.opening!.state!.id, opening.id, 'a changed save re-enters the opening environment gate');
      assert.equal(rig.graph.opening!.state!.environment!.reason, 'selection_required');
      assert.equal(choices.size, 0, 'saving must not remember a suggestion');
      if (suggestion === 'managed') await rememberEnvironment(context, { kind: 'managed' }, vscode.Uri.file(file));
      else await rememberEnvironment(context, { kind: 'project', interpreter: python }, vscode.Uri.file(file));
      assert.equal((await save()).phase, 'ready');
      assert.equal(runs, 1, 'the same saved profile remains usable after a saved selection');
      assert.ok(prepared > 0);
    } finally { rig.graph.dispose(); await rm(root, { recursive: true, force: true }); }
  });

  for (const selection of ['cancelled', 'ready', 'missing'] as const) test(`native environment browsing (${selection}) uses the original opening and Canvas library flow`, async () => {
    const accepted = selection !== 'cancelled';
    const root = await realpath(await mkdtemp(path.join(tmpdir(), 'tensorviz-browse-opening-')));
    const file = path.join(root, 'models', 'resid.py');
    await mkdir(path.dirname(file)); await writeFile(file, SAVED);
    state.workspaceFolders = [{ uri: { fsPath: root } }];
    let selected = false, installed = false, runs = 0;
    let finishInstall!: () => void;
    const installation = new Promise<void>(resolve => { finishInstall = resolve; });
    const profile = { kind: 'model', model: 'models.resid:Net', mode: 'eval', device: 'cpu', grad_enabled: false,
      init: { args: [], kwargs: {} }, inputs: { args: [{ kind: 'tensor', shape: [1, 8], dtype: 'float32',
        requires_grad: false, generator: { kind: 'full', value: 1 } }], kwargs: {} }, limits: { timeout_ms: 1000 } };
    const rig = makeGraph([], {
      environment: async () => ({ reason: selected ? selection === 'missing' && !installed ? 'missing_libraries' : 'ready' : 'selection_required',
        candidates: selected ? [{ id: 'browsed', interpreter: '/selected/python', source: 'Saved for this folder', selected: true,
          status: 'libraries_present', python: '3.12.2', torch: '2.7.1' }] : [],
        ...(selected ? { selectedId: 'browsed' } : {}), ...(selection === 'missing' && !installed ? { requirements: ['einops'] } : {}),
        detail: selected ? selection === 'missing' && !installed ? 'Required model libraries are missing: einops.' : 'Ready.' : 'Choose Python.' }),
      selectEnvironment: async target => {
        assert.equal(target.file, 'models/resid.py'); selected = accepted; return accepted;
      },
      recoverEnvironment: async (_target, environment, requested, _signal, progress) => {
        assert.equal(requested.kind, 'install_dependencies'); assert.equal(environment.selectedId, 'browsed');
        progress('Downloading and installing the reviewed libraries…');
        await installation; installed = true; return true;
      },
      profiles: {
        snapshot: async (_root, name) => (name
          ? { guards: {}, resolution: { status: 'resolved', spec: { profile, identity: { profile: name },
            files: [], interpreter: { executable: '/selected/python' }, dependencyScope: 'partial' } } }
          : { profiles: ['one'], profileDefinitions: { one: profile }, guards: {}, resolution: { status: 'needs_setup', requirements: [
            { profile: null, field: ['profiles'], code: 'ambiguous', phase: 'static', reason: 'Choose.', remedy: 'Choose.' },
          ] } }) as never,
        choose: async () => undefined,
      },
      trace: { trusted: async () => true, settings: () => ({ onSave: false, tier: 'real' }), worker: {
        run: async request => {
          assert.ok(selected); assert.equal(request.expectedInterpreter, '/selected/python'); runs++;
          return { ok: true, exitCode: 0, envelope: { protocol_version: 1, result: { graph: { nodes: [], edges: [] },
            diagnostics: { execution: { importedSources: [{ file: 'models/resid.py', realPath: file,
              bytes: Buffer.byteLength(SAVED), digest: createHash('sha256').update(SAVED).digest('hex') }] } } },
            spans: {}, error: null, worker: { python: 'test', torch: 'test', elapsed_ms: 1 } } };
        }, cancel() {},
      } },
    });
    try {
      openStubEditor(file, SAVED);
      await rig.graph.open({ uri: file, cls: 'Net' }, 'canvas'); await settle();
      const opening = rig.graph.opening!.state!;
      assert.equal(opening.phase, 'needs_setup');
      const browse = rig.graph.opening!.action(opening.id, 'browse-python', opening.target, { kind: 'environment' });
      await browse;
      assert.equal(rig.graph.opening!.state!.id, opening.id, 'the original MCP/help request must still identify this opening');
      assert.equal(rig.graph.opening!.state!.phase, selection === 'ready' ? 'ready' : 'needs_setup', JSON.stringify(rig.graph.opening!.state));
      assert.equal(runs, selection === 'ready' ? 1 : 0);
      if (selection === 'missing') {
        assert.equal(rig.graph.opening!.state!.environment!.selectedId, 'browsed');
        assert.deepEqual(rig.graph.opening!.state!.environment!.requirements, ['einops']);
        assert.equal(installed, false, 'Browse must select before any installation');
        const repair = rig.graph.opening!.action(opening.id, 'review-libraries', opening.target, { kind: 'install_dependencies' });
        await settle(); assert.equal(rig.graph.opening!.state!.phase, 'resolving');
        assert.equal(rig.graph.opening!.state!.detail, 'Downloading and installing the reviewed libraries…');
        assert.equal(runs, 0); finishInstall(); await repair;
        assert.equal(rig.graph.opening!.state!.id, opening.id); assert.equal(rig.graph.opening!.state!.phase, 'ready'); assert.equal(runs, 1);
      }
      assert.equal(rig.opened.length, 0, 'browsing keeps the installed Canvas surface');
    } finally { finishInstall(); rig.graph.dispose(); await rm(root, { recursive: true, force: true }); }
  });

  for (const sourceOnly of [false, true]) test(`environment change ${sourceOnly ? 'preserves source inspection' : 'resumes the retained model without another click'}`, async () => {
    const rig = makeGraph([], { profiles: {} as never,
      trace: { worker: { cancel() {} } as never, trusted: async () => true, settings: () => ({ onSave: false, tier: 'real' }) } });
    let openings = 0;
    rig.graph.opening!.open = async target => {
      openings++; rig.graph.opening!.sourceOnly(target);
      if (!sourceOnly) rig.graph.opening!.state!.phase = 'failed';
    };
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    await rig.graph.resumeAfterEnvironmentChange(); await settle();
    assert.equal(openings, sourceOnly ? 1 : 2);
    assert.equal(rig.opened.length, 0, 'environment selection must preserve the native Canvas surface');
  });

  for (const afterRefresh of [false, true]) {
    test(`reopening edited source restarts capture ${afterRefresh ? 'after' : 'before'} the static refresh`, async () => {
      const rig = makeGraph([], { profiles: {} as never, trace: {
        worker: { run: async () => { throw new Error('capture is observed at the opening seam'); }, cancel() {} },
        trusted: async () => true, settings: () => ({ onSave: false, tier: 'real' }),
      }, debounceMs: afterRefresh ? 0 : 10_000 });
      let openings = 0;
      rig.graph.opening!.open = async target => {
        openings++; rig.graph.opening!.sourceOnly(target);
        rig.graph.opening!.state!.phase = 'ready';
        rig.graph.opening!.state!.hasCapture = true;
      };
      openStubEditor(RESID, SAVED);
      await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
      await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas');
      assert.equal(openings, 1, 'unchanged reselect preserves the existing graph');
      const edited = openStubEditor(RESID, SAVED.replace('return x', 'return x * 0.5'));
      documentChanges.fire({ document: edited.document });
      if (afterRefresh) await settle();
      await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
      assert.equal(openings, 2, 'a source projection is not a fresh capture of the edited model');
      assert.equal(rig.graph.target!.sourceDigest, createHash('sha256').update(edited.document.getText()).digest('hex'));
    });
  }

  for (const afterRefresh of [false, true]) test(`Refresh rereads edited source ${afterRefresh ? 'after' : 'before'} an in-flight request target changes`, async () => {
    const rig = makeGraph([], { profiles: {} as never, trace: {
      worker: { run: async () => { throw new Error('capture is observed at the opening seam'); }, cancel() {} },
      trusted: async () => true, settings: () => ({ onSave: false, tier: 'real' }),
    }, debounceMs: afterRefresh ? 0 : 10_000 });
    let openings = 0;
    rig.graph.opening!.open = async target => {
      openings++; rig.graph.opening!.sourceOnly(target);
      rig.graph.opening!.state!.phase = 'ready';
    };
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    const opening = rig.graph.opening!.state!;
    const edited = openStubEditor(RESID, SAVED + '\ninvalid syntax !');
    documentChanges.fire({ document: edited.document });
    if (afterRefresh) await settle();
    rig.graph.handleCanvasMessage({ type: 'graphSetupRequest', openingId: opening.id,
      requestId: 'retry-edited-model', target: opening.target, action: { kind: 'retry' } } as never);
    await settle();
    assert.equal(openings, 2);
    assert.equal(rig.graph.target!.sourceDigest, createHash('sha256').update(edited.document.getText()).digest('hex'));
  });

  test('an import failure repairs compatibility and resolves the saved profile with the new interpreter before capture', async () => {
    const root = await realpath(await mkdtemp(path.join(tmpdir(), 'tensorviz-opening-compatibility-')));
    const file = path.join(root, 'models', 'resid.py');
    await mkdir(path.dirname(file)); await writeFile(file, SAVED);
    state.workspaceFolders = [{ uri: { fsPath: root } }];
    try {
      let python = '/original/python', repairs = 0;
      const capturedWith: Array<string | undefined> = [];
      const profile = { kind: 'model', model: 'models.resid:Net', mode: 'eval', device: 'cpu', grad_enabled: false,
        init: { args: [], kwargs: {} }, inputs: { args: [{ kind: 'tensor', shape: [1, 8], dtype: 'float32',
          requires_grad: false, generator: { kind: 'full', value: 1 } }], kwargs: {} }, limits: { timeout_ms: 1000 } };
      const rig = makeGraph([], {
        prepareDependencies: async (_target, _signal, missing, importFailure) => {
          if (!importFailure) return false;
          assert.equal(missing, undefined, 'an import error must not invent an absent package');
          repairs++; python = '/compatible/python'; return true;
        },
        profiles: {
          snapshot: async (_root, selected) => (selected
            ? { guards: {}, resolution: { status: 'resolved', spec: { profile, identity: { profile: selected },
              files: [], interpreter: { executable: python }, dependencyScope: 'partial' } } }
            : { profiles: ['one'], profileDefinitions: { one: profile }, guards: {}, resolution: { status: 'needs_setup', requirements: [
              { profile: null, field: ['profiles'], code: 'ambiguous', phase: 'static', reason: 'Choose.', remedy: 'Choose.' },
            ] } }) as never,
          choose: async () => undefined,
        },
        trace: { trusted: async () => true, settings: () => ({ onSave: false, tier: 'real' }), worker: {
          run: async request => {
            capturedWith.push(request.expectedInterpreter);
            if (python === '/original/python') return { ok: false, exitCode: 1, reason: 'import failed',
              envelope: { protocol_version: 1, result: null, spans: {}, worker: { python: 'test', torch: 'test', elapsed_ms: 1 },
                error: { type: 'ValueError', message: 'import failed', diagnostic: { id: 'setup', code: 'execution_failed', phase: 'import',
                  field: ['model'], names: ['ValueError'], remedy: 'review_dependencies', sourceBindingId: null } } } } as TraceOutcome;
            return { ok: true, exitCode: 0, envelope: { protocol_version: 1, result: { graph: { nodes: [], edges: [] },
              diagnostics: { execution: { importedSources: [{ file: 'models/resid.py', realPath: file,
                bytes: Buffer.byteLength(SAVED), digest: createHash('sha256').update(SAVED).digest('hex') }] } } },
              spans: {}, error: null, worker: { python: 'test', torch: 'test', elapsed_ms: 1 } } };
          }, cancel() {},
        } },
      });
      openStubEditor(file, SAVED);
      await rig.graph.open({ uri: file, cls: 'Net' }, 'canvas');
      for (let i = 0; i < 200 && ['resolving', 'capturing'].includes(rig.graph.opening!.state!.phase); i++) await settle();
      assert.equal(rig.graph.opening!.state!.phase, 'ready', JSON.stringify(rig.graph.opening!.state));
      assert.equal(repairs, 1);
      assert.deepEqual(capturedWith, ['/original/python', '/compatible/python']);
    } finally { await rm(root, { recursive: true, force: true }); }
  });

  for (const scenario of ['root', 'helper', 'root-failure', 'helper-failure', 'disabled', 'burst', 'dirty', 'unrelated', 'review', 'source-only', 'auto-open', 'retarget', 'dispose', 'typing-after-save', 'stopping-worker'] as const)
    test(`saved-source validation: ${scenario}`, async () => {
    const helper = scenario.startsWith('helper') || scenario === 'dirty';
    const succeeded = !scenario.endsWith('failure');
    const root = await realpath(await mkdtemp(path.join(tmpdir(), 'tensorviz-save-opening-')));
    const file = path.join(root, 'models', 'resid.py');
    const layer = path.join(root, 'models', 'layer.py');
    await mkdir(path.dirname(file));
    await writeFile(file, SAVED); if (helper) await writeFile(layer, SAVED);
    state.workspaceFolders = [{ uri: { fsPath: root } }];
    try {
      const profile = { kind: 'model', model: 'models.resid:Net', mode: 'eval', device: 'cpu', grad_enabled: false,
        init: { args: [], kwargs: {} }, inputs: { args: [{ kind: 'tensor', shape: [1, 8], dtype: 'float32',
          requires_grad: false, generator: { kind: 'full', value: 1 } }], kwargs: {} }, limits: { timeout_ms: 1000 } };
      let runs = 0, dependenciesReady = false;
      const rig = makeGraph([], {
        prepareDependencies: async () => { dependenciesReady = true; return true; },
        profiles: {
          snapshot: async (_root, selected) => (selected
            ? { guards: {}, resolution: { status: 'resolved', spec: { profile, identity: { profile: selected },
              files: [], interpreter: { executable: '/python' }, dependencyScope: 'partial' } } }
            : { profiles: ['one'], profileDefinitions: { one: profile }, guards: {}, resolution: { status: 'needs_setup', requirements: [
              { profile: null, field: ['profiles'], code: 'ambiguous', phase: 'static', reason: 'Choose.', remedy: 'Choose.' },
            ] } }) as never,
          choose: async () => undefined,
        },
        trace: { trusted: async () => true, settings: resource => { if (resource) assert.equal(resource.fsPath, helper ? layer : file); return { onSave: scenario !== 'disabled', tier: 'real' }; }, worker: {
          run: async () => {
            assert.ok(dependenciesReady, 'Canvas must finish guided dependency setup before constructing or capturing the model');
            runs++;
            if (runs > 1 && !succeeded) return { ok: false, timedOut: true, exitCode: 1, reason: 'timeout', detail: 'Timed out.' };
            return { ok: true, exitCode: 0, envelope: { protocol_version: 1,
            result: { graph: { nodes: [], edges: [] }, diagnostics: { execution: { importedSources: [file, ...(helper ? [layer] : [])].map(absolute => {
              const source = readFileSync(absolute); return { file: path.relative(root, absolute).split(path.sep).join('/'),
                realPath: absolute, bytes: source.length, digest: createHash('sha256').update(source).digest('hex') };
            }) } } }, spans: {}, error: null,
            worker: { python: 'test', torch: 'test', elapsed_ms: 1 } } }; }, cancel() {},
        } },
      });
      openStubEditor(file, SAVED);
      const changedFile = helper ? layer : file;
      if (helper) openStubEditor(changedFile, SAVED);
      const messages: any[] = [];
      rig.graph.attachCanvas(message => { messages.push(message); });
      await rig.graph.open({ uri: file, cls: 'Net', sourceOnly: scenario === 'source-only' }, 'canvas', scenario === 'auto-open' ? 'automatic' : 'human');
      for (let i = 0; i < 200 && ['resolving', 'capturing'].includes(rig.graph.opening!.state!.phase); i++) await settle();
      assert.equal(rig.graph.opening!.state!.phase, scenario === 'source-only' ? 'source_only' : scenario === 'auto-open' ? 'awaiting_execution' : 'ready', JSON.stringify(rig.graph.opening!.state));
      const beforeRuns = runs;
      const edited = openStubEditor(changedFile, SAVED.replace('return x', 'return x + 1'));
      documentChanges.fire({ document: edited.document }); await settle();
      assert.equal(rig.graph.opening!.state!.phase, 'cancelled', 'editing alone cannot certify old measurements');
      await writeFile(changedFile, edited.document.getText());
      const internals = rig.graph as any;
      if (scenario === 'dirty') editorState.dirtyDocuments.add(file);
      const review = scenario === 'review' ? mock.getter(internals.sourceEdits, 'reviewing', () => true) : undefined;
      if (scenario === 'stopping-worker') internals.executionReview.pending = { target: internals.sync.currentTarget() };
      const save = scenario === 'unrelated' ? openStubEditor(path.join(root, 'other.py'), '# unrelated').document : edited.document;
      saveChanges.fire(save);
      if (scenario === 'burst') { saveChanges.fire(save); saveChanges.fire(save); }
      if (scenario === 'retarget') internals.projectOperationGeneration++;
      if (scenario === 'dispose') rig.graph.dispose();
      if (scenario === 'typing-after-save') { editorState.dirtyDocuments.add(changedFile); documentChanges.fire({ document: edited.document }); }
      if (scenario === 'dirty' || scenario === 'stopping-worker') {
        await settle(); assert.equal(runs, beforeRuns, 'dirty code and a stopping worker cannot overlap a new run');
        editorState.dirtyDocuments.delete(file);
        if (scenario === 'stopping-worker') { internals.executionReview.pending = undefined; await new Promise(resolve => setTimeout(resolve, 110)); }
        else saveChanges.fire(edited.document);
      }
      const executes = ['root', 'helper', 'root-failure', 'helper-failure', 'burst', 'dirty', 'stopping-worker'].includes(scenario);
      if (!executes) {
        await settle(); await settle(); assert.equal(runs, beforeRuns, 'this save must not execute');
        review?.mock.restore();
        rig.graph.dispose(); return;
      }
      for (let i = 0; i < 200 && (runs < 2 || ['resolving', 'capturing'].includes(rig.graph.opening!.state!.phase)); i++) await settle();
      assert.equal(runs, 2, 'saving the selected model or a runtime import executes one fresh validation');
      assert.equal(rig.graph.opening!.state!.phase, succeeded ? 'ready' : 'failed', JSON.stringify(rig.graph.opening!.state));
      assert.equal(rig.graph.opening!.state!.hasCapture, succeeded);
      assert.equal(rig.graph.opening!.state!.execution?.phase, succeeded ? 'succeeded' : 'failed');
      saveChanges.fire(edited.document); await settle();
      assert.equal(runs, 2, 'saving unchanged code does not rerun');
      if (helper && !succeeded) {
        const retry = openStubEditor(layer, SAVED.replace('return x', 'return x + 2'));
        documentChanges.fire({ document: retry.document });
        await writeFile(layer, retry.document.getText()); saveChanges.fire(retry.document);
        for (let i = 0; i < 200 && runs < 3; i++) await settle();
        assert.equal(runs, 3, 'a failed run must not forget its known runtime imports');
      }
      rig.graph.dispose();
    } finally { await rm(root, { recursive: true, force: true }); }
  });

  /** A real opening of a saved model with trace on save, a counting worker and the native Canvas. */
  async function savedRunRig(root: string, overrides: Partial<LocalBrowserGraphOptions> = {}) {
    const file = path.join(root, 'models', 'resid.py');
    await mkdir(path.dirname(file)); await writeFile(file, SAVED);
    state.workspaceFolders = [{ uri: { fsPath: root } }];
    const profile = { kind: 'model', model: 'models.resid:Net', mode: 'eval', device: 'cpu', grad_enabled: false,
      init: { args: [], kwargs: {} }, inputs: { args: [{ kind: 'tensor', shape: [1, 8], dtype: 'float32',
        requires_grad: false, generator: { kind: 'full', value: 1 } }], kwargs: {} }, limits: { timeout_ms: 1000 } };
    const counts = { runs: 0 };
    const rig = makeGraph([], {
      prepareDependencies: async () => true,
      profiles: {
        snapshot: async (_root, selected) => (selected
          ? { guards: {}, resolution: { status: 'resolved', spec: { profile, identity: { profile: selected },
            files: [], interpreter: { executable: '/python' }, dependencyScope: 'partial' } } }
          : { profiles: ['one'], profileDefinitions: { one: profile }, guards: {}, resolution: { status: 'needs_setup', requirements: [
            { profile: null, field: ['profiles'], code: 'ambiguous', phase: 'static', reason: 'Choose.', remedy: 'Choose.' },
          ] } }) as never,
        choose: async () => undefined,
      },
      trace: { trusted: async () => true, settings: () => ({ onSave: true, tier: 'real' }), worker: {
        run: async () => {
          counts.runs++;
          const source = readFileSync(file);
          return { ok: true, exitCode: 0, envelope: { protocol_version: 1,
            result: { graph: { nodes: [], edges: [] }, diagnostics: { execution: { importedSources: [{ file: 'models/resid.py',
              realPath: file, bytes: source.length, digest: createHash('sha256').update(source).digest('hex') }] } } },
            spans: {}, error: null, worker: { python: 'test', torch: 'test', elapsed_ms: 1 } } };
        }, cancel() {},
      } },
      ...overrides,
    });
    const messages: any[] = [];
    rig.graph.attachCanvas(message => { messages.push(message); });
    const idle = async () => {
      for (let i = 0; i < 200 && ['resolving', 'saving', 'capturing'].includes(rig.graph.opening!.state!.phase); i++) await settle();
    };
    /** Edit and save the model, then wait for trace on save to run it. */
    const saveEdit = async () => {
      const edited = openStubEditor(file, SAVED.replace('return x', 'return x + 1'));
      documentChanges.fire({ document: edited.document }); await settle();
      await writeFile(file, edited.document.getText());
      saveChanges.fire(edited.document);
      for (let i = 0; i < 200 && counts.runs < 2; i++) await settle();
      await idle();
      assert.equal(counts.runs, 2, 'the save runs the model');
      assert.equal(rig.graph.opening!.state!.phase, 'ready', JSON.stringify(rig.graph.opening!.state));
      return edited;
    };
    openStubEditor(file, SAVED);
    await rig.graph.open({ uri: file, cls: 'Net' }, 'canvas'); await idle();
    assert.equal(rig.graph.opening!.state!.phase, 'ready', JSON.stringify(rig.graph.opening!.state));
    return { rig, file, counts, messages, idle, saveEdit };
  }

  test('after a save-triggered run, Run model waits at its review without claiming to run, and dismissing the review restores the result', async () => {
    const root = await realpath(await mkdtemp(path.join(tmpdir(), 'tensorviz-run-after-save-')));
    try {
      const { rig, counts, messages, saveEdit } = await savedRunRig(root);
      await saveEdit();
      const saved = messages.filter(message => message.type === 'validationState').at(-1);
      assert.equal(saved.status, 'passed');
      assert.equal(saved.evidence.origin, 'automatic', 'the save-triggered run is reported as trace on save');
      // Canvas sends a manual validation, not a reopening, once the graph has a current capture.
      assert.equal(rig.graph.opening!.state!.hasCapture, true);
      const target = rig.graph.target!, review = (rig.graph as any).executionReview as ExecutionReview;
      const runModel = (requestId: string) => {
        const from = messages.length;
        rig.graph.handleCanvasMessage({ type: 'validateRequest', target, requestId } as never);
        return () => messages.slice(from);
      };
      const states = (sent: any[]) => sent.filter(message => message.type === 'validationState').map(message => [message.runId, message.status, message.detail]);

      const dismissed = runModel('run-model-dismissed'); await settle();
      assert.equal(review.state!.phase, 'review');
      assert.deepEqual(states(dismissed()), [], 'a review waiting for Run has not started the model');
      rig.graph.handleCanvasMessage({ type: 'graphSetupRequest', openingId: rig.graph.opening!.state!.id, target, requestId: 'dismiss-review',
        action: { kind: 'execution', reviewId: review.state!.id, action: 'cancel', device: 'cpu' } } as never);
      await settle();
      assert.equal(dismissed().find(message => message.type === 'validateResult')?.status, 'cancelled');
      assert.deepEqual(states(dismissed()), [[saved.runId, 'passed', saved.detail]], 'dismissing the review shows the save-triggered result again');
      assert.equal((rig.graph as any).validationSummaries.snapshot(target).runId, saved.runId, 'a reconnecting view replays that result too');
      assert.equal(counts.runs, 2, 'nothing ran');

      const approved = runModel('run-model-approved'); await settle();
      assert.deepEqual(states(approved()), []);
      approveExecution(rig.graph);
      for (let i = 0; i < 200 && !approved().some(message => message.type === 'validateResult'); i++) await settle();
      assert.equal(counts.runs, 3);
      const progress = states(approved());
      assert.deepEqual(progress[0].slice(0, 2), ['run-model-approved', 'running'], 'the validation runs once Run is chosen');
      assert.deepEqual(progress.at(-1)!.slice(0, 2), ['run-model-approved', 'passed']);
    } finally { await rm(root, { recursive: true, force: true }); }
  });

  test('after a save-triggered run, Change inputs captures its saved setup without another Run click', async () => {
    const root = await realpath(await mkdtemp(path.join(tmpdir(), 'tensorviz-inputs-after-save-')));
    try {
      const { rig, counts, messages, idle, saveEdit } = await savedRunRig(root);
      await saveEdit();
      const discover = mock.method(rig.graph.setupHost!, 'discover', async () => ({ discovery: { parents: [] } }));
      const save = mock.method(rig.graph.setupHost!, 'save', async () => ({}));
      try {
        const opening = rig.graph.opening!.state!;
        rig.graph.handleCanvasMessage({ type: 'graphSetupRequest', openingId: opening.id, target: opening.target,
          requestId: 'change-inputs', action: { kind: 'change_inputs' } } as never);
        for (let i = 0; i < 200 && rig.graph.opening!.state!.phase !== 'needs_setup'; i++) await settle();
        const form = rig.graph.opening!.state!;
        assert.notEqual(form.id, opening.id);
        rig.graph.handleCanvasMessage({ type: 'graphSetupRequest', openingId: form.id, target: form.target,
          requestId: 'save-inputs', action: { kind: 'submit', draft: form.draft } } as never);
        await settle(); await idle();
        assert.equal(rig.graph.opening!.state!.phase, 'ready', 'saving changed inputs is the person asking to run the model');
        assert.equal(counts.runs, 3);
        assert.equal(save.mock.callCount(), 1);
        assert.equal(messages.filter(message => message.type === 'validationState').at(-1).evidence.origin, 'manual');
      } finally { discover.mock.restore(); save.mock.restore(); }
    } finally { await rm(root, { recursive: true, force: true }); }
  });

  test('an agent opening that starts while a save-triggered run finishes keeps the agent actor', async () => {
    const root = await realpath(await mkdtemp(path.join(tmpdir(), 'tensorviz-agent-after-save-')));
    let releaseRun = () => {}, releaseRead = () => {};
    try {
      const { rig, file, counts, messages, idle } = await savedRunRig(root);
      const worker = (rig.graph as any).options.trace.worker, run = worker.run;
      const held = new Promise<void>(resolve => { releaseRun = resolve; });
      const gated = mock.method(worker, 'run', async (...args: unknown[]) => { await held; return run(...args); });
      const edited = openStubEditor(file, SAVED.replace('return x', 'return x + 1'));
      documentChanges.fire({ document: edited.document }); await settle();
      await writeFile(file, edited.document.getText());
      saveChanges.fire(edited.document);
      for (let i = 0; i < 200 && gated.mock.callCount() < 1; i++) await settle();
      const savedRun = rig.graph.opening!.state!;
      assert.equal(savedRun.phase, 'capturing', 'the save-triggered run is executing');
      // The agent's open_graph sets its actor, then reads the document before it starts its opening.
      const vscode = require('../vscodeStub'), original = vscode.workspace.openTextDocument;
      const readHeld = new Promise<void>(resolve => { releaseRead = resolve; });
      let first = true;
      const read = mock.method(vscode.workspace, 'openTextDocument', async (uri: any) => {
        if (first) { first = false; await readHeld; }
        return original(uri);
      });
      try {
        const agent = rig.graph.open({ uri: file, cls: 'Net', restart: true }, 'canvas', 'agent'); await settle();
        releaseRun(); await idle();
        assert.equal(rig.graph.opening!.state!.id, savedRun.id);
        assert.equal(rig.graph.opening!.state!.phase, 'ready', 'the save-triggered run finished inside the agent opening');
        releaseRead(); await agent; await idle();
        assert.notEqual(rig.graph.opening!.state!.id, savedRun.id);
        assert.equal(counts.runs, 3, 'the agent opening runs the model');
        assert.equal(messages.filter(message => message.type === 'validationState').at(-1).evidence.origin, 'mcp',
          'the finished save run must not relabel the agent opening as a person');
      } finally { read.mock.restore(); gated.mock.restore(); }
    } finally { releaseRun(); releaseRead(); await rm(root, { recursive: true, force: true }); }
  });

  test('an unchanged save while an accepted change is opening does not restart it', async () => {
    const root = await realpath(await mkdtemp(path.join(tmpdir(), 'tensorviz-accept-save-')));
    let gate: Promise<void> = Promise.resolve(), checks = 0, release = () => {};
    try {
      const { rig, file, counts, idle } = await savedRunRig(root, {
        environment: async () => { checks++; await gate; return { reason: 'ready', candidates: [], detail: 'Ready.' } as never; } });
      const edited = openStubEditor(file, SAVED.replace('return x', 'return x + 1'));
      editorState.dirtyDocuments.add(file);
      documentChanges.fire({ document: edited.document }); await settle();
      gate = new Promise(resolve => { release = resolve; });
      const vscode = require('../vscodeStub'), original = vscode.workspace.openTextDocument;
      const read = mock.method(vscode.workspace, 'openTextDocument', async (uri: any) => Object.assign(await original(uri), { save: async () => {
        await writeFile(file, edited.document.getText());
        editorState.dirtyDocuments.delete(file);
        saveChanges.fire(edited.document);
        return true;
      } }));
      try {
        const detail = await (rig.graph as any).sourceEdits.options.accepted({ target: rig.graph.target!, file: 'models/resid.py', current: () => true });
        assert.match(detail, /Saved the accepted change/);
        await settle();
        const accepted = rig.graph.opening!.state!;
        assert.equal(accepted.phase, 'resolving');
        assert.equal(checks, 2, 'the accepted change is checking its environment');
        saveChanges.fire(edited.document); // Cmd+S on the model the accept already saved.
        await settle(); await new Promise(resolve => setTimeout(resolve, 150));
        assert.equal(rig.graph.opening!.state!.id, accepted.id, 'an unchanged save must not restart the accepted opening');
        assert.equal(checks, 2);
        release(); await idle();
        assert.equal(rig.graph.opening!.state!.phase, 'ready', JSON.stringify(rig.graph.opening!.state));
        assert.equal(counts.runs, 2, 'the accepted change runs once');
      } finally { read.mock.restore(); }
    } finally { release(); await rm(root, { recursive: true, force: true }); }
  });

  test('a buffer reload that only resets the document version keeps the completed capture current', async () => {
    const root = await realpath(await mkdtemp(path.join(tmpdir(), 'tensorviz-version-opening-')));
    const file = path.join(root, 'models', 'resid.py');
    await mkdir(path.dirname(file));
    await writeFile(file, SAVED);
    state.workspaceFolders = [{ uri: { fsPath: root } }];
    try {
      const profile = { kind: 'model', model: 'models.resid:Net', mode: 'eval', device: 'cpu', grad_enabled: false,
        init: { args: [], kwargs: {} }, inputs: { args: [{ kind: 'tensor', shape: [1, 8], dtype: 'float32',
          requires_grad: false, generator: { kind: 'full', value: 1 } }], kwargs: {} }, limits: { timeout_ms: 1000 } };
      let runs = 0;
      const rig = makeGraph([], {
        prepareDependencies: async () => true,
        profiles: {
          snapshot: async (_root, selected) => (selected
            ? { guards: {}, resolution: { status: 'resolved', spec: { profile, identity: { profile: selected },
              files: [], interpreter: { executable: '/python' }, dependencyScope: 'partial' } } }
            : { profiles: ['one'], profileDefinitions: { one: profile }, guards: {}, resolution: { status: 'needs_setup', requirements: [
              { profile: null, field: ['profiles'], code: 'ambiguous', phase: 'static', reason: 'Choose.', remedy: 'Choose.' },
            ] } }) as never,
          choose: async () => undefined,
        },
        trace: { trusted: async () => true, settings: () => ({ onSave: false, tier: 'real' }), worker: {
          run: async () => {
            runs++;
            // VS Code released the unshown buffer and read the same saved bytes again:
            // its version restarts at 1 with no change event and an unchanged digest.
            editorState.versions.set(file, 1);
            const source = readFileSync(file);
            return { ok: true, exitCode: 0, envelope: { protocol_version: 1,
              result: { graph: { nodes: [], edges: [] }, diagnostics: { execution: { importedSources: [{ file: 'models/resid.py',
                realPath: file, bytes: source.length, digest: createHash('sha256').update(source).digest('hex') }] } } },
              spans: {}, error: null, worker: { python: 'test', torch: 'test', elapsed_ms: 1 } } };
          }, cancel() {},
        } },
      });
      // Edited and saved back to the same text earlier in the session: version 3.
      for (let i = 0; i < 3; i++) openStubEditor(file, SAVED);
      rig.graph.attachCanvas(() => {});
      await rig.graph.open({ uri: file, cls: 'Net' }, 'canvas');
      for (let i = 0; i < 200 && ['resolving', 'capturing'].includes(rig.graph.opening!.state!.phase); i++) await settle();
      const opening = rig.graph.opening!.state!;
      assert.equal(runs, 1);
      assert.equal(opening.phase, 'ready', JSON.stringify(opening));
      assert.equal(opening.hasCapture, true);
      assert.equal(opening.cancellationReason, undefined);
      assert.equal(opening.target.documentVersion, 1, 'the opening follows the reloaded buffer');
      assert.equal(opening.target.sourceDigest, createHash('sha256').update(SAVED).digest('hex'));
      rig.graph.dispose();
    } finally { await rm(root, { recursive: true, force: true }); }
  });

  test('an imported layer edit invalidates opening even when the root source is unchanged', async () => {
    const rig = makeGraph([], { profiles: {} as never, trace: {
      worker: { run: async () => { throw new Error('no capture expected'); }, cancel() {} },
      trusted: async () => true, settings: () => ({ onSave: false, tier: 'real' }),
    } });
    openStubEditor(RESID, SAVED);
    const layer = path.join(ROOT, 'models', 'layer.py');
    openStubEditor(layer, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net', sourceOnly: true }, 'canvas'); await settle();
    const rootDigest = rig.graph.target!.sourceDigest;
    rig.graph.opening!.state!.phase = 'ready';
    rig.graph.opening!.state!.hasCapture = true;
    const edited = openStubEditor(layer, SAVED.replace('return x', 'return x * 0.5'));
    documentChanges.fire({ document: edited.document });
    assert.equal(rig.graph.opening!.state!.phase, 'cancelled', 'invalidate before the debounced projection');
    assert.equal(rig.graph.opening!.state!.hasCapture, false);
    await settle();
    assert.equal(rig.graph.target!.sourceDigest, rootDigest);
    assert.equal(rig.graph.opening!.state!.phase, 'cancelled');
  });

  test('explicit browser action opens the selected class without resetting its revision or following another editor', async () => {
    const rig = makeGraph([]);
    openStubEditor(RESID, EDITED);
    await rig.graph.open({ uri: RESID, cls: 'Unsaved' }, 'canvas'); await settle();
    const selected = rig.graph.target;
    openStubEditor('/other/other.py', SAVED);
    await rig.graph.openSelectedInBrowser(); await settle();
    assert.deepEqual(rig.graph.target, selected);
    assert.equal(rig.projectionCalls.length, 1);
    assert.equal(rig.mintCalls.length, 1); assert.equal(rig.opened.length, 1);
    await rig.graph.openSelectedInBrowser();
    assert.equal(rig.opened.length, 2, 'explicit action opens the selected page even if an existing tab is reported attached');
    assert.equal(rig.mintCalls.length, 1, 'same authenticated session');
    assert.equal(rig.projectionCalls.length, 1);
    rig.transports[0].fire({ type: 'snapshotRequest' });
    assert.deepEqual(rig.transports[0].published.at(-1)?.state && (rig.transports[0].published.at(-1)!.state as any).activeTarget, selected);
  });

  test('MCP browser handoff retains the graph without recapture or system-browser launch', async () => {
    const rig = makeGraph([]);
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    const selected = rig.graph.target!;
    const link = await rig.graph.prepareBrowserView(selected, new AbortController().signal);
    assert.match(link.pageUrl, /single-use/);
    assert.doesNotMatch(link.pageUrl, /host-key|view-credential|test-token/);
    assert.deepEqual(rig.graph.target, selected);
    assert.equal(rig.projectionCalls.length, 1); assert.equal(rig.opened.length, 0);
    assert.equal(rig.mintCalls.length, 1);
    await rig.graph.prepareBrowserView(selected, new AbortController().signal);
    assert.equal(rig.mintCalls.length, 1);
    const cancelled = new AbortController(); cancelled.abort();
    await assert.rejects(rig.graph.prepareBrowserView(selected, cancelled.signal), /cancelled/);
    await assert.rejects(rig.graph.prepareBrowserView({ ...selected, backendRevision: 900 }, new AbortController().signal), /graph changed/);
    rig.transports[0].setPhase('ended');
    assert.equal(rig.graph.browserSessionCurrent(link.workspaceSessionId), false);
  });

  test('no-model browser action explains how to select a graph without starting anything', async () => {
    const rig = makeGraph([]);
    openStubEditor(RESID, SAVED);
    await rig.graph.openSelectedInBrowser();
    assert.equal(rig.ensureCalls(), 0); assert.equal(rig.mintCalls.length, 0);
    assert.equal(rig.graph.target, undefined);
    assert.ok(infoMessages.some(message => message.includes('Open as Graph')));
    const messages: any[] = [];
    rig.graph.attachCanvas(message => { messages.push(message); }); rig.graph.replayCanvas();
    assert.deepEqual(messages[0].state, emptyLocalBrowserSnapshotState());
  });

  test('project browser action replays the selected project without reindexing or inferring another root', async () => {
    const rig = makeGraph([]);
    openStubEditor(RESID, SAVED);
    await rig.graph.openProject({ folder: ROOT }, 'canvas'); await settle();
    const calls = rig.fetchCalls.length;
    assert.equal(calls, 1, 'initial project was indexed');
    await rig.graph.openProject({ folder: ROOT }, 'canvas');
    openStubEditor('/other/other.py', SAVED);
    await rig.graph.openSelectedInBrowser();
    assert.equal(rig.fetchCalls.length, calls, 'browser opening and repeated native opening preserve project state');
    assert.equal(rig.graph.hasSelection, true); assert.equal(rig.graph.target, undefined);
    assert.equal(rig.mintCalls.length, 1);
    rig.transports[0].fire({ type: 'snapshotRequest' }); await settle();
    const index = rig.transports[0].published.find(message => message.type === 'projectIndex') as any;
    assert.equal(index.project.workspaceFolder, ROOT);
  });

  test('a project selection retries after backend startup failed instead of reusing a missing index', async () => {
    let available = false;
    const rig = makeGraph([], { ensureBackend: async () => available });
    openStubEditor(RESID, SAVED);
    await rig.graph.openProject({ folder: ROOT }, 'canvas');
    assert.equal(rig.fetchCalls.length, 0);
    available = true;
    await rig.graph.openProject({ folder: ROOT }, 'canvas'); await settle();
    assert.ok(rig.fetchCalls.some(call => call.url.endsWith('/project/index')));
  });

  test('switching source editors preserves the selected model; editing that source still refreshes', async () => {
    const rig = makeGraph([]);
    const editor = openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    const selected = rig.graph.target;
    const other = openStubEditor('/ws/other.py', SAVED);
    selectionChanges.fire({ textEditor: other, selections: [{ active: { line: 4, character: 0 } }] }); await settle();
    assert.deepEqual(rig.graph.target, selected);
    assert.equal(rig.projectionCalls.length, 1);
    editorState.documents.set(RESID, SAVED + '# changed\n');
    editorState.versions.set(RESID, 2);
    documentChanges.fire({ document: editor.document }); await settle();
    assert.equal(rig.graph.target?.cls, 'Net');
    assert.equal(rig.graph.target?.file, 'models/resid.py');
    assert.notEqual(rig.graph.target?.sourceDigest, selected?.sourceDigest);
    assert.equal(rig.projectionCalls.length, 2);
  });

  test('native MCP selection and focus are isolated from browser identities and disconnects', async () => {
    const rig = makeGraph([]);
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    const target = rig.graph.target!, posts: any[] = [];
    rig.graph.attachCanvas(message => { posts.push(message); return true; });
    rig.graph.setCanvasView('native', true);
    const envelope = { protocolVersion: 1 as const, workspaceSessionId: 'native-session', messageId: 'msg' };
    rig.graph.handleCanvasMessage({ ...envelope, type: 'viewSelection', viewId: 'native', target, nodeIds: ['net'], edgeIds: [] });
    assert.deepEqual(rig.graph.mcpViews.selection('native', target).nodeIds, ['net']);
    const focus = rig.graph.mcpViews.focus('owner', { viewId: 'native', target, nodeIds: ['fwd'], edgeIds: [] }, new AbortController().signal);
    const request = posts.find(message => message.type === 'focusView');
    assert.ok(request, 'native publication succeeds without a browser transport receipt');
    await rig.graph.openSelectedInBrowser();
    assert.equal(rig.transports[0].published.filter(message => message.type === 'focusView').length, 0);
    rig.transports[0].fire({ ...envelope, type: 'viewSelection', viewId: 'native', target, nodeIds: ['spoof'], edgeIds: [] });
    rig.transports[0].fire({ ...envelope, type: 'viewFocusResult', viewId: 'native', target, requestId: request.requestId, ok: true, detail: 'Spoof.' });
    rig.transports[0].setPhase('ended');
    assert.deepEqual(rig.graph.mcpViews.selection('native', target).nodeIds, ['net']);
    rig.graph.handleCanvasMessage({ ...envelope, type: 'viewFocusResult', viewId: 'native', target, requestId: request.requestId, ok: true, detail: 'Focused.' });
    assert.equal((await focus).focused, true);
    assert.deepEqual(rig.graph.mcpViews.selection('native', target).nodeIds, ['fwd']);
    const hidden = rig.graph.mcpViews.focus('owner', { viewId: 'native', target, nodeIds: ['net'], edgeIds: [] }, new AbortController().signal);
    rig.graph.setCanvasView('native', false);
    await assert.rejects(hidden, /closed or is hidden/);
    assert.deepEqual(rig.graph.mcpViews.list(), []);
    rig.graph.setCanvasView('native', true);
    assert.equal(rig.graph.mcpViews.list().length, 1);
    assert.throws(() => rig.graph.mcpViews.selection('native', target), /current graph/);
  });

  test('SSH opens on the forwarded origin and renews authentication when the tunnel changes', async () => {
    let external = 'http://127.0.0.1:43127', resolutions = 0;
    const rig = makeGraph([{ classes: [NET] }], { remoteName: () => 'ssh-remote',
      resolveLaunchOrigin: async origin => {
        assert.equal(origin, 'http://127.0.0.1:8631'); resolutions++; return external;
      } });
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net', sourceOnly: true }, 'canvas');
    await settle();
    assert.equal(rig.mintCalls.length, 0, 'native SSH Canvas uses its bridge');
    const target = rig.graph.target;
    await rig.graph.openSelectedInBrowser();
    assert.equal(rig.mintCalls[0].backendBaseUrl, 'http://127.0.0.1:8631');
    assert.equal(rig.mintCalls[0].launchOrigin, external);
    assert.ok(rig.opened[0].startsWith('http://127.0.0.1:8631/#'), 'openExternal forwards the internal URL itself');
    await rig.graph.openSelectedInBrowser();
    assert.equal(rig.mintCalls.length, 1, 'unchanged tunnel reuses authentication');
    external = 'http://127.0.0.1:53128';
    await rig.graph.openSelectedInBrowser();
    assert.equal(rig.mintCalls.length, 2);
    assert.equal(rig.mintCalls[1].launchOrigin, external);
    assert.equal(rig.transports[0].isDisposed(), true);
    assert.deepEqual(rig.graph.target, target, 'reopening never retargets the graph');
    assert.equal(resolutions, 3);
  });

  test('SSH forwarding failure does not mint or launch an unusable page', async () => {
    const rig = makeGraph([{ classes: [NET] }], { remoteName: () => 'ssh-remote',
      resolveLaunchOrigin: async () => { throw new Error('tunnel closed'); } });
    await rig.graph.openProject();
    assert.equal(rig.mintCalls.length, 0);
    assert.equal(rig.opened.length, 0);
    assert.match(state.warningMessages[0], /forwarding is unavailable/);
  });

  test('a backend restart during concurrent forwarding cannot reuse the preceding launch session', async () => {
    let token = 'first-launch';
    const resolves: Array<(origin: string) => void> = [];
    const mints: MintLocalBrowserSessionOptions[] = [];
    let releaseMint!: () => void;
    const rig = makeGraph([], { remoteName: () => 'ssh-remote',
      backend: { origin: () => 'http://127.0.0.1:8631', launchToken: () => token },
      resolveLaunchOrigin: () => new Promise(resolve => resolves.push(resolve)),
      mintSession: async options => {
        mints.push(options);
        if (mints.length === 1) await new Promise<void>(resolve => { releaseMint = resolve; });
        return { workspaceSessionId: 'fixture', launchOrigin: options.launchOrigin, expiresAt: 'later',
          hostCookieKey: 'host', viewCredential: 'view', pageUrl: `${options.launchOrigin}/#fixture` };
      } });
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    const first = rig.graph.openSelectedInBrowser(); await settle();
    token = 'second-launch';
    const second = rig.graph.openSelectedInBrowser(); await settle();
    resolves[0]('http://127.0.0.1:43127'); await settle();
    resolves[1]('http://127.0.0.1:53128'); await settle();
    releaseMint(); await Promise.all([first, second]);
    assert.equal(mints.length, 2);
    assert.equal(mints[1].launchToken, 'second-launch');
    assert.equal(mints[1].launchOrigin, 'http://127.0.0.1:53128');
    assert.equal(rig.transports[0].isDisposed(), true);
  });

  test('refuses an unsupported remote window without touching the network or the backend', async () => {
    const remote = makeGraph([{ classes: [NET] }], { remoteName: () => 'dev-container' });
    openStubEditor(RESID, SAVED);

    await remote.graph.open({ uri: RESID });

    assert.equal(remote.fetchCalls.length, 0);
    assert.equal(remote.mintCalls.length, 0);
    assert.equal(remote.opened.length, 0);
    assert.equal(remote.ensureCalls(), 0);
    assert.equal(state.warningMessages.length, 1);
    assert.match(
      state.warningMessages[0],
      /Remote SSH.*not supported/,
      'the refusal names the supported host and unsupported remote kind',
    );
  });

  test('with no editor and no argument, opens the project rather than demanding a file', async () => {
    // "Open a Python file with an nn.Module class first" was the chooser's demand
    // worded as a refusal: it made a file the price of a graph. The workspace is a
    // subject in its own right, so an empty editor is not an error any more.
    const noEditor = makeGraph([{ classes: [NET] }]);

    await noEditor.graph.open();

    assert.deepEqual(state.warningMessages, [], 'nothing to apologise for');
    assert.equal(noEditor.opened.length, 1, 'the project opened');
    assert.equal(noEditor.graph.target, undefined);
  });

  test('refuses a document that is not a file on disk', async () => {
    // Only when a CLASS is named: that names a target which must be anchored to a
    // real file. Without one the same untitled buffer simply opens the project,
    // which is the test above.
    const untitled = makeGraph([{ classes: [NET] }]);

    await untitled.graph.open({ uri: 'untitled:scratch.py', cls: 'Net' });

    assert.equal(untitled.ensureCalls(), 0);
    assert.equal(state.warningMessages.length, 1);
    assert.match(state.warningMessages[0], /file saved on this disk/);
  });

  test('a retarget reuses the live session; an ended session is reminted', async () => {
    const rig = makeGraph([]);
    openStubEditor(RESID, SAVED);

    await rig.graph.open({ uri: RESID, cls: 'Net' });
    assert.equal(rig.mintCalls.length, 1);
    assert.equal(rig.opened.length, 1);

    await rig.graph.open({ uri: RESID, cls: 'Gate' });
    // Same session, no second tab: the retarget went out as a live activeTarget
    // (amid the sync's own projection publishes, which this assertion ignores).
    assert.equal(rig.mintCalls.length, 1);
    assert.equal(rig.opened.length, 1);
    const retargets = rig.transports[0].published.filter((m) => m.type === 'activeTarget');
    assert.equal(retargets.length, 1);
    const retargetTarget = retargets[0].target as { cls?: string; backendRevision?: number };
    assert.equal(retargetTarget.cls, 'Gate');
    assert.equal(retargetTarget.backendRevision, 0, 'a retarget names no projection until one lands');

    rig.transports[0].setPhase('ended');
    await rig.graph.open({ uri: RESID, cls: 'Gate' });
    assert.equal(rig.mintCalls.length, 2, 'an ended session must be minted afresh');
    assert.equal(rig.transports.length, 2);
    assert.equal(rig.opened.length, 2, 'a new session means a new tab: the old one cannot resume');
  });

  test('a backend restart replaces a reconnecting session, including when its port is reused', async () => {
    let origin = 'http://127.0.0.1:8631';
    let token = 'first-launch';
    const rig = makeGraph([], { backend: { origin: () => origin, launchToken: () => token } });
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' });

    rig.transports[0].setPhase('reconnecting');
    origin = 'http://127.0.0.1:8632';
    token = 'second-launch';
    await rig.graph.open({ uri: RESID, cls: 'Net' });
    assert.ok(rig.transports[0].isDisposed(), 'stop retrying the old backend');
    assert.equal(rig.mintCalls.length, 2);
    assert.equal(rig.mintCalls[1].backendBaseUrl, origin);
    assert.equal(rig.opened.length, 2, 'open a page with credentials for the new backend');

    rig.transports[1].setPhase('reconnecting');
    token = 'third-launch';
    await rig.graph.open({ uri: RESID, cls: 'Net' });
    assert.ok(rig.transports[1].isDisposed());
    assert.equal(rig.mintCalls.length, 3, 'a new launch on the same port also needs a new session');
    assert.equal(rig.mintCalls[2].launchToken, token);
  });

  test('a view asking for state gets the minimal replay before the first projection, the full one after', async () => {
    const rig = makeGraph([], { deferFirstParse: true });
    openStubEditor(RESID, SAVED);

    await rig.graph.open({ uri: RESID, cls: 'Net' });
    const base = rig.graph.target;
    assert.ok(base);

    // Before any projection: the contract's minimal replay — the empty state,
    // then the target as its own ordered message.
    rig.transports[0].fire({ type: 'snapshotRequest' });

    assert.deepEqual(rig.transports[0].published, [
      { type: 'snapshot', state: emptyLocalBrowserSnapshotState() },
      { type: 'activeTarget', target: base },
    ]);
    // The snapshot state is the contract's empty state, exactly: no target, no
    // projection, no sync reading, connected.
    assert.deepEqual(emptyLocalBrowserSnapshotState(), {
      activeTarget: null,
      projection: null,
      traced: false,
      greyed: false,
      syncState: null,
      connectionState: 'connected',
    });

    // Release the projection: it lands live (projection before its syncState),
    // and the next view asking is restored from it — one atomic snapshot whose
    // state carries the target with the real backend revision, not the base 0.
    rig.releaseFirstParse();
    await settle();

    rig.transports[0].fire({ type: 'snapshotRequest' });

    const tail = rig.transports[0].published.slice(2);
    assert.deepEqual(tail.map((m) => m.type), ['projection', 'syncState', 'snapshot']);
    const live = tail[0] as unknown as {
      target: { backendRevision: number; cls: string };
      projection: { traced: boolean; greyed: boolean };
      traced: boolean;
    };
    assert.equal(live.target.backendRevision, 1);
    assert.equal(live.target.cls, 'Net');
    assert.equal(live.traced, false);
    assert.deepEqual(tail[1], { type: 'syncState', target: live.target, state: 'source-only' });
    const restored = (tail[2] as unknown as { state: Record<string, unknown> }).state;
    assert.equal(restored.syncState, 'source-only');
    assert.equal(restored.traced, false);
    assert.equal(restored.greyed, false);
    assert.ok(restored.projection, 'the graph the editor would show');
    assert.equal(
      (restored.activeTarget as { backendRevision: number }).backendRevision,
      1,
      'the restore names the revision of the projection it carries',
    );
    assert.ok(!('reason' in restored), 'a healthy state names no stale reason');
  });

  test('a reconnect snapshot restores a no-capture validation refusal', async () => {
    const rig = makeGraph([]);
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' });
    await settle();
    const target = rig.graph.target;
    rig.transports[0].fire({ type: 'validateRequest', target, requestId: 'restore-run' });
    await settle();
    const result = rig.transports[0].published.find(m => m.type === 'validateResult')!;
    assert.equal(result.status, 'refused');
    rig.transports[0].fire({ type: 'snapshotRequest' });
    const snapshot = rig.transports[0].published.at(-1) as any;
    assert.equal(snapshot.type, 'snapshot');
    assert.equal(snapshot.state.validation.runId, 'restore-run');
    assert.equal(snapshot.state.validation.detail, result.detail);
    assert.ok(!('projectionRevision' in snapshot.state.validation));
  });

  test('native validation publishes and replays the canonical attempt without a browser session', async () => {
    const rig = makeGraph([]), messages: any[] = [];
    rig.graph.attachCanvas(message => { messages.push(message); });
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' }, 'canvas'); await settle();
    rig.graph.handleCanvasMessage({ type: 'validateRequest', target: rig.graph.target!, requestId: 'native-attempt',
      workspaceSessionId: 'native', messageId: 'request', protocolVersion: 1 });
    await settle();
    const summary = messages.filter(message => message.type === 'validationState').at(-1);
    assert.equal(summary.status, 'refused');
    assert.ok(summary.evidence, 'the native canvas needs the same ledger attribution as browser and MCP');
    rig.graph.replayCanvas();
    assert.deepEqual(messages.at(-1).state.validation.evidence, summary.evidence);
    assert.equal(messages.at(-1).state.validation.runId, 'native-attempt');
    assert.equal(rig.mintCalls.length, 0);
  });

  test('a view asking to reveal a node moves the editor cursor, is acked, and cannot echo back', async () => {
    const rig = makeGraph([], { spans: SPANS });
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' });
    await settle();
    const live = liveTarget(rig.transports[0].published);

    rig.transports[0].fire({ type: 'revealNode', messageId: 'nav-1', target: live, nodeId: 'fwd' });
    await settle();

    assert.deepEqual(
      rig.transports[0].published.filter((m) => m.type === 'ack'),
      [{ type: 'ack', target: live, ackId: 'nav-1', ok: true }],
    );
    // `fwd` starts at span line 5, column 4: 0-based for the editor.
    assert.equal(residEditor().selection.active.line, 4);
    assert.equal(residEditor().selection.active.character, 4);

    // The selection event the reveal itself caused must not bounce a focusNode
    // back at the page that asked.
    selectionChanges.fire({ textEditor: residEditor(), selections: [residEditor().selection] });
    assert.ok(!rig.transports[0].published.some((m) => m.type === 'focusNode'));
  });

  test('a stale, foreign, or nameless reveal is refused by ack and moves nothing', async () => {
    const rig = makeGraph([], { spans: SPANS });
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' });
    await settle();
    const live = liveTarget(rig.transports[0].published);

    const stale = { ...live, backendRevision: 0 };
    rig.transports[0].fire({ type: 'revealNode', messageId: 'stale-1', target: stale, nodeId: 'fwd' });
    await settle();
    const foreign = { ...live, cls: 'Gate' };
    rig.transports[0].fire({ type: 'revealNode', messageId: 'foreign-1', target: foreign, nodeId: 'fwd' });
    await settle();
    rig.transports[0].fire({ type: 'revealNode', messageId: 'ghost-1', target: live, nodeId: 'ghost' });
    await settle();

    // The ack names the asking message's id, per the contract — including the
    // identity the view asked about, so a page can tell its stale self-apart.
    assert.deepEqual(
      rig.transports[0].published.filter((m) => m.type === 'ack'),
      [
        {
          type: 'ack',
          target: stale,
          ackId: 'stale-1',
          ok: false,
          errorCode: 'STALE_TARGET',
          detail:
            'the page is not showing this window’s current target state; ' +
            'wait for the refresh and click again',
        },
        {
          type: 'ack',
          target: foreign,
          ackId: 'foreign-1',
          ok: false,
          errorCode: 'STALE_TARGET',
          detail:
            'the page is not showing this window’s current target state; ' +
            'wait for the refresh and click again',
        },
        {
          type: 'ack',
          target: live,
          ackId: 'ghost-1',
          ok: false,
          errorCode: 'INVALID_MESSAGE',
          detail: 'no node named ghost in the graph the page is showing',
        },
      ],
    );
    // The editor's caret never left the top of the file.
    assert.equal(residEditor().selection.active.line, 0);
    assert.equal(residEditor().selection.active.character, 0);
  });

  test('terminal focus uses the current graph and refuses unknown or stale nodes', async () => {
    const rig = makeGraph([], { spans: SPANS });
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' });
    await settle();
    const live = liveTarget(rig.transports[0].published);
    const messages: any[] = [];
    const canvas = rig.graph.attachCanvas(message => { messages.push(message); });

    rig.graph.focusSourceNode('fwd');
    rig.graph.focusSourceNode('unknown');
    openStubEditor(RESID, SAVED + '\n# changed');
    rig.graph.focusSourceNode('fwd');
    const expected = [{ type: 'focusNode', target: live, nodeId: 'fwd' }];
    assert.deepEqual(messages.filter(m => m.type === 'focusNode'), expected);
    assert.deepEqual(rig.transports[0].published.filter(m => m.type === 'focusNode'), expected);
    canvas.dispose();
  });

  test('moving the editor cursor focuses the node on the page', async () => {
    const rig = makeGraph([], { spans: SPANS });
    openStubEditor(RESID, SAVED);
    await rig.graph.open({ uri: RESID, cls: 'Net' });
    await settle();
    const live = liveTarget(rig.transports[0].published);

    // `return x`: span line 6, covered only by `fwd` here.
    const editor = residEditor();
    editor.selection = { active: { line: 5, character: 0 } };
    selectionChanges.fire({ textEditor: editor, selections: [editor.selection] });

    assert.deepEqual(
      rig.transports[0].published.filter((m) => m.type === 'focusNode'),
      // One span on this line, so the honest instance set is the singleton
      // (nodeIds is how a shared definition's line would name them all).
      [{ type: 'focusNode', target: live, nodeId: 'fwd', nodeIds: ['fwd'] }],
    );
  });

  test('the dev server is the launch origin when one is configured', async () => {
    const dev = makeGraph([], { launchOrigin: () => 'http://localhost:3000' });
    openStubEditor(RESID, SAVED);

    await dev.graph.open({ uri: RESID, cls: 'Net' });

    assert.equal(dev.mintCalls[0].launchOrigin, 'http://localhost:3000');
    assert.deepEqual(dev.opened, [
      'http://localhost:3000/#session=ws-test&credential=view-credential',
    ]);
  });

  test('a refused connect drops the session: the next open mints again and warns once', async () => {
    const failed = makeGraph([], {}, new Error('terminal refusal'));
    openStubEditor(RESID, SAVED);

    await failed.graph.open({ uri: RESID, cls: 'Net' });
    // The rejection handler runs a microtask after open() returns.
    await new Promise((resolve) => setImmediate(resolve));

    assert.equal(state.warningMessages.length, 1);
    assert.match(state.warningMessages[0], /ended before the page could connect/);

    await failed.graph.open({ uri: RESID, cls: 'Net' });
    assert.equal(failed.mintCalls.length, 2);
  });

  test('two invocations landing during the mint fold into one session and one tab', async () => {
    let releaseMint!: () => void;
    const mintGate = new Promise<void>((resolve) => {
      releaseMint = resolve;
    });
    const rig = makeGraph([], {
      mintSession: (async (options: MintLocalBrowserSessionOptions) => {
        rig.mintCalls.push(options);
        await mintGate;
        return {
          workspaceSessionId: 'ws-test',
          launchOrigin: options.launchOrigin,
          expiresAt: '2026-01-01T00:00:00.000Z',
          hostCookieKey: 'host-key',
          viewCredential: 'view-credential',
          pageUrl: `${options.launchOrigin}/#session=ws-test&credential=view-credential`,
        };
      }) as never,
    });
    openStubEditor(RESID, SAVED);

    const first = rig.graph.open({ uri: RESID, cls: 'Net' });
    // The first call is parked in its mint; the second must fold, not mint again.
    const second = rig.graph.open({ uri: RESID, cls: 'Gate' });
    releaseMint();
    await Promise.all([first, second]);

    assert.equal(rig.mintCalls.length, 1, 'a second invocation during the mint must not mint again');
    assert.equal(rig.opened.length, 1, 'a second invocation during the mint must not open a second tab');
    const retargets = rig.transports[0].published.filter((m) => m.type === 'activeTarget');
    assert.equal(retargets.length, 1, 'the folded call retargets live, exactly once');
    assert.equal((retargets[0].target as { cls?: string }).cls, 'Gate');
    assert.equal(rig.graph.target?.cls, 'Gate', 'the folded call is a retarget to what the user last pointed at');
  });
});

// ── tensorviz.openProject ──────────────────────────────────────────────────
//
// The project surface's own way in (step 2.5). What these pin is that the
// session opens with NO class target: the subject is the workspace, and a class
// is something revealed inside it afterwards rather than chosen to get in.
suite('localBrowserGraph.openProject', () => {
  setup(() => {
    reset();
    resetEditors();
    state.workspaceFolders = [{ uri: { fsPath: ROOT } }];
  });

  teardown(() => {
    for (const rig of made.splice(0)) {
      rig.graph.dispose();
    }
  });

  test('opens a session for the workspace folder without naming a class', async () => {
    const rig = makeGraph([{ classes: [NET] }]);
    state.workspaceFolders = [{ uri: { fsPath: ROOT } }];

    await rig.graph.openProject();

    assert.equal(rig.mintCalls.length, 1, 'one session minted for the project');
    assert.equal(rig.opened.length, 1, 'the page was opened');
    assert.equal(
      rig.graph.target,
      undefined,
      'no class target: the project is the subject, and inventing one here would make ' +
        'every later staleness check compare against a class nobody chose',
    );
  });

  test('refuses a window with no folder, before the backend is touched', async () => {
    const rig = makeGraph([{ classes: [NET] }]);
    state.workspaceFolders = undefined;

    await rig.graph.openProject();

    assert.equal(rig.ensureCalls(), 0, 'nothing starts for a window with no project');
    assert.equal(rig.mintCalls.length, 0);
    assert.equal(state.warningMessages.length, 1);
    assert.match(state.warningMessages[0], /Open a folder first/);
  });

  test('refuses an unsupported remote window without touching the network', async () => {
    const rig = makeGraph([{ classes: [NET] }], { remoteName: () => 'dev-container' });
    state.workspaceFolders = [{ uri: { fsPath: ROOT } }];

    await rig.graph.openProject();

    assert.equal(rig.mintCalls.length, 0);
    assert.equal(rig.ensureCalls(), 0);
    assert.match(state.warningMessages[0], /not supported/);
  });

  test('an unnamed class opens the project instead of asking which one', async () => {
    // What the Quick Pick used to do. The file has three nn.Module classes and the
    // cursor is inside none of them — formerly the exact trigger for the picker.
    const rig = makeGraph([{ classes: [NET, GATE, UNSAVED] }]);
    openStubEditor(RESID, SAVED);
    // An answer left on the queue: if a picker were still shown it would consume this.
    state.quickPickAnswers.push('Gate');

    await rig.graph.open({ uri: RESID });

    assert.equal(rig.opened.length, 1, 'a page opened, unasked');
    assert.equal(rig.graph.target, undefined, 'and it is the project, not a class');
    assert.deepEqual(state.quickPickAnswers, ['Gate'], 'no picker was shown');
  });

  test('a second invocation re-indexes in place rather than opening a second page', async () => {
    const rig = makeGraph([{ classes: [NET] }]);
    state.workspaceFolders = [{ uri: { fsPath: ROOT } }];

    await rig.graph.openProject();
    await rig.graph.openProject();

    assert.equal(rig.mintCalls.length, 1, 'the live session is reused');
    assert.equal(rig.opened.length, 1, 'no second browser tab');
  });
});

/** Exercise the shared controllers with a real native-selected projection.
 * Only backend responses, the trace worker and the user's diff decision are fake. */
suite('LocalBrowserGraph browser transport ownership', () => {
  const owner = 'mcp-session';
  const edit = { kind: 'remove_layer' as const, nodeId: 'fwd' };
  const inputs = [{ shape: [1, 8], dtype: 'float32' as const, fill: 'ones' as const }];
  const after = SAVED.replace('return x', 'return x + 1');
  const digest = (text: string) => createHash('sha256').update(text).digest('hex');
  let rig: GraphRig, service: McpGraphService, source: LocalBrowserSourceEdits;
  let messages: any[];
  let preparing: { signal: AbortSignal; finish(): void } | undefined;
  let review: { signal: AbortSignal; finish(approved: boolean): void } | undefined;
  let execution: { cancelled: boolean; finish(): void } | undefined;

  setup(async () => {
    reset(); resetEditors();
    state.workspaceFolders = [{ uri: { fsPath: ROOT } }];
    openStubEditor(RESID, SAVED);
    messages = []; preparing = undefined; review = undefined; execution = undefined;
    mock.method(sourceReview, 'reviewSource', async (_title: string, artifact: CodeArtifact, signal: AbortSignal) => {
      assert.equal(artifact.before, SAVED); assert.equal(artifact.after, after);
      return new Promise<boolean>(resolve => {
        review = { signal, finish: resolve };
        signal.addEventListener('abort', () => resolve(false), { once: true });
      });
    });
    const backend = makeFetch([], { spans: SPANS,
      document: { view: {}, semantic: { nodes: [{ node_id: 'fwd' }], edges: [] } } });
    const profile = { kind: 'model', model: 'models.resid:Net', mode: 'eval', device: 'cpu', grad_enabled: false,
      init: { args: [], kwargs: {} }, inputs: { args: [{ kind: 'tensor', shape: [1, 8], dtype: 'float32',
        requires_grad: false, generator: { kind: 'full', value: 1 } }], kwargs: {} }, limits: { timeout_ms: 1000 } };
    rig = makeGraph([], {
      profiles: {
        snapshot: async (_root, selected) => (selected
          ? { guards: {}, resolution: { status: 'resolved', spec: { profile, identity: { profile: selected },
            files: [], interpreter: { executable: '/python' }, dependencyScope: 'partial' } } }
          : { profiles: ['one'], profileDefinitions: { one: profile }, guards: {}, resolution: { status: 'needs_setup', requirements: [
            { profile: null, field: ['profiles'], code: 'ambiguous', phase: 'static', reason: 'Choose.', remedy: 'Choose.' },
          ] } }) as never,
        choose: async () => undefined,
      },
      fetchImpl: async (url, init) => {
        if (String(url).endsWith('/buffer-classes')) {
          return new Promise<Response>((resolve, reject) => {
            const signal = init!.signal!;
            preparing = { signal, finish: () => resolve(new Response(JSON.stringify({ classes: [NET] }))) };
            signal.addEventListener('abort', () => reject(new Error('cancelled')), { once: true });
          });
        }
        if (String(url).endsWith('/apply')) {
          return new Response(JSON.stringify({ edits: [{ range: { start: [6, 0], end: [7, 0] },
            newText: '        return x + 1\n' }], refused: [], spanmap: {}, source_digest: digest(after) }));
        }
        return backend.fetchImpl(url, init);
      },
      trace: { trusted: async () => true, settings: () => ({ onSave: false, tier: 'real' }), worker: {
        run: () => new Promise<TraceOutcome>(resolve => {
          // A worker timeout is a completed failed run without changing the projection.
          execution = { cancelled: false, finish: () => resolve({ ok: false, timedOut: true, exitCode: 1,
            reason: 'timeout', detail: 'Deferred test worker timed out.' }) };
        }),
        cancel: () => { if (execution) { execution.cancelled = true; execution.finish(); } },
      } },
    });
    rig.graph.attachCanvas(message => { messages.push(message); });
    await rig.graph.open({ uri: RESID, cls: 'Net', sourceOnly: true }, 'canvas'); await settle();
    service = new McpGraphService('host', () => rig.graph.mcpModel);
    // Observe the real reservation, so a retained MCP record cannot hide an aborted review.
    source = (rig.graph as unknown as { sourceEdits: LocalBrowserSourceEdits }).sourceEdits;
    assert.ok(rig.graph.mcpModel?.current);
  });
  teardown(async () => {
    for (const item of made.splice(0)) item.graph.dispose();
    execution?.finish(); preparing?.finish(); review?.finish(false);
    await settle(); mock.restoreAll();
  });
  const ref = () => service.reference(rig.graph.mcpModel!);
  const startRun = () => rig.graph.mcpRuns.start(owner, service, ref(), inputs).runId as string;
  const startProposal = () => rig.graph.mcpProposals.start(owner, service, ref(), edit).proposalId as string;
  async function transition(phase: string) {
    // Connecting must be emitted by openSelectedInBrowser itself, as in production.
    if (!rig.transports.length) await rig.graph.openSelectedInBrowser();
    if (phase !== 'connecting') rig.transports[0].setPhase(phase);
    await settle();
  }
  function ui(type: 'validateRequest' | 'sourceEditRequest', requestId: string): LocalBrowserMessage {
    const common = { requestId, target: rig.graph.target!, protocolVersion: 1 as const,
      workspaceSessionId: 'ui-session', messageId: requestId };
    return type === 'validateRequest' ? { ...common, type } : { ...common, type, edit };
  }

  test('native candidate ownership and replies survive browser status changes', async () => {
    rig.graph.setCanvasView('native-view', true);
    rig.graph.handleCanvasMessage({ ...ui('sourceEditRequest', 'native-proposal'), viewId: 'native-view' } as LocalBrowserMessage);
    await settle(); assert.ok(preparing); preparing.finish(); await settle(); assert.ok(review);
    assert.ok(messages.some(message => message.type === 'sourceEditReview' && message.requestId === 'native-proposal'));
    await transition('reconnecting');
    assert.equal(review.signal.aborted, false);
    const request: LocalBrowserMessage = { type: 'candidateRequest', viewId: 'native-view', proposalId: 'native-proposal',
      action: { kind: 'list' }, requestId: 'candidate-list', target: rig.graph.target!,
      workspaceSessionId: 'ui-session', messageId: 'candidate-message', protocolVersion: 1 };
    rig.graph.handleCanvasMessage(request);
    assert.deepEqual(messages.at(-1).result, { candidates: [], page: null });
    assert.equal(rig.transports[0].published.some(message => message.type === 'candidateResult'), false,
      'native review results are not broadcast to browser views');
    review.finish(false); await settle();
    assert.equal(messages.at(-1).type, 'sourceEditResult');
    assert.equal(messages.at(-1).requestId, 'native-proposal');
  });

  for (const phase of ['connecting', 'reconnecting', 'ended']) {
    test(`MCP validation survives browser ${phase}`, async () => {
      if (phase !== 'connecting') await rig.graph.openSelectedInBrowser();
      const run = startRun(); await settle();
      assert.ok(execution, 'validation must reach the deferred worker');
      await transition(phase);
      assert.equal(execution.cancelled, false, 'browser status must not abort MCP execution');
      assert.equal(rig.graph.mcpRuns.get(owner, run).status, 'running');
      execution.finish(); await settle();
      assert.equal(rig.graph.mcpRuns.get(owner, run).status, 'failed');
    });

    for (const stage of ['preparing', 'review']) {
      test(`MCP ${stage} and its active source request survive browser ${phase}`, async () => {
        if (phase !== 'connecting') await rig.graph.openSelectedInBrowser();
        const proposal = startProposal(); await settle();
        assert.ok(preparing, 'proposal must enter source preparation');
        const requestId = source.activeRequestId;
        assert.equal(requestId, `mcp-${proposal}`);
        if (stage === 'review') { preparing.finish(); await settle(); assert.ok(review, 'full diff must reach review'); }
        await transition(phase);
        assert.equal(preparing.signal.aborted, false, 'browser status must not cancel the source reservation');
        assert.equal(source.activeRequestId, requestId);
        assert.equal(rig.graph.mcpProposals.get(owner, proposal).status, stage);
        if (!review) { preparing.finish(); await settle(); }
        assert.ok(review); assert.equal(review.signal.aborted, false);
        review.finish(false); await settle();
        assert.equal(rig.graph.mcpProposals.get(owner, proposal).status, 'cancelled');
        assert.equal(source.activeRequestId, undefined);
        assert.equal(editorState.appliedEdits.length, 0);
      });
    }

    for (const native of [false, true]) {
      test(`${native ? 'native' : 'browser'} UI cancellation policy survives browser ${phase}`, async () => {
        await rig.graph.openSelectedInBrowser();
        // Renderer-chosen MCP-looking IDs do not establish MCP ownership.
        const send = (message: LocalBrowserMessage) => native
          ? rig.graph.handleCanvasMessage(message) : rig.transports[0].fire(message);
        send(ui('validateRequest', 'mcp-ui-run')); send(ui('sourceEditRequest', 'mcp-ui-review'));
        await settle(); approveExecution(rig.graph); await settle(); assert.ok(execution); assert.ok(preparing);
        preparing.finish(); await settle(); assert.ok(review);
        const sourceRequest = source.activeRequestId;
        assert.ok(sourceRequest);
        rig.transports[0].setPhase(phase); await settle();
        assert.equal(execution.cancelled, !native);
        assert.equal(review.signal.aborted, !native);
        assert.equal(source.activeRequestId, native ? sourceRequest : undefined);
        if (native) { execution.finish(); review.finish(false); await settle(); }
        assert.equal(messages.find(message => message.type === 'validateResult')?.status, native ? 'failed' : 'cancelled');
        assert.equal(messages.find(message => message.type === 'sourceEditResult')?.status, native ? 'cancelled' : undefined,
          'disconnected browser review replies are revoked; native review replies remain deliverable');
      });
    }
  }

  test('completed MCP runs and proposals survive browser opening, reconnect, disconnect and remint', async () => {
    const run = startRun(), proposal = startProposal(); await settle();
    assert.ok(execution); assert.ok(preparing);
    execution.finish(); preparing.finish(); await settle(); assert.ok(review);
    review.finish(false); await settle();
    const runResult = rig.graph.mcpRuns.get(owner, run), proposalResult = rig.graph.mcpProposals.get(owner, proposal);
    assert.equal(runResult.status, 'failed'); assert.equal(proposalResult.status, 'cancelled');
    assert.ok(runResult.finishedAt); assert.ok(proposalResult.finishedAt);
    for (const phase of ['connecting', 'reconnecting', 'connected', 'ended']) {
      await transition(phase);
      assert.deepEqual(rig.graph.mcpRuns.get(owner, run), runResult);
      assert.deepEqual(rig.graph.mcpProposals.get(owner, proposal), proposalResult);
    }
    await rig.graph.openSelectedInBrowser();
    assert.equal(rig.mintCalls.length, 2);
    assert.deepEqual(rig.graph.mcpRuns.get(owner, run), runResult);
    assert.deepEqual(rig.graph.mcpProposals.get(owner, proposal), proposalResult);
    rig.graph.mcpRuns.cancelOwner(owner); rig.graph.mcpProposals.cancelOwner(owner);
    assert.throws(() => rig.graph.mcpRuns.get(owner, run), /unavailable/);
    assert.throws(() => rig.graph.mcpProposals.get(owner, proposal), /unavailable/);
  });

  test('an unaccepted MCP proposal cannot shelter the active browser review from disconnect', async () => {
    await rig.graph.openSelectedInBrowser();
    rig.transports[0].fire(ui('sourceEditRequest', 'browser-review'));
    await settle(); assert.ok(preparing); preparing.finish(); await settle(); assert.ok(review);
    // The MCP record exists, but has not acquired the shared source reservation.
    const sourceRequest = source.activeRequestId;
    assert.ok(sourceRequest);
    const proposal = startProposal();
    assert.equal(source.activeRequestId, sourceRequest);
    rig.transports[0].setPhase('reconnecting'); await settle();
    assert.equal(review.signal.aborted, true);
    assert.equal(source.activeRequestId, undefined);
    assert.equal(rig.graph.mcpProposals.get(owner, proposal).status, 'refused');
  });

  test('HTTP MCP session termination cancels its retained graph work after browser disconnect', async () => {
    let sessionId!: string;
    const endpoint = new GraphMcpServer((server, id) => {
      sessionId = id;
      registerMcpValidationTools(server, service, rig.graph.mcpRuns, id);
      registerMcpProposalTools(server, service, rig.graph.mcpProposals, id);
      return () => { rig.graph.mcpRuns.cancelOwner(id); rig.graph.mcpProposals.cancelOwner(id); };
    });
    const client = new Client({ name: 'ownership-test', version: '1' });
    try {
      await endpoint.start();
      const connection = endpoint.connection!;
      const transport = new StreamableHTTPClientTransport(new URL(connection.url));
      await client.connect(transport);
      const run = (await client.callTool({ name: 'validate_model', arguments: { ...ref(), inputs } })).structuredContent as { runId: string };
      const proposal = (await client.callTool({ name: 'propose_edit', arguments: { ...ref(), edit } })).structuredContent as { proposalId: string };
      await settle(); assert.ok(execution); assert.ok(preparing);
      preparing.finish(); await settle(); assert.ok(review);
      await transition('ended');
      assert.equal(rig.graph.mcpRuns.get(sessionId, run.runId).status, 'running');
      assert.equal(rig.graph.mcpProposals.get(sessionId, proposal.proposalId).status, 'review');
      await transport.terminateSession(); await settle();
      assert.equal(execution.cancelled, true); assert.equal(review.signal.aborted, true);
      assert.equal(source.activeRequestId, undefined);
      assert.throws(() => rig.graph.mcpRuns.get(sessionId, run.runId), /unavailable/);
      assert.throws(() => rig.graph.mcpProposals.get(sessionId, proposal.proposalId), /unavailable/);
    } finally { await client.close(); await endpoint.close(); }
  });

  for (const teardown of ['owner', 'window']) {
    test(`${teardown} teardown still aborts MCP execution and active review and deletes their records`, async () => {
      const run = startRun(), proposal = startProposal(); await settle();
      assert.ok(execution); assert.ok(preparing);
      preparing.finish(); await settle(); assert.ok(review); assert.ok(source.activeRequestId);
      rig.graph.mcpRuns.cancelOwner('other-session'); rig.graph.mcpProposals.cancelOwner('other-session');
      assert.equal(execution.cancelled, false); assert.equal(review.signal.aborted, false);
      if (teardown === 'window') rig.graph.dispose();
      else { rig.graph.mcpRuns.cancelOwner(owner); rig.graph.mcpProposals.cancelOwner(owner); }
      await settle();
      assert.equal(execution.cancelled, true); assert.equal(review.signal.aborted, true);
      assert.equal(source.activeRequestId, undefined);
      assert.throws(() => rig.graph.mcpRuns.get(owner, run), /unavailable/);
      assert.throws(() => rig.graph.mcpProposals.get(owner, proposal), /unavailable/);
      assert.equal(editorState.appliedEdits.length, 0);
    });
  }
});
