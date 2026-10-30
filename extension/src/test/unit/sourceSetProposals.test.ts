import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { LocalBrowserSourceEdits, type AgentSourceSetEdit, type SourceEditsOptions, type SourceProposalReview } from '../../localBrowserSourceEdits';
import { McpProposals, registerCanvasCodeTools, registerMcpProposalTools } from '../../mcpProposals';
import { GraphMcpServer } from '../../mcpServer';
import { McpGraphService, type McpModel } from '@tensorviz/graph-host/mcpGraph';
import type { ResolvedExecutionPlan } from '../../executionPlan';
import type { CandidateRunResult } from '@tensorviz/graph-host/candidateExecution';
import { sha256 } from '../../sourceChanges';
import { SOURCE_SET_LIMITS } from '@tensorviz/graph-host/workspaceSourceCore';
import { canonicalFixture } from '../../../../tests/fixtures/semantic.cjs';
import { reset, resetEditors, editorState, state } from '../vscodeStub';
const vscode = require('../vscodeStub');

const before = 'from config import WIDTH\nclass Model:\n    width = WIDTH\n';
const after = before + '    output_width = WIDTH\n', config = 'WIDTH = 12\n', changedConfig = 'WIDTH = 16\n';
const tick = () => new Promise<void>(resolve => setImmediate(resolve));
async function until(check: () => boolean) { for (let i = 0; i < 100 && !check(); i++) await tick(); assert.ok(check()); }

suite('reviewed source-set proposals', () => {
  let root: string, host: LocalBrowserSourceEdits, model: McpModel, dependencies: Map<string, string>, capture: ReturnType<typeof canonicalFixture>['capture'];
  let review: NonNullable<SourceEditsOptions['reviewSources']>, syntax: boolean, checked: (() => void) | undefined;
  let current: boolean, captureAvailable: boolean, plan: ResolvedExecutionPlan | undefined, candidateCalls: any[], requests: any[], draftInput: any;
  let token: string;
  let observationReads: unknown[][];
  let candidateResult: CandidateRunResult | undefined;
  const edit = (): AgentSourceSetEdit => ({ kind: 'replace_source_set', sources: [
    { file: 'model.py', beforeDigest: sha256(before), source: after },
    { file: 'config.py', beforeDigest: sha256(config), source: changedConfig },
  ] });
  const propose = (change = edit(), signal?: AbortSignal, onReview?: (value: SourceProposalReview) => void) =>
    host.proposeSourceSet({ requestId: 'source-set', target: model.target, edit: change }, signal, onReview);
  setup(() => {
    reset(); resetEditors(); root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'tensorviz-source-set-')));
    dependencies = new Map([['model.py', sha256(before)], ['config.py', sha256(config)]]);
    for (const [file, source] of [['model.py', before], ['config.py', config]]) {
      fs.writeFileSync(path.join(root, file), source); editorState.documents.set(path.join(root, file), source);
    }
    model = { current: true, target: { workspaceFolder: root, file: 'model.py', cls: 'Model', sourceDigest: sha256(before), documentVersion: 1, backendRevision: 1 },
      projection: { document: { semantic: { nodes: [], edges: [] } }, spanmap: {} } as any };
    capture = canonicalFixture().capture;
    capture.source.files = [...dependencies].map(([file, digest]) => ({ ...capture.source.files[0], path: file, digest, role: 'source' }));
    plan = { id: capture.attemptId, source: capture.source, captureLimits: capture.limits } as ResolvedExecutionPlan;
    current = true; captureAvailable = true; syntax = true; checked = undefined; token = 'private'; candidateCalls = []; requests = []; draftInput = undefined; candidateResult = undefined; observationReads = [];
    review = async (_title, _files, _signal, _candidate, shown) => { shown?.(async () => {}, () => {}); return false; };
    host = new LocalBrowserSourceEdits({ origin: () => 'http://localhost:9999', token: () => token,
      context: () => current ? { projection: model.projection, dependencies } : undefined,
      // CaptureInspector reads defensive copies from the attempt ledger.
      capture: () => { if (!captureAvailable) throw new Error('evicted'); return structuredClone(capture); }, capturePlan: () => structuredClone(plan),
      captureObservations: (target, id, sourceId) => { observationReads.push([target, id, sourceId]); return undefined; },
      runCandidate: async request => { candidateCalls.push(request); return candidateResult ?? { outcome: { ok: false, reason: 'fixture-only refusal', detail: 'No Python runs in this unit fixture.' } }; },
      reviewSources: (...args) => review(...args), publish() {}, channel: { appendLine() {} },
      draft: input => { draftInput = input; return { show() {}, dispose() {} }; },
      fetchImpl: async (_url, init) => { requests.push(JSON.parse(String(init?.body))); checked?.(); return new Response(JSON.stringify(syntax
        ? { classes: [] } : { detail: { error: 'syntax_error', message: 'invalid syntax', line: 3, column: 1 } }), { status: syntax ? 200 : 400 }); },
    });
  });
  teardown(() => { host.dispose(); fs.rmSync(root, { recursive: true, force: true }); reset(); resetEditors(); });

  for (const limit of ['files', 'oneFileBytes', 'totalBytes'] as const) {
    for (const excess of [0, 1]) {
      test(`one-file proposals ${excess ? 'refuse an inventory over' : 'accept an inventory at'} the ${limit} bound without truncation`, async () => {
        const add = (file: string, source: string) => {
          fs.writeFileSync(path.join(root, file), source);
          editorState.documents.set(path.join(root, file), source);
          dependencies.set(file, sha256(source));
        };
        if (limit === 'files') {
          while (dependencies.size < SOURCE_SET_LIMITS.maxSources + excess) add(`dependency_${dependencies.size}.py`, '# dependency\n');
        } else if (limit === 'oneFileBytes') {
          add('config.py', '#'.repeat(SOURCE_SET_LIMITS.maxSourceBytes + excess));
        } else {
          let remaining = SOURCE_SET_LIMITS.maxTotalBytes + excess - Buffer.byteLength(before) - Buffer.byteLength(config);
          while (remaining > 0) {
            const size = Math.min(remaining, SOURCE_SET_LIMITS.maxSourceBytes);
            add(`dependency_${dependencies.size}.py`, '#'.repeat(size)); remaining -= size;
          }
        }
        const originalInventory = new Map(dependencies);
        let shown = 0;
        review = async (_title, files, _signal, _candidate, onShown) => {
          shown++; assert.deepEqual(files.map(item => item.path), ['model.py']);
          onShown?.(async () => {}, () => {}); return false;
        };
        const value = edit(); value.sources = value.sources.slice(0, 1);
        const result = await propose(value);
        assert.equal(result.status, excess ? 'refused' : 'cancelled');
        assert.equal(shown, excess ? 0 : 1);
        if (excess) assert.match(result.detail, /review source inventory exceeds/i);
        assert.deepEqual(dependencies, originalInventory, 'Review must not trim the graph-owned dependency inventory.');
        assert.equal(editorState.appliedEdits.length, 0);
        assert.equal(candidateCalls.length, 0);
      });
    }
  }

  test('one native acceptance applies every exact replacement in a single unsaved WorkspaceEdit', async () => {
    const metadata: SourceProposalReview[] = [];
    review = async (_title, files, _signal, candidate, shown) => {
      assert.equal(editorState.appliedEdits.length, 0); assert.equal(candidate, undefined);
      assert.equal(metadata.length, 0, 'review is not ready until all native diffs are shown');
      shown?.(async () => {}, () => {});
      assert.deepEqual(metadata[0]?.files?.map(item => item.file), ['model.py', 'config.py']);
      assert.deepEqual(files.map(item => item.after), [after, changedConfig]);
      return true;
    };
    assert.equal((await propose(edit(), undefined, value => { metadata.push(value); })).status, 'applied');
    assert.equal(editorState.appliedEdits.length, 1);
    assert.deepEqual(editorState.appliedEdits[0].replacements.map(item => item.newText), [after, changedConfig]);
    assert.deepEqual(requests.map(item => item.file).sort(), ['config.py', 'model.py']);
    assert.ok(requests.every(item => item.check_compilation === true));
    assert.deepEqual(draftInput.artifacts.map((item: any) => item.path), ['model.py', 'config.py']);
    assert.equal(fs.readFileSync(path.join(root, 'model.py'), 'utf8'), before);
    assert.equal(fs.readFileSync(path.join(root, 'config.py'), 'utf8'), config);
    assert.equal(host.appliedSource('source-set'), null, 'Source-only writes have no base capture to compare.');
  });

  test('candidate execution is unavailable before all diffs open and uses the exact complete set afterwards', async () => {
    const value = edit(); value.capture = { captureId: capture.id, sourceId: capture.source.id };
    review = async (_title, _files, _signal, candidate, shown, decided) => {
      assert.ok(candidate); assert.equal(candidateCalls.length, 0);
      assert.throws(() => host.runCandidate('source-set'), /no open native review/);
      shown?.(async () => {}, () => {});
      host.runCandidate('source-set'); await tick();
      assert.equal(candidateCalls.length, 1);
      assert.equal(candidateCalls[0].candidate.version, 2);
      assert.deepEqual(candidateCalls[0].candidate.reviewedSources, value.sources);
      assert.deepEqual(candidateCalls[0].candidate.plan, plan);
      decided?.(); assert.throws(() => host.runCandidate('source-set'), /no open native review/);
      return false;
    };
    assert.equal((await propose(value)).status, 'cancelled');
    assert.equal(editorState.appliedEdits.length, 0);
    assert.equal(host.appliedSource('source-set'), null, 'Discard cannot confirm an Apply receipt.');
  });

  test('a capture-backed Apply confirms every reviewed file; source-only writes and revocation cannot invent evidence', async () => {
    const value = edit(); value.capture = { captureId: capture.id, sourceId: capture.source.id };
    review = async (_title, _files, _signal, _candidate, shown) => {
      assert.equal(host.appliedSource('source-set'), null); shown?.(async () => {}, () => {}); return true;
    };
    assert.equal((await propose(value)).status, 'applied');
    assert.deepEqual(host.appliedSource('source-set')!.files, value.sources.map(item => ({
      file: item.file, beforeDigest: item.beforeDigest, afterDigest: sha256(item.source),
    })));
    host.forgetCandidates('source-set'); assert.equal(host.appliedSource('source-set'), null);
  });

  test('owner revocation while native Apply settles preserves its actual outcome but cannot restore a receipt', async () => {
    const value = edit(); value.capture = { captureId: capture.id, sourceId: capture.source.id };
    review = async (_title, _files, _signal, _candidate, shown) => { shown?.(async () => {}, () => {}); return true; };
    const original = vscode.workspace.applyEdit;
    vscode.workspace.applyEdit = async (workspaceEdit: any) => { const applied = await original(workspaceEdit); host.forgetCandidates('source-set'); return applied; };
    try { assert.equal((await propose(value)).status, 'applied'); assert.equal(host.appliedSource('source-set'), null); }
    finally { vscode.workspace.applyEdit = original; }
  });

  test('acceptance hands the complete reviewed set to saving after cleanup and guards every replacement', async () => {
    const original = vscode.workspace.applyEdit;
    let calls = 0;
    vscode.workspace.applyEdit = async (workspaceEdit: any) => {
      const applied = await original(workspaceEdit);
      if (applied) for (const item of edit().sources) {
        editorState.documents.set(path.join(root, item.file), item.source);
        editorState.versions.set(path.join(root, item.file), 2);
      }
      return applied;
    };
    review = async (_title, _files, _signal, _candidate, shown) => { shown?.(async () => {}, () => {}); return true; };
    (host as unknown as { options: SourceEditsOptions }).options.accepted = async change => {
      calls++;
      assert.equal(host.reviewing, false, 'opening must not cancel its own source reservation');
      assert.deepEqual(change.files, ['model.py', 'config.py']);
      assert.equal(draftInput.canRevise, true);
      assert.equal(change.current(), true);
      editorState.documents.set(path.join(root, 'config.py'), 'WIDTH = 99\n');
      assert.equal(change.current(), false, 'the accepted model file cannot hide a changed configuration file');
      return 'Accepted files changed before saving; no capture started.';
    };
    try {
      const outcome = await propose();
      assert.equal(outcome.status, 'applied'); assert.equal(calls, 1);
      assert.match(outcome.detail, /no capture started/);
    } finally { vscode.workspace.applyEdit = original; }
  });

  test('the final source-set result keeps copied runtime evidence separate from applied buffers', async () => {
    const value = edit(); value.capture = { captureId: capture.id, sourceId: capture.source.id };
    candidateResult = { outcome: { ok: true, executionCurrent: true, recheckExecution: async () => true, exitCode: 0,
      envelope: { protocol_version: 1, result: {}, error: null, spans: {}, worker: {} } }, evidence: {
      execution: { id: 'candidate:source-set', version: 2, plan } as any,
      capture: structuredClone(capture), baseCurrent: true, candidateCurrent: true,
    } };
    const metadata: SourceProposalReview[] = [], originalApply = vscode.workspace.applyEdit;
    vscode.workspace.applyEdit = async (workspaceEdit: any) => {
      const applied = await originalApply(workspaceEdit);
      for (const item of value.sources) {
        const file = path.join(root, item.file);
        editorState.documents.set(file, item.source); editorState.versions.set(file, 2); editorState.dirtyDocuments.add(file);
      }
      return applied;
    };
    review = async (_title, _files, _signal, candidate, shown) => {
      shown?.(async () => {}, () => {});
      await candidate!.run(); assert.equal(candidate!.snapshot().freshness, 'current');
      return true;
    };
    try {
      const outcome = await propose(value, undefined, item => metadata.push(item));
      assert.equal(outcome.status, 'applied'); assert.match(outcome.detail, /Copied candidate runtime: succeeded/);
      assert.deepEqual(observationReads, [[model.target, capture.id, capture.source.id]], 'baseline lookup must name this proposal’s exact displayed capture');
      assert.match(outcome.detail, /Semantic postconditions: not checked/); assert.match(outcome.detail, /Save and Validate/);
      assert.equal(metadata.at(-1)?.candidate?.freshness, 'stale');
      assert.deepEqual(metadata.at(-1)?.files?.map(item => item.file), ['model.py', 'config.py']);
    } finally { vscode.workspace.applyEdit = originalApply; }
  });

  for (const change of ['configBytes', 'configVersion', 'modelBytes', 'inventory', 'trust', 'token', 'target', 'cancel', 'symlink', 'inode', 'capture', 'plan', 'captureConditions', 'planContents']) {
    test(`the complete source set refuses Apply after ${change} changes`, async () => {
      const controller = new AbortController(), value = edit();
      value.capture = { captureId: capture.id, sourceId: capture.source.id };
      review = async (_title, _files, _signal, _candidate, shown) => {
        shown?.(async () => {}, () => {});
        const file = path.join(root, 'config.py');
        if (change === 'configBytes') editorState.documents.set(file, '# changed');
        if (change === 'configVersion') editorState.versions.set(file, 2);
        if (change === 'modelBytes') editorState.documents.set(path.join(root, 'model.py'), '# changed');
        if (change === 'inventory') dependencies.set('new.py', '0'.repeat(64));
        if (change === 'trust') state.isTrusted = false;
        if (change === 'token') token = 'another-backend';
        if (change === 'target') current = false;
        if (change === 'cancel') controller.abort();
        if (change === 'symlink' || change === 'inode') {
          fs.renameSync(file, path.join(root, 'original.py'));
          if (change === 'symlink') fs.symlinkSync(path.join(root, 'original.py'), file); else fs.writeFileSync(file, config);
        }
        if (change === 'capture') captureAvailable = false;
        if (change === 'plan') plan = undefined;
        if (change === 'captureConditions') capture.inputDigest = '0'.repeat(64);
        if (change === 'planContents') plan = { ...plan!, id: 'another-plan' };
        return true;
      };
      assert.notEqual((await propose(value, controller.signal)).status, 'applied');
      assert.equal(editorState.appliedEdits.length, 0);
    });
  }

  test('syntax failures and changed source during preparation never open or replace a native review', async () => {
    let shown = 0; review = async () => { shown++; return true; };
    syntax = false; assert.equal((await propose()).status, 'refused');
    syntax = true; checked = () => editorState.documents.set(path.join(root, 'config.py'), '# changed while parsing');
    assert.equal((await propose()).status, 'refused');
    assert.equal(shown, 0); assert.equal(editorState.appliedEdits.length, 0);
  });

  test('a source-set draft revision preserves every pending file and invalid revisions leave it open', async () => {
    const graph = new McpGraphService('source-set', () => model), proposals = new McpProposals(host), ref = graph.reference(model);
    const artifacts: string[][] = []; let decide!: (value: boolean) => void;
    review = async (_title, files, signal, _candidate, shown) => {
      artifacts.push(files.map(item => item.after));
      shown?.(async () => {}, () => {});
      return new Promise(resolve => { decide = resolve; signal.addEventListener('abort', () => resolve(false), { once: true }); });
    };
    try {
      const first = await proposals.startSourceSet('owner', graph, ref, edit()), id = String(first.proposalId);
      await until(() => proposals.get('owner', id).status === 'review');
      const followup = proposals.beginRevision(`mcp-${id}`, new AbortController().signal);
      assert.deepEqual(followup.files?.map(item => item.file), ['model.py', 'config.py']);
      assert.deepEqual(followup.files.map(item => [item.source, item.baseDigest, item.sourceTruncated]),
        edit().sources.map(item => [item.source, item.beforeDigest, false]), 'follow-ups receive the complete pending files without an agent read');
      assert.equal((await proposals.readSource(graph, ref, 'owner', 'config.py')).pendingDraft?.source, changedConfig);
      assert.equal((await proposals.readSource(graph, ref, 'other', 'config.py')).pendingDraft, undefined);
      const incomplete = edit(); incomplete.sources.pop();
      await assert.rejects(proposals.startSourceSet('owner', graph, ref, incomplete), /Preserve every pending file/);
      syntax = false; await assert.rejects(proposals.startSourceSet('owner', graph, ref, edit()), /syntax error/);
      assert.equal(proposals.get('owner', id).status, 'review'); assert.equal(artifacts.length, 1);
      syntax = true;
      const revised = edit(); revised.sources[0].source += '    heads = 4\n';
      const second = await proposals.startSourceSet('owner', graph, ref, revised);
      await until(() => proposals.get('owner', String(second.proposalId)).status === 'review');
      assert.equal(proposals.get('owner', id).status, 'cancelled');
      assert.deepEqual(artifacts[1], revised.sources.map(item => item.source));
      decide(true); await until(() => proposals.get('owner', String(second.proposalId)).status === 'applied');
      assert.equal(editorState.appliedEdits.length, 1);
    } finally { proposals.clear(); await tick(); }
  });

  test('oversized pending sources retain their manifest and require an explicit read', async () => {
    const graph = new McpGraphService('source-set', () => model), proposals = new McpProposals(host), ref = graph.reference(model);
    review = async (_title, _files, signal, _candidate, shown) => { shown?.(async () => {}, () => {});
      return new Promise(resolve => signal.addEventListener('abort', () => resolve(false), { once: true })); };
    try {
      const large = edit(); large.sources[0].source += `# ${'界'.repeat(12_000)}\n`;
      const first = await proposals.startSourceSet('owner', graph, ref, large), id = String(first.proposalId);
      await until(() => proposals.get('owner', id).status === 'review');
      const context = proposals.beginRevision(`mcp-${id}`, new AbortController().signal);
      assert.equal(context.files.length, 2);
      assert.equal(context.files[0].source, undefined); assert.equal(context.files[0].sourceTruncated, true);
      assert.equal(context.files[0].baseDigest, large.sources[0].beforeDigest);
      assert.equal(context.files[1].source, changedConfig); assert.equal(context.files[1].sourceTruncated, false);
      assert.equal((await proposals.readSource(graph, ref, 'owner', 'model.py')).pendingDraft?.source, large.sources[0].source);
      assert.equal(editorState.appliedEdits.length, 0);
    } finally { proposals.clear(); await tick(); }
  });

  test('explicitly reverting one pending file retains the other changes without writing the reverted file', async () => {
    const graph = new McpGraphService('source-set', () => model), proposals = new McpProposals(host), ref = graph.reference(model);
    let decide!: (value: boolean) => void;
    review = async (_title, _files, signal, _candidate, shown) => { shown?.(async () => {}, () => {});
      return new Promise(resolve => { decide = resolve; signal.addEventListener('abort', () => resolve(false), { once: true }); }); };
    try {
      const first = await proposals.startSourceSet('owner', graph, ref, edit());
      await until(() => proposals.get('owner', String(first.proposalId)).status === 'review');
      proposals.beginRevision(`mcp-${first.proposalId}`, new AbortController().signal);
      const next = edit(); next.sources.pop();
      const second = await proposals.startSourceSet('owner', graph, ref, next, ['config.py']);
      await until(() => proposals.get('owner', String(second.proposalId)).status === 'review');
      decide(true); await until(() => proposals.get('owner', String(second.proposalId)).status === 'applied');
      assert.equal(editorState.appliedEdits[0].replacements.length, 1);
      assert.equal(editorState.appliedEdits[0].replacements[0].newText, after);
    } finally { proposals.clear(); await tick(); }
  });

  for (const change of ['capture', 'plan', 'token', 'inventory', 'inode']) {
    test(`a ${change} change during revision checking retains the prior review`, async () => {
      const graph = new McpGraphService('source-set', () => model), proposals = new McpProposals(host), ref = graph.reference(model);
      review = async (_title, _files, signal, _candidate, shown) => { shown?.(async () => {}, () => {});
        return new Promise(resolve => signal.addEventListener('abort', () => resolve(false), { once: true })); };
      try {
        const firstEdit = edit(); firstEdit.capture = { captureId: capture.id, sourceId: capture.source.id };
        const first = await proposals.startSourceSet('owner', graph, ref, firstEdit), id = String(first.proposalId);
        await until(() => proposals.get('owner', id).status === 'review');
        proposals.beginRevision(`mcp-${id}`, new AbortController().signal);
        checked = () => {
          checked = undefined;
          if (change === 'capture') captureAvailable = false;
          if (change === 'plan') plan = undefined;
          if (change === 'token') token = 'replaced';
          if (change === 'inventory') dependencies.set('new.py', '0'.repeat(64));
          if (change === 'inode') {
            fs.renameSync(path.join(root, 'config.py'), path.join(root, 'original.py'));
            fs.writeFileSync(path.join(root, 'config.py'), config);
          }
        };
        await assert.rejects(proposals.startSourceSet('owner', graph, ref, edit()));
        assert.equal(proposals.get('owner', id).status, 'review');
        assert.equal(editorState.appliedEdits.length, 0);
      } finally { proposals.clear(); await tick(); }
    });
  }

  for (const selection of ['entry', 'imported', 'both'] as const) test('actual Canvas MCP reviews the exact ' + selection + ' file set without source bodies', async () => {
    const graph = new McpGraphService('source-set', () => model), proposals = new McpProposals(host), ref = graph.reference(model);
    review = async (_title, _files, signal, _candidate, shown) => { shown?.(async () => {}, () => {});
      return new Promise(resolve => signal.addEventListener('abort', () => resolve(false), { once: true })); };
    const endpoint = new GraphMcpServer(server => { registerMcpProposalTools(server, graph, proposals, 'owner'); registerCanvasCodeTools(server, graph, proposals, 'owner'); });
    const client = new Client({ name: 'source-set', version: '1' });
    try {
      const { url } = await endpoint.start(); await client.connect(new StreamableHTTPClientTransport(new URL(url)));
      const tools = await client.listTools();
      assert.ok(tools.tools.some(tool => tool.name === 'propose_source_set'));
      assert.ok(!tools.tools.some(tool => tool.name === 'propose_code_edit'), 'freeform edits require explicit file targets');
      const sources = edit().sources.filter(item => selection === 'both' || item.file === (selection === 'entry' ? 'model.py' : 'config.py'));
      const result = await client.callTool({ name: 'propose_source_set', arguments: { ...ref, sources } });
      assert.notEqual(result.isError, true);
      const id = String((result.structuredContent as any).proposalId);
      await until(() => proposals.get('owner', id).status === 'review');
      const detail = await client.callTool({ name: 'get_proposal', arguments: { proposalId: id } });
      assert.deepEqual((detail.structuredContent as any).review.files.map((item: any) => item.file), sources.map(item => item.file));
      assert.doesNotMatch(JSON.stringify(detail.structuredContent), /output_width|WIDTH = 16/);
      assert.equal(editorState.appliedEdits.length, 0); assert.equal(candidateCalls.length, 0);
    } finally { await client.close(); await endpoint.close(); proposals.clear(); await tick(); }
  });
});
