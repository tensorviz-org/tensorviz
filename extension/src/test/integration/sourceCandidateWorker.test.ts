import * as assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import { existsSync } from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import type * as vscode from 'vscode';
import { TraceWorker, type WorkerEnvelope } from '../../worker';
import { candidateRequest, candidateEvidence, type SourceCandidateInput } from '@tensorviz/graph-host/candidateExecution';
import type { CandidateRequest, ExecutionSnapshot } from '../../executionPlan';
import { executionSpecSchema } from '@tensorviz/graph-host/executionSpec';
import { CandidateReview } from '../../candidateReview';
import { reset, resetEditors } from '../vscodeStub';

const repo = path.resolve(__dirname, '../../../..'), backend = path.join(repo, 'backend');
const python = process.env.TENSORVIZ_TEST_PYTHON || path.join(repo, 'venv/bin/python');
const transport = require('@tensorviz/graph-host/executionPlanCore') as typeof import('@tensorviz/graph-host/executionPlanCore');
const configFile = 'width_config.py', config = 'WIDTH = 12\nHEADS = 3\n';
const source = `import torch
from width_config import WIDTH, HEADS
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(4, WIDTH)
        self.down = torch.nn.Linear(12, 4)
    def forward(self, x):
        x = self.up(x).reshape(x.shape[0], HEADS, -1)
        return self.down(x.flatten(1))
`;

suite('exact source-set candidate with the real selected interpreter', function () {
  this.timeout(40000);
  let root: string, file: string, worker: TraceWorker, input: SourceCandidateInput;
  let requests: CandidateRequest[], temporary: string[];
  const originalTransport = transport.candidateRequestFile;
  suiteSetup(async () => {
    reset(); resetEditors();
    assert.ok(existsSync(python), 'Set TENSORVIZ_TEST_PYTHON to a Torch-capable interpreter.');
    root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-source-candidate-')));
    file = path.join(root, 'model.py');
    await fs.writeFile(file, source); await fs.writeFile(path.join(root, configFile), config);
    await fs.writeFile(path.join(root, 'unchanged.py'), '# tracked but not part of the proposal\n');
    worker = new TraceWorker(backend, { appendLine() {} } as unknown as vscode.OutputChannel,
      { interpreter: async () => python });
    let snapshot!: ExecutionSnapshot;
    const outcome = await worker.run({ workspace: root, file, cls: 'Model', timeoutMs: 20000, attemptId: 'source-base',
      execution: executionSpecSchema.parse({ inputs: { args: [{ shape: [2, 4], dtype: 'float32', fill: 'ones' }] } }),
      onResolvedPlan(value) { snapshot = value as ExecutionSnapshot; } });
    assert.equal(outcome.ok, true, JSON.stringify(outcome));
    input = { version: 2, plan: snapshot.plan, capture: (outcome.envelope!.result as any).diagnostics.semantic_capture,
      reviewedSources: [
        { file: 'model.py', beforeDigest: snapshot.plan.source.files.find(item => item.path === 'model.py')!.digest,
          source: source.replace('Linear(12, 4)', 'Linear(16, 4)') },
        { file: configFile, beforeDigest: snapshot.plan.source.files.find(item => item.path === configFile)!.digest,
          source: 'WIDTH = 16\nHEADS = 4\n' },
      ] };
  });
  setup(() => {
    requests = []; temporary = [];
    transport.candidateRequestFile = async request => {
      const result = await originalTransport(request);
      requests.push(structuredClone(request)); temporary.push(result.file); return result;
    };
  });
  teardown(() => {
    transport.candidateRequestFile = originalTransport;
    for (const file of temporary) assert.equal(existsSync(path.dirname(file)), false, 'private candidate folder must be removed');
  });
  suiteTeardown(async () => {
    worker?.dispose(); if (root) await fs.rm(root, { recursive: true, force: true });
    reset(); resetEditors();
  });
  const run = (candidate = input, signal = new AbortController().signal) => worker.runCandidate({ workspace: root,
    file, cls: 'Model', candidate, current: async () => true }, signal);

  test('host accepts the exact two-file worker capture and keeps recipe checks unclaimed', async () => {
    const result = await run();
    assert.equal(result.outcome.ok, true, JSON.stringify(result));
    assert.equal(result.evidence?.execution.version, 2);
    assert.equal(result.evidence?.capture?.status, 'complete');
    assert.equal(result.evidence?.baseCurrent, true);
    assert.equal(result.evidence?.candidateCurrent, true);
    assert.equal(result.evidence?.postconditions, undefined);
    assert.equal(await fs.readFile(file, 'utf8'), source);
    assert.equal(await fs.readFile(path.join(root, configFile), 'utf8'), config);
    const review = new CandidateReview(input.capture, new AbortController().signal, () => true, async () => result);
    await review.run();
    assert.equal(review.snapshot().runtime, 'succeeded');
    assert.equal(review.snapshot().inputs, 'same');
    assert.equal(review.snapshot().state, 'different');
    assert.equal(review.snapshot().semanticPostconditions, 'not_checked');
    await review.finish();

    for (const change of ['missing', 'extra', 'duplicate', 'order', 'before', 'after', 'hiddenFile', 'setup',
      'base', 'run', 'version', 'captureSource', 'capturePlan', 'workerPlan', 'recipeVerdict', 'unchangedSource', 'unreviewedChange']) {
      const envelope: WorkerEnvelope = structuredClone(result.outcome.envelope!);
      const execution = envelope.worker.candidateExecution as any;
      if (change === 'missing') execution.changes.pop();
      if (change === 'extra') execution.changes.push({ ...execution.changes[0], file: 'unreviewed.py' });
      if (change === 'duplicate') execution.changes.push(execution.changes[0]);
      if (change === 'order') execution.changes.reverse();
      if (change === 'before') execution.changes[0].beforeDigest = '0'.repeat(64);
      if (change === 'after') execution.changes[0].afterDigest = '0'.repeat(64);
      if (change === 'hiddenFile') execution.plan.source.files.push({ ...execution.plan.source.files[0], path: 'hidden.py' });
      if (change === 'setup') execution.plan.execution.seed++;
      if (change === 'base') execution.baseCaptureId = 'another-capture';
      if (change === 'run') execution.id = execution.plan.id = 'another-run';
      if (change === 'version') execution.version = 1;
      if (change === 'captureSource') (envelope.result as any).diagnostics.semantic_capture.source = input.plan.source;
      if (change === 'capturePlan') (envelope.result as any).diagnostics.execution.planDigest = 'invalid';
      if (change === 'workerPlan') envelope.worker.planId = 'another-run';
      if (change === 'recipeVerdict') envelope.worker.candidatePostconditions = { status: 'verified' };
      if (change === 'unchangedSource') execution.plan.source.files[0].digest = input.plan.source.files[0].digest;
      if (change === 'unreviewedChange') execution.plan.source.files.find((item: any) => item.path === 'unchanged.py').digest = '0'.repeat(64);
      assert.equal(candidateEvidence(envelope, requests[0]), undefined, change);
    }
  });

  test('each changed file is independently bound and all source limits apply before transport', () => {
    for (const change of ['empty', 'duplicate', 'untracked', 'before', 'unchanged', 'nonPython', 'sourceType',
      'fileBytes', 'utf8Bytes', 'totalBytes', 'fileCount', 'extraField', 'capture', 'source', 'surrogate']) {
      const candidate = structuredClone(input) as any;
      if (change === 'empty') candidate.reviewedSources = [];
      if (change === 'duplicate') candidate.reviewedSources.push(candidate.reviewedSources[0]);
      if (change === 'untracked') candidate.reviewedSources[1].file = 'untracked.py';
      if (change === 'before') candidate.reviewedSources[1].beforeDigest = '0'.repeat(64);
      if (change === 'unchanged') candidate.reviewedSources[1].source = config;
      if (change === 'nonPython') candidate.reviewedSources[1].file = 'tensorviz.toml';
      if (change === 'sourceType') candidate.reviewedSources[1].source = 4;
      if (change === 'fileBytes') candidate.reviewedSources[1].source = '#'.repeat(262145);
      if (change === 'utf8Bytes') candidate.reviewedSources[1].source = '🙂'.repeat(65537);
      if (change === 'totalBytes') {
        candidate.reviewedSources = Array.from({ length: 5 }, (_, i) => ({
          ...candidate.reviewedSources[0], file: `file${i}.py`, source: '#'.repeat(262144) }));
      }
      if (change === 'fileCount') candidate.reviewedSources = Array.from({ length: 17 }, (_, i) => ({
        ...candidate.reviewedSources[0], file: `file${i}.py` }));
      if (change === 'totalBytes' || change === 'fileCount') {
        candidate.plan.source.files.push(...candidate.reviewedSources.map((item: any) => ({
          ...candidate.plan.source.files[0], path: item.file, digest: item.beforeDigest })));
        candidate.capture.source = structuredClone(candidate.plan.source);
      }
      if (change === 'extraField') candidate.intent = { kind: 'set_literal' };
      if (change === 'capture') candidate.capture.attemptId = 'another-run';
      if (change === 'source') candidate.capture.source.id = 'another-source';
      if (change === 'surrogate') candidate.reviewedSources[1].source = '\ud800';
      assert.throws(() => candidateRequest(candidate), /candidate_/, change);
    }
    assert.equal(requests.length, 0);
  });

  test('request snapshots cannot be changed while an asynchronous candidate is running', () => {
    const candidate = structuredClone(input), request = candidateRequest(candidate);
    candidate.reviewedSources[1].source += '\nprivate_late_edit()';
    assert.doesNotMatch(request.reviewedSources[1].source, /private_late_edit/);
    assert.notEqual(request.id, candidate.plan.id);
  });

  test('documented exact byte and file boundaries accept without executing source', () => {
    const bounded = (count: number, size: number) => {
      const candidate = structuredClone(input);
      candidate.reviewedSources = Array.from({ length: count }, (_, i) => ({
        ...candidate.reviewedSources[0], file: `boundary${i}.py`, source: '#'.repeat(size) }));
      candidate.plan.source.files.push(...candidate.reviewedSources.map(item => ({
        ...candidate.plan.source.files[0], path: item.file, digest: item.beforeDigest })));
      candidate.capture.source = structuredClone(candidate.plan.source);
      return candidate;
    };
    const perFile = bounded(1, 262144);
    assert.doesNotThrow(() => candidateRequest(perFile));
    perFile.reviewedSources[0].source += '#';
    assert.throws(() => candidateRequest(perFile), /candidate_/);
    const total = bounded(5, 209715);
    total.reviewedSources[0].source += '#'; // exactly 1 MiB, every file below its own cap
    assert.doesNotThrow(() => candidateRequest(total));
    total.reviewedSources[0].source += '#';
    assert.throws(() => candidateRequest(total), /candidate_/);
    const count = bounded(16, 1);
    assert.doesNotThrow(() => candidateRequest(count));
    count.reviewedSources.push({ ...count.reviewedSources[0], file: 'boundary16.py' });
    count.plan.source.files.push({ ...count.plan.source.files[0], path: 'boundary16.py' });
    count.capture.source = structuredClone(count.plan.source);
    assert.throws(() => candidateRequest(count), /candidate_/);
    assert.equal(requests.length, 0);
  });

  test('broken head divisibility produces a failed partial capture, never a verified recipe', async () => {
    const candidate = structuredClone(input);
    candidate.reviewedSources[1].source = 'WIDTH = 16\nHEADS = 3\n';
    const result = await run(candidate);
    assert.equal(result.outcome.ok, false);
    assert.equal(result.evidence?.capture?.status, 'partial');
    assert.equal(result.evidence?.postconditions, undefined);
    assert.equal(result.evidence?.baseCurrent, true);
  });

  test('either stale file and cancelled launches cannot execute the proposed source', async () => {
    for (const changed of ['model.py', configFile]) {
      const location = path.join(root, changed), before = await fs.readFile(location, 'utf8');
      try {
        await fs.appendFile(location, '# changed after review');
        const result = await run();
        assert.equal(result.outcome.ok, false, changed);
        assert.equal(result.evidence, undefined, changed);
      } finally { await fs.writeFile(location, before); }
    }
    const cancelled = new AbortController(); cancelled.abort();
    const result = await run(input, cancelled.signal);
    assert.equal(result.outcome.ok, false);
    assert.equal(result.evidence, undefined);
  });
});
