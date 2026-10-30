import * as assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import { existsSync } from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { createHash } from 'node:crypto';
import type * as vscode from 'vscode';
import { TraceWorker } from '../../worker';
import type { ExecutionSnapshot } from '../../executionPlan';
import { executionSpecSchema } from '@tensorviz/graph-host/executionSpec';
import { candidateEvidence, type SourceCandidateInput } from '@tensorviz/graph-host/candidateExecution';
import { CandidateHistory } from '../../candidateHistory';
import { CandidateReview } from '../../candidateReview';
import { isLocalBrowserMessage } from '@tensorviz/contracts/protocol';

const repo = path.resolve(__dirname, '../../../..'), backend = path.join(repo, 'backend');
const python = process.env.TENSORVIZ_TEST_PYTHON || path.join(repo, 'venv/bin/python');
const sha = (text: string | Buffer) => createHash('sha256').update(text).digest('hex');
const source = `import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left = torch.nn.Linear(3, 2, bias=False)
        self.right = self.left
    def forward(self, x):
        return self.left(x) + self.right(x)
`;

suite('checkpoint evidence for exact reviewed candidates', function () {
  this.timeout(40000);
  let root: string, file: string, worker: TraceWorker, base: SourceCandidateInput, checkpointDigest: string;
  suiteSetup(async () => {
    assert.ok(existsSync(python), 'Set TENSORVIZ_TEST_PYTHON to a Torch-capable interpreter.');
    root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-candidate-checkpoint-')));
    file = path.join(root, 'model.py'); await fs.writeFile(file, source);
    await promisify(execFile)(python, ['-c', `import torch,sys
weight=torch.ones(2,3,dtype=torch.float64)
torch.save({'left.weight':weight,'right.weight':weight},sys.argv[1])`, path.join(root, 'weights.pt')], {timeout:20000});
    checkpointDigest = sha(await fs.readFile(path.join(root, 'weights.pt')));
    worker = new TraceWorker(backend, {appendLine() {}} as unknown as vscode.OutputChannel, {interpreter:async () => python});
    let snapshot!: ExecutionSnapshot;
    const outcome = await worker.run({workspace:root, file, cls:'Model', timeoutMs:20000,
      execution:executionSpecSchema.parse({inputs:{args:[{shape:[2,3],dtype:'float32',fill:'ones'}]},
        checkpoint:{file:'weights.pt',strict:true}}), onResolvedPlan(value) { snapshot = value as ExecutionSnapshot; }});
    assert.equal(outcome.ok, true, JSON.stringify(outcome));
    base = {version:2, plan:snapshot.plan, capture:(outcome.envelope!.result as any).diagnostics.semantic_capture, reviewedSources:[]};
  });
  suiteTeardown(async () => {
    worker?.dispose(); if (root) await fs.rm(root, {recursive:true,force:true});
  });

  for (const kind of ['converted', 'size_mismatch', 'forward_failure', 'custom_loader_failure'] as const) {
    test(`${kind} retains the actual load outcome separately from forward success`, async () => {
      const reviewed = kind === 'size_mismatch' ? source.replace('Linear(3, 2', 'Linear(3, 4')
        : kind === 'forward_failure' ? source.replace('return self.left(x) + self.right(x)', 'raise RuntimeError("candidate_forward_failed")')
          : kind === 'custom_loader_failure' ? source.replace('    def forward', '    def load_state_dict(self, state_dict, strict=True):\n        raise RuntimeError("custom_loader_failure")\n    def forward')
          : source.replace('return self.left(x) + self.right(x)', 'return self.left(x) + self.right(x) + 1');
      const loadFailed = kind === 'size_mismatch' || kind === 'custom_loader_failure';
      const candidate: SourceCandidateInput = {...base, reviewedSources:[{file:'model.py', beforeDigest:sha(source),source:reviewed}]};
      const result = await worker.runCandidate({workspace:root,file,cls:'Model',candidate,current:async () => true}, new AbortController().signal);
      assert.equal(result.outcome.ok, kind === 'converted', JSON.stringify(result));
      assert.equal(result.outcome.executionCurrent, true, 'An unchanged failed checkpoint load needs its own freshness proof, independent of a forward result.');
      const report = result.evidence?.checkpoint;
      assert.ok(report, 'The exact candidate must retain its loader report, even without a forward capture.');
      assert.equal(report.loadStatus, loadFailed ? 'failed' : 'succeeded');
      assert.equal(report.sourceDigest, sha(reviewed));
      assert.equal(report.checkpoint.digest, checkpointDigest);
      assert.equal(report.entries.length, 2);
      assert.equal(report.entries[0].shape, kind === 'size_mismatch' ? 'different' : 'same');
      assert.equal(report.entries[0].dtype, 'different', 'float64 checkpoint metadata remains distinct from converted float32 model state');
      assert.equal(report.entries[0].registered!.objectGroup, report.entries[1].registered!.objectGroup);
      if (loadFailed) {
        assert.equal(result.outcome.envelope!.result, null);
        assert.equal(result.evidence!.capture, undefined, 'a load failure cannot borrow a forward capture');
      } else assert.ok(result.outcome.envelope!.result);

      const history = new CandidateHistory();
      const review = new CandidateReview(base.capture, new AbortController().signal, () => true, async () => result,
        {history,requestId:'candidate-review'});
      const summary = await review.run(), runId = summary.runId!;
      const record = history.get('candidate-review', runId);
      const page = history.inspectCheckpointText('candidate-review', runId, {offset:0,limit:1});
      assert.equal(summary.checkpoint!.loadStatus, report.loadStatus);
      assert.equal(summary.freshness, 'current');
      assert.match(review.resultText(), /Checkpoint digest:/);
      assert.equal(isLocalBrowserMessage({type:'candidateResult',protocolVersion:1,workspaceSessionId:'session',messageId:'result',
        viewId:'view',requestId:'read',proposalId:'proposal', target:{workspaceFolder:root,file:'model.py',cls:'Model',documentVersion:1,
          sourceDigest:sha(source),backendRevision:1},result:{candidates:[record],page:null,checkpoint:page},detail:'Candidate checkpoint evidence.'}), true);
      assert.notDeepEqual(history.inspectCheckpointText('candidate-review', runId, {offset:1,limit:1}), page);
      await review.finish();
      assert.equal(history.get('candidate-review', runId).review, 'historical');
      assert.equal(history.inspectCheckpoint('candidate-review', runId).checkpoint!.loadStatus, report.loadStatus);

      const request = {...candidate, id:result.evidence!.execution.id};
      const original = result.outcome.envelope!;
      if (loadFailed) {
        assert.equal(original.worker.candidatePlanCurrent, true);
        assert.equal(original.worker.candidatePlanDigest, original.worker.planDigest);
        const absent = structuredClone(original);
        delete absent.worker.candidatePlanCurrent; delete absent.worker.candidatePlanDigest;
        assert.equal(candidateEvidence(absent, request)?.candidateCurrent, false, 'older missing receipts remain uncertified');
        const stale = structuredClone(original); stale.worker.candidatePlanCurrent = false;
        assert.equal(candidateEvidence(stale, request)?.candidateCurrent, false);
        for (const defect of ['plan', 'digest', 'missing_digest', 'type', 'result']) {
          const envelope = structuredClone(original);
          if (defect === 'plan') envelope.worker.planId = 'another-run';
          if (defect === 'digest') envelope.worker.candidatePlanDigest = '0'.repeat(64);
          if (defect === 'missing_digest') delete envelope.worker.candidatePlanDigest;
          if (defect === 'type') (envelope.worker as any).candidatePlanCurrent = 'true';
          if (defect === 'result') envelope.result = { diagnostics: { execution: { resolvedPlan: result.evidence!.execution.plan,
            planDigest: envelope.worker.planDigest, planCurrent: true } } };
          assert.equal(candidateEvidence(envelope, request), undefined, defect);
        }
      }
      const raw = (value: typeof original) => (value.result as any)?.diagnostics.execution.checkpointCompatibility
        ?? value.worker.checkpointCompatibility;
      for (const mutation of ['base_source', 'checkpoint_digest', 'checkpoint_file', 'strict', 'key', 'forged_comparison', 'conflict', 'load_status']) {
        const envelope = structuredClone(original), value = raw(envelope);
        if (mutation === 'base_source') value.sourceDigest = sha(source);
        if (mutation === 'checkpoint_digest') value.checkpoint.digest = '0'.repeat(64);
        if (mutation === 'checkpoint_file') value.checkpoint.file = 'other.pt';
        if (mutation === 'strict') value.checkpoint.strict = false;
        if (mutation === 'key') value.checkpoint.key = 'other';
        if (mutation === 'forged_comparison') value.entries[0].shape = 'unknown';
        if (mutation === 'conflict') {
          envelope.worker.checkpointCompatibility = {...value, sourceDigest:'0'.repeat(64)};
          if (envelope.result === null) envelope.result = {diagnostics:{execution:{checkpointCompatibility:value}}};
        }
        if (mutation === 'load_status') {
          if (loadFailed) continue; // a successful load followed by setup failure is possible
          value.loadStatus = 'failed';
        }
        assert.equal(candidateEvidence(envelope, request), undefined, mutation);
      }
      assert.equal(await fs.readFile(file, 'utf8'), source);
      assert.equal(sha(await fs.readFile(path.join(root, 'weights.pt'))), checkpointDigest);
    });
  }

  for (const mutation of ['source', 'import']) test(`a loader with changed ${mutation} identity remains stale with its failed load evidence`, async () => {
    const mutate = mutation === 'source'
      ? '        Path(__file__).write_text("# changed during checkpoint loading\\n")\n'
      : '        import importlib.util\n'
        + '        extra = Path(__file__).with_name("_candidate_dynamic.py")\n'
        + '        extra.write_text("FLAG = 1\\n")\n'
        + '        spec = importlib.util.spec_from_file_location("_candidate_dynamic", extra)\n'
        + '        spec.loader.exec_module(importlib.util.module_from_spec(spec))\n';
    const reviewed = source.replace('    def forward',
      '    def load_state_dict(self, state_dict, strict=True):\n'
      + '        from pathlib import Path\n'
      + mutate
      + '        raise RuntimeError("loader changed candidate source")\n'
      + '    def forward');
    const candidate: SourceCandidateInput = { ...base, reviewedSources: [{ file: 'model.py', beforeDigest: sha(source), source: reviewed }] };
    const result = await worker.runCandidate({ workspace: root, file, cls: 'Model', candidate, current: async () => true }, new AbortController().signal);
    assert.equal(result.outcome.ok, false);
    assert.equal(result.outcome.envelope!.result, null);
    assert.equal(result.outcome.envelope!.worker.candidatePlanCurrent, false);
    if (mutation === 'import') {
      const imported = (result.outcome.envelope!.worker as unknown as { importedSources: { file: string }[] }).importedSources;
      assert.ok(imported.some(item => item.file === '_candidate_dynamic.py'));
    }
    assert.equal(result.evidence?.checkpoint?.loadStatus, 'failed');
    assert.equal(result.evidence?.candidateCurrent, false);
    assert.equal(result.outcome.executionCurrent, false);
    assert.equal(await fs.readFile(file, 'utf8'), source);
    assert.equal(sha(await fs.readFile(path.join(root, 'weights.pt'))), checkpointDigest);
  });
});
