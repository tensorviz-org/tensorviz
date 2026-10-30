import * as assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
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
import { executionObservation } from '@tensorviz/graph-host/executionObservation';
import { CandidateHistory } from '../../candidateHistory';
import { CandidateReview } from '../../candidateReview';

const source = `import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2., 3.]))
    def forward(self, x):
        return x * self.weight
`;
const sha = (text: string) => createHash('sha256').update(text).digest('hex');

suite('exact candidate numerical evidence', function () {
  this.timeout(60_000);
  test('exact reviewed outputs remain independently inspectable without diagnostic execution', async () => {
    const python = process.env.TENSORVIZ_TEST_PYTHON;
    assert.ok(python, 'A real Torch interpreter is required; never skip this oracle.');
    const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-candidate-observations-')));
    const file = path.join(root, 'model.py'), backend = path.resolve(__dirname, '../../../../backend');
    const worker = new TraceWorker(backend, { appendLine() {} } as unknown as vscode.OutputChannel, { interpreter: async () => python });
    try {
      const marker = path.join(root, 'calls.txt');
      const modelSource = source.replace('        return x * self.weight',
        `        with open(${JSON.stringify(marker)}, 'a') as stream:\n            stream.write('forward\\n')\n        return x * self.weight`);
      await fs.writeFile(file, modelSource);
      await promisify(execFile)(python, ['-c', 'import torch,sys; torch.save({"args":(torch.tensor([-1.,2.]),)},sys.argv[1])', path.join(root, 'batch.pt')]);
      let snapshot!: ExecutionSnapshot;
      const baseline = await worker.run({ workspace: root, file, cls: 'Model', timeoutMs: 25_000,
        execution: executionSpecSchema.parse({ inputs: { fixture: 'batch.pt' }, mode: 'eval', seed: 4 }),
        onResolvedPlan(value) { snapshot = value as ExecutionSnapshot; } });
      assert.equal(baseline.ok, true, JSON.stringify(baseline));
      const capture = (baseline.envelope!.result as any).diagnostics.semantic_capture;
      const baselineObservation = executionObservation(baseline.envelope!, snapshot.plan, 'passed', capture);
      assert.ok(baselineObservation);
      const history = new CandidateHistory();
      const ids: string[] = [];
      for (const [kind, expression] of [['reordered', 'torch.flip(x * self.weight, [0])'],
        ['nonfinite', 'x * self.weight / 0']] as const) {
        const reviewed = modelSource.replace('return x * self.weight', `return ${expression}`);
        const candidate: SourceCandidateInput = { version: 2, plan: snapshot.plan, capture,
          reviewedSources: [{ file: 'model.py', beforeDigest: sha(modelSource), source: reviewed }] };
        const result = await worker.runCandidate({ workspace: root, file, cls: 'Model', candidate, current: async () => true }, new AbortController().signal);
        assert.equal(result.outcome.ok, true, `${kind}: ${result.outcome.ok ? 'passed' : result.outcome.reason}`);
        assert.ok(result.evidence?.observations, 'Exact-candidate evidence must preserve numerical observations.');
        const review: CandidateReview = new CandidateReview(capture, new AbortController().signal, () => true, async () => result,
          { history, requestId: 'owner', baseline: baselineObservation });
        const summary = await review.run(); ids.push(summary.runId!);
        const output: any = history.inspectObservations('owner', summary.runId!, { section: 'outputs', limit: 1 });
        assert.equal(output.baselineExecutionId, snapshot.plan.id);
        assert.equal(output.candidateExecutionId, result.evidence.execution.id);
        assert.equal(output.evidence.leaves.items[0].values, undefined);
        assert.deepEqual(output.evidence.leaves.items[0].shape, [2]);
        assert.equal(output.comparison.allClose, false);
        if (kind === 'reordered') assert.equal(output.comparison.items[0].maxAbsDifference, 8);
        else assert.equal(output.comparison.items[0].status, 'nonfinite');
        await review.finish();
        assert.equal(history.get('owner', summary.runId!).review, 'historical');
        for (const key of ['sourceDigest', 'planDigest', 'resolvedPlan']) {
          const bad = structuredClone(result.outcome.envelope!);
          (bad.result as any).diagnostics.execution[key] = key === 'resolvedPlan' ? {} : '0'.repeat(64);
          const evidence = candidateEvidence(bad, { ...candidate, id: result.evidence.execution.id });
          assert.equal(evidence?.observations, undefined, key);
        }
      }
      assert.equal(await fs.readFile(file, 'utf8'), modelSource, 'Private candidate execution must not edit the live source.');
      await fs.writeFile(file, '# edited after all candidate runs\n');
      for (const runId of ids) {
        assert.ok(history.inspectObservations('owner', runId, { section: 'outputs' }).evidence.outputs);
        assert.throws(() => history.inspectObservations('other-owner', runId), /no longer retained/);
      }
      assert.deepEqual((await fs.readFile(marker, 'utf8')).trim().split('\n'),
        Array.from({ length: 3 }, () => 'forward'), 'Observation reads never rerun a reference or forward.');
      history.forget('owner'); assert.throws(() => history.inspectObservations('owner', ids[0]), /no longer retained/);
    } finally { worker.dispose(); await fs.rm(root, { recursive: true, force: true }); }
  });
});
