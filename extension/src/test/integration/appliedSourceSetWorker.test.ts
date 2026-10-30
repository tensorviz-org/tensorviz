import * as assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import * as os from 'node:os';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import type * as vscode from 'vscode';
import { TraceWorker } from '../../worker';
import type { ExecutionSnapshot } from '../../executionPlan';
import { executionSpecSchema } from '@tensorviz/graph-host/executionSpec';
import { AppliedSourceReceipts } from '../../appliedSourceReceipts';
import { AppliedSourceInspector, type AppliedCapture } from '../../appliedSourceInspection';
import { sha256 } from '../../sourceChanges';
import { isAppliedSourceObservation } from '@tensorviz/contracts/protocol';
import { reset, resetEditors, editorState } from '../vscodeStub';

suite('applied source-set evidence with actual CPU captures', function () {
  this.timeout(60_000);
  test('two saved replacements match a later capture, while further dependency edits and stale files cannot borrow its result', async () => {
    const python = process.env.TENSORVIZ_TEST_PYTHON;
    assert.ok(python, 'A real Torch interpreter is required; never skip this oracle.');
    reset(); resetEditors();
    const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-applied-source-set-')));
    const file = path.join(root, 'model.py'), marker = path.join(root, 'forwards.txt');
    const worker = new TraceWorker(path.resolve(__dirname, '../../../../backend'), { appendLine() {} } as unknown as vscode.OutputChannel,
      { interpreter: async () => python });
    try {
      const model = `import torch\nfrom config import WIDTH\nfrom helper import transform\nclass Model(torch.nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.projection = torch.nn.Linear(2, WIDTH)\n    def forward(self, x):\n        with open(${JSON.stringify(marker)}, 'a') as stream:\n            stream.write('forward\\n')\n        return transform(self.projection(x))\n`;
      const changed = model.replace('return transform(self.projection(x))', 'return torch.sigmoid(transform(self.projection(x)))');
      const before = new Map([['model.py', model], ['config.py', 'WIDTH = 2\n'], ['helper.py', 'def transform(x):\n    return x + 1\n']]);
      for (const [name, source] of before) await fs.writeFile(path.join(root, name), source);
      await promisify(execFile)(python, ['-c', 'import torch,sys; torch.save({"args":(torch.ones(1,2),)},sys.argv[1])', path.join(root, 'inputs.pt')]);
      const execution = executionSpecSchema.parse({ inputs: { fixture: 'inputs.pt' }, mode: 'eval', seed: 4 });
      const run = async (): Promise<AppliedCapture> => {
        let snapshot!: ExecutionSnapshot; const startedAt = Date.now();
        const outcome = await worker.run({ workspace: root, file, cls: 'Model', timeoutMs: 25_000, execution,
          onResolvedPlan(value) { snapshot = value as ExecutionSnapshot; } });
        assert.equal(outcome.ok, true, outcome.ok ? undefined : outcome.reason);
        return { capture: (outcome.envelope!.result as any).diagnostics.semantic_capture, plan: snapshot.plan, startedAt };
      };
      const baseline = await run();
      assert.ok(baseline.capture.source.files.some(item => item.path === 'helper.py'), 'The real capture must bind the unchanged helper.');
      const replacements = new Map([['model.py', changed], ['config.py', 'WIDTH = 3\n']]);
      const files = [...replacements].map(([name, source]) => ({ file: name, beforeDigest: sha256(before.get(name)!), afterDigest: sha256(source) }));
      const target = { workspaceFolder: root, file: 'model.py', cls: 'Model', documentVersion: 1, sourceDigest: sha256(model), backendRevision: 1 };
      const receipts = new AppliedSourceReceipts(), reservation = receipts.reserve('owner', target, { version: 2, base: baseline.capture, files })!;
      assert.ok(reservation); assert.equal(receipts.summary('owner'), null);
      // This fixture simulates the native writer acknowledgment; installed Apply is a separate acceptance gate.
      for (const [name, source] of replacements) {
        editorState.documents.set(path.join(root, name), source); editorState.dirtyDocuments.add(path.join(root, name));
      }
      receipts.confirm('owner', reservation);
      let live: AppliedCapture | undefined;
      let setupChecks = 0;
      const inspector = new AppliedSourceInspector(receipts, { origin: () => 'http://localhost:9000', token: () => 'fixture-only', trusted: () => true,
        live: () => live, checkSetup: async (_target, selected, signal) => { setupChecks++;
          return worker.checkCapturedSetup(root, selected.plan!, selected.capture.planDigest!, signal); },
        fetchImpl: async () => { throw new Error('Inspection cannot import or execute a reconciliation engine.'); } });
      assert.equal((await inspector.inspect('owner')).source, 'reviewed_unsaved'); assert.equal(setupChecks, 0);
      for (const [name, source] of replacements) {
        await fs.writeFile(path.join(root, name), source); editorState.dirtyDocuments.delete(path.join(root, name));
      }
      assert.equal((await inspector.inspect('owner')).unavailable, 'capture_unavailable');
      live = await run();
      const checked = await inspector.inspect('owner'); assert.ok(isAppliedSourceObservation(checked));
      assert.equal(checked.captureFreshness, 'current'); assert.equal(checked.sourceSetComparison!.reason, 'reviewed_source');
      assert.equal(checked.sourceSetComparison!.inputs, 'same'); assert.equal(checked.sourceSetComparison!.state, 'different');
      assert.equal(checked.sourceSetComparison!.semanticPostconditions, 'not_checked');
      assert.ok(checked.sourceSetComparison!.comparison!.observedFieldCount > 0);
      assert.notEqual(checked.sourceSetComparison!.comparison!.equivalent, true);
      await fs.writeFile(path.join(root, 'helper.py'), 'def transform(x):\n    return x + 2\n');
      const stale = await inspector.inspect('owner'); assert.equal(stale.sourceSetComparison, undefined);
      assert.notEqual(stale.captureFreshness, 'current');
      live = await run();
      const different = await inspector.inspect('owner'); assert.equal(different.sourceSetComparison!.reason, 'dependencies_changed');
      assert.equal(different.sourceSetComparison!.comparison, null);
      editorState.documents.set(path.join(root, 'config.py'), before.get('config.py')!);
      const restored = await inspector.inspect('owner'); assert.equal(restored.source, 'changed');
      assert.equal(restored.sourceSetComparison, undefined);
      assert.deepEqual((await fs.readFile(marker, 'utf8')).trim().split('\n'), ['forward', 'forward', 'forward'], 'Inspection never adds a forward.');
      assert.equal(await fs.readFile(file, 'utf8'), changed); assert.equal(await fs.readFile(path.join(root, 'config.py'), 'utf8'), 'WIDTH = 3\n');
      receipts.forget('owner'); await assert.rejects(inspector.inspect('owner'), /confirmed Apply receipt/);
    } finally { worker.dispose(); reset(); resetEditors(); await fs.rm(root, { recursive: true, force: true }); }
  });
});
