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
import { executionSpecSchema } from '@tensorviz/graph-host/executionSpec';
import { captureRunEvidence, inspectEvidence } from '@tensorviz/graph-host/mcpRunEvidence';

const repo = path.resolve(__dirname, '../../../..'), backend = path.join(repo, 'backend');
const python = process.env.TENSORVIZ_TEST_PYTHON || path.join(repo, 'venv/bin/python');
const source = `import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left = torch.nn.Linear(3, 2, bias=False)
        self.right = self.left
        self.register_buffer('transient', torch.ones(1), persistent=False)
    def forward(self, x):
        return self.left(x) + self.right(x)
`;

suite('checkpoint evidence with the real worker', function () {
  this.timeout(40000);
  let root: string, file: string, worker: TraceWorker;
  suiteSetup(async () => {
    assert.ok(existsSync(python), 'Set TENSORVIZ_TEST_PYTHON to a Torch-capable interpreter.');
    root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-checkpoint-')));
    file = path.join(root, 'model.py'); await fs.writeFile(file, source);
    await promisify(execFile)(python, ['-c', `import torch, sys
from pathlib import Path
root=Path(sys.argv[1])
for width in (3, 4):
    weight=torch.ones(2, width, dtype=torch.float64)
    torch.save({'left.weight':weight, 'right.weight':weight}, root/f'width{width}.pt')
torch.save({'left.weight':torch.ones(2,3), 'extra':torch.ones(1)}, root/'missing.pt')
`, root], { timeout: 20000 });
    worker = new TraceWorker(backend, { appendLine() {} } as unknown as vscode.OutputChannel, { interpreter: async () => python });
  });
  suiteTeardown(async () => { worker?.dispose(); if (root) await fs.rm(root, { recursive: true, force: true }); });

  for (const [checkpointFile, strict, success] of [['width3.pt', true, true], ['width4.pt', true, false], ['missing.pt', false, true]] as const) {
    test(`${checkpointFile} keeps actual loading separate from stored metadata`, async () => {
      const execution = executionSpecSchema.parse({ inputs: { args: [{shape:[2,3], dtype:'float32', fill:'ones'}] },
        checkpoint: {file:checkpointFile, strict} });
      const outcome = await worker.run({workspace:root, file, cls:'Model', execution, timeoutMs:20000});
      assert.equal(outcome.ok, success, JSON.stringify(outcome));
      const evidence = captureRunEvidence(outcome, root, undefined, {
        sourceDigest: createHash('sha256').update(source).digest('hex'), checkpoint: execution.checkpoint });
      const page = inspectEvidence({}, evidence, 'checkpoint', 0, 20) as any;
      assert.ok(page.checkpoint, JSON.stringify(outcome));
      assert.equal(page.checkpoint.loadStatus, success ? 'succeeded' : 'failed');
      assert.equal(page.checkpoint.checkpoint.digest, createHash('sha256').update(await fs.readFile(path.join(root, checkpointFile))).digest('hex'));
      assert.equal(page.items.some((item: any) => item.name === 'transient'), false);
      const left = page.items.find((item: any) => item.name === 'left.weight'), right = page.items.find((item: any) => item.name === 'right.weight');
      assert.equal(left.registered.objectGroup, right.registered.objectGroup);
      if (checkpointFile === 'missing.pt') {
        assert.equal(right.presence, 'registered_only');
        assert.equal(page.items.find((item: any) => item.name === 'extra').presence, 'checkpoint_only');
      } else {
        assert.equal(left.stored.objectGroup, right.stored.objectGroup);
        assert.equal(left.shape, success ? 'same' : 'different');
        assert.equal(left.dtype, 'different');
      }
      if (success) {
        const executionState = (outcome.envelope!.result as any).diagnostics.execution.state.parametersAndBuffers.items;
        assert.equal(executionState['left.weight'].dtype, 'torch.float32', 'stored float64 is converted by the actual loader');
      } else assert.equal(outcome.envelope!.result, null, 'no previous forward is borrowed for a failed load');
      assert.equal(await fs.readFile(file, 'utf8'), source);
    });
  }
});
