const { test } = require('node:test');
const assert = require('node:assert/strict');
const { mkdtemp, mkdir, writeFile, rm } = require('node:fs/promises');
const { createHash, randomUUID } = require('node:crypto');
const os = require('node:os');
const path = require('node:path');
const { TraceWorkerCore, readHostSnapshot, lookupProfile } = require('../../out/graph-host.cjs');

const python = process.env.TENSORVIZ_TEST_PYTHON;
assert.ok(python && path.isAbsolute(python), 'Set TENSORVIZ_TEST_PYTHON to a Python 3.11+ environment with Torch and backend dependencies.');
const backend = process.env.TENSORVIZ_TEST_BACKEND ?? path.resolve(__dirname, '../../../backend');

test('shared host captures, detects saved edits, and recaptures the same recipe without VS Code', { timeout: 90000 }, async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'tensorviz-independent-capture-'));
  const file = path.join(root, 'model.py');
  const model = activation => `from torch import nn\nfrom services import width\nfrom models import height\nfrom utils import identity\nclass Tiny(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(width, height)\n        self.activation = nn.${activation}()\n    def forward(self, x):\n        return identity(self.activation(self.fc(x)))\n`;
  const session = randomUUID();
  const signal = new AbortController().signal;
  const options = {
    snapshot: (_root, profile, nextSignal) => readHostSnapshot(python, backend, {
      root, documents: {}, executable: python, environmentRevision: 0, session,
      workspaceId: createHash('sha256').update(root).digest('hex'), runId: randomUUID(), generation: 1, profile,
    }, nextSignal),
    choose: async () => undefined,
  };
  const worker = new TraceWorkerCore(backend, { appendLine() {} }, {
    trusted: () => true, documents: () => ({}), interpreter: async () => python,
  });
  const capture = async lookup => {
    assert.equal(lookup.ok, true, JSON.stringify(lookup));
    return worker.run({ workspace: root, file, cls: 'Tiny', tier: 'real',
      execution: lookup.run.execution, profileSpec: lookup.guard.spec,
      expectedInterpreter: python, timeoutMs: lookup.run.timeoutMs,
      beforeLaunch: () => lookup.guard.current(signal) });
  };
  try {
    for (const [name, source] of Object.entries({ services: 'width = 4\n', models: 'height = 3\n', utils: 'def identity(x): return x\n' })) {
      await mkdir(path.join(root, name));
      await writeFile(path.join(root, name, '__init__.py'), source);
    }
    await writeFile(file, model('ReLU'));
    await writeFile(path.join(root, 'tensorviz.toml'), `version = 1\n[profiles.tiny]\nkind = "model"\nmodel = "model:Tiny"\nmode = "eval"\ngrad_enabled = false\ndevice = "cpu"\ninit = { args = [], kwargs = {} }\ninputs = { args = [{kind = "tensor", shape = [2, 4], dtype = "float32", requires_grad = false, generator = {kind = "full", value = 1.0}}], kwargs = {} }\n`);
    const before = await lookupProfile(options, root, { file: 'model.py', cls: 'Tiny' }, signal);
    const first = await capture(before);
    assert.equal(first.ok, true, JSON.stringify(first));
    const original = first.envelope.result.diagnostics.semantic_capture;
    assert.equal(original.status, 'complete');
    assert.ok(original.invocations.length > 0);
    await writeFile(file, model('Sigmoid'));
    assert.equal(await before.guard.current(signal), false, 'a saved source edit must invalidate the old execution guard');
    const after = await lookupProfile(options, root, { file: 'model.py', cls: 'Tiny' }, signal);
    assert.deepEqual(after.run.execution.inputs, before.run.execution.inputs);
    const second = await capture(after);
    assert.equal(second.ok, true, JSON.stringify(second));
    const current = second.envelope.result.diagnostics.semantic_capture;
    assert.equal(current.status, 'complete');
    assert.notEqual(current.source.id, original.source.id);
    assert.match(JSON.stringify(current.operations), /sigmoid/i);
  } finally {
    worker.dispose();
    await rm(root, { recursive: true, force: true });
  }
});
