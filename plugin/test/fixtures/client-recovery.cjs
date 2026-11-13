// Exercise the installed client's own MCP bridge, without a model/provider turn.
// Callers own this disposable project and restore it before uninstall checks.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');

exports.checkClientRecovery = async ({ call, project, python, model, recipe, initial }) => {
  const modelFile = path.join(project, 'model.py'), recipeFile = path.join(project, 'tensorviz.toml');
  assert.equal(await fs.readFile(modelFile, 'utf8'), model);
  assert.equal(await fs.readFile(recipeFile, 'utf8'), recipe);
  const open = () => call('open_graph', { workspaceRoot: project, file: 'model.py', cls: 'Tiny', interpreter: python, waitMs: 0 });
  const capture = async () => {
    let value = await open();
    const deadline = Date.now() + 45000;
    while (!['ready', 'failed', 'blocked', 'cancelled', 'needs_setup', 'source_changed'].includes(value.status) && Date.now() < deadline) {
      await new Promise(resolve => setTimeout(resolve, 100));
      value = await call('open_graph', { requestId: value.requestId, waitMs: 1000 });
    }
    return value;
  };
  try {
    await fs.writeFile(modelFile, model.replace('nn.Linear(4, 3)', 'nn.Linear(4, 2)').replace('return self.fc(x)', 'return self.fc(x).sigmoid()'));
    const changed = await capture();
    assert.equal(changed.status, 'ready');
    assert.notEqual(changed.models[0].revision, initial.models[0].revision, 'an edited model must not reuse stale capture evidence');
    const context = await call('get_model_context', { modelId: changed.models[0].modelId, revision: changed.models[0].revision });
    const fc = context.modules.find(module => module.paths.includes('fc'));
    assert.deepEqual(fc.calls[0].inputs.tensors[0].shape, [2, 4]);
    assert.deepEqual(fc.calls[0].outputs.tensors[0].shape, [2, 2]);
    assert.equal(fc.calls[0].parameterCount, 10);

    await fs.writeFile(recipeFile, recipe.replace('shape = [2, 4]', 'shape = [2, 5]'));
    const mismatch = await capture();
    assert.equal(mismatch.status, 'failed');
    assert.equal(mismatch.validated, false);
    assert.equal(mismatch.nextAction.kind, 'inspect_failure');

    await fs.writeFile(recipeFile, recipe);
    await fs.writeFile(modelFile, model.replace('return self.fc(x)', "raise RuntimeError('synthetic client recovery failure')"));
    const exception = await capture();
    assert.equal(exception.status, 'failed');
    assert.equal(exception.validated, false);
    assert.equal(exception.nextAction.kind, 'inspect_failure');

    await fs.writeFile(modelFile, model);
    const recovered = await capture();
    assert.equal(recovered.status, 'ready');
    assert.equal(recovered.validated, true);

    await fs.writeFile(modelFile, model.replace('return self.fc(x)', 'import time\n  time.sleep(20)\n  return self.fc(x)'));
    const pending = await open();
    assert.ok(pending.requestId);
    assert.notEqual(pending.status, 'ready');
    const cancelled = await call('open_graph', { requestId: pending.requestId, action: 'cancel' });
    assert.equal(cancelled.status, 'cancelled');
    console.log('Edited measured output [2,3] -> [2,2], rejected shape/model failures, recovered and cancelled through the installed client.');
  } finally {
    await fs.writeFile(modelFile, model);
    await fs.writeFile(recipeFile, recipe);
  }
};
