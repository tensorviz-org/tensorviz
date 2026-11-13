const { copyBuiltPlugin } = require('../fixtures/plugin.cjs');
const { readyService } = require('../fixtures/service.cjs');
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { StandaloneHost } = require('../../out/host/host.cjs');
const { ApprovalGate } = require('../../out/host/approval.cjs');

const python = process.env.TENSORVIZ_TEST_PYTHON;
assert.ok(python && path.isAbsolute(python), 'Set an explicit compatible test interpreter.');
const delay = () => new Promise(resolve => setTimeout(resolve, 100));

for (const change of ['helper', 'recipe', 'input-editor']) test(`saved invocation freshness: ${change}`, { timeout: 60000 }, async () => {
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-idle-freshness-')));
  const plugin = path.join(root, 'plugin'), project = path.join(root, 'project');
  let host;
  try {
    await fs.mkdir(project);
    await copyBuiltPlugin(plugin);
    await readyService(plugin, path.join(root, 'state'));
    const model = 'from torch import nn\nfrom helper import WIDTH\nclass Tiny(nn.Module):\n def __init__(self):\n  super().__init__()\n  self.fc = nn.Linear(4, WIDTH)\n def forward(self, x): return self.fc(x)\n';
    const recipe = 'version = 1\n[profiles.tiny]\nkind = "model"\nmodel = "model:Tiny"\nmode = "eval"\ndevice = "cpu"\ngrad_enabled = false\ninit = { args = [], kwargs = {} }\ninputs = { args = [{ kind = "tensor", shape = [2, 4], dtype = "float32", requires_grad = false, generator = { kind = "full", value = 1 } }], kwargs = {} }\n';
    await fs.writeFile(path.join(project, 'model.py'), model);
    await fs.writeFile(path.join(project, 'helper.py'), 'WIDTH = 3\n');
    await fs.writeFile(path.join(project, 'tensorviz.toml'), recipe);
    let approvals = 0;
    host = new StandaloneHost(path.join(plugin, 'backend'), path.join(plugin, 'browser'), new ApprovalGate(async message => {
      assert.ok(message.includes(project) && message.includes(python));
      approvals++;
      return true;
    }, path.join(root, 'state')), path.join(root, 'state'));
    const approve = host.approvals.approve.bind(host.approvals);
    let captures = 0;
    host.approvals.approve = async (plan, signal) => {
      if (change === 'input-editor' && plan.operation === 'capture' && ++captures === 2) await new Promise(resolve => setTimeout(resolve, 2000));
      return approve(plan, signal);
    };
    const initial = host.openGraph.invoke(host.id, { workspaceRoot: project, file: 'model.py', cls: 'Tiny', interpreter: python });
    let opening = initial;
    for (let n = 0; n < 400 && !['ready', 'failed', 'blocked', 'cancelled', 'needs_setup'].includes(opening.status); n++) {
      await delay(); opening = host.openGraph.invoke(host.id, { requestId: initial.requestId });
    }
    assert.equal(opening.status, 'ready', JSON.stringify(opening));
    assert.ok(host.graph.list().models[0].current && host.graph.list().models[0].traced);
    assert.equal(approvals, 1);
    if (change === 'input-editor') {
      const ready = host.opening.state;
      await host.opening.action(ready.id, 'change-inputs', ready.target, { kind: 'change_inputs' });
      const pending = host.opening.state;
      assert.equal(pending.phase, 'needs_setup');
      const draft = structuredClone(pending.draft);
      draft.profile.inputs.args[0].shape = [3, 4];
      await host.opening.action(pending.id, 'save-inputs', pending.target, { kind: 'submit', draft });
      assert.equal(host.opening.state.phase, 'ready', JSON.stringify(host.opening.state));
      assert.ok(host.graph.list().models[0].current && host.graph.list().models[0].traced);
      const graph = host.graph.require(host.graph.list().models[0]);
      assert.ok(graph.nodes.some(node => host.graph.node(node, graph.model.projection).shapes.output?.join(',') === '3,3'));
      assert.equal(approvals, 1, 'Saving changed inputs and recapturing reuse plugin permission');
      assert.equal(await fs.readFile(path.join(project, 'model.py'), 'utf8'), model);
      await host.approvals.permissions.save('denied');
      await assert.rejects(host.inspectEnvironment({ workspaceRoot: project, interpreter: python }, new AbortController().signal), { code: 'APPROVAL_DECLINED' });
      return;
    }
    if (change === 'helper') await fs.writeFile(path.join(project, 'helper.py'), 'WIDTH = 2\n');
    else await fs.writeFile(path.join(project, 'tensorviz.toml'), recipe.replace('shape = [2, 4]', 'shape = [3, 4]'));
    // graph.list is deliberately a raw snapshot read. Calling an MCP tool's
    // complete freshness guard here would hide a missing idle-browser update.
    for (let n = 0; n < 70 && host.graph.list().models[0].current; n++) await delay();
    assert.equal(host.graph.list().models[0].current, false, 'The idle watcher must invalidate changed source or inputs, even when static lowering omits them');
    assert.equal(host.model.projection.greyed, true);
    assert.equal(host.opening.state.cancellationReason, 'source_changed');
    assert.equal(approvals, 1, 'Saved changes must invalidate without executing or requesting a new capture');
    assert.equal(await fs.readFile(path.join(project, 'model.py'), 'utf8'), model);
    assert.equal(await fs.readFile(path.join(project, 'tensorviz.toml'), 'utf8'), change === 'recipe' ? recipe.replace('shape = [2, 4]', 'shape = [3, 4]') : recipe);
  } finally {
    host?.dispose();
    await fs.rm(root, { recursive: true, force: true });
  }
});
