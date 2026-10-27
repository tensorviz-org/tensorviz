const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');
const { ElicitRequestSchema } = require('@modelcontextprotocol/sdk/types.js');

const python = process.env.TENSORVIZ_TEST_PYTHON;
assert.ok(python && path.isAbsolute(python), 'Set the explicit test interpreter.');
const source = required => `from torch import nn\nclass Tiny(nn.Module):\n def __init__(self${required ? ', width, variant: str' : ''}):\n  super().__init__()\n  self.fc = nn.Linear(${required ? 'width' : '4'}, 3)\n def forward(self, x):\n  return self.fc(x)\n`;
const draft = { profileName: 'tiny', profile: { kind: 'model', model: 'model:Tiny', mode: 'eval', device: 'cpu', grad_enabled: false,
  init: { args: [], kwargs: { width: 4, variant: 'test' } }, inputs: { args: [{ kind: 'tensor', shape: [2, 4], dtype: 'float32', requires_grad: false,
    generator: { kind: 'full', value: 1 } }], kwargs: {} } } };

async function fixture(required, decide = async () => true) {
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-setup-integration-')));
  const project = path.join(root, 'project'), plugin = path.join(root, 'plugin');
  await fs.mkdir(project); await fs.writeFile(path.join(project, 'model.py'), source(required));
  await fs.cp(path.resolve(__dirname, '../../../plugins/tensorviz'), plugin, { recursive: true });
  const prompts = [];
  let client;
  const connect = async () => {
    client = new Client({ name: 'synthetic-setup-test', version: '1' }, { capabilities: { elicitation: { form: {} } } });
    client.setRequestHandler(ElicitRequestSchema, async request => {
      assert.ok(request.params.message.includes(project)); assert.ok(request.params.message.includes(python));
      prompts.push(request.params.message);
      const decision = await decide(request.params.message, project);
      return typeof decision === 'boolean' ? { action: 'accept', content: { confirm: decision } } : decision;
    });
    await client.connect(new StdioClientTransport({ command: process.execPath, args: [path.join(plugin, 'runtime/host.cjs')], cwd: root,
      env: { PATH: '', TENSORVIZ_DATA_DIR: path.join(root, 'state') }, stderr: 'inherit' }));
  };
  await connect();
  const call = async (name, args = {}) => {
    const reply = await client.callTool({ name, arguments: args });
    assert.ok(!reply.isError, JSON.stringify(reply.structuredContent));
    return reply.structuredContent;
  };
  const poll = async initial => {
    let reply = initial;
    for (let count = 0; !['ready', 'failed', 'cancelled', 'blocked', 'needs_setup'].includes(reply.status) && count < 450; count++) {
      await new Promise(resolve => setTimeout(resolve, 100));
      reply = await call('open_graph', { requestId: reply.requestId });
    }
    return reply;
  };
  return { project, call, poll, prompts,
    open: (args = {}) => call('open_graph', { workspaceRoot: project, file: 'model.py', cls: 'Tiny', interpreter: python, ...args }),
    restart: async () => { await client.close(); await connect(); },
    close: async () => { await client.close(); await fs.rm(root, { recursive: true, force: true }); } };
}

test('first-use deterministic setup is approved, saved once and captured without a checkout', { timeout: 90000 }, async () => {
  const f = await fixture(false);
  try {
    const result = await f.poll(await f.open());
    assert.equal(result.status, 'ready', JSON.stringify(result));
    assert.equal(result.validated, true);
    const config = await fs.readFile(path.join(f.project, 'tensorviz.toml'), 'utf8');
    assert.match(config, /model:Tiny/);
    assert.equal(f.prompts.length, 1, 'Inspection, setup and capture share one plugin consent');
    await f.restart();
    const again = await f.poll(await f.open());
    assert.equal(again.status, 'ready', JSON.stringify(again));
    assert.equal(await fs.readFile(path.join(f.project, 'tensorviz.toml'), 'utf8'), config);
    assert.equal(f.prompts.length, 1, 'Permission survives a new host and recapture');
  } finally { await f.close(); }
});

test('missing constructor/input values continue the same opening and preserve other profiles and comments', { timeout: 90000 }, async () => {
  const f = await fixture(true);
  try {
    const existing = '# My experiment notes\nversion = 1\n\n[profiles.other]\nkind = "model"\nmodel = "other:Other"\nmode = "eval"\ndevice = "cpu"\ngrad_enabled = false\ninit = { args = [], kwargs = {} }\ninputs = { args = [], kwargs = {} }\n';
    await fs.writeFile(path.join(f.project, 'tensorviz.toml'), existing);
    const pending = await f.poll(await f.open());
    assert.equal(pending.status, 'needs_setup', JSON.stringify(pending));
    assert.equal(pending.validated, false);
    assert.equal(pending.models[0].traced, false, 'setup exposes only an unexecuted source view');
    const handoff = await f.call('open_graph_view', { modelId: pending.models[0].modelId, revision: pending.models[0].revision });
    assert.equal(handoff.status, 'awaiting_browser');
    assert.equal(handoff.displayed, false);
    assert.equal(await fs.readFile(path.join(f.project, 'tensorviz.toml'), 'utf8'), existing);
    const help = await f.call('suggest_graph_inputs');
    assert.ok(help.inputContract);
    assert.ok(help.fields.init.some(field => field.name === 'width'));
    await f.call('configure_graph_setup', { openingId: pending.opening.id, sourceDigest: pending.opening.target.sourceDigest, draft });
    const result = await f.poll(await f.call('open_graph', { requestId: pending.requestId }));
    assert.equal(result.status, 'ready', JSON.stringify(result));
    const config = await fs.readFile(path.join(f.project, 'tensorviz.toml'), 'utf8');
    assert.ok(config.startsWith(existing));
    assert.match(config, /profiles.tiny/);
  } finally { await f.close(); }
});

test('an external config edit during input setup refuses the stale write after consent', { timeout: 90000 }, async () => {
  const f = await fixture(true);
  try {
    const pending = await f.poll(await f.open());
    assert.equal(pending.status, 'needs_setup');
    await fs.writeFile(path.join(f.project, 'tensorviz.toml'), '# concurrent user edit\nversion = 1\n');
    await f.call('configure_graph_setup', { openingId: pending.opening.id, sourceDigest: pending.opening.target.sourceDigest, draft });
    const result = await f.poll(await f.call('open_graph', { requestId: pending.requestId }));
    assert.equal(result.status, 'needs_setup', JSON.stringify(result));
    assert.equal(result.error.code, 'SETUP_CHANGED');
    assert.equal(await fs.readFile(path.join(f.project, 'tensorviz.toml'), 'utf8'), '# concurrent user edit\nversion = 1\n');
    assert.equal(f.prompts.length, 1);
  } finally { await f.close(); }
});

test('declining plugin permission ends continuation without saving or capturing', { timeout: 90000 }, async () => {
  const f = await fixture(false, async () => false);
  try {
    const result = await f.poll(await f.open());
    assert.equal(result.status, 'blocked', JSON.stringify(result));
    assert.equal(result.continuationRequired, false);
    assert.equal(result.error.code, 'APPROVAL_DECLINED');
    await assert.rejects(fs.stat(path.join(f.project, 'tensorviz.toml')), { code: 'ENOENT' });
    assert.equal(f.prompts.length, 1);
    await f.restart();
    assert.equal((await f.poll(await f.open())).error.code, 'APPROVAL_DECLINED');
    assert.equal(f.prompts.length, 1, 'Declining does not cause repeated permission prompts');
  } finally { await f.close(); }
});

test('native dismissal is recoverable in the same opening without a terminal command', { timeout: 90000 }, async () => {
  let attempts = 0;
  const f = await fixture(false, async () => ++attempts === 1 ? { action: 'cancel' } : true);
  try {
    const stopped = await f.poll(await f.open());
    assert.equal(stopped.error.code, 'APPROVAL_CANCELLED');
    assert.equal(stopped.nextAction.kind, 'permission_required');
    const result = await f.poll(await f.call('open_graph', stopped.nextAction.arguments));
    assert.equal(result.requestId, stopped.requestId);
    assert.equal(result.status, 'ready', JSON.stringify(result));
    assert.equal(f.prompts.length, 2);
    await f.restart();
    assert.equal((await f.poll(await f.open())).status, 'ready');
    assert.equal(f.prompts.length, 2);
  } finally { await f.close(); }
});

test('a fresh project offers a previously used Python and remembers the explicit choice', { timeout: 90000 }, async () => {
  const f = await fixture(false);
  try {
    assert.equal((await f.poll(await f.open())).status, 'ready');
    const second = path.join(path.dirname(f.project), 'second project');
    await fs.mkdir(second); await fs.writeFile(path.join(second, 'model.py'), source(false));
    const choice = await f.poll(await f.call('open_graph', { workspaceRoot: second, file: 'model.py', cls: 'Tiny' }));
    assert.equal(choice.nextAction.kind, 'select_environment');
    assert.ok(choice.candidates.includes(python));
    await assert.rejects(fs.stat(path.join(second, 'tensorviz.toml')), { code: 'ENOENT' });
    const result = await f.poll(await f.call('open_graph', { ...choice.nextAction.arguments, interpreter: python }));
    assert.equal(result.status, 'ready', JSON.stringify(result));
    await f.restart();
    const again = await f.poll(await f.call('open_graph', { workspaceRoot: second, file: 'model.py', cls: 'Tiny' }));
    assert.equal(again.status, 'ready', JSON.stringify(again));
  } finally { await f.close(); }
});

test('ambiguous profiles request selection and the choice survives a new host without editing TOML', { timeout: 90000 }, async () => {
  const f = await fixture(false);
  try {
    const profile = name => `[profiles.${name}]\nkind = "model"\nmodel = "model:Tiny"\nmode = "eval"\ndevice = "cpu"\ngrad_enabled = false\ninit = { args = [], kwargs = {} }\ninputs = { args = [{ kind = "tensor", shape = [2, 4], dtype = "float32", requires_grad = false, generator = { kind = "full", value = 1 } }], kwargs = {} }\n`;
    const config = `version = 1\n\n${profile('first')}\n${profile('second')}`;
    await fs.writeFile(path.join(f.project, 'tensorviz.toml'), config);
    const choice = await f.poll(await f.open());
    assert.equal(choice.error.code, 'PROFILE_REQUIRED', JSON.stringify(choice));
    assert.equal(choice.nextAction.kind, 'select_profile');
    assert.deepEqual(choice.profiles, ['first', 'second']);
    assert.equal(f.prompts.some(text => /requests (save setup|capture)/.test(text)), false);
    const selected = await f.poll(await f.open({ profile: 'second' }));
    assert.equal(selected.status, 'ready', JSON.stringify(selected));
    assert.equal(selected.opening.draft.profileName, 'second');
    await f.restart();
    const restored = await f.poll(await f.open());
    assert.equal(restored.status, 'ready', JSON.stringify(restored));
    assert.equal(restored.opening.draft.profileName, 'second');
    assert.equal(await fs.readFile(path.join(f.project, 'tensorviz.toml'), 'utf8'), config);
  } finally { await f.close(); }
});

test('one plugin permission covers runtime input preparation and a single forward capture', { timeout: 90000 }, async () => {
  const f = await fixture(false, async (message, project) => {
    await assert.rejects(fs.stat(path.join(project, 'calls.txt')), { code: 'ENOENT' });
    return true;
  });
  try {
    const marker = JSON.stringify(path.join(f.project, 'calls.txt'));
    await fs.writeFile(path.join(f.project, 'model.py'), `from torch import nn\nclass Tiny(nn.Module):\n def __init__(self):\n  super().__init__()\n  with open(${marker}, 'a') as log: log.write('construct\\n')\n  layers = []\n  for _ in range(1): layers.append(nn.Linear(4, 3))\n  self.fc = nn.Sequential(*layers)\n def forward(self, x):\n  with open(${marker}, 'a') as log: log.write('forward\\n')\n  return self.fc(x)\n`);
    const result = await f.poll(await f.open());
    assert.equal(result.status, 'ready', JSON.stringify(result));
    assert.equal(f.prompts.length, 1);
    assert.equal(await fs.readFile(path.join(f.project, 'calls.txt'), 'utf8'), 'construct\nconstruct\nforward\n');
  } finally { await f.close(); }
});


test('Python example inputs save and recapture in a copied plugin without literal tensor serialization', { timeout: 90000 }, async () => {
  const f = await fixture(true);
  try {
    await fs.appendFile(path.join(f.project, 'model.py'), `\nimport torch\ndef graph_inputs():\n return {"x": torch.ones(1, 64, 80)}\n`);
    const pending = await f.poll(await f.open());
    assert.equal(pending.status, 'needs_setup', JSON.stringify(pending));
    const inputDraft = { profileName: 'tiny', profile: { ...draft.profile,
      init: { args: [], kwargs: { width: 80, variant: 'test' } },
      inputs: { factory: { file: 'model.py', callable: 'graph_inputs', kwargs: {} } } } };
    await f.call('configure_graph_setup', { openingId: pending.opening.id, sourceDigest: pending.opening.target.sourceDigest, draft: inputDraft });
    const ready = await f.poll(await f.call('open_graph', { requestId: pending.requestId }));
    assert.equal(ready.status, 'ready', JSON.stringify(ready));
    const config = await fs.readFile(path.join(f.project, 'tensorviz.toml'), 'utf8');
    assert.match(config, /graph_inputs/); assert.doesNotMatch(config, /generator|values/);
    await f.restart();
    const reopened = await f.poll(await f.open());
    assert.equal(reopened.status, 'ready', JSON.stringify(reopened));
    assert.equal(await fs.readFile(path.join(f.project, 'tensorviz.toml'), 'utf8'), config);
    assert.equal(f.prompts.length, 1);
  } finally { await f.close(); }
});
