const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { createHash } = require('node:crypto');
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');
const { ElicitRequestSchema } = require('@modelcontextprotocol/sdk/types.js');

// Uses the real compiled archive, never a source-development host. Only the
// copied test launcher's curl command is injected; production has no file URL mode.
test('compiled Linux archive launches through verified local-file transport, captures a CPU MLP and reuses cache', { timeout: 120000 }, async () => {
  assert.equal(`${process.platform}-${process.arch}`, 'linux-x64');
  const python = process.env.TENSORVIZ_TEST_PYTHON, runtime = process.env.TENSORVIZ_TEST_RUNTIME;
  assert.ok(python && path.isAbsolute(python) && runtime && path.isAbsolute(runtime));
  const { runtimeArchive } = await import('../../scripts/native-platforms.mjs');
  const metadata = JSON.parse(await fs.readFile(path.join(runtime, 'plugin.json'), 'utf8'));
  const archive = path.resolve(__dirname, '../../dist', runtimeArchive({ version: metadata.version, platform: 'linux-x64' }));
  const lock = JSON.parse(await fs.readFile(`${archive}.json`, 'utf8'));
  assert.equal(lock.sha256, createHash('sha256').update(await fs.readFile(archive)).digest('hex'));
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-native-launch-capture-')));
  const home = path.join(root, 'home'), tmp = path.join(root, 'tmp'), state = path.join(root, 'state');
  const plugin = path.join(root, 'plugin'), project = path.join(root, 'project'), bin = path.join(root, 'bin');
  let client, stderr = '';
  try {
    for (const folder of [home, tmp, state, bin, path.join(plugin, 'scripts'), path.join(project, 'proj')]) await fs.mkdir(folder, { recursive: true });
    const url = 'https://127.0.0.1:1/runtime.tar.gz';
    await fs.writeFile(path.join(plugin, 'runtime-lock.json'), JSON.stringify({ formatVersion: 2, version: lock.version,
      runtimes: { 'linux-x64': { ...lock, url } } }));
    await fs.copyFile(path.resolve(__dirname, '../../integration/launch.sh'), path.join(plugin, 'scripts/launch.sh'));
    await fs.writeFile(path.join(bin, 'curl'), '#!/bin/sh\nprevious=""\nfor arg do\n [ "$previous" != --output ] || output="$arg"\n previous="$arg"\ndone\n[ "$arg" = https://127.0.0.1:1/runtime.tar.gz ] || exit 99\necho download >> "$DOWNLOAD_LOG"\ncp "$FIXTURE_ARCHIVE" "$output"\n', { mode: 0o755 });
    // Seed only the test-owned service cache with the explicit test venv. This
    // tests native ABI/service/model execution, not Python/package downloading.
    const manifest = JSON.parse(await fs.readFile(path.join(runtime, 'runtime-manifest.json'), 'utf8'));
    const requirements = (await fs.readFile(path.join(runtime, 'backend/requirements-standalone.txt'), 'utf8'))
      .split(/\r?\n/).map(line => line.trim()).filter(line => line && !line.startsWith('#'));
    const digest = value => createHash('sha256').update(value).digest('hex');
    const pin = manifest.platforms['linux-x64'].sha256;
    const key = digest(JSON.stringify([pin, requirements])).slice(0, 24);
    const service = path.join(state, 'service', key); await fs.mkdir(service, { recursive: true });
    const owner = JSON.stringify({ role: 'service', runtime: pin, requirements: digest(JSON.stringify(requirements)) });
    await fs.symlink(path.dirname(path.dirname(python)), path.join(service, 'venv'), 'dir');
    for (const file of ['owner.json', 'ready.json']) await fs.writeFile(path.join(service, file), owner);
    await fs.writeFile(path.join(project, 'proj/mlp.py'), 'from torch import nn\nclass MLP(nn.Module):\n def __init__(self):\n  super().__init__()\n  self.layers = nn.Sequential(nn.Linear(4,8), nn.ReLU(), nn.Linear(8,3))\n def forward(self,x): return self.layers(x)\n');
    await fs.writeFile(path.join(project, 'tensorviz.toml'), 'version = 1\n[profiles.mlp]\nkind = "model"\nmodel = "proj.mlp:MLP"\nmode = "eval"\ngrad_enabled = false\ndevice = "cpu"\ninit = { args = [], kwargs = {} }\ninputs = { args = [{kind = "tensor", shape = [2,4], dtype = "float32", requires_grad = false, generator = {kind = "full", value = 1.0}}], kwargs = {} }\n');
    const env = { HOME: home, TMPDIR: tmp, PATH: `${bin}:/usr/bin:/bin`, TENSORVIZ_DATA_DIR: state,
      FIXTURE_ARCHIVE: archive, DOWNLOAD_LOG: path.join(root, 'downloads') };
    const connect = async () => {
      client = new Client({ name: 'native-local-acceptance', version: '1' }, { capabilities: { elicitation: { form: {} } } });
      client.setRequestHandler(ElicitRequestSchema, async request => {
        assert.ok(request.params.message.startsWith('Allow TensorViz to work with local models without asking again?'));
        assert.ok(request.params.message.includes(project) && request.params.message.includes(python));
        return { action: 'accept', content: { confirm: true } };
      });
      const transport = new StdioClientTransport({ command: '/bin/sh', args: [path.join(plugin, 'scripts/launch.sh')],
        env, cwd: root, stderr: 'pipe' });
      transport.stderr?.on('data', bytes => stderr += bytes);
      await client.connect(transport);
      assert.equal((await client.listTools()).tools.length, 29);
    };
    await connect();
    const call = async (name, args) => {
      const result = await client.callTool({ name, arguments: args });
      assert.ok(!result.isError, JSON.stringify(result.structuredContent)); return result.structuredContent;
    };
    let capture = await call('open_graph', { workspaceRoot: project, file: 'proj/mlp.py', cls: 'MLP', interpreter: python });
    for (let n = 0; n < 600 && !['ready', 'failed', 'blocked', 'cancelled', 'needs_setup'].includes(capture.status); n++) {
      await new Promise(resolve => setTimeout(resolve, 100));
      capture = await call('open_graph', { requestId: capture.requestId });
    }
    assert.equal(capture.status, 'ready', JSON.stringify(capture)); assert.equal(capture.validated, true);
    const model = capture.models[0];
    const graph = await call('get_graph', { modelId: model.modelId, revision: model.revision, mode: 'graph', limit: 200 });
    assert.ok(graph.nodes.some(node => node.shapes.output?.join(',') === '2,3'), 'Real CPU capture must observe MLP output [2,3]');
    await client.close(); client = undefined;
    await connect(); await client.close(); client = undefined;
    assert.equal(await fs.readFile(env.DOWNLOAD_LOG, 'utf8'), 'download\n');
    const lockContainer = `tensorviz-locks-${process.getuid()}`;
    const leftovers = await fs.readdir(tmp);
    assert.deepEqual(leftovers.filter(name => name !== lockContainer), [],
      'Native sessions and unexpected temporary files must be cleaned');
    for (const name of leftovers) {
      const directory = path.join(tmp, name), info = await fs.lstat(directory);
      assert.ok(info.isDirectory() && !info.isSymbolicLink(), 'Lock container must be a real directory');
      assert.equal(info.uid, process.getuid(), 'Lock container must belong to this test user');
      assert.equal(info.mode & 0o077, 0, 'Lock container must remain private');
      assert.deepEqual(await fs.readdir(directory), [], 'No active lock state may remain after shutdown');
    }
    console.log(`Compiled MLP CPU capture passed: [2,4] -> [2,3]; archive ${lock.sha256}; cache reused.`);
  } catch (error) {
    if (stderr) console.error(stderr);
    throw error;
  } finally { await client?.close(); await fs.rm(root, { recursive: true, force: true }); }
});
