/** Real Unix installed-catalog acceptance without optional model frameworks.
 * Requires the native archive/lock built on THIS runner, pinned official Codex,
 * a service-only Python, and a temporary localhost TLS certificate. No auth or LLM.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {execFileSync, execFile, spawn} from 'node:child_process';
import {promisify} from 'node:util';
import {createInterface} from 'node:readline';
import https from 'node:https';
import {exportPublic} from '../../scripts/export-public.mjs';
import {nativePlatform} from '../../scripts/native-platforms.mjs';

assert.ok(['darwin', 'linux'].includes(process.platform), 'A real Unix runner is required');
const required = name => { assert.ok(process.env[name], `Set ${name}`); return path.resolve(process.env[name]); };
const lockFile = required('TENSORVIZ_PRODUCTION_RUNTIME_LOCK');
const codex = required('TENSORVIZ_TEST_CODEX');
const python = required('TENSORVIZ_SERVICE_TEST_PYTHON');
const certificate = required('TENSORVIZ_TEST_TLS_CERT'), key = required('TENSORVIZ_TEST_TLS_KEY');
const native = path.resolve(process.env.TENSORVIZ_NATIVE_RUNTIME ?? 'plugin/out/native-runtime');
const output = path.resolve('plugin/out/unix-production-mcp');
await fs.mkdir(output, {recursive: true});
const target = nativePlatform();
const lock = JSON.parse(await fs.readFile(lockFile, 'utf8'));
assert.equal(lock.formatVersion, 1); assert.equal(lock.platform, target.key);
const archive = await fs.readFile(lockFile.replace(/\.json$/, ''));
assert.equal(archive.length, lock.bytes);
assert.equal(crypto.createHash('sha256').update(archive).digest('hex'), lock.sha256);
const info = JSON.parse(execFileSync(path.join(native, target.host), ['--verify-native-package'], {encoding: 'utf8', timeout: 30000}));
assert.equal(info.sourceIncluded, false); assert.equal(info.platform, process.platform); assert.equal(info.architecture, process.arch);
const release = JSON.parse(await fs.readFile(path.join(native, 'release-manifest.json'), 'utf8'));
assert.equal(release.platform, target.key); assert.equal(release.version, lock.version); assert.equal(release.coreSourceIncluded, false);
execFileSync(python, ['-I', '-c', 'import importlib.util; assert importlib.util.find_spec("torch") is None'], {stdio: 'inherit', timeout: 30000});
// Probe the actual compiled backend in the explicitly clean interpreter. This is
// distinct from launcher discovery: missing/broken Torch must be a capture error.
for (const broken of [false, true]) {
  const code = `import sys, types\nsys.path.insert(0, sys.argv[1])\nimport _tensorviz_backend\n${broken ? "sys.modules['torch'] = types.ModuleType('torch')\n" : ''}sys.exit(_tensorviz_backend.run('services/ide/environment_setup.py', ['probe-capture']))`;
  try { execFileSync(python, ['-I', '-B', '-c', code, path.join(native, 'backend')], {encoding: 'utf8', timeout: 30000}); assert.fail('Missing/broken framework probe unexpectedly succeeded'); }
  catch (error) {
    assert.equal(error.status, 1, error.stderr || error.message);
    assert.equal(error.stderr, '', 'Expected capture error, not a startup traceback');
    const failure = JSON.parse(error.stdout);
    assert.equal(failure.code, broken ? 'CAPTURE_RUNTIME_UNSUPPORTED' : 'MISSING_DEPENDENCIES');
    assert.match(failure.message, broken ? /TensorViz started/ : /runtime is available/);
  }
}
const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'TensorViz Unix 测试 space ')));
let server, child, lines, primaryError, diagnostics = '', downloads = 0, nextId = 0, unexpectedRequests = 0, forcedStop = false;
const pending = new Map();
const stop = async () => {
  if (!child || child.exitCode !== null || child.signalCode !== null) return;
  child.stdin.end();
  await Promise.race([new Promise(resolve => child.once('exit', resolve)), new Promise(resolve => setTimeout(resolve, 5000))]);
  if (child.exitCode === null && child.signalCode === null) {
    forcedStop = true;
    try { process.kill(-child.pid, 'SIGTERM'); } catch {}
    await Promise.race([new Promise(resolve => child.once('exit', resolve)), new Promise(resolve => setTimeout(resolve, 3000))]);
  }
  if (child.exitCode === null && child.signalCode === null) { try { process.kill(-child.pid, 'SIGKILL'); } catch {} }
};
try {
  const home = path.join(root, 'home'), state = path.join(root, 'data'), codexHome = path.join(root, 'codex-home'), project = path.join(root, 'empty-project');
  for (const dir of [home, state, codexHome, project]) await fs.mkdir(dir, {mode: 0o700});
  const cache = process.platform === 'darwin' ? path.join(home, 'Library/Caches/dev.tensorviz/runtime') : path.join(root, 'xdg-cache/dev.tensorviz/runtime');
  const env = {HOME: home, PATH: '/usr/bin:/bin', TMPDIR: root, XDG_CACHE_HOME: path.join(root, 'xdg-cache'),
    CODEX_HOME: codexHome, TENSORVIZ_DATA_DIR: state, UV_CACHE_DIR: path.join(root, 'uv-cache')};
  server = https.createServer({key: await fs.readFile(key), cert: await fs.readFile(certificate), minVersion: 'TLSv1.2'}, (req, res) => {
    if (req.url !== '/runtime.tar.gz') { res.writeHead(404).end(); return; }
    downloads++; res.writeHead(200, {'content-type': 'application/gzip', 'content-length': archive.length}); res.end(archive);
  });
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
  const entry = {...lock, url: `https://127.0.0.1:${server.address().port}/runtime.tar.gz`};
  const catalog = path.join(root, 'catalog');
  await exportPublic(catalog, {runtimeLock: {formatVersion: 2, version: lock.version, runtimes: {[target.key]: entry}}});
  // Only inject isolated test state/CA. The production command/arguments remain
  // unchanged, and Codex itself must resolve the installed catalog transport.
  const manifestPath = path.join(catalog, 'plugins/tensorviz/mcp.json');
  const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
  assert.equal(manifest.mcpServers.tensorviz.command, 'sh');
  manifest.mcpServers.tensorviz.env = {...env, CURL_CA_BUNDLE: certificate};
  await fs.writeFile(manifestPath, JSON.stringify(manifest));
  // The CLI itself may be an official npm wrapper. Only the installed MCP
  // launcher gets the restricted system PATH, not the test client.
  const cenv = {...process.env, ...env, PATH: process.env.PATH};
  delete cenv.OPENAI_API_KEY; delete cenv.CODEX_API_KEY;
  const cli = args => execFileSync(codex, args, {env: cenv, cwd: project, encoding: 'utf8', timeout: 30000});
  const codexVersion = cli(['--version']).trim();
  cli(['plugin', 'marketplace', 'add', catalog, '--json']);
  const installed = JSON.parse(cli(['plugin', 'add', 'tensorviz@tensorviz-standalone', '--json']));
  assert.equal(installed.version, lock.version);
  const discovered = JSON.parse(cli(['mcp', 'list', '--json'])).find(x => x.name === 'tensorviz');
  assert.ok(discovered?.enabled); assert.equal(discovered.transport.command, 'sh');
  assert.equal(discovered.transport.env.PLUGIN_ROOT, installed.installedPath);
  assert.equal(await fs.stat(cache).catch(() => null), null, 'Installation must not prefetch runtime');
  // Verify TLS rejects the certificate before any request/download or execution.
  await assert.rejects(promisify(execFile)('/bin/sh', [path.join(installed.installedPath, 'scripts/launch.sh'), '--verify-native-package'], {env, cwd: project, timeout: 30000}), error => {
    assert.match(String(error.stderr), /certificate|SSL|TLS|curl: \(60\)/i, 'Failure must be TLS validation, not an unrelated launcher error');
    return true;
  });
  assert.equal(downloads, 0); assert.deepEqual(await fs.readdir(cache), []);
  child = spawn(codex, ['app-server', '--stdio'], {env: cenv, cwd: project, detached: true, stdio: ['pipe', 'pipe', 'pipe']});
  child.stderr.on('data', chunk => { diagnostics = (diagnostics + chunk.toString()).slice(-1024 * 1024); });
  child.on('error', error => { for (const entry of pending.values()) { clearTimeout(entry.timer); entry.reject(error); } pending.clear(); });
  lines = createInterface({input: child.stdout});
  lines.on('line', line => {
    let message; try { message = JSON.parse(line); } catch { return; }
    if (message.id !== undefined && pending.has(message.id)) {
      const entry = pending.get(message.id); pending.delete(message.id); clearTimeout(entry.timer);
      message.error ? entry.reject(new Error(JSON.stringify(message.error))) : entry.resolve(message.result);
    } else if (message.id !== undefined) {
      unexpectedRequests++;
      child.stdin.write(JSON.stringify({id: message.id, error: {code: -32601, message: 'No consent, model execution or external agent call is authorized by this static test.'}}) + '\n');
    }
  });
  const request = (method, params) => new Promise((resolve, reject) => {
    const id = ++nextId;
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`${method} timed out`)); }, 30000);
    pending.set(id, {resolve, reject, timer}); child.stdin.write(JSON.stringify({id, method, params}) + '\n');
  });
  const start = Date.now();
  await request('initialize', {clientInfo: {name: 'tensorviz-unix-production-acceptance', version: '1'}, capabilities: {experimentalApi: true}});
  child.stdin.write(JSON.stringify({method: 'initialized'}) + '\n');
  const started = await request('thread/start', {cwd: project, ephemeral: true, approvalPolicy: 'on-request', sandbox: 'workspace-write'});
  const threadId = started.thread.id;
  const status = await request('mcpServerStatus/list', {threadId});
  assert.equal(status.data.length, 1);
  const mcp = status.data.find(item => Object.values(item.tools).some(tool => tool.name === 'open_graph'));
  assert.ok(mcp, 'Installed TensorViz server did not expose its tools');
  const tools = Object.values(mcp.tools);
  for (const name of ['list_models', 'get_model_context', 'open_graph', 'open_graph_panel', 'get_execution_environment']) assert.ok(tools.some(t => t.name === name), name);
  for (const name of ['get_pending_permission', 'present_permission', 'get_machine_info', 'read_project_file', 'request_project_command', 'get_project_command']) assert.ok(!tools.some(t => t.name === name), `Private tool leaked: ${name}`);
  const handshakeMs = Date.now() - start;
  // Codex may initialize multiple transports concurrently. Each launcher uses
  // its own verified temporary download and atomically commits the cache.
  assert.ok(downloads >= 1, 'Cold installed startup must download the runtime');
  const coldDownloadCount = downloads;
  assert.equal((await fs.stat(path.join(cache, lock.sha256 + '.tar.gz'))).size, lock.bytes);
  assert.equal(crypto.createHash('sha256').update(await fs.readFile(path.join(cache, lock.sha256 + '.tar.gz'))).digest('hex'), lock.sha256);
  const call = async (tool, args) => { const result = await request('mcpServer/tool/call', {threadId, server: mcp.name, tool, arguments: args}); assert.ok(!result.isError, JSON.stringify(result)); return result.structuredContent; };
  const listing = await call('list_models', {});
  assert.equal(listing.host.kind, 'standalone'); assert.equal(listing.host.version, lock.version); assert.equal(listing.workspaceSelectionRequired, true);
  const unopened = await call('get_model_context', {modelId: 'unopened', captureId: 'missing', sourceId: 'missing', revision: 'a'.repeat(64)});
  assert.equal(unopened.error.code, 'NO_MODEL');
  assert.equal(unexpectedRequests, 0, 'Static discovery must not request consent or execute agents/models');
  const saved = await fs.readdir(state, {recursive: true});
  assert.ok(!saved.some(p => /(^|\/)(permissions|workspaces|runtimes|venvs)(\/|$)/.test(p)), 'Discovery must not provision or persist permission');
  await stop();
  assert.equal(forcedStop, false, 'Client must exit after EOF without forced termination');
  assert.deepEqual((await fs.readdir(root)).filter(p => p.startsWith('tensorviz-native.')), [], 'Extracted runtime leaked after client exit');
  await promisify(execFile)('/bin/sh', [path.join(installed.installedPath, 'scripts/launch.sh'), '--prepare-runtime'],
    {env: {...env, CURL_CA_BUNDLE: certificate}, cwd: project, timeout: 30000});
  assert.equal(downloads, coldDownloadCount, 'Warm preparation must reuse the verified cached archive without another HTTPS request');
  cli(['plugin', 'remove', 'tensorviz@tensorviz-standalone']);
  assert.equal(JSON.parse(cli(['mcp', 'list', '--json'])).length, 0, 'Plugin MCP registration must be removed');
  const result = {platform: target.key, version: lock.version, sourceRevision: release.sourceRevision, runtimeSha256: lock.sha256, codexVersion, toolCount: tools.length, handshakeMs, coldDownloadCount,
    passed: ['real native archive', 'installed production Unix catalog', 'untrusted TLS rejected', 'cold verified HTTPS download', 'verified warm cache reuse', 'Codex app-server MCP discovery', 'safe static tools', 'no Torch or model execution', 'compiled missing/broken Torch diagnostics', 'no consent or service provisioning', 'EOF cleanup', 'plugin removal'],
    limitations: ['No browser UI or model capture', 'Uses local HTTPS trusted CA, not public CDN', 'Linux shell metadata needs system Python3; no caller Python used by the runtime']};
  await fs.writeFile(path.join(output, 'result.json'), JSON.stringify(result, null, 2)); console.log(JSON.stringify(result));
} catch (error) { primaryError = error; await fs.writeFile(path.join(output, 'failure.txt'), String(error.stack ?? error)); throw error; }
finally {
  for (const entry of pending.values()) { clearTimeout(entry.timer); entry.reject(new Error('Acceptance shutting down')); } pending.clear();
  await stop(); lines?.close();
  await fs.writeFile(path.join(output, 'stderr.txt'), diagnostics);
  if (server) { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); }
  await fs.rm(root, {recursive: true, force: true}).catch(error => { if (!primaryError) throw error; });
}
