import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {nativePlatform} from './native-platforms.mjs';
import {stageWindowsArmPython} from './stage-windows-arm-python.mjs';
import {buildBootstrap, validateWindowsPE} from './windows-bootstrap.mjs';

const run = (command, args) => execFileSync(command, args, {stdio: 'inherit'});
const emit = async values => {
  assert.ok(process.env.GITHUB_ENV, 'CI environment output is required');
  for (const [key, value] of Object.entries(values)) {
    assert.ok(!/[\r\n]/.test(value));
    await fs.appendFile(process.env.GITHUB_ENV, `${key}=${value}\n`);
  }
};
const pythonIn = name => path.resolve('plugin/out', name, process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const target = nativePlatform();
assert.equal(target.key, process.env.TENSORVIZ_RELEASE_PLATFORM, 'Native runner must match the release matrix');
const mode = process.argv[2];
if (mode === 'preflight') {
  const requested = target.key === 'win32-arm64' ? await stageWindowsArmPython() : '3.12.13';
  for (const name of ['compiler-venv', 'service-venv']) run('uv', ['venv', '--python', requested, `plugin/out/${name}`]);
  const compiler = pythonIn('compiler-venv'), service = pythonIn('service-venv');
  run('uv', ['pip', 'install', '--python', compiler, '-r', 'plugin/compiler-requirements.txt']);
  const abi = JSON.parse(execFileSync(compiler, ['-c', "import importlib.util,json,platform; s=importlib.util.spec_from_file_location('compiler','plugin/scripts/compile-backend.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(json.dumps(dict(abi=m.python_abi(),machine=platform.machine())))"], {encoding: 'utf8'}));
  assert.equal(abi.abi, target.abi, 'Compiler Python ABI must match the native target');
  run('uv', ['pip', 'install', '--only-binary', ':all:', '--python', service, '-r', 'backend/requirements-standalone.txt']);
  run(service, ['-I', '-c', 'import importlib.util; assert importlib.util.find_spec("torch") is None']);
  const values = {
    TENSORVIZ_COMPILER_PYTHON: compiler, TENSORVIZ_NO_TORCH_PYTHON: compiler,
    TENSORVIZ_SERVICE_TEST_PYTHON: service,
    TENSORVIZ_SERVICE_TEST_BACKEND: path.resolve('plugin/out/native-runtime/backend'),
    TENSORVIZ_TEST_BACKEND: path.resolve('plugin/out/native-runtime/backend'),
    TENSORVIZ_TEST_RUNTIME: path.resolve('plugin/out/native-runtime'),
  };
  if (['linux-x64', 'darwin-arm64'].includes(target.key)) {
    run('uv', ['venv', '--python', requested, 'plugin/out/test-venv']);
    values.TENSORVIZ_TEST_PYTHON = pythonIn('test-venv');
    run('uv', ['pip', 'install', '--python', values.TENSORVIZ_TEST_PYTHON, 'torch==2.14.1', ...(target.os === 'linux' ? ['--index-url', 'https://download.pytorch.org/whl/cpu'] : [])]);
    run('uv', ['pip', 'install', '--python', values.TENSORVIZ_TEST_PYTHON, '-r', 'backend/requirements-standalone.txt']);
  }
  await emit(values);
} else if (mode === 'clients') {
  const find = async directory => {
    for (const entry of await fs.readdir(directory, {withFileTypes: true})) {
      const file = path.join(directory, entry.name);
      if (entry.isDirectory()) { const result = await find(file); if (result) return result; }
      else if (entry.isFile() && entry.name === (target.os === 'win32' ? 'codex.exe' : 'codex')) return file;
    }
  };
  const codex = await find(path.resolve('.bootstrap-codex/node_modules'));
  assert.ok(codex, 'Official native Codex executable is required');
  const version = execFileSync(codex, ['--version'], {encoding: 'utf8'}).trim();
  assert.match(version, /\b0\.169\.0\b/);
  await emit({TENSORVIZ_TEST_CODEX: codex});
} else if (mode === 'bootstrap') {
  const lock = path.resolve('plugin/dist', `tensorviz-runtime-${process.env.TENSORVIZ_RELEASE_VERSION}-${target.key}.tar.gz.json`);
  const values = {TENSORVIZ_PRODUCTION_RUNTIME_LOCK: lock};
  if (target.os === 'win32') values.TENSORVIZ_WINDOWS_BOOTSTRAP = await buildBootstrap({
    launcher: path.resolve('plugin/integration/launch.cjs'), pkg: path.resolve('.bootstrap-build/node_modules/@yao-pkg/pkg/lib-es5/bin.js'),
    output: path.resolve('plugin/out/windows-production-bootstrap'), arch: target.arch,
  });
  await emit(values);
} else if (mode === 'attest-windows') {
  assert.equal(target.os, 'win32');
  const receipt = JSON.parse(await fs.readFile('plugin/out/windows-production-mcp/result.json', 'utf8'));
  const lock = JSON.parse(await fs.readFile(process.env.TENSORVIZ_PRODUCTION_RUNTIME_LOCK, 'utf8'));
  assert.equal(receipt.version, process.env.TENSORVIZ_RELEASE_VERSION);
  assert.equal(receipt.sourceRevision, process.env.TENSORVIZ_REVIEWED_SOURCE_SHA);
  assert.equal(receipt.arch, target.arch); assert.equal(receipt.platform, 'win32');
  assert.equal(receipt.runtimeSha256, lock.sha256);
  const binary = await fs.readFile(process.env.TENSORVIZ_WINDOWS_BOOTSTRAP);
  validateWindowsPE(binary, target.arch);
  assert.equal(receipt.bootstrapSha256, sha256(binary), 'Bootstrap differs from the installed binary tested');
  const notices = await fs.readFile('plugin/out/native-runtime/DEPENDENCY_NOTICES.md');
  const basename = `plugin/dist/tensorviz-bootstrap-${target.arch}`;
  await fs.writeFile(`${basename}.exe`, binary);
  await fs.writeFile(`${basename}.NOTICES.md`, notices);
  await fs.writeFile(`${basename}.exe.json`, JSON.stringify({formatVersion: 1, arch: target.arch,
    version: receipt.version, sourceRevision: receipt.sourceRevision, sha256: sha256(binary),
    bytes: binary.length, noticesSha256: sha256(notices)}, null, 2) + '\n');
} else throw new Error(`Unknown release setup mode: ${mode}`);
