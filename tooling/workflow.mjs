#!/usr/bin/env node
/** Local extension development, installed-package QA, and exact-artifact release. */
import { createHash } from 'node:crypto';
import { execFileSync, spawn } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { verifyRendererLicense } from '../plugin/scripts/renderer-license.mjs';

export const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export const QA_CHECKS = ['workflow', 'shared-and-contracts', 'extension-compile', 'extension-units',
  'extension-python-integrations', 'canvas-host-viewer', 'python-discovery-source-runtime',
  'vsix-build-and-archive', 'staged-assets', 'installed-capture-and-render'];
const state = join(root, '.tensorviz-workflow');
const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const children = new Set();
const json = file => JSON.parse(readFileSync(file, 'utf8'));
const save = (file, value) => { mkdirSync(dirname(file), { recursive: true }); writeFileSync(file, JSON.stringify(value, null, 2) + '\n'); };
export const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const digest = file => sha256(readFileSync(file));
const git = (...args) => execFileSync('git', args, { cwd: root, encoding: 'utf8' }).trim();

// Committed extension version. package.json and both package-lock.json version fields must agree.
function packageVersion(repoRoot, ref = 'HEAD') {
  const show = file => JSON.parse(execFileSync('git', ['show', `${ref}:${file}`], { cwd: repoRoot, encoding: 'utf8' }));
  const manifest = show('extension/package.json'), lock = show('extension/package-lock.json');
  if (!/^\d+\.\d+\.\d+$/.test(manifest.version || '') || lock.name !== manifest.name
    || lock.version !== manifest.version || lock.packages?.['']?.version !== manifest.version) {
    throw new Error('Extension package.json and both package-lock.json versions must match.');
  }
  return manifest.version;
}

export function temporaryEditorRoot(kind) {
  // macOS Unix sockets have a 103-character path limit. Worktree and TMPDIR
  // paths can exceed it before VS Code even appends its main-process socket.
  return mkdtempSync(join(process.platform === 'win32' ? tmpdir() : '/tmp', `tv-${kind}-`));
}

export function dependencyFingerprint(directory) {
  return sha256([process.versions.node.split('.')[0], ...['package.json', 'package-lock.json']
    .map(name => readFileSync(join(directory, name), 'utf8'))].join('\0'));
}

export function sourceFingerprint() {
  const files = execFileSync('git', ['ls-files', '-co', '--exclude-standard', '-z'], { cwd: root, encoding: 'utf8', maxBuffer: 8 * 1024 * 1024 });
  const hash = createHash('sha256');
  for (const file of [...new Set(files.split('\0').filter(Boolean))].sort()) {
    hash.update(file + '\0');
    hash.update(existsSync(join(root, file)) ? readFileSync(join(root, file)) : '<deleted>');
    hash.update('\0');
  }
  return hash.digest('hex');
}

function run(command, args, { cwd = root, env = {}, background = false } = {}) {
  console.log(`\n> ${command} ${args.join(' ')}`);
  const child = spawn(command, args, { cwd, env: { ...process.env, ...env }, stdio: 'inherit', detached: process.platform !== 'win32' });
  children.add(child);
  const finished = new Promise((done, fail) => {
    child.once('error', fail);
    child.once('exit', (code, signal) => { children.delete(child); code === 0 ? done() : fail(new Error(`${command} failed (${signal ?? code}).`)); });
  });
  return background ? { child, finished } : finished;
}

function stopChildren() {
  for (const child of children) {
    try { process.platform === 'win32' ? child.kill() : process.kill(-child.pid, 'SIGTERM'); } catch { /* Already exited. */ }
  }
}

export async function prepare() {
  if (process.platform === 'win32') throw new Error('The one-command workflow currently supports macOS/Linux. See docs/local-setup.md for native Windows.');
  if (Number(process.versions.node.split('.')[0]) < 22) throw new Error('Use Node.js 22 or newer.');
  mkdirSync(state, { recursive: true });
  const stampFile = join(state, 'dependencies.json');
  const stamps = existsSync(stampFile) ? json(stampFile) : {};
  for (const directory of ['', 'extension', 'canvas']) {
    const cwd = join(root, directory);
    const expected = dependencyFingerprint(cwd);
    if (!existsSync(join(cwd, 'node_modules')) || stamps[directory] !== expected) {
      await run(npm, ['ci'], { cwd });
      stamps[directory] = expected;
      save(stampFile, stamps);
    }
  }
  const venv = join(state, 'venv');
  const python = join(venv, process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
  if (!existsSync(python)) {
    const base = process.env.TENSORVIZ_DEV_PYTHON || 'python3.12';
    await run(base, ['-c', 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12 to create the development environment"']);
    await run(base, ['-m', 'venv', venv]);
  }
  const requirements = ['requirements.txt', 'requirements-cpu.txt', 'requirements-test.txt'];
  const expected = sha256(requirements.map(file => readFileSync(join(root, 'backend', file), 'utf8')).join('\0'));
  if (stamps.python !== expected) {
    await run(python, ['-m', 'pip', 'install', '-r', 'backend/requirements-cpu.txt', '-r', 'backend/requirements-test.txt']);
    stamps.python = expected;
    save(stampFile, stamps);
  }
  await run(python, ['-c', 'import torch, pytest, server_v2; from uvicorn import Config; Config(server_v2.app, log_config=None).load()'], { cwd: join(root, 'backend') });
  return python;
}

/** Resolve the actual editor, not a `code` alias that may start Cursor. */
export function editorPaths(executable) {
  const actual = realpathSync(executable);
  const app = actual.includes('.app/') ? actual.slice(0, actual.indexOf('.app/') + 4) : undefined;
  const product = app ? join(app, 'Contents/Resources/app/product.json')
    : join(dirname(actual), 'resources/app/product.json');
  if (!existsSync(product) || json(product).applicationName !== 'code') {
    throw new Error('TENSORVIZ_VSCODE_EXECUTABLE must point to stock VS Code, not Cursor or a custom editor.');
  }
  return { executable: actual, cli: app ? join(app, 'Contents/Resources/app/bin/code')
    : process.platform === 'win32' ? join(dirname(actual), 'bin/code.cmd') : actual };
}

async function editor() {
  if (process.env.TENSORVIZ_VSCODE_EXECUTABLE) return editorPaths(process.env.TENSORVIZ_VSCODE_EXECUTABLE);
  if (process.platform === 'darwin') {
    const mac = ['Code', 'Electron'].map(name => `/Applications/Visual Studio Code.app/Contents/MacOS/${name}`).find(existsSync);
    if (mac) return editorPaths(mac);
  }
  // Downloads/caches stock stable VS Code without using the shell's code alias.
  const { downloadAndUnzipVSCode } = await import('../extension/node_modules/@vscode/test-electron/out/index.js');
  return editorPaths(await downloadAndUnzipVSCode());
}

async function freePort() {
  const server = createServer();
  await new Promise((done, fail) => { server.once('error', fail); server.listen(0, '127.0.0.1', done); });
  const port = server.address().port;
  await new Promise(done => server.close(done));
  return port;
}

async function waitForPage(port) {
  const deadline = Date.now() + 120000;
  while (Date.now() < deadline) {
    try { if ((await fetch(`http://127.0.0.1:${port}`, { signal: AbortSignal.timeout(1000) })).ok) return; } catch { /* Server is compiling. */ }
    await new Promise(done => setTimeout(done, 500));
  }
  throw new Error('The Canvas development server did not become ready within two minutes.');
}

async function dev(fresh) {
  const python = await prepare();
  const code = await editor();
  await run(npm, ['run', 'clean', '--prefix', 'extension']);
  await run(npm, ['run', 'dev:build', '--prefix', 'extension']);
  const pointer = join(state, 'development.json');
  const previous = !fresh && existsSync(pointer) ? json(pointer) : undefined;
  const directory = previous?.directory ?? join(state, `dev-${Date.now()}`);
  const workspace = join(directory, 'workspace');
  if (!existsSync(workspace)) cpSync(join(root, 'examples/tensorviz-test'), workspace, { recursive: true });
  const profile = previous?.profile && existsSync(previous.profile) ? previous.profile : join(temporaryEditorRoot('dev'), 'profile');
  const sourceProfile = join(dirname(profile), 'source-profile');
  for (const editorProfile of [profile, sourceProfile]) {
    const settings = join(editorProfile, 'User/settings.json');
    if (!existsSync(settings)) save(settings, {
      'window.title': 'tensorViz Development — ${activeEditorShort}',
      'tensorviz.pythonPath': python, 'tensorviz.analytics.enabled': false,
      'workbench.startupEditor': 'none', 'extensions.autoUpdate': false,
    });
  }
  save(pointer, { directory, workspace, python, profile });
  const frontPort = await freePort();
  const backendPort = await freePort();
  const env = { BACKEND_PYTHON: python, TENSORVIZ_TEST_PYTHON: python, TENSORVIZ_TEST_PRESEED_MANAGED: '1',
    TENSORVIZ_DEV_SERVER_URL: `http://127.0.0.1:${frontPort}/`, TENSORVIZ_BACKEND_PORT: String(backendPort) };
  const watchers = [
    run(npm, ['run', 'build:shared', '--', '--watch'], { background: true }),
    run(npm, ['run', 'watch', '--prefix', 'extension'], { background: true }),
    run(npm, ['start', '--prefix', 'canvas'], { background: true, env: { ...env,
      BROWSER: 'none', HOST: '127.0.0.1', PORT: String(frontPort), REACT_APP_IDE: 'true', REACT_APP_AUTH_DISABLED: 'true' } }),
  ];
  const stopped = Promise.race(watchers.map(watch => watch.finished.then(() => { throw new Error('A development watcher stopped.'); })));
  // Attach the rejection handler before waiting for the frontend to compile.
  await Promise.race([waitForPage(frontPort), stopped]);
  const args = ['--user-data-dir', profile, '--extensions-dir', join(directory, 'extensions'), '--new-window'];
  // Separate processes avoid losing the development-host request while a new
  // source window is still initializing its main-process IPC service.
  await run(code.cli, ['--user-data-dir', sourceProfile, '--extensions-dir', join(directory, 'source-extensions'), '--new-window', root], { env });
  await run(code.cli, [...args, `--extensionDevelopmentPath=${join(root, 'extension')}`, workspace, join(workspace, 'models/simple.py')], { env });
  console.log(`\nDevelopment is ready. Profile: ${profile}\nCanvas hot reload is active. Reload the development window after host changes.\nPython changes: stop and rerun npm run dev. Ctrl+C stops this command's watchers.\nYour normal VS Code profile and installed extensions were not changed.`);
  await stopped;
}

export function assertReleaseReceipt(receipt, { commit, source, clean, artifactHash, version, platform = process.platform, arch = process.arch }) {
  if (receipt.schema !== 1 || receipt.status !== 'passed' || !QA_CHECKS.every(check => receipt.checks?.includes(check))) throw new Error('A successful package QA receipt is required.');
  if (!clean || !receipt.clean || receipt.commit !== commit || receipt.source !== source) throw new Error('Source changed or was dirty at QA. Commit the intended changes and rerun npm run qa.');
  if (receipt.sha256 !== artifactHash) throw new Error('The QA artifact changed. Refusing to publish untested bytes.');
  if (!version || receipt.version !== version) throw new Error('The QA package version must match the committed package.json and lockfile. Prepare the version before QA.');
  if (receipt.platform !== platform || receipt.arch !== arch) throw new Error('Use the machine/platform that produced this QA receipt.');
  if (!receipt.renderer || !Number.isFinite(Date.parse(receipt.renderer.expires)) || Date.parse(receipt.renderer.expires) <= Date.now()) throw new Error('QA needs a current production renderer license. Set TLDRAW_LICENSE_KEY and rerun npm run qa.');
}

async function qa() {
  rmSync(join(state, 'qa.json'), { force: true });
  const python = await prepare();
  const code = await editor();
  const commit = git('rev-parse', 'HEAD');
  const source = sourceFingerprint();
  const clean = !git('status', '--porcelain');
  const directory = join(state, `qa-${Date.now()}`);
  mkdirSync(directory, { recursive: true });
  const env = { TENSORVIZ_TEST_PYTHON: python, BACKEND_PYTHON: python, KMP_DUPLICATE_LIB_OK: 'TRUE',
    TENSORVIZ_DEV_SERVER_URL: '', TENSORVIZ_BACKEND_PORT: '', TENSORVIZ_TEST_NO_BACKEND: '', TENSORVIZ_TEST_ONLY: '' };
  const checks = [];
  const check = async (name, command, args, options = {}) => { await run(command, args, { env, ...options }); checks.push(name); };
  await check('workflow', npm, ['run', 'test:workflow']);
  await check('shared-and-contracts', 'make', ['shared-check', 'docs-links']);
  await check('extension-compile', npm, ['run', 'pretest', '--prefix', 'extension']);
  await check('extension-units', npm, ['run', 'test:unit', '--prefix', 'extension', '--', '--timeout', '15000', '--forbid-pending']);
  await check('extension-python-integrations', npm, ['run', 'test:integration', '--prefix', 'extension', '--', '--forbid-pending']);
  await check('canvas-host-viewer', npm, ['run', 'test:ci', '--prefix', 'canvas', '--', 'src/__tests__/canvas/', 'src/__tests__/hosts/', 'src/__tests__/viewer/']);
  await check('python-discovery-source-runtime', python, ['-m', 'pytest', 'backend/tests/discovery', 'backend/tests/source_review', 'backend/tests/runtime', '-q']);
  const key = process.env.TLDRAW_LICENSE_KEY || process.env.REACT_APP_TLDRAW_LICENSE_KEY;
  const renderer = key ? await verifyRendererLicense(key) : null;
  const buildEnv = { ...env, ...(key ? { REACT_APP_TLDRAW_LICENSE_KEY: key } : {}) };
  await check('vsix-build-and-archive', npm, ['run', 'vsix', '--prefix', 'extension'], { env: buildEnv });
  await check('staged-assets', npm, ['run', 'check:staged', '--prefix', 'extension']);
  const { name, version } = json(join(root, 'extension/package.json'));
  const artifact = join(directory, `${name}-${version}.vsix`);
  cpSync(join(root, 'extension', `${name}-${version}.vsix`), artifact);
  if (key) for (const entry of ['app/static/js/*.js', 'app/viewer/static/js/*.js']) {
    const shipped = execFileSync('unzip', ['-p', artifact, `extension/media/${entry}`], { maxBuffer: 64 * 1024 * 1024 });
    if (!shipped.includes(Buffer.from(key))) throw new Error('The production renderer key is missing from the packaged Canvas or viewer.');
  }
  const installed = temporaryEditorRoot('qa');
  await check('installed-capture-and-render', process.execPath, ['extension/scripts/qa-host.mjs', '--root', installed,
    '--vsix', artifact, '--vscode', code.executable, '--debug-port', String(await freePort()), '--driver', 'extension/scripts/qa-workflow.cjs']);
  cpSync(join(installed, 'evidence'), join(directory, 'installed-evidence'), { recursive: true });
  if (sourceFingerprint() !== source || git('rev-parse', 'HEAD') !== commit) throw new Error('Source changed during QA. Rerun against a stable checkout.');
  const receipt = { schema: 1, status: 'passed', version, artifact, sha256: digest(artifact), commit, source, clean,
    platform: process.platform, arch: process.arch, renderer, checks, completedAt: new Date().toISOString() };
  save(join(directory, 'receipt.json'), receipt);
  save(join(state, 'qa.json'), receipt);
  console.log(`\nPackage QA passed. Artifact: ${artifact}\nSHA-256: ${receipt.sha256}\n${clean && renderer ? 'Ready for release eligibility checks.' : 'Local QA only: production requires a clean commit and a licensed candidate.'}`);
}

/** `npm run release` only checks a QA receipt. Marketplace and GitHub publication
 *  stay behind `npm run release:publish`, which passes --publish and still
 *  refuses to run without VSCE_PAT and GH_TOKEN or GITHUB_TOKEN. */
export function maintainerPublication(args, env = process.env) {
  if (!args.includes('--publish')) return false;
  if (!(env.VSCE_PAT || '').trim() || !((env.GH_TOKEN || env.GITHUB_TOKEN || '').trim())) {
    throw new Error('Maintainer publication requires --publish plus VSCE_PAT and GH_TOKEN (or GITHUB_TOKEN). Nothing was published.');
  }
  return true;
}

async function release(args) {
  const file = join(state, 'qa.json');
  if (!existsSync(file)) throw new Error('Run npm run qa successfully before release.');
  const receipt = json(file);
  assertReleaseReceipt(receipt, { commit: git('rev-parse', 'HEAD'), source: sourceFingerprint(), clean: !git('status', '--porcelain'), artifactHash: digest(receipt.artifact), version: packageVersion(root) });
  console.log(`Verified ${receipt.version} from ${receipt.commit}: ${receipt.sha256}`);
  if (!maintainerPublication(args)) {
    console.log('Local release checks passed. Nothing published. Maintainer publication is npm run release:publish, which requires VSCE_PAT and GH_TOKEN.');
    return;
  }
  await run('git', ['fetch', 'origin', 'main']);
  await run('git', ['merge-base', '--is-ancestor', receipt.commit, 'origin/main']);
  await run('gh', ['auth', 'status']);
  // No rebuild or lifecycle scripts: publish precisely the archive inspected above.
  await run(npm, ['exec', '--no', '--', 'vsce', 'publish', '--packagePath', receipt.artifact], { cwd: join(root, 'extension'), env: { ...process.env, VSCE_PAT: process.env.VSCE_PAT } });
  const notes = join(dirname(receipt.artifact), 'release-notes.md');
  writeFileSync(notes, `Extension ${receipt.version}\n\nCommit: ${receipt.commit}\nSHA-256: ${receipt.sha256}\n\nLocal QA: ${receipt.checks.join(', ')}.\nInstalled acceptance: ${receipt.platform}/${receipt.arch}. Other platforms were not certified by this run.\n`);
  await run('gh', ['release', 'create', `v${receipt.version}`, receipt.artifact, '--target', receipt.commit,
    '--title', `tensorViz ${receipt.version}`, '--notes-file', notes]);
  console.log('Published the QA artifact to Marketplace and the versioned GitHub release.');
}

async function main() {
  const [command, ...args] = process.argv.slice(2);
  if (args.includes('--help')) { console.log('npm run dev [-- --fresh]\nnpm run qa\nnpm run release [-- --check]\nnpm run release:publish'); return; }
  const allowed = command === 'dev' ? ['--fresh'] : command === 'release' ? ['--check', '--publish'] : [];
  if (!['dev', 'qa', 'release'].includes(command) || args.some(arg => !allowed.includes(arg)) || (args.includes('--check') && args.includes('--publish'))) throw new Error('Unknown command or option. Use --help.');
  mkdirSync(state, { recursive: true });
  const lock = join(state, 'running.json');
  if (existsSync(lock)) {
    const previous = json(lock);
    let alive = true;
    try { process.kill(previous.pid, 0); } catch (error) { if (error.code === 'ESRCH') alive = false; else throw error; }
    if (alive) throw new Error(`${previous.command} is already running (PID ${previous.pid}). Stop it before ${command}; build outputs are shared.`);
    rmSync(lock);
  }
  writeFileSync(lock, JSON.stringify({ pid: process.pid, command }), { flag: 'wx' });
  const cleanup = () => { stopChildren(); rmSync(lock, { force: true }); };
  process.once('SIGINT', () => { cleanup(); process.exit(130); });
  process.once('SIGTERM', () => { cleanup(); process.exit(143); });
  try { await (command === 'dev' ? dev(args.includes('--fresh')) : command === 'qa' ? qa() : release(args)); }
  finally { cleanup(); }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => { console.error(`\nworkflow: ${error.message}`); process.exitCode = 1; });
}
