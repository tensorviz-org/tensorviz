import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { writeBundleNotices } from '../../tooling/packaging/bundle-notices.mjs';
import { verifyRendererLicense } from './renderer-license.mjs';
import { pluginVersion } from './release-version.mjs';

const root = fileURLToPath(new URL('../..', import.meta.url));
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');

export function rendererLicenseKey(environment) {
  const key = (environment.REACT_APP_TLDRAW_LICENSE_KEY || environment.TLDRAW_LICENSE_KEY || '').replace(/[\u200B-\u200D\uFEFF\r\n]/g, '');
  if (!key) throw new Error('A production renderer license is required. The key is never printed.');
  return key;
}

export async function noticeSupplements(repository) {
  const missing = {};
  for (const item of JSON.parse(await fs.readFile(path.join(repository, 'plugin/licenses/sources.json'), 'utf8'))) {
    const text = await fs.readFile(path.join(repository, 'plugin/licenses', item.file), 'utf8');
    for (const name of item.packages) missing[name] = { text, source: item.source, license: item.license };
  }
  return missing;
}

export async function rendererInputs(repository) {
  const inputs = [path.join(repository, 'canvas/node_modules/@tldraw/assets/package.json')];
  const visit = modules => { for (const item of modules ?? []) {
    if (item.nameForCondition) inputs.push(item.nameForCondition);
    visit(item.modules); visit(item.children);
  } };
  for (const file of ['browser-modules.json', 'panel-modules.json']) {
    visit(JSON.parse(await fs.readFile(path.join(repository, 'plugin/out', file), 'utf8')).modules);
  }
  return inputs;
}

async function inventory(directory, relative = '') {
  const files = Object.create(null);
  for (const item of await fs.readdir(path.join(directory, relative), { withFileTypes: true })) {
    const name = path.posix.join(relative, item.name);
    if (!/^[A-Za-z0-9_./-]+$/.test(name)) throw new Error('Unsafe renderer artifact path.');
    if (item.isDirectory()) Object.assign(files, await inventory(directory, name));
    else if (item.isFile()) files[name] = sha256(await fs.readFile(path.join(directory, name)));
    else throw new Error('Renderer artifact contains a link or special file.');
  }
  return files;
}

export async function assertRendererKey(directory, key) {
  for (const item of await fs.readdir(directory, { withFileTypes: true })) {
    const file = path.join(directory, item.name);
    if (item.isDirectory() && await assertRendererKey(file, key)) return true;
    if (item.isFile() && item.name.endsWith('.js') && (await fs.readFile(file, 'utf8')).includes(key)) return true;
  }
  return false;
}

export async function writeRendererArtifact(browser, notices, directory, { version, sourceRevision }) {
  await fs.rm(directory, { recursive: true, force: true });
  await fs.mkdir(directory, { recursive: true });
  await fs.cp(browser, path.join(directory, 'browser'), { recursive: true });
  await fs.copyFile(notices, path.join(directory, 'DEPENDENCY_NOTICES.md'));
  const files = await inventory(directory);
  const manifest = { formatVersion: 1, version, sourceRevision, rendererDistributionCoverage: 'owner-confirmed', files };
  await fs.writeFile(path.join(directory, 'renderer.json'), JSON.stringify(manifest, null, 2) + '\n');
  return manifest;
}

export async function verifyRendererArtifact(directory, { version, sourceRevision, key }) {
  const files = await inventory(directory);
  const manifest = JSON.parse(await fs.readFile(path.join(directory, 'renderer.json'), 'utf8'));
  if (manifest.formatVersion !== 1 || manifest.version !== version || manifest.sourceRevision !== sourceRevision
    || !/^[a-f0-9]{40}$/.test(sourceRevision ?? '') || manifest.rendererDistributionCoverage !== 'owner-confirmed') {
    throw new Error('Shared renderer provenance differs from the reviewed source/version.');
  }
  delete files['renderer.json'];
  const names = Object.keys(files).sort();
  if (JSON.stringify(names) !== JSON.stringify(Object.keys(manifest.files).sort())
    || !['browser/index.html', 'browser/panel.html', 'DEPENDENCY_NOTICES.md'].every(file => files[file])
    || names.some(file => file !== 'DEPENDENCY_NOTICES.md' && !file.startsWith('browser/'))
    || names.some(file => files[file] !== manifest.files[file])) throw new Error('Shared renderer file inventory/checksums differ.');
  if (!key || !await assertRendererKey(path.join(directory, 'browser'), key)) throw new Error('Shared renderer does not embed the production license.');
  return manifest;
}

export async function installRendererArtifact(directory, destination, expected) {
  await verifyRendererArtifact(directory, expected);
  await fs.rm(destination, { recursive: true, force: true });
  await fs.cp(path.join(directory, 'browser'), destination, { recursive: true });
}

export async function writeNativeDependencyNotices(repository, output, { artifact, ...expected } = {}) {
  const host = JSON.parse(await fs.readFile(path.join(repository, 'plugin/out/host-inputs.json'), 'utf8'));
  const inputs = Object.keys(host.inputs).map(file => path.resolve(repository, 'plugin', file));
  if (artifact) await verifyRendererArtifact(artifact, expected);
  else inputs.push(...await rendererInputs(repository));
  writeBundleNotices(repository, inputs, output, { missing: await noticeSupplements(repository) });
  if (artifact) await fs.appendFile(output, '\n' + await fs.readFile(path.join(artifact, 'DEPENDENCY_NOTICES.md'), 'utf8'));
}

async function buildRendererArtifact(directory) {
  const version = pluginVersion(), sourceRevision = process.env.TENSORVIZ_REVIEWED_SOURCE_SHA;
  const git = args => execFileSync('git', args, { cwd: root, encoding: 'utf8' }).trim();
  if (!/^[a-f0-9]{40}$/.test(sourceRevision ?? '') || git(['rev-parse', 'HEAD']) !== sourceRevision
    || git(['status', '--porcelain', '--untracked-files=no'])) throw new Error('Shared renderer requires a clean reviewed source checkout.');
  if (process.env.TENSORVIZ_RELEASE_VERSION !== version || process.env.TLDRAW_DISTRIBUTION_CONFIRMED !== 'true') throw new Error('Shared renderer release version/coverage differs.');
  const key = rendererLicenseKey(process.env);
  await verifyRendererLicense(key);
  const env = { ...process.env, TLDRAW_LICENSE_KEY: key, REACT_APP_TLDRAW_LICENSE_KEY: key };
  for (const args of [['canvas/scripts/build-plugin.cjs'], ['canvas/scripts/build-plugin.cjs', '--panel']]) {
    execFileSync(process.execPath, args, { cwd: root, env, stdio: 'inherit' });
  }
  const notices = path.join(root, 'plugin/out/renderer-dependency-notices.md');
  writeBundleNotices(root, await rendererInputs(root), notices,
    { title: 'Bundled renderer dependency notices', missing: await noticeSupplements(root) });
  await writeRendererArtifact(path.join(root, 'plugins/tensorviz/browser'), notices, directory, { version, sourceRevision });
  await verifyRendererArtifact(directory, { version, sourceRevision, key });
  if (git(['status', '--porcelain', '--untracked-files=no'])) throw new Error('Shared renderer build changed reviewed source.');
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 3) throw new Error('Usage: node renderer-artifact.mjs OUTPUT_DIRECTORY');
  await buildRendererArtifact(path.resolve(process.argv[2]));
}
