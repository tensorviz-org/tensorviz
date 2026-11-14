import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { writeBundleNotices } from '../../tooling/packaging/bundle-notices.mjs';
import { nativePlatform, nativePlatforms, runtimeArchive, nativeArchiveTool } from './native-platforms.mjs';
import { validateRuntimeLock } from './export-public.mjs';

const repository = fileURLToPath(new URL('../..', import.meta.url));
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const json = async file => JSON.parse(await fs.readFile(file, 'utf8'));

export async function auditNativeRuntime(directory, platform = nativePlatform().key) {
  if (!Object.hasOwn(nativePlatforms, platform)) throw new Error(`Unsupported compiled runtime target: ${platform}`);
  const raw = nativePlatforms[platform];
  const target = nativePlatform(raw.os, raw.arch);
  const files = {};
  const walk = async relative => {
    for (const entry of await fs.readdir(path.join(directory, relative), { withFileTypes: true })) {
      const name = path.posix.join(relative, entry.name);
      if (!/^[A-Za-z0-9_.\/-]+$/.test(name) || /(?:^|\/)(?:\.git|src|source|node_modules|__pycache__|\.env[^/]*)(?:\/|$)/.test(name)
        || /\.(?:py|pyc|pyo|ts|tsx|map|c|h|o|os|pem|key|cjs)$/.test(name)) throw new Error(`Core source or unsafe file in runtime: ${name}`);
      if (entry.isDirectory()) await walk(name);
      else if (entry.isFile()) {
        const bytes = await fs.readFile(path.join(directory, name));
        if (bytes.includes(Buffer.from(repository))) throw new Error(`Private build path in runtime: ${name}`);
        files[name] = digest(bytes);
      }
      else throw new Error(`Runtime contains a link or special file: ${name}`);
    }
  };
  await walk('');
  for (const file of [target.host, 'plugin.json', 'runtime-manifest.json', 'backend/native-backend.json',
    `backend/${target.backend}`, 'backend/requirements-project.txt',
    'backend/requirements-standalone.txt', 'browser/index.html', 'browser/panel.html',
    'DEPENDENCY_NOTICES.md', 'licenses/node.txt', 'licenses/pkg.txt', 'licenses/nuitka.txt', 'licenses/tldraw-5.0.1.md']) {
    if (!files[file]) throw new Error(`Native runtime is incomplete: ${file}`);
  }
  return files;
}

async function notices(directory, target = nativePlatform()) {
  const host = await json(path.join(repository, 'plugin/out/host-inputs.json'));
  const inputs = Object.keys(host.inputs).map(file => path.resolve(repository, 'plugin', file));
  const visit = modules => { for (const item of modules ?? []) {
    if (item.nameForCondition) inputs.push(item.nameForCondition);
    visit(item.modules); visit(item.children);
  } };
  for (const file of ['browser-modules.json', 'panel-modules.json']) visit((await json(path.join(repository, 'plugin/out', file))).modules);
  inputs.push(path.join(repository, 'canvas/node_modules/@tldraw/assets/package.json'));
  const missing = {};
  for (const item of await json(path.join(repository, 'plugin/licenses/sources.json'))) {
    const text = await fs.readFile(path.join(repository, 'plugin/licenses', item.file), 'utf8');
    for (const name of item.packages) missing[name] = { text, source: item.source, license: item.license };
  }
  writeBundleNotices(repository, inputs, path.join(directory, 'DEPENDENCY_NOTICES.md'), { missing });
  await fs.copyFile(path.join(repository, 'LICENSE'), path.join(directory, 'LICENSE'));
  await fs.copyFile(path.join(repository, 'plugin/node_modules/@yao-pkg/pkg/LICENSE'), path.join(directory, 'licenses/pkg.txt'));
  const python = process.env.TENSORVIZ_COMPILER_PYTHON;
  if (!python || !path.isAbsolute(python)) throw new Error('Set TENSORVIZ_COMPILER_PYTHON for compiler runtime notices.');
  const nuitkaLicense = execFileSync(python, ['-c', "import importlib.metadata; print(importlib.metadata.distribution('nuitka').locate_file('nuitka-2.8.10.dist-info/licenses/LICENSE.txt'))"], { encoding: 'utf8' }).trim();
  await fs.copyFile(nuitkaLicense, path.join(directory, 'licenses/nuitka.txt'));
  const { node } = JSON.parse(execFileSync(path.join(directory, target.host), ['--verify-native-package'], { encoding: 'utf8' }));
  // Exact upstream Node release license, including the bundled dependency notices.
  const nodeLicense = await fs.readFile(path.join(repository, 'plugin/licenses', `node-${node}.txt`), 'utf8');
  if (!nodeLicense.includes('Copyright Node.js contributors')) throw new Error('Bundled Node license is unavailable.');
  await fs.writeFile(path.join(directory, 'licenses/node.txt'), nodeLicense);
}

// pkg hosts have large version-symbol tables (observed: 1,418,885 bytes).
// Keep a finite bound and inspect the complete output; never truncate the gate.
export function verifyLinuxGlibc(file, command = execFileSync, architecture = 'x64') {
  const ceiling = { x64: 31, arm64: 35 }[architecture];
  if (!ceiling) throw new Error(`Unsupported Linux architecture for ABI audit: ${architecture}`);
  const versions = command('readelf', ['--version-info', file], { encoding: 'utf8', maxBuffer: 8 * 1024 * 1024 });
  for (const match of versions.matchAll(/GLIBC_(\d+)\.(\d+)/g)) {
    if (Number(match[1]) > 2 || Number(match[1]) === 2 && Number(match[2]) > ceiling) throw new Error(`Linux ${architecture} runtime requires glibc newer than 2.${ceiling}: ${file}`);
  }
}

export async function packageNative(url) {
  const directory = path.join(repository, 'plugin/out/native-runtime');
  const { version } = await json(path.join(directory, 'plugin.json'));
  const backend = await json(path.join(directory, 'backend/native-backend.json'));
  const target = nativePlatform();
  const check = JSON.parse(execFileSync(path.join(directory, target.host), ['--verify-native-package'], { encoding: 'utf8' }));
  if (check.sourceIncluded !== false || check.platform !== target.os || check.architecture !== target.arch || backend.abi !== target.abi || backend.architecture !== target.machine) throw new Error('Native host verification failed.');
  if (target.os === 'linux') {
    for (const file of [target.host, path.join('backend', backend.binary)]) {
      verifyLinuxGlibc(path.join(directory, file), execFileSync, target.arch);
    }
  }
  await notices(directory, target);
  await fs.rm(path.join(directory, 'release-manifest.json'), { force: true });
  const files = await auditNativeRuntime(directory, target.key);
  const manifest = { formatVersion: 1, version, platform: target.key, pythonABI: backend.abi,
    sourceRevision: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repository, encoding: 'utf8' }).trim(),
    workingTreeChanged: execFileSync('git', ['status', '--porcelain', '--untracked-files=no'], { cwd: repository, encoding: 'utf8' }).trim().length > 0,
    coreSourceIncluded: false, rendererDistributionCoverage: process.env.TLDRAW_DISTRIBUTION_CONFIRMED === 'true'
      ? 'owner-confirmed' : 'pending-owner-confirmation', files };
  await fs.writeFile(path.join(directory, 'release-manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  const archive = path.join(repository, 'plugin/dist', runtimeArchive({ version, platform: target.key }));
  await fs.mkdir(path.dirname(archive), { recursive: true });
  execFileSync(nativeArchiveTool(), ['-czf', archive, '-C', directory, ...Object.keys(files), 'release-manifest.json'], {
    env: { ...process.env, COPYFILE_DISABLE: '1' }, maxBuffer: 1024 * 1024,
  });
  const bytes = await fs.readFile(archive);
  const lock = validateRuntimeLock({ formatVersion: 1, version, platform: target.key, pythonABI: backend.abi,
    sha256: digest(bytes), bytes: bytes.length, url });
  await fs.writeFile(`${archive}.json`, JSON.stringify(lock, null, 2) + '\n');
  console.log(JSON.stringify({ archive, lock: `${archive}.json`, files: Object.keys(files).length, bytes: bytes.length }));
  return lock;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 3) throw new Error('Usage: node plugin/scripts/package-native.mjs HTTPS_RUNTIME_URL');
  await packageNative(process.argv[2]);
}
