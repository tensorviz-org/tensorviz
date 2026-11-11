import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { writeBundleNotices } from '../../tooling/packaging/bundle-notices.mjs';
import { verifyRendererLicense } from './renderer-license.mjs';
import { pluginVersion } from './release-version.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const plugin = path.join(root, 'plugins/tensorviz');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const json = async file => JSON.parse(await fs.readFile(file, 'utf8'));
const forbidden = /(?:^|\/)(?:node_modules|__pycache__|\.env[^/]*|\.git|\.gitignore|\.tensorviz|\.DS_Store|venv|\.venv|tests?)(?:\/|$)|\.(?:pyc|map|pem|key|vsix)$/;

async function files(directory, prefix = '') {
  const result = [];
  for (const entry of await fs.readdir(path.join(directory, prefix), { withFileTypes: true })) {
    const name = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isSymbolicLink()) throw new Error(`Release contains a symbolic link: ${name}`);
    if (forbidden.test(name)) throw new Error(`Release contains an excluded file: ${name}`);
    if (entry.isDirectory()) result.push(...await files(directory, name));
    else if (entry.isFile()) result.push(name);
    else throw new Error(`Release contains a non-regular file: ${name}`);
  }
  return result.sort();
}

export async function verifyRelease(directory) {
  const manifest = await json(path.join(directory, 'release-manifest.json'));
  if (manifest.formatVersion !== 1 || manifest.name !== 'tensorviz' || manifest.compatibility.node !== '>=22') throw new Error('Unknown release compatibility contract.');
  if (manifest.target !== undefined && manifest.target !== 'cursor') throw new Error('Unknown release target.');
  const inventory = (await files(directory)).filter(file => file !== 'release-manifest.json');
  if (JSON.stringify(inventory) !== JSON.stringify(Object.keys(manifest.files).sort())) throw new Error('Release inventory does not match its manifest.');
  for (const file of inventory) {
    const bytes = await fs.readFile(path.join(directory, file));
    if (file.startsWith('plugins/') && bytes.length >= 5 * 1024 * 1024) throw new Error(`Plugin file reaches Claude's 5 MiB directory limit: ${file}`);
    if (hash(bytes) !== manifest.files[file]) throw new Error(`Release integrity mismatch: ${file}`);
  }
  const manifests = manifest.target === 'cursor' ? ['plugin.json', '.cursor-plugin/plugin.json'] : ['plugin.json', '.codex-plugin/plugin.json', '.claude-plugin/plugin.json'];
  for (const name of manifests) {
    if ((await json(path.join(directory, 'plugins/tensorviz', name))).version !== manifest.version) throw new Error(`Plugin version mismatch: ${name}`);
  }
  if (manifest.target === 'cursor') {
    const cursor = await json(path.join(directory, 'plugins/tensorviz/.cursor-plugin/plugin.json'));
    const mcp = await json(path.join(directory, 'plugins/tensorviz/mcp.json'));
    const catalog = await json(path.join(directory, '.cursor-plugin/marketplace.json'));
    if (cursor.mcpServers !== './mcp.json' || mcp.mcpServers.tensorviz.command !== 'node' || mcp.mcpServers.tensorviz.args[0] !== '${CURSOR_PLUGIN_ROOT}/scripts/launch.cjs'
      || catalog.plugins?.[0]?.source !== 'plugins/tensorviz') throw new Error('Invalid Cursor plugin path contract.');
  }
  for (const file of ['runtime/host.cjs', 'runtime/permissions.cjs', 'backend/standalone_server.py', 'browser/index.html', 'browser/panel.html', 'scripts/launch.sh', 'scripts/launch.cjs', 'scripts/install.cjs', 'scripts/install.sh', 'scripts/codex-config.cjs', 'scripts/codex-connections.cjs', 'scripts/permissions.cjs', 'scripts/codex-permissions.cjs', 'LICENSE', 'DEPENDENCY_NOTICES.md', 'licenses/tldraw-5.0.1.md']) {
    if (!manifest.files[`plugins/tensorviz/${file}`]) throw new Error(`Required release file missing: ${file}`);
  }
  return manifest;
}

/** A local build can carry a release version (TENSORVIZ_RELEASE_VERSION), as the host and
 *  Canvas builds do, so it can replace a published release, which is numbered from the base
 *  version by its release run. Tracked manifests keep the base; staged copies are stamped. */
export async function stampVersion(pluginRoot, version) {
  for (const name of ['plugin.json', '.codex-plugin/plugin.json', '.claude-plugin/plugin.json']) {
    const file = path.join(pluginRoot, name);
    const manifest = await json(file);
    if (manifest.version === version) continue;
    manifest.version = version;
    await fs.writeFile(file, JSON.stringify(manifest, null, 2) + '\n');
  }
}

const runtimeBuildHint = 'The plugin runtime has not been built. From plugin/, run npm run build, npm run build:browser, and npm run build:panel before packaging.';

async function readBuilt(file) {
  try { return await json(file); }
  catch (error) {
    if (error.code === 'ENOENT') throw new Error(runtimeBuildHint);
    throw error;
  }
}

async function assertProductionLicense() {
  const key = (process.env.REACT_APP_TLDRAW_LICENSE_KEY || process.env.TLDRAW_LICENSE_KEY || '').replace(/[\u200B-\u200D\uFEFF\r\n]/g, '');
  if (!key) throw new Error('Release packaging requires REACT_APP_TLDRAW_LICENSE_KEY or TLDRAW_LICENSE_KEY. Dev packaging uses --allow-source-development. The key is not printed.');
  const license = await verifyRendererLicense(key);
  const directory = path.join(plugin, 'browser');
  let embedded = false;
  const visit = async dir => {
    let entries;
    try { entries = await fs.readdir(dir, { withFileTypes: true }); }
    catch (error) {
      if (error.code === 'ENOENT') throw new Error('The plugin browser bundle is missing. Build it with npm run build:browser before --release. The license key is not printed.');
      throw error;
    }
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) await visit(full);
      else if (entry.name.endsWith('.js') && (await fs.readFile(full, 'utf8')).includes(key)) embedded = true;
    }
  };
  await visit(directory);
  if (!embedded) throw new Error('The plugin browser bundle does not embed the renderer license. Rebuild the browser and panel with the key set. The key is not printed.');
  console.log(`package: renderer license (${license.type}, expires ${license.expires}) embedded in the plugin browser.`);
  return license.type;
}

async function pack({ release = false } = {}) {
  const productionTldrawLicense = release ? await assertProductionLicense() : 'unverified';
  const version = pluginVersion();
  // The host refuses to start beside a manifest of another version.
  const built = (await readBuilt(path.join(root, 'plugin/out/host-version.json'))).version;
  if (built !== version) throw new Error(`The plugin host was built as ${built}, not ${version}. Build it again with the same TENSORVIZ_RELEASE_VERSION before packaging.`);
  const tracked = new Set(execFileSync('git', ['ls-files', '-z'], { cwd: root, encoding: 'utf8' }).split('\0').filter(Boolean));
  const host = await readBuilt(path.join(root, 'plugin/out/host-inputs.json'));
  const browser = await readBuilt(path.join(root, 'plugin/out/browser-modules.json'));
  const inputs = Object.keys(host.inputs).map(file => path.resolve(root, 'plugin', file));
  const visit = modules => { for (const module of modules ?? []) {
    if (module.nameForCondition) inputs.push(module.nameForCondition);
    visit(module.modules); visit(module.children);
  } };
  visit(browser.modules);
  visit((await readBuilt(path.join(root, 'plugin/out/panel-modules.json'))).modules);
  if (inputs.filter(file => file.includes('/canvas/node_modules/')).length < 10) throw new Error('Browser dependency inventory is incomplete. Rebuild browser assets.');
  // Fonts, icons and translations are copied deliberately outside webpack.
  inputs.push(path.join(root, 'canvas/node_modules/@tldraw/assets/package.json'));
  for (const file of ['LICENSE', 'runtime/licenses/tldraw-5.0.1.md', 'package.json', 'package-lock.json', 'packages/contracts/package.json', 'packages/contracts/tsconfig.json',
    'packages/graph-host/package.json', 'packages/graph-host/tsconfig.json',
    'plugin/package.json', 'plugin/package-lock.json', 'plugin/tsconfig.json',
    'plugin/scripts/build.mjs', 'plugin/scripts/package.mjs', 'tooling/packaging/bundle-notices.mjs',
    'canvas/package.json', 'canvas/package-lock.json', 'canvas/tsconfig.json',
    'canvas/scripts/build-plugin.cjs', 'canvas/scripts/plugin-license.cjs', 'runtime/python/manifest.json', 'runtime/backend-profiles.json', 'tooling/runtime/stage-backend.mjs', 'tooling/runtime/assets.mjs', 'plugin/packaging/archive/standalone-install.md', 'plugin/packaging/archive/standalone-permissions.md', 'plugin/packaging/archive/standalone-cursor.md', 'plugin/packaging/archive/standalone-panel.md', 'plugin/packaging/archive/package-names.md',
    'plugin/packaging/archive/standalone-t3.md', ...[...tracked].filter(file => file.startsWith('plugin/licenses/'))]) inputs.push(path.join(root, file));
  const temporary = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-package-'));
  const output = path.join(root, 'plugin/dist'), staging = path.join(temporary, `tensorviz-${version}`);
  const destination = path.join(staging, 'plugins/tensorviz');
  const copy = async (from, to) => { await fs.mkdir(path.dirname(to), { recursive: true }); await fs.copyFile(from, to); };
  try {
    // Only tracked plugin metadata/skills/scripts and freshly built runtime trees.
    for (const file of tracked) if (file.startsWith('plugins/tensorviz/') && !file.endsWith('/.gitignore')) {
      await copy(path.join(root, file), path.join(staging, file));
    }
    await stampVersion(destination, version);
    for (const folder of ['runtime', 'backend', 'browser']) {
      let built;
      try { built = await files(path.join(plugin, folder)); }
      catch (error) {
        if (error.code === 'ENOENT') throw new Error(runtimeBuildHint);
        throw error;
      }
      for (const file of built) await copy(path.join(plugin, folder, file), path.join(destination, folder, file));
    }
    await copy(path.join(plugin, 'runtime-manifest.json'), path.join(destination, 'runtime-manifest.json'));
    await copy(path.join(root, 'LICENSE'), path.join(destination, 'LICENSE'));
    await copy(path.join(root, 'runtime/licenses/tldraw-5.0.1.md'), path.join(destination, 'licenses/tldraw-5.0.1.md'));
    await copy(path.join(root, 'plugin/packaging/archive/standalone-install.md'), path.join(staging, 'README.md'));
    await copy(path.join(root, 'plugin/packaging/archive/standalone-install.md'), path.join(staging, 'standalone-install.md'));
    await copy(path.join(root, 'plugin/packaging/archive/standalone-permissions.md'), path.join(staging, 'standalone-permissions.md'));
    await copy(path.join(root, 'plugin/packaging/archive/standalone-cursor.md'), path.join(staging, 'standalone-cursor.md'));
    await copy(path.join(root, 'plugin/packaging/archive/standalone-panel.md'), path.join(staging, 'standalone-panel.md'));
    await copy(path.join(root, 'plugin/packaging/archive/standalone-t3.md'), path.join(staging, 'standalone-t3.md'));
    await copy(path.join(root, 'plugin/packaging/archive/package-names.md'), path.join(staging, 'package-names.md'));
    const notices = path.join(destination, 'DEPENDENCY_NOTICES.md');
    const missing = {};
    for (const item of await json(path.join(root, 'plugin/licenses/sources.json'))) {
      const text = await fs.readFile(path.join(root, 'plugin/licenses', item.file), 'utf8');
      for (const name of item.packages) missing[name] = { text, source: item.source, license: item.license };
    }
    writeBundleNotices(root, inputs, notices, { title: 'Bundled tensorViz plugin dependency notices', missing });
    const catalog = { name: 'tensorviz-standalone', plugins: [{ name: 'tensorviz', source: { source: 'local', path: './plugins/tensorviz' },
      policy: { installation: 'AVAILABLE', authentication: 'ON_INSTALL' }, category: 'Productivity' }] };
    await fs.mkdir(path.join(staging, '.agents/plugins'), { recursive: true });
    await fs.writeFile(path.join(staging, '.agents/plugins/marketplace.json'), JSON.stringify(catalog, null, 2) + '\n');
    await fs.mkdir(path.join(staging, '.claude-plugin'), { recursive: true });
    await fs.writeFile(path.join(staging, '.claude-plugin/marketplace.json'), JSON.stringify({ name: 'tensorviz-standalone',
      description: 'tensorViz tools for exploring local PyTorch models and their tensor shapes.',
      owner: { name: 'TensorViz' }, plugins: [{ name: 'tensorviz', source: './plugins/tensorviz' }] }, null, 2) + '\n');
    const sources = {};
    for (const input of [...new Set(inputs)].sort()) {
      if (input.includes('/node_modules/')) continue;
      const relative = path.relative(root, input).split(path.sep).join('/');
      if (!tracked.has(relative)) throw new Error(`Bundle includes untracked source: ${relative}`);
      sources[relative] = hash(await fs.readFile(input));
    }
    for (const file of await files(path.join(plugin, 'backend'))) {
      const relative = `backend/${file}`;
      if (!tracked.has(relative)) throw new Error(`Backend includes untracked source: ${relative}`);
      const bytes = await fs.readFile(path.join(root, relative));
      if (hash(bytes) !== hash(await fs.readFile(path.join(plugin, 'backend', file)))) throw new Error(`Stale staged backend: ${relative}`);
      sources[relative] = hash(bytes);
    }
    const baseManifest = { formatVersion: 1, name: 'tensorviz', version, status: 'evaluation-only',
      productionTldrawLicense, sourceRevision: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim(),
      workingTreeDirty: !!execFileSync('git', ['status', '--porcelain', '--untracked-files=no'], { cwd: root, encoding: 'utf8' }).trim(),
      compatibility: { node: '>=22', python: '>=3.11', browserProtocol: 1, executionPlan: 1, graphDocument: 2,
        verified: [{ platform: 'darwin-arm64', client: 'Codex 0.157.1', capture: 'cpu', installation: 'migration-update-rollback' },
          { platform: 'darwin-arm64', client: 'Claude Code 2.1.283', capture: 'cpu', installation: 'fresh' }] }, sources };
    const archiveRelease = async (directory, metadata) => {
      const inventory = {};
      for (const file of await files(directory)) if (file !== 'release-manifest.json') inventory[file] = hash(await fs.readFile(path.join(directory, file)));
      await fs.writeFile(path.join(directory, 'release-manifest.json'), JSON.stringify({ ...metadata, files: inventory }, null, 2) + '\n');
      await verifyRelease(directory);
      await fs.mkdir(output, { recursive: true });
      const archive = path.join(output, `${path.basename(directory)}.tar.gz`);
      execFileSync('tar', ['-czf', archive, '-C', temporary, path.basename(directory)], { env: { ...process.env, COPYFILE_DISABLE: '1' } });
      await fs.writeFile(`${archive}.sha256`, `${hash(await fs.readFile(archive))}  ${path.basename(archive)}\n`);
      console.log(`Verified ${Object.keys(inventory).length} release files: ${archive}`);
    };
    await archiveRelease(staging, baseManifest);

    // Cursor does not expand PLUGIN_ROOT. Derive a client-specific package from
    // the very same host/backend/browser bytes rather than changing Codex's config.
    const cursorRoot = path.join(temporary, `tensorviz-cursor-${version}`);
    await fs.rename(staging, cursorRoot);
    const cursorPlugin = path.join(cursorRoot, 'plugins/tensorviz');
    for (const name of ['.agents', '.claude-plugin', 'plugins/tensorviz/.codex-plugin', 'plugins/tensorviz/.claude-plugin', 'plugins/tensorviz/.mcp.json']) {
      await fs.rm(path.join(cursorRoot, name), { recursive: true, force: true });
    }
    const portable = await json(path.join(cursorPlugin, 'plugin.json'));
    await fs.mkdir(path.join(cursorPlugin, '.cursor-plugin'), { recursive: true });
    await fs.writeFile(path.join(cursorPlugin, '.cursor-plugin/plugin.json'), JSON.stringify({ name: portable.name,
      version, description: portable.description, author: { name: portable.author.name }, license: portable.license,
      logo: portable.extensions['com.openai'].interface.logo,
      skills: './skills/', mcpServers: './mcp.json' }, null, 2) + '\n');
    const mcp = await json(path.join(cursorPlugin, 'mcp.json'));
    mcp.mcpServers.tensorviz.args = ['${CURSOR_PLUGIN_ROOT}/scripts/launch.cjs'];
    await fs.writeFile(path.join(cursorPlugin, 'mcp.json'), JSON.stringify(mcp, null, 2) + '\n');
    await fs.mkdir(path.join(cursorRoot, '.cursor-plugin'), { recursive: true });
    await fs.writeFile(path.join(cursorRoot, '.cursor-plugin/marketplace.json'), JSON.stringify({ name: 'tensorviz-standalone',
      owner: { name: 'TensorViz' }, plugins: [{ name: 'tensorviz', source: 'plugins/tensorviz' }] }, null, 2) + '\n');
    await copy(path.join(root, 'plugin/packaging/archive/standalone-cursor.md'), path.join(cursorRoot, 'README.md'));
    await archiveRelease(cursorRoot, { ...baseManifest, target: 'cursor', compatibility: { ...baseManifest.compatibility,
      verified: [], verificationPending: ['Cursor installation, capture, display and update acceptance'] } });
  } finally { await fs.rm(temporary, { recursive: true, force: true }); }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const release = process.argv.includes('--release');
  const sourceDevelopment = process.argv.includes('--allow-source-development');
  if (process.argv.length !== 3 || release === sourceDevelopment) {
    throw new Error('Source packaging needs --allow-source-development. Release packaging needs --release and a production tldraw license in the environment. The compiled private publisher is not in this tree; see docs/plugin-build.md. export-public.mjs still assembles a client archive when you supply a runtime lock.');
  }
  try { await pack({ release }); }
  catch (error) {
    const expected = error.code === 'ENOENT'
      || /plugin runtime has not been built|browser bundle is missing|renderer license|Release packaging requires/.test(error.message || '');
    console.error(expected ? error.message : error.stack);
    process.exit(1);
  }
}
