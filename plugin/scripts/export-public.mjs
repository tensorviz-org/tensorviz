import { createHash } from 'node:crypto';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { distributionRepository, pluginVersion } from './release-version.mjs';

const root = fileURLToPath(new URL('../..', import.meta.url));
const metadata = ['plugin.json', '.codex-plugin/plugin.json', '.claude-plugin/plugin.json',
  'assets/logo.png', 'skills/model-workflow/SKILL.md'];
const documents = ['README.md', 'BUILD.md', 'PRIVACY.md', 'TERMS.md', 'SUPPORT.md', 'docs/installation.md'];
export const publicFiles = [...documents, 'LICENSE',
  '.agents/plugins/marketplace.json', '.claude-plugin/marketplace.json', '.cursor-plugin/marketplace.json',
  ...['plugins/tensorviz', 'cursor/plugins/tensorviz'].flatMap(client =>
    [...metadata, ...documents, 'scripts/launch.sh', 'runtime-lock.json', 'mcp.json', '.mcp.json'].map(file => `${client}/${file}`)),
  'cursor/plugins/tensorviz/.cursor-plugin/plugin.json'];
const json = (file, value) => fs.writeFile(file, JSON.stringify(value, null, 2) + '\n');

export function validateRuntimeLock(lock) {
  if (lock?.formatVersion !== 1 || lock.platform !== 'darwin-arm64'
    || !/^\d+\.\d+\.\d+$/.test(lock.version) || !/^[a-f0-9]{64}$/.test(lock.sha256)
    || !Number.isSafeInteger(lock.bytes) || lock.bytes < 1 || lock.bytes > 256 * 1024 * 1024
    || lock.pythonABI !== 'cpython-312-darwin') throw new Error('Invalid compiled runtime lock.');
  const url = new URL(lock.url);
  if (url.protocol !== 'https:' || url.username || url.password || url.hash) throw new Error('The runtime URL must use HTTPS without credentials.');
  return lock;
}

/** Export a closed integration allowlist. Never traverse build/source inventories. */
export async function exportPublic(destination, options = {}) {
  if ('sourceOnly' in options) throw new Error('The public core-source export is retired. Supply a compiled runtimeLock.');
  const lock = validateRuntimeLock(options.runtimeLock);
  const version = pluginVersion();
  if (lock.version !== version) throw new Error('Runtime and plugin versions differ.');
  destination = path.resolve(destination);
  await fs.mkdir(destination, { recursive: false });
  const files = {};
  const copy = async (source, relative) => {
    const stat = await fs.lstat(source);
    if (!stat.isFile()) throw new Error(`Integration input must be a regular file: ${relative}`);
    await fs.mkdir(path.dirname(path.join(destination, relative)), { recursive: true });
    await fs.copyFile(source, path.join(destination, relative));
  };
  const document = async (file, relative) => {
    await copy(path.join(root, 'plugin/packaging/public', file), relative);
    const target = path.join(destination, relative);
    await fs.writeFile(target, (await fs.readFile(target, 'utf8')).replaceAll('{{VERSION}}', version));
  };
  try {
    for (const file of documents) await document(file, file);
    await copy(path.join(root, 'LICENSE'), 'LICENSE');
    for (const client of ['plugins/tensorviz', 'cursor/plugins/tensorviz']) {
      for (const file of metadata) await copy(path.join(root, 'plugins/tensorviz', file), `${client}/${file}`);
      for (const file of documents) await document(file, `${client}/${file}`);
      await copy(path.join(root, 'plugin/integration/launch.sh'), `${client}/scripts/launch.sh`);
      await fs.chmod(path.join(destination, client, 'scripts/launch.sh'), 0o755);
      await json(path.join(destination, client, 'runtime-lock.json'), lock);
      for (const file of ['plugin.json', '.codex-plugin/plugin.json', '.claude-plugin/plugin.json']) {
        const target = path.join(destination, client, file), manifest = JSON.parse(await fs.readFile(target, 'utf8'));
        manifest.version = version;
        manifest.homepage = 'https://tensorviz.dev';
        manifest.repository = `https://github.com/${distributionRepository}`;
        const ui = manifest.interface ?? manifest.extensions?.['com.openai']?.interface;
        if (ui) Object.assign(ui, {
          longDescription: 'A local TensorViz host for PyTorch architecture, tensor shapes and model changes. Downloads a verified compiled runtime automatically. This preview supports macOS Apple Silicon and CPython 3.12. VS Code and Node.js are not required.',
          category: 'Developer Tools',
          websiteURL: 'https://tensorviz.dev',
          supportURL: `https://github.com/${distributionRepository}/blob/main/SUPPORT.md`,
          privacyPolicyURL: `https://github.com/${distributionRepository}/blob/main/PRIVACY.md`,
          termsOfServiceURL: `https://github.com/${distributionRepository}/blob/main/TERMS.md`,
          defaultPrompt: ['Explain this PyTorch model and show its TensorViz graph.'],
        });
        await json(target, manifest);
      }
      for (const [file, variable] of [['mcp.json', client.startsWith('cursor/') ? 'CURSOR_PLUGIN_ROOT' : 'PLUGIN_ROOT'], ['.mcp.json', client.startsWith('cursor/') ? 'CURSOR_PLUGIN_ROOT' : 'CLAUDE_PLUGIN_ROOT']]) {
        await json(path.join(destination, client, file), {
          ...(file === 'mcp.json' ? { $schema: 'https://agent-plugins.org/schemas/1.0.0/mcp.schema.json' } : {}),
          mcpServers: { tensorviz: {
          ...(file === 'mcp.json' ? { type: 'stdio' } : {}),
          command: file === 'mcp.json' ? 'sh' : '/bin/sh', args: [`\${${variable}}/scripts/launch.sh`],
        } } });
      }
    }
    const catalogs = {
      '.agents/plugins/marketplace.json': { name: 'tensorviz-standalone', plugins: [{ name: 'tensorviz',
        source: { source: 'local', path: './plugins/tensorviz' }, policy: { installation: 'AVAILABLE', authentication: 'ON_INSTALL' }, category: 'Developer Tools' }] },
      '.claude-plugin/marketplace.json': { name: 'tensorviz-standalone', owner: { name: 'TensorViz' }, plugins: [{ name: 'tensorviz', source: './plugins/tensorviz' }] },
      '.cursor-plugin/marketplace.json': { name: 'tensorviz', owner: { name: 'TensorViz' }, plugins: [{ name: 'tensorviz', source: 'cursor/plugins/tensorviz' }] },
    };
    const cursor = path.join(destination, 'cursor/plugins/tensorviz');
    await fs.mkdir(path.join(cursor, '.cursor-plugin'));
    const cursorManifest = JSON.parse(await fs.readFile(path.join(cursor, '.claude-plugin/plugin.json'), 'utf8'));
    await json(path.join(cursor, '.cursor-plugin/plugin.json'), { ...cursorManifest, skills: './skills/', mcpServers: './mcp.json' });
    for (const [file, value] of Object.entries(catalogs)) {
      await fs.mkdir(path.dirname(path.join(destination, file)), { recursive: true });
      await json(path.join(destination, file), value);
    }
    const inventory = async relative => {
      for (const entry of await fs.readdir(path.join(destination, relative), { withFileTypes: true })) {
        const file = path.posix.join(relative, entry.name);
        if (entry.isDirectory()) await inventory(file);
        else files[file] = createHash('sha256').update(await fs.readFile(path.join(destination, file))).digest('hex');
      }
    };
    await inventory('');
    await json(path.join(destination, 'PUBLICATION.json'), { formatVersion: 2, version,
      sourceRevision: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim(),
      status: 'compiled-integration', runtime: lock, files });
    return { destination, version, files: Object.keys(files).length };
  } catch (error) { await fs.rm(destination, { recursive: true, force: true }); throw error; }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [destination, lockFile, ...rest] = process.argv.slice(2);
  if (!destination || !lockFile || rest.length || lockFile.startsWith('--')) throw new Error('Usage: node plugin/scripts/export-public.mjs OUTPUT RUNTIME_LOCK_JSON');
  console.log(JSON.stringify(await exportPublic(destination, { runtimeLock: JSON.parse(await fs.readFile(lockFile, 'utf8')) })));
}
