import { build as esbuild } from 'esbuild';
const build = options => esbuild({ conditions: ['tensorviz-source'], ...options });
import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { pluginVersion } from './release-version.mjs';

import { stageBackend } from '../../tooling/runtime/stage-backend.mjs';
import { stageRuntimeAssets } from '../../tooling/runtime/assets.mjs';

const root = fileURLToPath(new URL('..', import.meta.url));
const version = pluginVersion();
const result = await build({
  absWorkingDir: root,
  entryPoints: ['src/main.ts'],
  outfile: '../plugins/tensorviz/runtime/host.cjs',
  bundle: true,
  platform: 'node',
  target: 'node22',
  format: 'cjs',
  minifySyntax: true,
  metafile: true,
  logLevel: 'info',
  define: { TENSORVIZ_VERSION: JSON.stringify(version) },
});
if (Object.keys(result.metafile.inputs).some(file => /(?:^|\/)vscode(?:\/|$)/.test(file))) {
  throw new Error('The plugin host must not include VS Code.');
}
if (Object.keys(result.metafile.inputs).some(file => /(?:^|\/)(?:extension|canvas)\/src\//.test(file))) {
  throw new Error('The plugin bundle must use shared packages, never another application\'s source.');
}
await mkdir(path.join(root, 'out'), { recursive: true });
await writeFile(path.join(root, 'out/host-inputs.json'), JSON.stringify(result.metafile, null, 2) + '\n');
// Packaging stamps manifests with this version and refuses a host built as another.
await writeFile(path.join(root, 'out/host-version.json'), JSON.stringify({ version }) + '\n');
await build({ absWorkingDir: root, entryPoints: ['src/permissions.ts'], outfile: '../plugins/tensorviz/runtime/permissions.cjs',
  bundle: true, platform: 'node', target: 'node22', format: 'cjs' });

// Exercise the real shared dependency graph without an editor-module substitute.
await build({
  absWorkingDir: root,
  entryPoints: ['../packages/graph-host/src/index.ts'],
  outfile: 'out/graph-host.cjs',
  bundle: true,
  platform: 'node',
  target: 'node22',
  format: 'cjs',
  logLevel: 'info',
});

await build({ absWorkingDir: root, entryPoints: {
  approval: 'src/approval.ts', assets: 'src/assets.ts', workspace: 'src/workspace.ts', environment: 'src/environment.ts',
  host: 'src/host.ts', graphPanel: 'src/graphPanel.ts', review: 'src/review.ts', ownedTemporary: '../packages/graph-host/src/ownedTemporary.ts', protocol: '../packages/contracts/src/protocol.ts',
  setupFiles: 'src/setupFiles.ts', locks: 'src/locks.ts', profileChoices: 'src/profileChoices.ts', dependencies: 'src/dependencies.ts', runtime: 'src/runtime.ts',
  backendPython: '../packages/graph-host/src/backendPython.ts', website: 'src/website.ts',
},
  outdir: 'out/host', bundle: true, platform: 'node', target: 'node22', format: 'cjs', outExtension: { '.js': '.cjs' } });

// The neutral runtime profile owns the tracked backend selection.
const repository = path.resolve(root, '..');
stageBackend(repository, path.join(repository, 'plugins/tensorviz/backend'), 'standalone');
stageRuntimeAssets(repository, path.join(repository, 'plugins/tensorviz'));
