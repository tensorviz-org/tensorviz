import { build } from 'esbuild';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readFileSync, writeFileSync } from 'node:fs';
import { writeBundleNotices } from '../../tooling/packaging/bundle-notices.mjs';
import { annotatePublicFirebaseConfig } from './public-firebase-config.mjs';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const result = await build({
  absWorkingDir: root, entryPoints: ['src/extension.ts'], outfile: 'out/extension.js',
  conditions: ['tensorviz-source'], bundle: true, platform: 'node', format: 'cjs', target: 'node18', external: ['vscode'],
  // The UMD entry passes require through a factory, hiding its relative imports
  // from esbuild. Bundle the ESM entry so packaged installs contain the parser.
  alias: { 'jsonc-parser': 'jsonc-parser/lib/esm/main.js' },
  sourcemap: false, metafile: true,
});
if (!Object.keys(result.metafile.inputs).some(path => path.includes('@modelcontextprotocol/sdk/'))) {
  throw new Error('The standard MCP SDK was not bundled into the extension.');
}
const bundle = resolve(root, 'out/extension.js');
writeFileSync(bundle, annotatePublicFirebaseConfig(readFileSync(bundle, 'utf8'),
  readFileSync(resolve(root, 'src/websiteAccount.ts'), 'utf8')));
writeBundleNotices(root, Object.keys(result.metafile.inputs), resolve(root, 'DEPENDENCY_NOTICES.txt'));
console.log('Bundled the extension and standard MCP SDK; no separate Node.js install is needed.');
