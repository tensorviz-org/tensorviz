// Read-only alignment gate. Preparing launch must not bump the live 0.1.x train.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
export const unifiedVersionFiles = Object.freeze([
  'extension/package.json', 'extension/package-lock.json',
  'canvas/package.json', 'canvas/package-lock.json',
  'plugin/package.json', 'plugin/package-lock.json',
  'plugins/tensorviz/plugin.json', 'plugins/tensorviz/.codex-plugin/plugin.json',
  'plugins/tensorviz/.claude-plugin/plugin.json',
]);
export function validateUnifiedSourceVersions(documents, expectedVersion) {
  if (!/^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(expectedVersion ?? '')) throw new Error('Explicit stable release version required');
  for (const name of unifiedVersionFiles) {
    const document = documents[name];
    if (!document || document.version !== expectedVersion) throw new Error(`Source version mismatch: ${name}`);
    if (name.endsWith('package-lock.json') && document.packages?.['']?.version !== expectedVersion)
      throw new Error(`Lockfile root version mismatch: ${name}`);
  }
  return {version: expectedVersion, files: unifiedVersionFiles.length};
}
export async function checkUnifiedSourceVersions(root, expectedVersion) {
  const documents = Object.fromEntries(await Promise.all(unifiedVersionFiles.map(async name =>
    [name, JSON.parse(await fs.readFile(path.join(root, name), 'utf8'))])));
  return validateUnifiedSourceVersions(documents, expectedVersion);
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 4) throw new Error('Usage: node unified-source-versions.mjs SOURCE_ROOT EXPECTED_VERSION');
  console.log(JSON.stringify(await checkUnifiedSourceVersions(path.resolve(process.argv[2]), process.argv[3])));
}
