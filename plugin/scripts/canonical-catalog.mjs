// Offline cutover preparation only. Never mutate a checkout or publish a ref.
import {createHash} from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

export const catalogPrefix = 'distribution/plugin-catalog';
export const catalogPaths = ['.agents/plugins/marketplace.json', '.claude-plugin/marketplace.json', '.cursor-plugin/marketplace.json'];
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const json = value => Buffer.from(JSON.stringify(value, null, 2) + '\n');
const safePath = value => typeof value === 'string' && /^[A-Za-z0-9_.\-/]+$/.test(value)
  && !value.split('/').some(part => !part || part === '.' || part === '..' || part.toLowerCase() === '.git');

/** Transform an already audited export into explicit Git tree additions.
 * No deletions, no root document overwrite, and no source plugin overwrite.
 * Runtime URLs and bytes stay immutable; redirecting downloads requires a new export.
 */
export function canonicalCatalog(publication, entries) {
  if (publication?.formatVersion !== 2 || !/^\d+\.\d+\.\d+$/.test(publication.version ?? '')
      || !/^[a-f0-9]{40}$/.test(publication.sourceRevision ?? '')) throw new Error('Invalid publication provenance');
  const expected = Object.keys(publication.files ?? {}).sort();
  if (JSON.stringify(expected) !== JSON.stringify(Object.keys(entries).sort())) throw new Error('Inventory mismatch');
  const output = Object.create(null);
  const folded = new Set();
  for (const name of expected) {
    if (!safePath(name) || folded.has(name.toLowerCase())) throw new Error(`Unsafe or colliding path: ${name}`);
    folded.add(name.toLowerCase());
    const bytes = entries[name];
    if (!Buffer.isBuffer(bytes) || hash(bytes) !== publication.files[name]) throw new Error(`Checksum mismatch: ${name}`);
    output[`${catalogPrefix}/${name}`] = {bytes: Buffer.from(bytes), mode: name.endsWith('/scripts/launch.sh') ? '100755' : '100644'};
  }
  // Original export remains byte-identical under a dedicated namespace.
  output[`${catalogPrefix}/PUBLICATION.json`] = {bytes: json(publication), mode: '100644'};
  for (const name of catalogPaths) {
    if (!entries[name]) throw new Error(`Missing client catalog: ${name}`);
    const catalog = JSON.parse(entries[name].toString('utf8'));
    if (!Array.isArray(catalog.plugins) || !catalog.plugins.length) throw new Error('Empty client catalog');
    const names = new Set();
    for (const plugin of catalog.plugins) {
      if (typeof plugin.name !== 'string' || names.has(plugin.name)) throw new Error('Invalid/duplicate plugin identity');
      names.add(plugin.name);
      const objectSource = typeof plugin.source === 'object' && plugin.source !== null;
      if (objectSource && plugin.source.source !== 'local') throw new Error('Non-local marketplace source');
      const original = objectSource ? plugin.source.path : plugin.source;
      const relative = typeof original === 'string' ? original.replace(/^\.\//, '') : '';
      if (!safePath(relative) || !Object.hasOwn(entries, `${relative}/plugin.json`)) throw new Error('Marketplace source is not an exported plugin');
      const destination = `./${catalogPrefix}/${relative}`;
      if (objectSource) plugin.source.path = destination; else plugin.source = destination;
    }
    output[name] = {bytes: json(catalog), mode: '100644'};
  }
  output[`${catalogPrefix}/CANONICAL.json`] = {mode: '100644', bytes: json({formatVersion: 1,
    version: publication.version, sourceRevision: publication.sourceRevision,
    repository: 'tensorviz-org/tensorviz', originalPublicationSha256: hash(json(publication)),
    files: Object.fromEntries(Object.entries(output).map(([name, value]) => [name, hash(value.bytes)]))})};
  return output;
}

/** Write a fresh staging directory, never an existing checkout. Caller reviews tree. */
export async function stageCanonicalCatalog(integration, destination) {
  const {auditIntegration} = await import('./release-candidate.mjs');
  const publication = await auditIntegration(integration);
  const entries = Object.fromEntries(await Promise.all(Object.keys(publication.files).map(async name =>
    [name, await fs.readFile(path.join(integration, name))])));
  const output = canonicalCatalog(publication, entries);
  await fs.mkdir(destination); // EEXIST is deliberate: cannot wipe a source tree.
  for (const [name, value] of Object.entries(output)) {
    const target = path.join(destination, name);
    await fs.mkdir(path.dirname(target), {recursive: true});
    await fs.writeFile(target, value.bytes, {flag: 'wx', mode: value.mode === '100755' ? 0o755 : 0o644});
  }
  return {version: publication.version, sourceRevision: publication.sourceRevision, additions: Object.keys(output).length};
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 4) throw new Error('Usage: node canonical-catalog.mjs AUDITED_INTEGRATION NEW_STAGING_DIRECTORY');
  console.log(JSON.stringify(await stageCanonicalCatalog(path.resolve(process.argv[2]), path.resolve(process.argv[3]))));
}
