import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { publicFiles, integrationFiles, validateRuntimeLock } from './export-public.mjs';
import { validateWindowsPE, windowsMcp } from './windows-bootstrap.mjs';
import { runtimeEntries, runtimeArchive } from './native-platforms.mjs';
import { distributionRepository } from './release-version.mjs';
import { auditArchive } from './archive-audit.mjs';

export const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const readJson = async file => JSON.parse(await fs.readFile(file, 'utf8'));
const pluginClients = runtime => [['plugins/tensorviz', 'tensorviz'], ['cursor/plugins/tensorviz', 'tensorviz-cursor'],
  ...runtimeEntries(runtime).filter(entry => entry.platform.startsWith('win32-')).map(entry => {
    const name = `tensorviz-windows-${entry.platform.slice(6)}`;
    return [`plugins/${name}`, name];
  })];

export async function auditIntegration(directory, { allowPrevious = false } = {}) {
  const publication = await readJson(path.join(directory, 'PUBLICATION.json'));
  const inventories = [integrationFiles(publication.runtime)];
  if (allowPrevious) {
    const original = publicFiles.filter(file => !file.endsWith('docs/installation.md'));
    inventories.push(publicFiles, original, [...original, 'docs/installation.md']);
  }
  const inventory = inventories.find(files => JSON.stringify(files.slice().sort())
    === JSON.stringify(Object.keys(publication.files).sort()));
  if (!inventory) throw new Error('Integration differs from the closed public file allowlist.');
  const expected = [...inventory, 'PUBLICATION.json'].sort();
  const found = [];
  const walk = async relative => {
    for (const item of await fs.readdir(path.join(directory, relative), { withFileTypes: true })) {
      const file = path.posix.join(relative, item.name);
      if (item.isDirectory()) await walk(file);
      else if (item.isFile()) found.push(file);
      else throw new Error(`Linked or special integration file: ${file}`);
    }
  };
  await walk('');
  if (JSON.stringify(found.sort()) !== JSON.stringify(expected)) {
    throw new Error('Integration differs from the closed public file allowlist.');
  }
  for (const file of inventory) {
    if (sha256(await fs.readFile(path.join(directory, file))) !== publication.files[file]) throw new Error(`Integration checksum mismatch: ${file}`);
  }
  validateRuntimeLock(publication.runtime);
  if (publication.runtime.version !== publication.version) throw new Error('Runtime and publication versions differ.');
  for (const client of ['plugins/tensorviz', 'cursor/plugins/tensorviz']) {
    if (JSON.stringify(await readJson(path.join(directory, client, 'runtime-lock.json'))) !== JSON.stringify(publication.runtime)) {
      throw new Error('Integration runtime locks differ.');
    }
  }
  for (const entry of runtimeEntries(publication.runtime).filter(entry => entry.platform.startsWith('win32-'))) {
    const arch = entry.platform.slice(6), root = path.join(directory, 'plugins', `tensorviz-windows-${arch}`);
    if (allowPrevious && !inventory.includes(`plugins/tensorviz-windows-${arch}/runtime-lock.json`)) continue;
    if (JSON.stringify(await readJson(path.join(root, 'runtime-lock.json'))) !== JSON.stringify(entry)) throw new Error('Windows runtime lock differs.');
    const bytes = await fs.readFile(path.join(root, 'scripts/tensorviz-bootstrap.exe'));
    validateWindowsPE(bytes, arch);
    const receipt = await readJson(path.join(root, 'bootstrap-provenance.json'));
    if (receipt.formatVersion !== 1 || receipt.arch !== arch || receipt.version !== publication.version
      || receipt.sourceRevision !== publication.sourceRevision || receipt.sha256 !== sha256(bytes)
      || receipt.bytes !== bytes.length || receipt.noticesSha256 !== sha256(await fs.readFile(path.join(root, 'DEPENDENCY_NOTICES.md'))))
      throw new Error('Windows bootstrap provenance mismatch.');
    if ((await fs.readFile(path.join(root, 'bootstrap-sha256.txt'), 'utf8')).trim() !== sha256(bytes)) throw new Error('Windows bootstrap checksum mismatch.');
    if (JSON.stringify(await readJson(path.join(root, 'mcp.json'))) !== JSON.stringify({mcpServers:{tensorviz:windowsMcp()}})) throw new Error('Windows MCP invocation differs.');
  }
  return publication;
}

export async function createReleaseCandidate(integration, archive) {
  // ZIP commands change cwd; resolve caller paths before building any assets.
  integration = path.resolve(integration);
  const archives = (Array.isArray(archive) ? archive : [archive]).map(file => path.resolve(file));
  const publication = await auditIntegration(integration);
  const { version } = publication;
  const filesAllowed = integrationFiles(publication.runtime);
  const expectedArchives = runtimeEntries(publication.runtime).map(runtimeArchive).sort();
  if (JSON.stringify(archives.map(file => path.basename(file)).sort()) !== JSON.stringify(expectedArchives)) throw new Error('Candidate runtime platforms differ from the lock.');
  const directory = path.join(path.dirname(archives[0]), `candidate-${version}`);
  await fs.rm(directory, { recursive: true, force: true });
  await fs.mkdir(directory);
  await fs.cp(integration, path.join(directory, 'integration'), { recursive: true });
  const files = {};
  for (const file of archives.flatMap(file => [file, `${file}.json`])) await fs.copyFile(file, path.join(directory, path.basename(file)));
  for (const [client, prefix] of pluginClients(publication.runtime)) {
    const names = filesAllowed.filter(file => file.startsWith(`${client}/`)).map(file => file.slice(client.length + 1));
    execFileSync('zip', ['-q', '-X', path.join(directory, `${prefix}-${version}.zip`), ...names], { cwd: path.join(integration, client) });
  }
  execFileSync('zip', ['-q', '-X', path.join(directory, `tensorviz-catalog-${version}.zip`), ...filesAllowed, 'PUBLICATION.json'], { cwd: integration });
  for (const item of await fs.readdir(directory, { withFileTypes: true })) {
    if (item.isFile()) files[item.name] = sha256(await fs.readFile(path.join(directory, item.name)));
  }
  const runtime = JSON.parse(execFileSync('tar', ['-xOf', archives[0], 'release-manifest.json'], { encoding: 'utf8', maxBuffer: 1024 * 1024 }));
  await fs.writeFile(path.join(directory, 'release.json'), JSON.stringify({ version,
    sourceRevision: publication.sourceRevision, rendererDistributionCoverage: runtime.rendererDistributionCoverage, files }, null, 2) + '\n');
  await verifyReleaseCandidate(directory);
  return directory;
}

export async function verifyReleaseCandidate(directory, { publishing = false } = {}) {
  const release = await readJson(path.join(directory, 'release.json'));
  const publication = await auditIntegration(path.join(directory, 'integration'));
  const { version, runtime } = publication;
  const filesAllowed = integrationFiles(runtime);
  if (release.version !== version || release.sourceRevision !== publication.sourceRevision
    || !/^[a-f0-9]{40}$/.test(release.sourceRevision)) throw new Error('Release provenance mismatch.');
  const runtimes = runtimeEntries(runtime);
  const archives = runtimes.map(runtimeArchive);
  const expected = [...archives.flatMap(archive => [archive, `${archive}.json`]), ...pluginClients(runtime).map(([,prefix]) => `${prefix}-${version}.zip`), `tensorviz-catalog-${version}.zip`].sort();
  if (JSON.stringify(Object.keys(release.files).sort()) !== JSON.stringify(expected)
    || JSON.stringify((await fs.readdir(directory)).sort()) !== JSON.stringify([...expected, 'integration', 'release.json'].sort())) throw new Error('Unexpected release assets.');
  for (const file of expected) {
    if (!(await fs.lstat(path.join(directory, file))).isFile()
      || sha256(await fs.readFile(path.join(directory, file))) !== release.files[file]) throw new Error(`Release checksum mismatch: ${file}`);
  }
  for (const [client, prefix] of pluginClients(publication.runtime)) {
    const zip = path.join(directory, `${prefix}-${version}.zip`);
    const names = filesAllowed.filter(file => file.startsWith(`${client}/`)).map(file => file.slice(client.length + 1)).sort();
    const { files: contents } = auditArchive(zip, 'zip');
    const actual = Object.keys(contents).sort();
    if (JSON.stringify(actual) !== JSON.stringify(names)) throw new Error('Submitted plugin ZIP differs from the public file allowlist.');
    for (const name of names) {
      if (contents[name] !== publication.files[`${client}/${name}`]) {
        throw new Error(`Submitted plugin ZIP checksum mismatch: ${name}`);
      }
    }
  }
  const catalog = path.join(directory, `tensorviz-catalog-${version}.zip`);
  const catalogNames = [...filesAllowed, 'PUBLICATION.json'].sort();
  const { files: catalogContents } = auditArchive(catalog, 'zip');
  const catalogActual = Object.keys(catalogContents).sort();
  if (JSON.stringify(catalogActual) !== JSON.stringify(catalogNames)) throw new Error('Submitted catalog ZIP differs from the public file allowlist.');
  for (const name of catalogNames) {
    if (catalogContents[name]
      !== sha256(await fs.readFile(path.join(directory, 'integration', name)))) throw new Error(`Submitted catalog ZIP checksum mismatch: ${name}`);
  }
  for (const entry of runtimes) {
    const archive = runtimeArchive(entry);
    if (release.files[archive] !== entry.sha256 || (await fs.stat(path.join(directory, archive))).size !== entry.bytes
      || JSON.stringify(await readJson(path.join(directory, `${archive}.json`))) !== JSON.stringify(entry)) throw new Error('Runtime lock mismatch.');
    const { files: contents, manifest } = auditArchive(path.join(directory, archive), 'runtime');
    const listed = Object.keys(contents);
    if (manifest.platform !== entry.platform || manifest.pythonABI !== entry.pythonABI || manifest.version !== version || manifest.sourceRevision !== release.sourceRevision || manifest.coreSourceIncluded !== false
      || manifest.rendererDistributionCoverage !== release.rendererDistributionCoverage
      || JSON.stringify(listed.slice().sort()) !== JSON.stringify([...Object.keys(manifest.files), 'release-manifest.json'].sort())) {
      throw new Error('Runtime manifest and candidate provenance differ.');
    }
    if (publishing && manifest.workingTreeChanged !== false) throw new Error('Publication requires a clean reviewed source working tree.');
    if (publishing && (release.rendererDistributionCoverage !== 'owner-confirmed'
      || entry.url !== `https://github.com/${distributionRepository}/releases/download/v${version}/${archive}`)) {
      throw new Error('Publication requires confirmed renderer coverage and the exact distribution release URL.');
    }
    for (const [file, digest] of Object.entries(manifest.files)) {
      if (!/^[a-f0-9]{64}$/.test(digest) || contents[file] !== digest) throw new Error(`Runtime internal checksum mismatch: ${file}`);
    }
  }
  return { ...release, publication, archive: archives[0], archives };
}

