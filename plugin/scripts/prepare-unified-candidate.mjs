// Assemble unsigned candidate bytes only. Publication separately verifies GitHub
// attestations, native job conclusions, migration receipts and exact source.
import fs from 'node:fs/promises';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {isDeepStrictEqual} from 'node:util';
import {verifyReleaseCandidate} from './release-candidate.mjs';
import {contentHash} from './source-release-admission.mjs';
import {prepareDocsArtifact} from './docs-release-artifact.mjs';
import {stageCanonicalCatalog} from './canonical-catalog.mjs';
import {verifyReleaseMigration} from './release-migration.mjs';
import {unifiedWorkflow} from './actions-release-admission.mjs';

export async function snapshotCandidateVsix(source, destination, version) {
  if (!(await fs.lstat(source)).isFile()) throw Error('Extension candidate must be a regular file');
  await fs.copyFile(source, destination, 1);
  // Inspect exactly the private snapshot whose bytes will enter the inventory.
  const pkg = JSON.parse(execFileSync('unzip', ['-p', destination, 'extension/package.json'], {encoding: 'utf8', maxBuffer: 1024 * 1024}));
  if (pkg.version !== version || pkg.name !== 'tensorviz-vscode' || pkg.publisher !== 'tensorviz') throw Error('Extension candidate identity mismatch');
  return contentHash(await fs.readFile(destination));
}

export async function prepareUnifiedCandidate({plugin, vsix, docs, migrations, output, sourceRevision, version, runId, runAttempt}) {
  if (!/^[1-9]\d*\.\d+\.\d+$/.test(version ?? '') || !/^[a-f0-9]{40}$/.test(sourceRevision ?? '')
      || !Number.isSafeInteger(runId) || runId < 1 || !Number.isSafeInteger(runAttempt) || runAttempt < 1) throw Error('Exact candidate build identity required');
  const release = await verifyReleaseCandidate(plugin, {publishing: true});
  if (release.version !== version || release.sourceRevision !== sourceRevision) throw Error('Plugin candidate source/version mismatch');
  const migrationReceipts = await Promise.all(['darwin-arm64', 'linux-x64'].map(async p => JSON.parse(await fs.readFile(path.join(migrations, `${p}.json`), 'utf8'))));
  verifyReleaseMigration({previous: {version: '0.1.48'}, initialMain: '8b0b539953c5abe693803c15854d953a00bcd5fe', release, receipts: migrationReceipts});
  await fs.mkdir(output); // Never overwrite another candidate or an existing checkout.
  await fs.cp(plugin, path.join(output, 'plugin'), {recursive: true});
  const captured = await verifyReleaseCandidate(path.join(output, 'plugin'), {publishing: true});
  if (!isDeepStrictEqual(captured, release)) throw Error('Plugin candidate changed during assembly');
  const assets = path.join(output, 'assets');
  await fs.mkdir(assets);
  const files = {};
  const copy = async (from, name, expected) => {
    const to = path.join(assets, name);
    await fs.copyFile(from, to, 1);
    const hash = contentHash(await fs.readFile(to));
    if (expected && hash !== expected) throw Error('Candidate changed during assembly');
    files[name] = hash;
  };
  for (const [name, hash] of Object.entries(release.files)) await copy(path.join(output, 'plugin', name), name, hash);
  await copy(path.join(output, 'plugin/release.json'), 'plugin-release.json');
  files[`tensorviz-${version}.vsix`] = await snapshotCandidateVsix(vsix, path.join(assets, `tensorviz-${version}.vsix`), version);
  const provenance = Buffer.from(JSON.stringify({schemaVersion: 1, version, sourceRevision, sha256: files[`tensorviz-${version}.vsix`]}, null, 2) + '\n');
  await fs.writeFile(path.join(assets, 'extension-provenance.json'), provenance, {flag: 'wx'});
  files['extension-provenance.json'] = contentHash(provenance);
  const docsName = `tensorviz-docs-${version}.zip`;
  files[docsName] = (await prepareDocsArtifact(docs, path.join(assets, docsName), {version, sourceRevision})).sha256;
  const manifest = {schemaVersion: 1, repository: 'tensorviz-org/tensorviz', version, sourceRevision,
    build: {workflow: unifiedWorkflow, runId, runAttempt}, files};
  await fs.writeFile(path.join(output, 'unified-release.json'), JSON.stringify(manifest, null, 2) + '\n', {flag: 'wx'});
  await fs.writeFile(path.join(output, 'admission.json'), JSON.stringify({schemaVersion: 2, method: 'github-actions', runId, runAttempt, migrationReceipts}, null, 2) + '\n', {flag: 'wx'});
  await stageCanonicalCatalog(path.join(output, 'plugin/integration'), path.join(output, 'catalog-overlay'));
  return {version, sourceRevision, assets: Object.keys(files).length, output};
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 7) throw Error('Usage: prepare-unified-candidate.mjs PLUGIN VSIX DOCS MIGRATIONS NEW_OUTPUT');
  console.log(JSON.stringify(await prepareUnifiedCandidate({plugin: path.resolve(process.argv[2]), vsix: path.resolve(process.argv[3]),
    docs: path.resolve(process.argv[4]), migrations: path.resolve(process.argv[5]), output: path.resolve(process.argv[6]),
    version: process.env.TENSORVIZ_RELEASE_VERSION, sourceRevision: process.env.TENSORVIZ_REVIEWED_SOURCE_SHA,
    runId: Number(process.env.GITHUB_RUN_ID), runAttempt: Number(process.env.GITHUB_RUN_ATTEMPT)})));
}
