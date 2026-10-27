import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, realpathSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { assertReleaseReceipt, dependencyFingerprint, editorPaths, maintainerPublication, sha256, QA_CHECKS, temporaryEditorRoot } from '../workflow.mjs';

const context = { commit: 'abc', source: 'source', clean: true, artifactHash: sha256('tested package') };
const candidate = () => ({ schema: 1, status: 'passed', checks: [...QA_CHECKS],
  commit: context.commit, source: context.source, clean: true, sha256: context.artifactHash,
  platform: process.platform, arch: process.arch, renderer: { expires: '2099-01-01T00:00:00Z' } });

test('editor profiles leave room for the macOS main-process Unix socket', () => {
  const directory = temporaryEditorRoot('qa');
  try { assert.ok(process.platform === 'win32' || Buffer.byteLength(join(directory, 'profile/1.14-main.sock')) < 103); }
  finally { rmSync(directory, { recursive: true, force: true }); }
});

test('default release does not publish, and publication requires explicit tokens', () => {
  assert.equal(maintainerPublication([], {}), false);
  assert.equal(maintainerPublication(['--check'], { VSCE_PAT: 'pat', GH_TOKEN: 'gh' }), false);
  assert.equal(maintainerPublication(['--publish'], { VSCE_PAT: 'pat', GH_TOKEN: 'gh' }), true);
  assert.equal(maintainerPublication(['--publish'], { VSCE_PAT: 'pat', GITHUB_TOKEN: 'gh' }), true);
  assert.throws(() => maintainerPublication(['--publish'], { VSCE_PAT: 'pat' }), /Nothing was published/);
  assert.throws(() => maintainerPublication(['--publish'], { GH_TOKEN: 'gh' }), /Nothing was published/);
});

test('release accepts the exact clean, tested, licensed candidate', () => {
  assert.doesNotThrow(() => assertReleaseReceipt(candidate(), context));
});

test('changed package bytes cannot inherit an earlier QA pass', () => {
  assert.throws(() => assertReleaseReceipt(candidate(), { ...context, artifactHash: sha256('rebuilt package') }), /untested bytes/);
});

test('dirty QA, later commits and later source changes all require another QA run', () => {
  for (const change of [{ clean: false }, { commit: 'new commit' }, { source: 'different source' }]) {
    assert.throws(() => assertReleaseReceipt(candidate(), { ...context, ...change }), /Source changed/);
  }
  assert.throws(() => assertReleaseReceipt({ ...candidate(), clean: false }, context), /dirty/);
});

test('a build or source-only host result cannot certify an installed capture', () => {
  for (const change of [{ status: 'failed' }, { checks: ['vsix-build-and-archive'] }, { schema: 0 }]) {
    assert.throws(() => assertReleaseReceipt({ ...candidate(), ...change }, context), /QA receipt/);
  }
});

test('a partial set of QA checks cannot be published even with a render receipt', () => {
  for (const missing of QA_CHECKS) {
    assert.throws(() => assertReleaseReceipt({ ...candidate(), checks: QA_CHECKS.filter(check => check !== missing) }, context), /QA receipt/);
  }
});

test('unlicensed and expired candidates remain local QA builds', () => {
  for (const renderer of [null, { expires: '2000-01-01' }, { expires: 'invalid' }]) {
    assert.throws(() => assertReleaseReceipt({ ...candidate(), renderer }, context), /renderer license/);
  }
});

test('dependency inputs changing invalidates bootstrap reuse', () => {
  const directory = mkdtempSync(join(tmpdir(), 'tensorviz-deps-'));
  try {
    writeFileSync(join(directory, 'package.json'), '{"name":"example"}');
    writeFileSync(join(directory, 'package-lock.json'), '{"lockfileVersion":3}');
    const before = dependencyFingerprint(directory);
    writeFileSync(join(directory, 'package-lock.json'), '{"lockfileVersion":3,"packages":{}}');
    assert.notEqual(dependencyFingerprint(directory), before);
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

test('a code alias targeting Cursor cannot silently replace stock VS Code', () => {
  const directory = mkdtempSync(join(tmpdir(), 'tensorviz-editor-'));
  try {
    const executable = join(directory, 'Editor');
    writeFileSync(executable, '');
    mkdirSync(join(directory, 'resources/app'), { recursive: true });
    const product = join(directory, 'resources/app/product.json');
    writeFileSync(product, '{"applicationName":"cursor"}');
    assert.throws(() => editorPaths(executable), /stock VS Code/);
    writeFileSync(product, '{"applicationName":"code"}');
    assert.equal(editorPaths(executable).executable, realpathSync(executable));
  } finally { rmSync(directory, { recursive: true, force: true }); }
});
