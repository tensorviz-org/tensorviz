import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { writeBundleNotices } from '../packaging/bundle-notices.mjs';

test('notices follow bundled scoped/nested package versions and exclude unused dependencies', t => {
  const root = mkdtempSync(join(tmpdir(), 'tensorviz-notices-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const fixture = (directory, name, version, notice) => {
    mkdirSync(join(root, directory), { recursive: true });
    writeFileSync(join(root, directory, 'package.json'), JSON.stringify({ name, version }));
    if (notice) writeFileSync(join(root, directory, 'LICENSE.md'), notice);
  };
  fixture('node_modules/@scope/a', '@scope/a', '1.0.0', 'Copyright A: original terms');
  fixture('node_modules/@scope/a/node_modules/b', 'b', '2.0.0', 'Copyright nested B');
  fixture('node_modules/b', 'b', '3.0.0', 'Copyright root B');
  fixture('node_modules/unused', 'unused', '4.0.0', 'Must not be included');
  const out = join(root, 'notices.txt');
  writeBundleNotices(root, [
    'src/extension.ts', 'node_modules/@scope/a/dist/one.js', 'node_modules/@scope/a/dist/two.js',
    'node_modules/@scope/a/node_modules/b/index.js', 'node_modules\\b\\index.js',
  ], out);
  const text = readFileSync(out, 'utf8');
  assert.equal(text.match(/Copyright A: original terms/g).length, 1);
  assert.match(text, /b@2.0.0/); assert.match(text, /b@3.0.0/);
  assert.match(text, /Copyright nested B/); assert.match(text, /Copyright root B/);
  assert.ok(!text.includes('unused')); assert.ok(!text.includes(root));
  fixture('node_modules/missing', 'missing', '1.0.0');
  assert.throws(() => writeBundleNotices(root, ['node_modules/missing/index.js'], out), /Missing license/);
  assert.throws(() => writeBundleNotices(root, ['src/extension.ts'], out), /No bundled dependency/);
  writeFileSync(join(root, 'node_modules/missing/package.json'), JSON.stringify({ name: 'missing', version: '1.0.0', license: 'MIT' }));
  const supplement = { license: 'MIT', source: 'https://example.test/pinned/LICENSE', text: 'Original upstream notice' };
  assert.throws(() => writeBundleNotices(root, ['node_modules/missing/index.js'], out,
    { missing: { 'missing@2.0.0': supplement } }), /Missing license/, 'a new version needs reviewed attribution');
  writeBundleNotices(root, ['node_modules/missing/index.js', 'node_modules/b/index.js'], out,
    { title: 'Standalone notices', missing: { 'missing@1.0.0': supplement, 'b@3.0.0': supplement } });
  const supplemented = readFileSync(out, 'utf8');
  assert.match(supplemented, /Original upstream notice/);
  assert.match(supplemented, /Copyright root B/, 'a supplied license is never replaced by a fallback');
});
