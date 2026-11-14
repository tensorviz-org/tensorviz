import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {removeUnixFixture} from './unix-fixture-cleanup.mjs';

test('fixture disposal uses bounded native retries for concurrent writes', async () => {
  let calls = 0;
  await removeUnixFixture('/owned/test-fixture', async (root, options) => {
    calls++;
    assert.equal(root, '/owned/test-fixture');
    assert.deepEqual(options, {recursive: true, force: true, maxRetries: 10, retryDelay: 100});
  });
  assert.equal(calls, 1, 'Node owns the bounded retry loop');
});

test('persistent cleanup failure propagates rather than becoming acceptance success', async () => {
  const error = Object.assign(new Error('directory still receiving writes'), {code: 'ENOTEMPTY'});
  await assert.rejects(removeUnixFixture('/owned/test-fixture', async () => { throw error; }), actual => actual === error);
});

test('real nested fixture is removed, including spaces and Unicode', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'fixture 测试 '));
  try {
    await fs.mkdir(path.join(root, 'nested', 'skills'), {recursive: true});
    await fs.writeFile(path.join(root, 'nested', 'skills', 'test.txt'), 'fixture');
    await removeUnixFixture(root);
    await assert.rejects(fs.stat(root), {code: 'ENOENT'});
  } finally { await fs.rm(root, {recursive: true, force: true}); }
});
