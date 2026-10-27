import * as assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { createHash } from 'node:crypto';
import { executionFileIdentities, verifyImportedSources } from '../../executionFiles';
import { executionSpecSchema } from '@tensorviz/graph-host/executionSpec';

suite('execution file identities', () => {
  let root: string;
  setup(async () => { root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-execution-files-')); });
  teardown(async () => { await fs.rm(root, { recursive: true, force: true }); });
  test('imported helpers require the actual compiled bytes and saved buffers', async () => {
    const source = 'WIDTH = 4\n';
    const file = path.join(root, 'helper.py'); await fs.writeFile(file, source);
    const entries = [{ file: 'helper.py', realPath: await fs.realpath(file), bytes: Buffer.byteLength(source), digest: createHash('sha256').update(source).digest('hex') }];
    await verifyImportedSources(root, entries, async () => false);
    await assert.rejects(() => verifyImportedSources(root, entries, async () => true), /unsaved/);
    await fs.writeFile(file, 'WIDTH = 8\n');
    await assert.rejects(() => verifyImportedSources(root, entries, async () => false), /changed/);
    await assert.rejects(() => verifyImportedSources(root, [], async () => false), /unavailable/);
  });
  test('fingerprints actual files and observes replacements', async () => {
    const spec = executionSpecSchema.parse({ inputs: { fixture: 'batch.pt' } });
    await fs.writeFile(path.join(root, 'batch.pt'), 'first');
    const before = await executionFileIdentities(root, spec);
    await fs.writeFile(path.join(root, 'batch.pt'), 'second');
    const after = await executionFileIdentities(root, spec);
    assert.notEqual(before[0].digest, after[0].digest);
    assert.equal(before[0].bytes, 5); assert.equal(after[0].bytes, 6);
    assert.ok(!JSON.stringify(before).includes('first'));
  });
  test('refuses symlink escapes, missing files and cancelled reads', async () => {
    const workspace = path.join(root, 'ws'); await fs.mkdir(workspace);
    await fs.writeFile(path.join(root, 'outside.pt'), 'fixture');
    await fs.symlink(path.join(root, 'outside.pt'), path.join(workspace, 'batch.pt'));
    const spec = executionSpecSchema.parse({ inputs: { fixture: 'batch.pt' } });
    await assert.rejects(() => executionFileIdentities(workspace, spec), /inside this workspace/);
    await fs.unlink(path.join(workspace, 'batch.pt'));
    await assert.rejects(() => executionFileIdentities(workspace, spec), /Cannot read/);
    await fs.writeFile(path.join(workspace, 'batch.pt'), 'fixture');
    const controller = new AbortController(); controller.abort();
    await assert.rejects(() => executionFileIdentities(workspace, spec, controller.signal));
  });
});
