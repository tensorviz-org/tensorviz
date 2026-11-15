import * as assert from 'node:assert/strict';
import { runProcess } from '@tensorviz/graph-host/traceWorkerCore';

suite('capture worker transport budget', () => {
  const run = (bytes: number, stdoutLimit?: number) => runProcess(process.execPath,
    ['-e', `process.stdout.write('x'.repeat(${bytes}))`],
    { cwd: process.cwd(), env: process.env, timeoutMs: 10000, stdoutLimit }, () => {});

  test('a supported worker envelope above the former 32 MiB limit arrives intact', async () => {
    const result = await run(33 * 1024 * 1024);
    assert.equal(result.code, 0);
    assert.equal(result.outputLimit, undefined);
    assert.equal(result.stdout.length, 33 * 1024 * 1024);
  });

  test('worker output still terminates above 64 MiB', async () => {
    const result = await run(65 * 1024 * 1024);
    assert.equal(result.outputLimit, 'stdout');
    assert.ok(Buffer.byteLength(result.stdout) <= 64 * 1024 * 1024);
  });

  test('explicit smaller worker limits remain authoritative', async () => {
    const result = await run(2 * 1024 * 1024, 1024 * 1024);
    assert.equal(result.outputLimit, 'stdout');
    assert.ok(Buffer.byteLength(result.stdout) <= 1024 * 1024);
  });
});
