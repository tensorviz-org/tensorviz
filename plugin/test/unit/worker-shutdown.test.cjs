const { test } = require('node:test');
const assert = require('node:assert/strict');
const { runProcess, TraceWorkerCore } = require(
  process.env.TENSORVIZ_TEST_WORKER_ADAPTER || '../../out/host/traceWorkerCore.cjs');

for (const mode of ['timeout', 'cancel']) {
  test(`worker ${mode} releases its parent pipe without a platform kill target`, async () => {
    let child, deadline;
    const worker = new TraceWorkerCore(process.cwd(), { appendLine() {} }, {});
    const run = runProcess(process.execPath,
      ['-e', "process.stdin.resume();process.stdin.on('end',()=>process.exit(0))"],
      { cwd: process.cwd(), env: process.env, timeoutMs: mode === 'timeout' ? 200 : 10000 },
      current => {
        child = current;
        // Exercise pipe cleanup independently of OS termination permissions.
        // The real child handle remains available for emergency test cleanup.
        current.pid = undefined;
        if (mode === 'cancel') {
          worker.current = current;
          setTimeout(() => worker.cancel(), 200);
        }
      });
    try {
      const result = await Promise.race([run, new Promise(resolve => {
        deadline = setTimeout(() => resolve({ deadline: true }), 2500);
      })]);
      assert.equal(result.deadline, undefined, 'Worker remained alive with its owner pipe open');
      assert.equal(result.code, 0);
      assert.equal(result.timedOut, mode === 'timeout');
    } finally {
      clearTimeout(deadline);
      if (child && child.exitCode === null) {
        child.stdin.end();
        child.kill();
        await run;
      }
    }
  });
}
