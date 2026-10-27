import * as assert from 'node:assert/strict';
import childProcess = require('node:child_process');
import { EventEmitter } from 'node:events';
import { mock } from 'node:test';
import type * as vscode from 'vscode';
import { TraceWorker } from '../../worker';

const envelope = JSON.stringify({ protocol_version: 1, result: { graph: { nodes: [], edges: [] } },
  spans: {}, error: null, worker: { elapsed_ms: 240_000 } });
const flush = async () => { for (let i = 0; i < 30; i++) await Promise.resolve(); };

/** Real TraceWorker/runProcess timers and generation guards; only the OS boundary
 * is simulated. No Python process or multi-minute wall-clock wait is needed. */
function rig(autoProbe = true) {
  const children: Array<EventEmitter & { pid: number; stdout: EventEmitter; stderr: EventEmitter; kill(): boolean }> = [];
  const killed: Array<{ pid: number; signal: string }> = [];
  mock.timers.enable();
  mock.method(process, 'kill', (pid: number, signal?: string | number) => {
    killed.push({ pid: Math.abs(pid), signal: String(signal) }); return true;
  });
  mock.method(childProcess, 'spawn', ((command: string, args: string[]) => {
    const child = Object.assign(new EventEmitter(), { pid: 1000 + children.length,
      stdout: new EventEmitter(), stderr: new EventEmitter(),
      kill() { killed.push({ pid: this.pid, signal: 'SIGTERM' }); return true; } });
    if (command === 'taskkill') {
      killed.push({ pid: Number(args[1]), signal: 'SIGKILL' });
      queueMicrotask(() => child.emit('close', 0));
      return child;
    }
    children.push(child);
    if (autoProbe && args[0] === '-c') queueMicrotask(() => {
      child.stdout.emit('data', Buffer.from('torch pydantic'));
      child.emit('close', 0);
    });
    return child;
  }) as typeof childProcess.spawn);
  const worker = new TraceWorker('/backend', { appendLine() {} } as unknown as vscode.OutputChannel,
    { interpreter: async () => '/python' });
  const complete = (index: number) => {
    children[index].stdout.emit('data', Buffer.from(envelope));
    children[index].emit('close', 0);
  };
  return { worker, children, killed, complete,
    dispose() { worker.dispose(); mock.timers.reset(); mock.restoreAll(); } };
}

suite('TraceWorker authoritative deadlines', () => {
  test('a cold interpreter probe does not spend the profile execution budget', async () => {
    const r = rig(false);
    try {
      const pending = r.worker.run({ file: '/ws/net.py', cls: 'Net', timeoutMs: 300_000 });
      await flush();
      mock.timers.tick(89_000);
      r.children[0].stdout.emit('data', Buffer.from('torch pydantic'));
      r.children[0].emit('close', 0);
      await flush();
      assert.equal(r.children.length, 2);
      mock.timers.tick(299_999);
      assert.deepEqual(r.killed, [], 'the profile receives its full budget after the bounded probe');
      r.complete(1);
      assert.equal((await pending).ok, true);
    } finally { r.dispose(); }
  });

  test('a profile completes after the old host/browser cutoffs and leaves no live deadline', async () => {
    const r = rig();
    try {
      const pending = r.worker.run({ file: '/ws/net.py', cls: 'Net', timeoutMs: 300_000 });
      await flush();
      assert.equal(r.children.length, 2, 'one bounded probe and one execution');
      mock.timers.tick(240_000);
      assert.deepEqual(r.killed, []);
      r.complete(1);
      const result = await pending;
      assert.ok(result.ok);
      assert.equal(result.envelope.worker.elapsed_ms, 240_000);
      mock.timers.tick(100_000);
      assert.deepEqual(r.killed, [], 'a completed execution cannot be killed by its old timer');
    } finally { r.dispose(); }
  });

  for (const timeoutMs of [100, 60_000, 300_000, 86_400_000, undefined]) {
    test(`enforces ${timeoutMs ?? 'default forward'} deadline and rejects output arriving after it`, async () => {
      const r = rig();
      try {
        const pending = r.worker.run({ file: '/ws/net.py', cls: 'Net', timeoutMs });
        await flush();
        mock.timers.tick((timeoutMs ?? 180_000) - 1);
        assert.deepEqual(r.killed, []);
        mock.timers.tick(1);
        assert.deepEqual(r.killed, [{ pid: r.children[1].pid, signal: process.platform === 'win32' ? 'SIGKILL' : 'SIGTERM' }]);
        // A process can print success while handling SIGTERM. That is too late to
        // certify a run, even when close reports exit 0.
        r.complete(1);
        const result = await pending;
        assert.ok(!result.ok && result.timedOut);
        assert.equal(result.envelope, undefined);
        mock.timers.tick(2_000);
        if (process.platform !== 'win32') {
          assert.deepEqual(r.killed[1], { pid: r.children[1].pid, signal: 'SIGKILL' });
        }
      } finally { r.dispose(); }
    });
  }

  test('cancelling a long run kills it; a superseded success cannot affect its replacement', async () => {
    const r = rig();
    try {
      const first = r.worker.run({ file: '/ws/net.py', cls: 'Net', timeoutMs: 300_000 });
      await flush();
      mock.timers.tick(240_000);
      r.worker.cancel();
      assert.deepEqual(r.killed, [{ pid: r.children[1].pid, signal: process.platform === 'win32' ? 'SIGKILL' : 'SIGTERM' }]);
      const second = r.worker.run({ file: '/ws/net.py', cls: 'Net', timeoutMs: 300_000 });
      await flush();
      assert.equal(r.children.length, 3, 'the successful probe is reused');
      r.complete(1);
      const stale = await first;
      assert.ok(!stale.ok && /superseded/.test(stale.reason));
      assert.equal(stale.envelope, undefined);
      mock.timers.tick(60_000);
      assert.ok(r.killed.every(entry => entry.pid === r.children[1].pid));
      r.complete(2);
      assert.equal((await second).ok, true);
      mock.timers.tick(300_000);
      assert.ok(r.killed.every(entry => entry.pid === r.children[1].pid));
    } finally { r.dispose(); }
  });
});
