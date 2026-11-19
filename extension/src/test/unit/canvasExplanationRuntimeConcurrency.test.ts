import * as assert from 'node:assert/strict';
import { CanvasAgentRuntime } from '../../canvasAgentRuntime';

suite('Detached explanation runtime isolation', () => {
  test('two runtimes enter concurrently and stopping one leaves the other owner and endpoint alive', async () => {
    const released: string[] = [];
    const jobs = ['node-a', 'node-b'].map(id => {
      const runtime = new CanvasAgentRuntime({ executable: async () => 'fixture',
        release: owner => released.push(owner), tools: () => new Map() });
      const fixture = runtime as any;
      let enter!: () => void, finish!: () => void;
      const entered = new Promise<void>(resolve => { enter = resolve; });
      const gate = new Promise<void>(resolve => { finish = resolve; });
      let url = '';
      fixture.startCodex = async (_executable: string, _input: unknown, endpoint: string) => {
        url = endpoint;
        return { process: { completed: new Promise(() => {}) }, sent: false };
      };
      fixture.turn = async () => { enter(); await gate; };
      return { id, runtime, fixture, controller: new AbortController(), entered, finish, get url() { return url; } };
    });
    const running = jobs.map(job => job.runtime.run({ provider: 'codex', cwd: '/fixture', intent: 'model_explanation',
      conversationId: job.id, context: 'Offline isolation fixture', messages: [] }, () => {}, job.controller.signal)
      .then(() => undefined, error => error));
    let timer: NodeJS.Timeout | undefined;
    try {
      await Promise.race([Promise.all(jobs.map(job => job.entered)), new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error('Both detached runtimes must enter before either completes.')), 2000);
      })]);
      clearTimeout(timer);
      const [a, b] = jobs;
      assert.notEqual(a.fixture.owner, b.fixture.owner);
      assert.notEqual(a.url, b.url);
      const ownerA = a.fixture.owner, ownerB = b.fixture.owner;
      a.controller.abort(); a.runtime.clear(); a.finish();
      await running[0];
      assert.equal(b.controller.signal.aborted, false);
      assert.ok(released.includes(ownerA));
      assert.equal(released.includes(ownerB), false);
      assert.equal((await fetch(b.url)).status, 401, 'B retains its own authenticated MCP endpoint after A stops.');
      b.finish();
      assert.equal(await running[1], undefined);
    } finally {
      clearTimeout(timer);
      for (const job of jobs) { job.finish(); job.runtime.clear(); }
      await Promise.all(running);
    }
  });
});
