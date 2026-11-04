import * as assert from 'node:assert/strict';
import { McpOpenGraph } from '@tensorviz/graph-host/mcpOpenGraph';
import type { GraphOpeningState } from '@tensorviz/contracts/protocol';

const tick = () => new Promise(resolve => setImmediate(resolve));

suite('opening completion and immediate retry', () => {
  test('runtime approval returns a prompt continuation before opening resolves and resumes the same request', async () => {
    let approve!: () => void, waiting = true, opens = 0;
    const approval = new Promise<void>(resolve => { approve = resolve; });
    const state: GraphOpeningState = { id: 'installed', target: { workspaceFolder: '/w', file: 'model.py', cls: 'Tiny',
      sourceDigest: 'a'.repeat(64), documentVersion: 0, backendRevision: 1 },
      phase: 'needs_setup', hasCapture: false, detail: 'Choose inputs.' };
    const service = new McpOpenGraph({
      open: async () => { opens++; await approval; waiting = false; return state; },
      pending: () => waiting ? { kind: 'runtime_install_approval', detail: 'Approve the runtime installation in VS Code.' } : undefined,
      state: () => waiting ? undefined : state, model: () => ({}), action: async () => true,
    });
    const started = service.invoke('agent', { file: '/w/model.py', cls: 'Tiny' });
    assert.equal(started.status, 'awaiting_runtime_install');
    assert.equal(started.validated, false); assert.equal(started.continuationRequired, true);
    const next = started.nextAction as { kind: string; instruction: string; arguments: { requestId: string } };
    assert.equal(next.kind, 'approve_runtime_install');
    assert.match(next.instruction, /user must approve.*VS Code/);
    assert.match(next.instruction, /Repair runtime/);
    assert.equal(service.invoke('agent', next.arguments).status, 'awaiting_runtime_install');
    assert.throws(() => service.invoke('agent', {}), /opening is in progress/);
    approve(); await tick();
    const resumed = service.invoke('agent', next.arguments);
    assert.equal(resumed.status, 'needs_setup'); assert.equal(resumed.requestId, started.requestId);
    assert.equal(resumed.pending, undefined); assert.equal(opens, 1);
  });
  for (const phase of ['ready', 'failed', 'cancelled'] as const) {
    test(`a ${phase} graph can reopen while its dependency inspection finishes`, async () => {
      let state: GraphOpeningState | undefined, opens = 0;
      let finishInspection!: () => void, inspectedSignal: AbortSignal | undefined;
      const inspection = new Promise<void>(resolve => { finishInspection = resolve; });
      const service = new McpOpenGraph({
        open: async () => {
          opens++;
          state = { id: `opening-${opens}`, target: { workspaceFolder: '/w', file: 'model.py', cls: 'Tiny',
            sourceDigest: 'a'.repeat(64), documentVersion: 0, backendRevision: opens },
            phase, hasCapture: phase === 'ready', detail: 'Capture finished.' };
          return state;
        },
        state: () => state, model: () => ({}), action: async () => true,
        dependencies: async (_state, signal) => {
          if (opens === 1) { inspectedSignal = signal; await inspection; }
          return {};
        },
      });
      const first = service.invoke('panel', {});
      await tick();
      assert.ok(inspectedSignal, 'capture completed and only dependency inspection remains');
      const second = service.invoke('panel', {});
      assert.notEqual(second.requestId, first.requestId);
      assert.equal(inspectedSignal.aborted, true, 'the superseded read is cancelled');
      finishInspection();
      await tick();
      assert.equal(opens, 2);
      assert.equal(state!.id, 'opening-2', 'a late inspection cannot replace the new opening');
      assert.equal(service.invoke('panel', { requestId: second.requestId as string }).status, phase);
    });
  }

  test('an executing opening still rejects a second capture', async () => {
    let finish!: () => void;
    const execution = new Promise<void>(resolve => { finish = resolve; });
    const state: GraphOpeningState = { id: 'running', target: { workspaceFolder: '/w', file: 'model.py', cls: 'Tiny',
      sourceDigest: 'a'.repeat(64), documentVersion: 0, backendRevision: 1 },
      phase: 'capturing', hasCapture: false, detail: 'Running forward.' };
    const service = new McpOpenGraph({ open: async () => { await execution; return state; },
      state: () => state, model: () => ({}), action: async () => true });
    service.invoke('panel', {});
    assert.throws(() => service.invoke('panel', {}), /opening is in progress/);
    finish();
    await tick();
    assert.throws(() => service.invoke('panel', {}), /opening is in progress/);
    service.cancelOwner('panel');
  });
});
