import * as assert from 'node:assert/strict';
import { ExecutionReview, executionStages } from '@tensorviz/graph-host/executionReview';
import { isGraphOpeningState, isGraphSetupAction, type ExecutionResources, type ExecutionReviewState, type LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';
import { GraphOpeningController } from '@tensorviz/graph-host/graphOpening';
import type { ExecutionSpec } from '@tensorviz/graph-host/executionSpec';

const target: LocalBrowserTargetIdentity = { workspaceFolder: '/ws', file: 'model.py', cls: 'Net', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 };
const resources: ExecutionResources = { sampledAt: Date.now(), host: 'training-host', remote: true, cpu: 'CPU', cores: 32,
  cpuPercent: 91, memoryFree: 1e9, memoryTotal: 16e9, devices: [{ id: 'cuda:0', name: 'Busy GPU', utilization: 100, memoryFree: 0, memoryTotal: 8e9 }], detail: 'Observations only.' };
const execution = { device: 'cuda:0', mode: 'eval', gradEnabled: false, inputs: { args: [], kwargs: {} } } as unknown as ExecutionSpec;
const tick = () => new Promise(resolve => setImmediate(resolve));
function harness(readings: ExecutionResources | null = resources) {
  let current = true;
  const states: ExecutionReviewState[] = [];
  const review = new ExecutionReview({ current: () => current, resources: async () => { if (!readings) throw new Error('unknown'); return readings; }, publish: state => states.push(state) });
  let n = 0;
  const action = (action: 'run' | 'cancel' | 'select' | 'refresh', device = review.state!.device) => review.action(target, `request${n++}`, { kind: 'execution', reviewId: review.state!.id, action, device });
  return { review, states, action, set current(value: boolean) { current = value; } };
}
suite('explicit execution review', () => {
  test('an unapproved opening invocation publishes source without constructing or capturing until explicit Run', async () => {
    const h = harness(); let captures = 0;
    const opening = new GraphOpeningController({ current: () => true, lookup: async () => ({ ok: true, profile: 'one', guard: { spec: { profile: {} } } } as never),
      discover: async () => ({} as never), save: async () => ({}), recover: async () => undefined, parent: async () => {},
      capture: (t, _p, signal) => h.review.run(t, signal, { kind: 'capture', interpreter: '/python', execution }, async () => {
        captures++; return { target, hasCapture: true, outcome: { status: 'passed' as const, detail: 'Captured.' } };
      }), publish: () => {} });
    const opened = opening.open(target); await tick();
    assert.equal(captures, 0, 'without approval, model code cannot run as a side effect of opening');
    assert.equal(h.review.state?.phase, 'review');
    assert.equal(h.action('run'), true); await opened; assert.equal(captures, 1);
    assert.equal(h.action('run'), false, 'the same approval cannot execute again');
  });
  test('an approved opening captures on the saved device without waiting for Run', async () => {
    const states: ExecutionReviewState[] = [], phases: string[] = []; let captures = 0;
    let opening!: GraphOpeningController;
    const review = new ExecutionReview({ current: () => true, resources: async () => resources, publish: state => { states.push(state); opening.execution(state); } });
    opening = new GraphOpeningController({ current: () => true, lookup: async () => ({ ok: true, profile: 'one', guard: { spec: { profile: {} } } } as never),
      discover: async () => ({} as never), save: async () => ({}), recover: async () => undefined, parent: async () => {},
      capture: (t, _p, signal) => review.run(t, signal, { kind: 'capture', interpreter: '/python', execution, approved: true }, async (_signal, device) => {
        captures++; assert.equal(device, 'cuda:0', 'the saved device runs; nothing is chosen for the user');
        return { target, hasCapture: true, outcome: { status: 'passed' as const, detail: 'Captured.' } };
      }), publish: state => phases.push(state.phase) });
    await opening.open(target);
    assert.equal(captures, 1); assert.equal(opening.state!.phase, 'ready'); assert.equal(review.state!.phase, 'succeeded');
    assert.ok(!states.some(state => state.phase === 'review'), 'no review waits for a Run click');
    assert.ok(!phases.includes('awaiting_execution'));
    assert.equal(review.action(target, 'late-run', { kind: 'execution', reviewId: review.state!.id, action: 'run', device: 'cuda:0' }), false, 'a replayed Run cannot execute again');
    assert.equal(captures, 1);
  });
  test('an approved invocation refuses review choices and stays cancellable while running', async () => {
    const h = harness(); let finish!: () => void, signal!: AbortSignal, calls = 0;
    const run = h.review.run(target, new AbortController().signal, { kind: 'capture', execution, interpreter: '/python', approved: true },
      async s => { calls++; signal = s; await new Promise<void>(resolve => { finish = resolve; }); });
    await tick();
    assert.equal(calls, 1); assert.equal(h.review.state!.phase, 'running');
    assert.equal(h.action('select', 'cpu'), false, 'the device cannot change once running');
    assert.equal(h.action('run'), false);
    assert.equal(h.action('cancel'), true); assert.equal(signal.aborted, true); assert.equal(h.review.state!.phase, 'cancelling');
    finish(); await run; assert.equal(h.review.state!.phase, 'cancelled'); assert.equal(calls, 1);
  });
  test('a run that settles before its first host reading still shows that reading; cancelling the opening stops it', async () => {
    const states: ExecutionReviewState[] = []; let finishReading!: () => void, readingSignal!: AbortSignal;
    const review = new ExecutionReview({ current: () => true, publish: state => states.push(state),
      resources: async (_interpreter, signal) => { readingSignal = signal; await new Promise<void>(resolve => { finishReading = resolve; }); return resources; } });
    await review.run(target, new AbortController().signal, { kind: 'capture', execution, interpreter: '/python', approved: true }, async () => ({ status: 'passed', detail: 'Done.' }));
    assert.equal(review.state!.phase, 'succeeded'); assert.equal(review.state!.resources === undefined, true);
    finishReading(); await tick();
    assert.equal(review.state!.resources?.host, 'training-host', 'the strip names the host after a fast run');
    assert.equal(states.at(-1)!.resources?.host, 'training-host'); assert.equal(states.at(-1)!.phase, 'succeeded');
    const opening = new AbortController();
    const run = review.run(target, opening.signal, { kind: 'capture', execution, interpreter: '/python', approved: true },
      signal => new Promise((_resolve, reject) => signal.addEventListener('abort', () => reject(new Error('Stopped.')))));
    await tick(); const stopped = assert.rejects(run); opening.abort(); await stopped;
    assert.equal(readingSignal.aborted, true, 'a cancelled opening stops its host reading');
  });
  test('approval never runs changed source and prepares inputs only on CPU', async () => {
    const stale = harness(); let calls = 0; stale.current = false;
    await assert.rejects(stale.review.run(target, new AbortController().signal, { kind: 'capture', execution, interpreter: '/python', approved: true },
      async () => { calls++; }), /Source changed/);
    assert.equal(calls, 0); assert.equal(stale.review.state!.phase, 'failed');
    const h = harness(); let device = '';
    await h.review.run(target, new AbortController().signal, { kind: 'prepare', interpreter: '/python', approved: true }, async (_signal, d) => { device = d; });
    assert.equal(device, 'cpu'); assert.equal(h.review.state!.phase, 'succeeded');
  });
  test('preparing unknown dimensions also waits before any model construction and is CPU-only', async () => {
    const h = harness(); let constructions = 0;
    const run = h.review.run(target, new AbortController().signal, { kind: 'prepare', interpreter: '/python' }, async (_signal, device) => { assert.equal(device, 'cpu'); constructions++; });
    await tick(); assert.equal(constructions, 0); assert.equal(h.review.state!.stages[0].calls, 0);
    assert.equal(h.action('select', 'cuda:0'), false); h.action('run'); await run; assert.equal(constructions, 1);
  });
  test('JAX preview describes tracing without claiming eager forward execution', async () => {
    const h = harness(); const run = h.review.run(target, new AbortController().signal,
      { kind: 'capture', interpreter: '/python', execution: { ...execution, framework: 'jax' } }, async () => {});
    assert.equal(h.review.state!.kind, 'trace'); assert.equal(h.review.state!.stages[0].calls, 1);
    h.action('run'); await run;
  });
  test('100% GPU activity does not block its selection or silently switch to CPU', async () => {
    const h = harness(); let actual = '';
    const run = h.review.run(target, new AbortController().signal, { kind: 'capture', execution, interpreter: '/python' }, async (_signal, device) => { actual = device; return { status: 'passed', detail: 'Done.' }; });
    await tick(); assert.equal(h.review.state!.device, 'cuda:0');
    assert.equal(h.action('run'), true); await run; assert.equal(actual, 'cuda:0');
  });
  test('unknown readings do not prevent an explicit CPU choice; configured GPU stays selectable', async () => {
    const h = harness(null); let actual = '';
    const run = h.review.run(target, new AbortController().signal, { kind: 'capture', execution, interpreter: '/python' }, async (_signal, device) => { actual = device; });
    await tick(); assert.equal(h.action('select', 'cpu'), true); assert.equal(h.action('select', 'cuda:0'), true);
    assert.equal(h.action('select', 'cpu'), true); assert.equal(h.action('run'), true); await run;
    assert.equal(actual, 'cpu'); assert.equal(execution.device, 'cuda:0', 'run choices never mutate saved settings');
  });
  test('cancel before Run executes nothing; no reading refresh can restart it', async () => {
    const h = harness(); let calls = 0;
    const run = h.review.run(target, new AbortController().signal, { kind: 'capture', execution, interpreter: '/python' }, async () => { calls++; });
    const stopped = assert.rejects(run, { name: 'AbortError' }); h.action('cancel'); await stopped;
    assert.equal(calls, 0); assert.equal(h.review.state!.phase, 'cancelled'); assert.equal(h.action('run'), false);
  });
  test('cancellation stays Cancelling until the worker settles', async () => {
    const h = harness(); let finish!: () => void, signal!: AbortSignal;
    const run = h.review.run(target, new AbortController().signal, { kind: 'capture', execution, interpreter: '/python' }, async s => { signal = s; await new Promise<void>(resolve => { finish = resolve; }); });
    h.action('run'); await tick(); h.action('cancel');
    assert.equal(signal.aborted, true); assert.equal(h.review.state!.phase, 'cancelling');
    finish(); await run; assert.equal(h.review.state!.phase, 'cancelled');
  });
  test('stale source and forged targets cannot authorize project execution', async () => {
    const h = harness(); let calls = 0;
    const run = h.review.run(target, new AbortController().signal, { kind: 'capture', execution, interpreter: '/python' }, async () => { calls++; });
    assert.equal(h.review.action({ ...target, cls: 'Other' }, 'foreign', { kind: 'execution', reviewId: h.review.state!.id, action: 'run', device: 'cuda:0' }), false);
    h.current = false; const stopped = assert.rejects(run, /Source changed/); assert.equal(h.action('run'), false); await stopped; assert.equal(calls, 0);
  });
  test('a capture review plans one call, refuses timing and passes no profiling job to the run', async () => {
    assert.deepEqual(executionStages('capture'), [{ label: 'Capture shapes and operations', calls: 1 }]);
    assert.equal(executionStages('prepare')[0].calls, 0);
    const h = harness(); const runs: unknown[][] = [];
    const run = h.review.run(target, new AbortController().signal, { kind: 'capture', execution, interpreter: '/python', profileName: 'saved.run' },
      async (...args: unknown[]) => { runs.push(args); });
    assert.equal(h.review.state!.profileName, 'saved.run');
    assert.equal('timing' in h.review.state!, false);
    assert.equal(h.review.action(target, 'timing', { kind: 'execution', reviewId: h.review.state!.id, action: 'configure', device: 'cuda:0',
      timing: { warmup: 2, iterations: 7, profileIterations: 4 } } as never), false, 'a timing request never configures or approves the run');
    assert.equal(h.review.state!.phase, 'review'); assert.equal(runs.length, 0);
    h.action('run'); await run;
    assert.equal(runs.length, 1); assert.equal(runs[0].length, 2, 'the run receives its signal and device only');
  });
  test('a measured manual invocation promotes an explicitly source-only opening', () => {
    const opening = new GraphOpeningController({ publish: () => {} } as never);
    opening.sourceOnly(target);
    opening.acceptMeasuredProjection({ ...target, backendRevision: 2 });
    assert.equal(opening.state!.phase, 'ready'); assert.equal(opening.state!.hasCapture, true);
  });
  test('review survives the strict opening wire validator and rejects executable payloads', async () => {
    const h = harness(); const run = h.review.run(target, new AbortController().signal, { kind: 'capture', execution, interpreter: '/python' }, async () => {});
    await tick(); assert.ok(isGraphOpeningState({ id: 'opening', target, phase: 'awaiting_execution', detail: 'Review', hasCapture: false, execution: h.review.state }));
    assert.equal(isGraphSetupAction({ kind: 'execution', reviewId: 'x', action: 'run', device: 'cuda:0;sh' }), false);
    assert.ok(isGraphSetupAction({ kind: 'edit_execution', reviewId: 'x', field: ['inputs', 'args', 0, 'shape'] }));
    for (const field of [[], ['profile'], ['mode', 'anything'], ['inputs', -1], ['inputs', 64]]) {
      assert.equal(isGraphSetupAction({ kind: 'edit_execution', reviewId: 'x', field }), false);
    }
    const stopped = assert.rejects(run); h.action('cancel'); await stopped;
  });

  test('a reused result is labelled on the finished review; Run fresh is announced before Run', async () => {
    const h = harness();
    const served = h.review.run(target, new AbortController().signal, { kind: 'capture', interpreter: '/python', execution, approved: true },
      async () => ({ status: 'passed' as const, detail: 'Reused identical result from 14:02 (no Python ran).', reused: { capturedAt: 1761400000000 } }));
    await served;
    const finished = h.review.state!;
    assert.equal(finished.phase, 'succeeded');
    assert.deepEqual(finished.reused, { capturedAt: 1761400000000 });
    assert.ok(isGraphOpeningState({ id: 'o', target, phase: 'ready', detail: finished.detail, hasCapture: true, execution: finished }));
    const fresh = h.review.run(target, new AbortController().signal, { kind: 'capture', interpreter: '/python', execution, fresh: true },
      async () => ({ status: 'passed' as const, detail: 'Validation passed.' }));
    await tick();
    assert.equal(h.review.state!.fresh, true);
    assert.equal(h.review.state!.reused, undefined, 'a new invocation never inherits the reused label');
    assert.match(h.review.state!.detail, /executes even if an identical result is available/);
    assert.ok(isGraphOpeningState({ id: 'o', target, phase: 'awaiting_execution', detail: 'x', hasCapture: true, execution: h.review.state! }));
    assert.equal(h.action('run'), true);
    await fresh;
    assert.equal(h.review.state!.reused, undefined);
    // A failed or cancelled result never carries the label.
    await h.review.run(target, new AbortController().signal, { kind: 'capture', interpreter: '/python', execution, approved: true },
      async () => ({ status: 'failed' as const, detail: 'x', reused: { capturedAt: 1 } }));
    assert.equal(h.review.state!.reused, undefined);
  });
});
