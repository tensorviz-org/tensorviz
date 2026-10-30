import * as assert from 'node:assert/strict';
import { ValidationSummaryStore } from '@tensorviz/graph-host/validationSummary';
import { ExecutionReview } from '@tensorviz/graph-host/executionReview';
import { LocalBrowserValidation } from '../../localBrowserValidation';
import { LocalBrowserSync } from '../../localBrowserSync';
import { LocalBrowserHostPublish } from '@tensorviz/graph-host/localBrowserHostTransport';
import { isLocalBrowserMessage, LocalBrowserValidateRequestMessage, LocalBrowserValidationOutcome } from '@tensorviz/contracts/protocol';

const request: LocalBrowserValidateRequestMessage = { type: 'validateRequest', protocolVersion: 1,
  workspaceSessionId: 'session', messageId: 'm1', requestId: 'v1',
  target: { workspaceFolder: '/ws', file: 'net.py', cls: 'Net', documentVersion: 1,
    sourceDigest: 'a'.repeat(64), backendRevision: 1 } };

suite('LocalBrowserValidation input source', () => {
  function harness(profiles?: unknown, measuredStatus?: string) {
    const runs: Array<{ execution: unknown; timeoutMs: unknown; guard: unknown; job: unknown; reuse: unknown }> = [];
    const sync = { validate: async (_t: unknown, _s: AbortSignal, _ti?: unknown, _p?: unknown,
      _m?: Function, _c?: unknown, execution?: unknown, _j?: unknown, timeoutMs?: unknown, guard?: unknown,
      _a?: unknown, _d?: unknown, reuse?: unknown) => {
      runs.push({ execution, timeoutMs, guard, job: _j, reuse });
      if (measuredStatus === 'reused') return { status: 'passed', detail: 'Reused identical result from 14:02 (no Python ran).', reused: { capturedAt: 5 } } as LocalBrowserValidationOutcome;
      return { status: 'passed', detail: 'ok' } as LocalBrowserValidationOutcome;
    } } as unknown as LocalBrowserSync;
    const replies: LocalBrowserHostPublish[] = [];
    const manager = new LocalBrowserValidation(sync, m => replies.push(m), profiles as never);
    return { runs, replies, manager };
  }
  const settle = async () => { for (let i = 0; i < 4; i++) await new Promise(r => setImmediate(r)); };
  const listing = (names: string[], profile?: unknown) => ({ profiles: names,
    profileDefinitions: Object.fromEntries(names.map(name => [name, profile])), guards: {},
    resolution: { status: 'needs_setup', requirements: [{ profile: null,
      field: names.length ? ['profiles'] : ['tensorviz.toml'], phase: 'static',
      code: names.length ? 'ambiguous' : 'missing', reason: 'Select saved settings.', remedy: 'Check the config.' }] } });

  test('a class with no saved profile refuses instead of searching inputs', async () => {
    const { runs, manager, replies } = harness({ snapshot: async () => listing([]) as never, choose: async () => undefined });
    manager.handle(request); await settle();
    assert.equal(runs.length, 0);
    assert.equal((replies[0] as { status?: string }).status, 'refused');
  });

  test('Run fresh bypasses reused results; a reused result reaches the page as a valid validateResult', async () => {
    const profile = { cwd: '.', tracked_files: [],
      limits: { timeout_ms: 1234, max_events: 1, max_tensors: 1, max_record_bytes: 1 },
      kind: 'model', model: 'net:Net', mode: 'eval', device: 'cpu', grad_enabled: false, seed: 0,
      init: { args: [], kwargs: {} },
      inputs: { args: [], kwargs: {} } };
    const profiles = { snapshot: async (_r: string, name: string | undefined) => (name === undefined
      ? listing(['only'], profile) : { profiles: ['only'], guards: {}, resolution: { status: 'resolved', spec: { profile, identity: { profile: name } } } }) as never,
      choose: async () => undefined };
    const { runs, manager, replies } = harness(profiles, 'reused');
    manager.handle(request); await settle();
    manager.handle({ ...request, messageId: 'm2', requestId: 'v2', fresh: true }); await settle();
    assert.deepEqual(runs.map(run => run.reuse), ['allow', 'bypass']);
    const result = replies.find(reply => (reply as { requestId?: string }).requestId === 'v1') as Record<string, unknown>;
    assert.deepEqual(result.reused, { capturedAt: 5 });
    assert.ok(isLocalBrowserMessage({ ...result, protocolVersion: 1, workspaceSessionId: 'session', messageId: 'r1' }));
  });

  test('a saved profile for the class overrides the automatic input', async () => {
    const profile = { cwd: '.', tracked_files: [],
      limits: { timeout_ms: 1234, max_events: 1, max_tensors: 1, max_record_bytes: 1 },
      kind: 'model', model: 'net:Net', mode: 'eval', device: 'cpu', grad_enabled: false, seed: null,
      init: { args: [], kwargs: {} },
      inputs: { args: [{ kind: 'tensor', shape: [1, 8], dtype: 'float32', requires_grad: false,
        generator: { kind: 'normal', mean: 0, std: 1, seed: 0 } }], kwargs: {} } };
    const { runs, manager } = harness({
      snapshot: async (_r: string, name: string | undefined) => (name === undefined
        ? listing(['only'], profile) : { profiles: ['only'], guards: {}, resolution: { status: 'resolved', spec: { profile, identity: { profile: name } } } }) as never,
      choose: async () => undefined });
    manager.handle(request);
    await settle();
    assert.equal(runs.length, 1);
    assert.ok(runs[0].execution, 'the profile pins the invocation');
    assert.equal(runs[0].timeoutMs, 1234, "and its own limit bounds the run");
    assert.ok(runs[0].guard, 'saved source/config/environment identity must reach the execution owner');
  });

  test('without a review, one capture runs with the saved timeout and no profiling job', async () => {
    const profile = { kind: 'model', model: 'net:Net', mode: 'eval', device: 'cpu', grad_enabled: false,
      cwd: '.', tracked_files: [], init: { args: [], kwargs: {} }, seed: 0,
      inputs: { args: [{ kind: 'tensor', shape: [1, 8], dtype: 'float32', requires_grad: false, generator: { kind: 'full', value: 1 } }], kwargs: {} },
      limits: { timeout_ms: 1234 } };
    const { runs, replies, manager } = harness({ snapshot: async (_r: string, name?: string) => name
      ? { guards: {}, resolution: { status: 'resolved', spec: { profile, identity: { profile: name } } } }
      : listing(['only'], profile), choose: async () => undefined }, 'measured');
    manager.handle({ ...request, measureLatency: true } as never); await settle();
    assert.equal(runs.length, 1);
    assert.equal(runs[0].job, undefined);
    assert.equal(runs[0].timeoutMs, 1234);
    assert.equal((replies[0] as { status: string }).status, 'passed');
    assert.equal('validationPassed' in replies[0], false);
  });

  test('a script profile naming the class refuses instead of quietly guessing', async () => {
    const profile = { cwd: '.', tracked_files: [],
      limits: { timeout_ms: 1, max_events: 1, max_tensors: 1, max_record_bytes: 1 },
      kind: 'script', script: 'train.py', args: [], capture: { model: 'net:Net', call: 1 } };
    const { runs, replies, manager } = harness({
      snapshot: async (_r: string, name: string | undefined) => (name === undefined
        ? listing(['training'], profile) : { profiles: ['training'], guards: {}, resolution: { status: 'resolved', spec: { profile, identity: { profile: name } } } }) as never,
      choose: async () => undefined });
    manager.handle(request);
    await settle();
    assert.equal(runs.length, 0, 'what the user wrote must not be replaced by a guess');
    assert.equal((replies[0] as { status?: string }).status, 'refused');
    assert.match((replies[0] as { detail: string }).detail, /script profile/);
  });
});

suite('LocalBrowserValidation request ownership', () => {
  test('duplicates do not run twice and replies retain the original target', async () => {
    let finish!: (result: LocalBrowserValidationOutcome) => void;
    let calls = 0;
    let signal!: AbortSignal;
    const sync = { validate: (_target: unknown, value: AbortSignal) => {
      calls++; signal = value; return new Promise<LocalBrowserValidationOutcome>(resolve => { finish = resolve; });
    } } as unknown as LocalBrowserSync;
    const replies: LocalBrowserHostPublish[] = [];
    const manager = new LocalBrowserValidation(sync, message => replies.push(message), { snapshot: async (_root, selected) => (selected ? { guards: {}, resolution: { status: 'resolved', spec: { profile: { kind: 'model', model: 'net:Net', init: { args: [], kwargs: {} }, inputs: { args: [], kwargs: {} }, mode: 'eval', device: 'cpu', grad_enabled: false, limits: { timeout_ms: 1000 } }, identity: { profile: selected } } } } : { profiles: ['one'], profileDefinitions: { one: { kind: 'model', model: 'net:Net' } }, guards: {}, resolution: { status: 'needs_setup', requirements: [{ profile: null, field: ['profiles'], phase: 'static', code: 'ambiguous', reason: 'Choose.', remedy: 'Choose.' }] } }) as never, choose: async () => undefined });
    manager.handle(request);
    manager.handle(request);
    manager.handle({ ...request, requestId: 'v2' });
    // The duplicate and the second request are refused synchronously, before any lookup.
    assert.equal(replies.length, 1);
    await new Promise(resolve => setImmediate(resolve));
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(calls, 1);
    assert.equal(replies[0].type, 'validateResult');
    manager.cancel('another-request');
    assert.equal(signal.aborted, false);
    manager.cancel('v1');
    assert.equal(signal.aborted, true);
    finish({ status: 'passed', detail: 'Late success.' });
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(replies.length, 2);
    assert.deepEqual(replies[1], { type: 'validateResult', requestId: 'v1', target: request.target,
      status: 'cancelled', detail: 'Validation cancelled.' });
  });

  test('cancelling profile resolution cannot launch a late automatic fallback', async () => {
    let release!: () => void;
    let runs = 0;
    const replies: LocalBrowserHostPublish[] = [];
    const manager = new LocalBrowserValidation({ validate: async () => { runs++; } } as unknown as LocalBrowserSync,
      message => replies.push(message), {
        snapshot: async () => {
          await new Promise<void>(resolve => { release = resolve; });
          throw new Error('cancelled snapshot');
        }, choose: async () => undefined,
      });
    manager.handle(request);
    manager.cancel(request.requestId);
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(replies.length, 1, 'cancel must finish before an uncooperative snapshot returns');
    assert.equal((replies[0] as { status: string }).status, 'cancelled');
    release();
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(runs, 0);
    assert.deepEqual(replies, [{ type: 'validateResult', requestId: request.requestId,
      target: request.target, status: 'cancelled', detail: 'Validation cancelled.' }]);
  });

  test('cancelled lookup releases ownership for retry and ignores the old late response', async () => {
    let release!: (value: never) => void;
    let calls = 0;
    let runs = 0;
    const replies: LocalBrowserHostPublish[] = [];
    const absent = { profiles: [], guards: {}, resolution: { status: 'needs_setup', requirements: [
      { profile: null, field: ['tensorviz.toml'], code: 'missing', phase: 'static', reason: 'Missing config.', remedy: 'Create config.' },
    ] } } as never;
    const manager = new LocalBrowserValidation({ validate: async () => { runs++; return { status: 'passed', detail: 'Ran.' }; } } as unknown as LocalBrowserSync,
      message => replies.push(message), {
        snapshot: async () => ++calls === 1 ? new Promise<never>(resolve => { release = resolve; }) : absent,
        choose: async () => undefined,
      });
    manager.handle(request);
    manager.cancel(request.requestId);
    await new Promise(resolve => setImmediate(resolve));
    manager.handle({ ...request, requestId: 'retry' });
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(runs, 0);
    assert.deepEqual(replies.map(reply => (reply as { status: string }).status), ['cancelled', 'refused']);
    release(absent);
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(runs, 0, 'neither absent profile may launch an automatic run');
    assert.equal(replies.length, 2);
  });
});

suite('Run model runs one capture', () => {
  const profile = { kind: 'model', model: 'net:Net', cwd: '.', tracked_files: [],
    init: { args: [], kwargs: {} }, inputs: { args: [], kwargs: {} },
    mode: 'eval', device: 'cpu', grad_enabled: false, seed: 0, limits: { timeout_ms: 1000 } };
  const profiles = { snapshot: async (_root: string, selected?: string) => selected
    ? { guards: {}, resolution: { status: 'resolved', spec: { profile, identity: { profile: selected }, interpreter: { executable: '/env/python' } } } }
    : { profiles: ['one'], profileDefinitions: { one: profile }, guards: {}, resolution: { status: 'needs_setup', requirements: [{ profile: null, field: ['profiles'], phase: 'static', code: 'ambiguous', reason: 'Choose.', remedy: 'Choose.' }] } },
    choose: async () => undefined };
  const settle = async () => { for (let i = 0; i < 5; i++) await new Promise(r => setImmediate(r)); };

  test('the review runs one capture and no profiling job, even when an older page asks for latency', async () => {
    const jobs: unknown[] = [], summaries: import('@tensorviz/contracts/protocol').LocalBrowserValidationSummary[] = [];
    const review = new ExecutionReview({ current: () => true, resources: async () => { throw new Error('Readings unavailable'); }, publish: () => {} });
    const sync = { validate: async (_t: unknown, _s: unknown, _ti: unknown, _p: unknown, measured: Function, _c: unknown, _e: unknown, job: unknown) => {
      jobs.push(job); measured(request.target, { revision: 2 });
      return { status: 'passed', detail: 'Passed.' };
    } } as unknown as LocalBrowserSync;
    const replies: LocalBrowserHostPublish[] = [];
    new LocalBrowserValidation(sync, message => replies.push(message), profiles as never, new ValidationSummaryStore(s => summaries.push(s)), review)
      .handle({ ...request, measureLatency: true } as never);
    await settle(); assert.equal(jobs.length, 0);
    assert.equal(review.state!.kind, 'capture'); assert.equal(review.state!.profileName, 'one');
    review.action(request.target, 'run', { kind: 'execution', action: 'run', reviewId: review.state!.id, device: 'cpu' });
    await settle(); assert.deepEqual(jobs, [undefined]);
    const reply = replies[0] as import('@tensorviz/contracts/protocol').LocalBrowserValidateResultMessage;
    assert.equal(reply.status, 'passed'); assert.equal('validationPassed' in reply, false);
    assert.ok(summaries.every(summary => !('validationPassed' in summary)));
  });
});
