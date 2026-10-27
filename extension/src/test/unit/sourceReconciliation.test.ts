import * as assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { reconcileAppliedCapture, type AppliedSourceEvidence } from '../../sourceReconciliation';
import type { CaptureSnapshot } from '@tensorviz/graph-host/semanticCapture';

const fixture = () => JSON.parse(readFileSync(join(__dirname, '../../../../tests/fixtures/contracts/source-reconciliation.json'), 'utf8'));
const input = () => {
  const { current, ...evidence } = fixture().request as AppliedSourceEvidence & { current: CaptureSnapshot };
  return { current, evidence };
};
const options = (fetchImpl: typeof fetch, signal = new AbortController().signal) =>
  ({ origin: 'http://localhost:9000', token: 'HOST_ONLY', signal, fetchImpl });
const run = (fetchImpl: typeof fetch, signal?: AbortSignal) => {
  const { current, evidence } = input(); return reconcileAppliedCapture(evidence, current, options(fetchImpl, signal));
};

suite('applied-source reconciliation boundary', () => {
  test('real Python live-capture fixture shares provenance, scalar witnesses and tracked-source scope', async () => {
    const result = await run(async (url, init) => {
      assert.equal(url, 'http://localhost:9000/api/ide/capture/reconcile');
      assert.equal(init?.redirect, 'error');
      assert.equal((init?.headers as Record<string, string>)['X-tensorViz-Token'], 'HOST_ONLY');
      assert.deepEqual(JSON.parse(init!.body as string), fixture().request);
      return new Response(JSON.stringify(fixture().result));
    });
    assert.deepEqual(result, fixture().result);
    assert.equal(result.dependencyScope, 'partial');
    assert.equal(result.postconditions?.executionId, 'live-after');
  });

  for (const change of ['base', 'capture', 'source', 'digest', 'dependency_scope', 'status', 'reason', 'inputs', 'state',
    'missing', 'execution', 'witness_capture', 'witness_count', 'witness_binding', 'witness_recipe']) {
    test(`refuses substituted ${change} evidence`, async () => {
      const value = fixture().result;
      if (change === 'base') value.baseCaptureId = 'other';
      if (change === 'capture') value.captureId = 'other';
      if (change === 'source') value.sourceId = 'other';
      if (change === 'digest') value.reviewedDigest = 'a'.repeat(64);
      if (change === 'dependency_scope') value.dependencyScope = 'complete';
      if (change === 'status') value.status = 'unavailable';
      if (change === 'reason') value.reason = 'source_reverted';
      if (change === 'inputs') value.inputs = 'different';
      if (change === 'state') value.state = 'different';
      if (change === 'missing') value.postconditions = null;
      if (change === 'execution') value.postconditions.executionId = 'other';
      if (change === 'witness_capture') value.postconditions.captureId = 'other';
      if (change === 'witness_count') value.postconditions.observedOperations++;
      if (change === 'witness_binding') value.postconditions.bindingId = 'other';
      if (change === 'witness_recipe') value.postconditions.recipe = 'constructor_scalar_v1';
      await assert.rejects(run(async () => new Response(JSON.stringify(value))), /mismatched/);
    });
  }

  for (const change of ['model', 'dependency', 'environment', 'same_capture', 'same_attempt', 'reverted', 'changed']) {
    test(`${change} source returns an unavailable observation, never a live semantic pass`, async () => {
      const { current, evidence } = input(), result = fixture().result;
      let reason: string;
      if (change === 'model') { current.source.modelId = 'other'; reason = 'different_model'; }
      else if (change === 'dependency') { current.source.files.push({ path: 'new.py', digest: 'a'.repeat(64), documentVersion: null, role: 'source' }); reason = 'dependencies_changed'; }
      else if (change === 'environment') { current.source.environmentDigest = 'a'.repeat(64); reason = 'environment_changed'; }
      else if (change === 'same_capture') { current.id = evidence.base.id; reason = 'capture_not_new'; }
      else if (change === 'same_attempt') { current.attemptId = evidence.base.attemptId; reason = 'capture_not_new'; }
      else { current.source.files[0].digest = change === 'reverted' ? evidence.base.source.files[0].digest : 'a'.repeat(64); reason = `source_${change}`; }
      Object.assign(result, { captureId: current.id, status: 'unavailable', reason, inputs: 'unavailable', state: 'unavailable', postconditions: null });
      const actual = await reconcileAppliedCapture(evidence, current, options(async () => new Response(JSON.stringify(result))));
      assert.equal(actual.reason, reason); assert.equal(actual.postconditions, null);
    });
  }

  test('checks a real mismatch independently from exact source identity', async () => {
    const { current, evidence } = input(), result = fixture().result;
    (current.operations[0].arguments.args[1] as { value: string }).value = '99';
    Object.assign(result.postconditions, { status: 'failed', reason: 'observed_literal_mismatch' });
    const actual = await reconcileAppliedCapture(evidence, current, options(async () => new Response(JSON.stringify(result))));
    assert.equal(actual.status, 'checked'); assert.equal(actual.postconditions?.status, 'failed');
  });

  test('caller mutation cannot redirect an in-flight source comparison', async () => {
    const { current, evidence } = input();
    const result = await reconcileAppliedCapture(evidence, current, options(async () => {
      evidence.reviewedSource = 'changed'; current.source.modelId = 'other';
      return new Response(JSON.stringify(fixture().result));
    }));
    assert.deepEqual(result, fixture().result);
  });

  test('cancels oversized streams and hides malformed or private backend responses', async () => {
    let cancelled = false;
    const stream = new ReadableStream<Uint8Array>({ start(controller) { controller.enqueue(new Uint8Array(65537)); }, cancel() { cancelled = true; } });
    await assert.rejects(run(async () => new Response(stream)), /response limit/);
    assert.equal(cancelled, true);
    for (const status of [200, 400, 413, 500]) {
      await assert.rejects(run(async () => new Response('PRIVATE_SOURCE', { status })), error => {
        assert.ok(error instanceof Error && !error.message.includes('PRIVATE_SOURCE')); return true;
      });
    }
  });

  test('cancellation stops a stalled body and cannot return late evidence', async () => {
    const controller = new AbortController(); let cancelled = false;
    const stream = new ReadableStream<Uint8Array>({ cancel() { cancelled = true; } });
    const promise = run(async () => {
      setImmediate(() => controller.abort()); return new Response(stream);
    }, controller.signal);
    await assert.rejects(promise, /cancelled/);
    assert.equal(cancelled, true);
  });

  test('already cancelled work never sends original source to the service', async () => {
    const controller = new AbortController(); controller.abort(); let calls = 0;
    await assert.rejects(run(async () => { calls++; return new Response(); }, controller.signal));
    assert.equal(calls, 0);
  });
});
