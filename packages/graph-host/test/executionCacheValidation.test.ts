import * as assert from 'node:assert/strict';
import { servableEntry, type StoredRunMeta } from '@tensorviz/graph-host/executionCache';
import type { ExecutionSnapshot } from '@tensorviz/graph-host/executionPlanCore';
import type { WorkerEnvelope } from '@tensorviz/graph-host/traceWorkerCore';
import { canonicalFixture } from '../../../tests/fixtures/semantic.cjs';

function fixture() {
  const { capture, plan } = canonicalFixture();
  const snapshot: ExecutionSnapshot = { plan, applicabilityDigest: capture.planDigest, guards: {} };
  const diagnostics = {
    execution: { planDigest: capture.planDigest, planCurrent: true, resolvedPlan: structuredClone(plan) },
    semantic_capture: capture, semantic_capture_error: null,
    reuse: { storable: true, reasons: [], untrackedReads: 0, environment: [] },
  };
  const envelope: WorkerEnvelope = { protocol_version: 1, error: null, spans: {}, worker: {}, result: { diagnostics } };
  const meta: StoredRunMeta = { attemptId: capture.attemptId, captureId: capture.id, plan,
    planDigest: capture.planDigest, capturedAt: 0 };
  return { capture, plan, snapshot, diagnostics,
    entry: () => ({ text: JSON.stringify(envelope), meta: JSON.stringify(meta) }) };
}

/** Count full validations, not wall time, so this regression stays deterministic. */
function counted(check: (calls: () => number) => void): void {
  const semantic = require('@tensorviz/graph-host/semanticCapture') as typeof import('@tensorviz/graph-host/semanticCapture');
  const original = semantic.semanticCapture;
  let calls = 0;
  semantic.semanticCapture = (...args: Parameters<typeof original>) => { calls++; return original(...args); };
  try { check(() => calls); }
  finally { semantic.semanticCapture = original; }
}

suite('execution cache unchanged-version validation', () => {
  test('validates every unchanged hit once and returns an isolated fresh parse', () => {
    const f = fixture(), entry = f.entry(), original = { ...entry };
    counted(calls => {
      const first = servableEntry(entry, f.snapshot, 1);
      assert.ok(first.ok);
      assert.equal(calls(), 1);
      (first.envelope.result as { diagnostics: typeof f.diagnostics }).diagnostics.semantic_capture.status = 'partial';
      first.reused.plan.source.files[0].documentVersion = 99;
      const second = servableEntry(entry, f.snapshot, 1);
      assert.ok(second.ok);
      assert.equal(calls(), 2, 'each hit still validates the stored capture');
      assert.deepEqual(second.envelope, JSON.parse(entry.text));
      assert.equal(second.reused.plan.source.files[0].documentVersion, 1);
      assert.deepEqual(entry, original, 'caller mutation must not change retained text');
    });
  });

  test('unchanged closed-document versions take one validation', () => {
    const f = fixture();
    f.capture.source.files[0].documentVersion = null;
    f.plan.source.files[0].documentVersion = null;
    f.diagnostics.execution.resolvedPlan.source.files[0].documentVersion = null;
    counted(calls => {
      assert.ok(servableEntry(f.entry(), f.snapshot, 1).ok);
      assert.equal(calls(), 1);
    });
  });

  test('a version change in a later source file still revalidates', () => {
    const f = fixture();
    const second = { ...f.capture.source.files[0], path: 'helper.py' };
    f.capture.source.files.push(structuredClone(second));
    f.plan.source.files.push(structuredClone(second));
    f.diagnostics.execution.resolvedPlan.source.files.push(structuredClone(second));
    const snapshot = structuredClone(f.snapshot);
    snapshot.plan.source.files[1].documentVersion = 9;
    counted(calls => {
      const result = servableEntry(f.entry(), snapshot, 1);
      assert.ok(result.ok);
      assert.equal(calls(), 2);
      assert.deepEqual(result.reused.plan.source.files.map(file => file.documentVersion), [1, 9]);
    });
  });

  for (const documentVersion of [3, null]) {
    test(`revalidates all rewritten copies for document version ${documentVersion}`, () => {
      const f = fixture(), entry = f.entry(), snapshot = structuredClone(f.snapshot);
      snapshot.plan.source.files[0].documentVersion = documentVersion;
      counted(calls => {
        const result = servableEntry(entry, snapshot, 1);
        assert.ok(result.ok);
        assert.equal(calls(), 2);
        const diagnostics = (result.envelope.result as { diagnostics: typeof f.diagnostics }).diagnostics;
        assert.deepEqual([diagnostics.execution.resolvedPlan, result.reused.plan]
          .map(plan => plan.source.files[0].documentVersion), [documentVersion, documentVersion]);
        assert.equal(diagnostics.semantic_capture.source.files[0].documentVersion, documentVersion);
        assert.equal(f.entry().text, entry.text);
        assert.equal(f.entry().meta, entry.meta);
      });
    });
  }

  test('a differing envelope plan copy still takes the rewrite path', () => {
    const f = fixture();
    f.diagnostics.execution.resolvedPlan.source.files[0].documentVersion = 9;
    counted(calls => {
      const result = servableEntry(f.entry(), f.snapshot, 1);
      assert.ok(result.ok);
      assert.equal(calls(), 2);
      const diagnostics = (result.envelope.result as { diagnostics: typeof f.diagnostics }).diagnostics;
      assert.equal(diagnostics.execution.resolvedPlan.source.files[0].documentVersion, 1);
    });
  });

  test('does not skip validation for a corrupt same-version capture', () => {
    const f = fixture();
    f.capture.operations[0].arguments.args[0] = { kind: 'value', id: 'missing' };
    counted(calls => {
      assert.deepEqual(servableEntry(f.entry(), f.snapshot, 1), { ok: false, reason: 'capture' });
      assert.equal(calls(), 1);
    });
  });

  test('still rejects source changes other than document versions', () => {
    const f = fixture(), snapshot = structuredClone(f.snapshot);
    snapshot.plan.source.files[0].digest = 'b'.repeat(64);
    assert.deepEqual(servableEntry(f.entry(), snapshot, 1), { ok: false, reason: 'source_files' });
  });

  test('still rejects an invalid document version after rewriting', () => {
    const f = fixture(), snapshot = structuredClone(f.snapshot);
    (snapshot.plan.source.files[0] as { documentVersion: unknown }).documentVersion = 'invalid';
    counted(calls => {
      assert.deepEqual(servableEntry(f.entry(), snapshot, 1), { ok: false, reason: 'capture' });
      assert.equal(calls(), 2);
    });
  });

  test('still enforces the capture byte limit after a longer version is inserted', () => {
    const f = fixture();
    // Find the fixed point because the byte limit itself is part of the capture JSON.
    while (f.capture.limits.bytes !== Buffer.byteLength(JSON.stringify(f.capture))) {
      f.capture.limits.bytes = Buffer.byteLength(JSON.stringify(f.capture));
    }
    f.plan.captureLimits = structuredClone(f.capture.limits);
    f.diagnostics.execution.resolvedPlan.captureLimits = structuredClone(f.capture.limits);
    const entry = f.entry(), snapshot = structuredClone(f.snapshot);
    assert.ok(servableEntry(entry, f.snapshot, 1).ok);
    snapshot.plan.source.files[0].documentVersion = Number.MAX_SAFE_INTEGER;
    counted(calls => {
      assert.deepEqual(servableEntry(entry, snapshot, 1), { ok: false, reason: 'capture' });
      assert.equal(calls(), 2);
    });
  });
});
