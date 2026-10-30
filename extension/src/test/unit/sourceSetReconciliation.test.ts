import * as assert from 'node:assert/strict';
import { canonicalFixture } from '../../../../tests/fixtures/semantic.cjs';
import { reconcileAppliedSourceSet, type AppliedSourceSetEvidence } from '../../sourceSetReconciliation';
import { AppliedSourceReceipts } from '../../appliedSourceReceipts';
import { AppliedSourceInspector } from '../../appliedSourceInspection';
import { isAppliedSourceObservation } from '@tensorviz/contracts/protocol';

const target = { workspaceFolder: '/work', file: 'model.py', cls: 'Model', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 };
function fixture() {
  const base = canonicalFixture().capture;
  base.source.files = ['model.py', 'config.py', 'helper.py'].map(path => ({ ...base.source.files[0], path, digest: 'a'.repeat(64) }));
  const files = base.source.files.slice(0, 2).map(item => ({ file: item.path, beforeDigest: item.digest, afterDigest: 'b'.repeat(64) }));
  const input: AppliedSourceSetEvidence = { version: 2, base, files };
  const after = structuredClone(base); after.id = 'after'; after.attemptId = 'after-attempt'; after.source.id = 'after-source';
  after.source.files.slice(0, 2).forEach(item => { item.digest = 'b'.repeat(64); });
  return { input, after };
}

suite('applied source-set capture correspondence', () => {
  test('compares exactly the changed set, preserves conditional facts and never claims requested semantics', () => {
    const { input, after } = fixture();
    let result = reconcileAppliedSourceSet(input, after);
    assert.equal(result.reason, 'reviewed_source'); assert.equal(result.semanticPostconditions, 'not_checked');
    assert.ok(result.comparison); assert.equal(result.inputs, 'same');
    after.values[0].shape = [99]; result = reconcileAppliedSourceSet(input, after);
    assert.ok(result.comparison!.observedFieldCount > 0);
    after.status = 'partial'; after.inputDigest = null;
    result = reconcileAppliedSourceSet(input, after);
    assert.equal(result.comparison!.coverageComplete, false); assert.notEqual(result.comparison!.equivalent, true);
    assert.equal(result.inputs, 'unavailable');
  });
  test('every reviewed file and unchanged tracked dependency owns its exact digest', () => {
    const controls: Array<[string, (after: ReturnType<typeof fixture>['after']) => void]> = [
      ['source_changed', after => { after.source.files[1].digest = 'c'.repeat(64); }],
      ['source_changed', after => { after.source.files[1].digest = 'a'.repeat(64); }],
      ['source_reverted', after => { after.source.files.slice(0, 2).forEach(file => { file.digest = 'a'.repeat(64); }); }],
      ['source_unavailable', after => { after.source.files.splice(1, 1); }],
      ['dependencies_changed', after => { after.source.files[2].digest = 'c'.repeat(64); }],
      ['dependencies_changed', after => { after.source.configDigest = 'c'.repeat(64); }],
      ['environment_changed', after => { after.source.environmentDigest = 'c'.repeat(64); }],
      ['different_model', after => { after.source.workspaceSessionId = 'other'; }],
      ['different_model', after => { after.source.modelId = 'other'; }],
      ['capture_not_new', after => { after.attemptId = fixture().input.base.attemptId; }],
    ];
    for (const [reason, change] of controls) {
      const { input, after } = fixture(); change(after); const result = reconcileAppliedSourceSet(input, after);
      assert.equal(result.reason, reason); assert.equal(result.comparison, null); assert.equal(result.inputs, 'unavailable');
    }
  });
  test('confirmed source-set receipts are immutable, bounded, revoked and never revived by late acknowledgments', () => {
    const { input } = fixture(), store = new AppliedSourceReceipts();
    const id = store.reserve('owner', target, input)!; assert.ok(id); assert.equal(store.summary('owner'), null);
    input.files[1].afterDigest = 'c'.repeat(64); store.confirm('owner', id);
    const first = store.summary('owner')!; assert.equal(first.files![1].afterDigest, 'b'.repeat(64));
    first.files![1].file = 'poison.py'; assert.equal(store.summary('owner')!.files![1].file, 'config.py');
    store.forget('owner'); store.confirm('owner', id); assert.equal(store.summary('owner'), null);
    const bad = fixture().input; bad.files[1].beforeDigest = 'd'.repeat(64);
    assert.equal(store.reserve('bad', target, bad), undefined);
    const small = new AppliedSourceReceipts({ records: 1, evidenceBytes: 1, totalBytes: 1, ttlMs: 100 });
    const limited = small.reserve('owner', target, fixture().input)!; small.confirm('owner', limited);
    assert.equal(small.summary('owner')!.unavailable, 'too_large'); assert.equal(small.read('owner')!.evidence, undefined);
  });
  test('shared current-source checks gate comparison and recheck revocation, setup and resource races', async () => {
    for (const race of ['none', 'receipt', 'setup', 'source', 'resources', 'capture']) {
      const { input, after } = fixture(), store = new AppliedSourceReceipts(undefined, () => 10);
      const id = store.reserve('owner', target, input)!; store.confirm('owner', id);
      let checks = 0, current = true, resources = true, live = { capture: after, startedAt: 20 };
      const inspector = new AppliedSourceInspector(store, { origin: () => 'http://localhost:9000', token: () => 'private', trusted: () => true,
        live: () => live, readSource: async () => ({ source: 'reviewed_saved', current: () => current }),
        captureGuard: async () => () => resources, fetchImpl: async () => { throw new Error('Source-set comparison must not contact a second engine.'); },
        checkSetup: async () => {
          if (++checks === 2) {
            if (race === 'receipt') store.forget('owner'); if (race === 'source') current = false;
            if (race === 'resources') resources = false; if (race === 'capture') live = { capture: { ...after, id: 'newer' }, startedAt: 30 };
            if (race === 'setup') return false;
          }
          return true;
        } });
      const result = await inspector.inspect('owner'); assert.ok(isAppliedSourceObservation(result));
      assert.equal(result.captureFreshness, race === 'none' ? 'current' : 'stale');
      assert.equal(result.sourceSetComparison?.reason, race === 'none' ? 'reviewed_source' : undefined);
      assert.equal(result.reconciliation, null);
    }
  });
});
