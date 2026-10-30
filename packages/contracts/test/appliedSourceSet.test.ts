import * as assert from 'node:assert/strict';
import { isAppliedSourceFiles, isAppliedSourceObservation, type AppliedSourceObservation } from '@tensorviz/contracts/protocol';

const file = { file: 'model.py', beforeDigest: 'a'.repeat(64), afterDigest: 'b'.repeat(64) };
const observation = (): AppliedSourceObservation => ({ receipt: { ...file, appliedAt: 1,
  files: [file, { ...file, file: 'blocks/config.py' }], unavailable: null }, checkedAt: 2,
  source: 'reviewed_saved', captureFreshness: 'current', reconciliation: null, unavailable: null,
  sourceSetComparison: { baseCaptureId: 'before', captureId: 'after', sourceId: 'source', reason: 'reviewed_source',
    dependencyScope: 'partial', inputs: 'same', state: 'different', semanticPostconditions: 'not_checked',
    comparison: { equivalent: null, coverageComplete: false, changedFieldCount: 0, observedFieldCount: 2,
      unresolvedFieldCount: 1, unmatchedRecordCount: 0, uncertainMatchCount: 0,
      connectionChangeCount: 0, unresolvedConnectionCount: 1 } } });

suite('applied source-set protocol evidence', () => {
  test('complete bounded source inventory refuses ambiguous paths, duplicate or unchanged files and private fields', () => {
    assert.ok(isAppliedSourceFiles([file]));
    assert.ok(isAppliedSourceFiles(Array.from({ length: 16 }, (_, i) => ({ ...file, file: `m${i}.py` }))));
    for (const bad of [[], [file, file], Array.from({ length: 17 }, (_, i) => ({ ...file, file: `m${i}.py` })),
      [{ ...file, beforeDigest: file.afterDigest }], [{ ...file, source: 'private source' }],
      ...['../model.py', '/model.py', 'a//b.py', './model.py', 'a/../b.py', 'C:/m.py', 'a\\b.py', 'a\nb.py', 'a\0b.py', 'a\x7fb.py', 'weights.bin']
        .map(name => [{ ...file, file: name }])]) assert.equal(isAppliedSourceFiles(bad), false, JSON.stringify(bad));
  });
  test('accepts separate source-set facts while refusing invented recipe success, incomplete equality and conflicting receipts', () => {
    assert.ok(isAppliedSourceObservation(observation()));
    const mutations: Array<(value: any) => void> = [
      v => { v.receipt.file = 'different.py'; }, v => { v.receipt.files = undefined; },
      v => { v.sourceSetComparison.semanticPostconditions = 'verified'; },
      v => { v.sourceSetComparison.comparison.equivalent = true; },
      v => { v.sourceSetComparison.comparison.equivalent = false; },
      v => { v.sourceSetComparison.comparison.privateSource = 'private'; },
      v => { v.sourceSetComparison.comparison.changedFieldCount = -1; },
      v => { v.sourceSetComparison.captureId = 'before'; }, v => { v.captureFreshness = 'stale'; },
      v => { v.source = 'reviewed_unsaved'; }, v => { v.unavailable = 'check_failed'; },
      v => { v.reconciliation = {}; }, v => { v.receipt.unavailable = 'evicted'; },
      v => { v.sourceSetComparison.reason = 'dependencies_changed'; },
    ];
    for (const mutate of mutations) { const value = observation(); mutate(value); assert.equal(isAppliedSourceObservation(value), false); }
    const refused = observation(); Object.assign(refused.sourceSetComparison!, { reason: 'dependencies_changed',
      comparison: null, inputs: 'unavailable', state: 'unavailable' });
    assert.ok(isAppliedSourceObservation(refused));
    const missing = observation(); delete missing.sourceSetComparison;
    missing.unavailable = 'source_not_saved'; missing.source = 'reviewed_unsaved'; missing.captureFreshness = 'stale';
    assert.ok(isAppliedSourceObservation(missing));
  });
});
