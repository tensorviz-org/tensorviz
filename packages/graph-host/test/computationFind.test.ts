import * as assert from 'node:assert/strict';
import { ComputationIndex } from '@tensorviz/graph-host/computationQuery';
import { inspectCapture } from '@tensorviz/graph-host/captureInspection';
import type { CaptureSnapshot } from '@tensorviz/graph-host/semanticCapture';
import { canonicalFixture } from '../../../tests/fixtures/semantic.cjs';

suite('canonical computation name discovery', () => {
  let capture: CaptureSnapshot;
  setup(() => { capture = canonicalFixture().capture; });
  const find = (fields: object = {}) => new ComputationIndex(capture).find({
    captureId: capture.id, sourceId: capture.source.id, text: 'add', ...fields });

  test('recorded operation names lead to exact queryable anchors without accepting guessed IDs', () => {
    const page = find(), index = new ComputationIndex(capture);
    assert.deepEqual(page.items.map(item => item.ref), [
      { kind: 'operation', id: 'add1' }, { kind: 'operation', id: 'add2' },
    ]);
    assert.deepEqual(page.items[0].matchedFields, ['target']);
    const query = { captureId: capture.id, sourceId: capture.source.id,
      anchors: [page.items[1].ref], direction: 'upstream', depth: 4 };
    assert.deepEqual(index.query(query).items.flatMap(item => item.kind === 'entity' ? [item.ref.id] : []),
      ['add2', 'a', 'add1', 'h']);
    assert.throws(() => index.query({ ...query, anchors: [{ kind: 'operation', id: 'aten.add.Tensor' }] }),
      { code: 'UNKNOWN_ENTITY' });
    assert.equal(inspectCapture(capture, { ...page.items[1].inspect, offset: 0, limit: 50 })
      .items.find(item => item.key === 'id')?.value, 'add2');
  });

  test('aliases stay together while same-type and repeated-name instances remain separate candidates', () => {
    capture.modules[0].paths = ['encoder.shared', 'decoder.shared'];
    capture.modules.push({ ...structuredClone(capture.modules[0]), id: 'module:other', paths: ['encoder.shared'] });
    const page = find({ text: 'shared' });
    assert.deepEqual(page.items.map(item => item.ref.id), ['module:0', 'module:other']);
    assert.deepEqual(page.items[0].matchedFields, ['paths[0]', 'paths[1]']);
    assert.deepEqual(find({ text: 'model.Model', kinds: ['module'] }).items.map(item => item.ref.id),
      ['module:0', 'module:other']);
    assert.deepEqual(find({ text: 'model.Model', kinds: ['definition'] }).items.map(item => item.ref.id), ['definition:0']);
    assert.deepEqual((page.items[0].record as any).paths, ['encoder.shared', 'decoder.shared']);
  });

  test('matching is literal and case-sensitive, with explicit kind filters and no fuzzy fallback', () => {
    capture.modules[0].paths = ['layers[0].Add'];
    assert.equal(find({ text: 'Add' }).total, 1);
    assert.equal(find({ text: '[0]' }).items[0].ref.id, 'module:0');
    assert.equal(find({ text: '.*' }).total, 0);
    assert.equal(find({ text: 'ADD' }).total, 0);
    assert.equal(find({ text: 'add', kinds: ['module'] }).total, 0);
    assert.deepEqual(find({ kinds: ['operation', 'operation'] }).kinds, ['operation']);
  });

  test('only observed state names are discoverable and partial coverage never establishes absence', () => {
    capture.values[0].state = [{ kind: 'parameter', name: 'left.weight' }, { kind: 'parameter', name: 'right.weight' },
      { kind: 'buffer', name: 'weight_mean' }];
    capture.status = 'partial'; capture.coverage.omissions = ['opaque_kernel'];
    assert.deepEqual(find({ text: 'weight', kinds: ['parameter'] }).items.map(item => item.ref.id),
      ['left.weight', 'right.weight']);
    assert.deepEqual(find({ text: 'weight', kinds: ['buffer'] }).items.map(item => item.ref.id), ['weight_mean']);
    const missing = find({ text: 'unobserved' });
    assert.equal(missing.total, 0); assert.equal(missing.nextOffset, null);
    assert.equal(missing.evidence.status, 'partial');
    assert.equal(missing.evidence.appliesToCurrentRuntime, 'not_established');
    assert.equal(missing.evidence.planDigest, capture.planDigest);
    assert.match(missing.evidence.limitation, /No match does not establish absence/);
    assert.deepEqual(missing.evidence.coverageInspection.path, ['coverage']);
  });

  test('byte-limited pages reconstruct every matching entity without duplication', () => {
    for (let i = 0; i < 140; i++) capture.operations.push({ ...structuredClone(capture.operations[0]),
      id: `extra:${i}`, schemaSignature: 's'.repeat(1500) });
    const index = new ComputationIndex(capture), ids: string[] = [];
    let offset = 0, pages = 0;
    for (;;) {
      const page = index.find({ captureId: capture.id, sourceId: capture.source.id, text: 'add', offset, limit: 100 });
      assert.ok(Buffer.byteLength(JSON.stringify(page)) < 48 * 1024);
      assert.ok(page.items.length > 0 && page.items.length < 100, 'The byte cap must actually constrain the page.');
      ids.push(...page.items.map(item => item.ref.id)); pages++;
      if (page.nextOffset === null) { assert.equal(ids.length, page.total); break; }
      assert.ok(page.nextOffset > offset); offset = page.nextOffset;
    }
    assert.ok(pages > 2); assert.equal(ids.length, 142); assert.equal(new Set(ids).size, 142);
    assert.equal(find({ offset: 1000 }).nextOffset, null);
  });

  test('large aliased records retain exact inspection pointers and disclose bounded match metadata', () => {
    capture.modules[0].paths = Array.from({ length: 300 }, (_, i) => `long.module.alias.${i}`);
    const item = find({ text: 'alias' }).items[0];
    assert.equal(item.omitted, 'record_bytes'); assert.equal(item.record, undefined);
    assert.equal(item.matchedFields.length, 16); assert.equal(item.omittedMatchedFields, 284);
    const exact = inspectCapture(capture, { ...item.inspect, path: [...item.inspect.path, 'paths'], offset: 299, limit: 1 });
    assert.equal(exact.items[0].value, 'long.module.alias.299');
  });

  test('foreign identities and unbounded or ambiguous request syntax are rejected without mutation', () => {
    const before = structuredClone(capture);
    for (const fields of [{ captureId: 'foreign' }, { sourceId: 'foreign' }])
      assert.throws(() => find(fields), { code: 'SOURCE_MISMATCH' });
    for (const fields of [{ text: '' }, { text: ' \n' }, { text: 'x'.repeat(257) }, { limit: 101 },
      { offset: -1 }, { kinds: [] }, { kinds: ['value'] }, { regex: true }])
      assert.throws(() => find(fields), { code: 'INVALID_QUERY' });
    find(); assert.deepEqual(capture, before);
  });
});
