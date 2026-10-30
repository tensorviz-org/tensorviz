import * as assert from 'node:assert/strict';
import { ComputationIndex } from '@tensorviz/graph-host/computationQuery';
import { inspectCapture } from '@tensorviz/graph-host/captureInspection';
import type { CaptureSnapshot } from '@tensorviz/graph-host/semanticCapture';
import { canonicalFixture } from '../../../tests/fixtures/semantic.cjs';

suite('canonical computation query', () => {
  let capture: CaptureSnapshot;
  setup(() => { capture = canonicalFixture().capture; });
  const query = (fields: object = {}) => new ComputationIndex(capture).query({ captureId: capture.id,
    sourceId: capture.source.id, anchors: [{ kind: 'operation', id: 'add2' }], ...fields });
  const entityIds = (page: ReturnType<typeof query>) => page.items.flatMap(item => item.kind === 'entity' ? [item.ref.id] : []);

  test('upstream and downstream follow recorded values rather than equal tensor shapes', () => {
    assert.deepEqual(entityIds(query({ direction: 'upstream' })), ['add2', 'a', 'add1', 'h']);
    const downstream = query({ anchors: [{ kind: 'value', id: 'a' }], direction: 'downstream' });
    assert.deepEqual(entityIds(downstream), ['a', 'add2', 'b']);
    assert.equal(downstream.traversal.completeWithinRecordedGraph, true);
    assert.equal(downstream.evidence.appliesToCurrentRuntime, 'not_established');
    assert.equal(downstream.evidence.planDigest, capture.planDigest);
  });

  test('depth boundaries retain each operand use and can be expanded without changing capture', () => {
    const page = query({ direction: 'upstream', depth: 0 });
    assert.deepEqual(entityIds(page), ['add2']);
    const boundaries = page.items.filter(item => item.kind === 'boundary');
    assert.equal(boundaries.length, 2);
    assert.deepEqual(boundaries.map(item => item.via.path), [['args', 0], ['args', 1]]);
    assert.ok(boundaries.every(item => item.reason === 'depth_limit' && item.ref.id === 'a'));
    assert.equal(page.traversal.completeWithinRecordedGraph, false);
    assert.deepEqual(entityIds(query({ direction: 'upstream', anchors: [boundaries[0].ref] })), ['a', 'add1', 'h']);
  });

  test('entity limits disclose excluded dependencies and omitted module seeds', () => {
    const page = query({ maxEntities: 1, depth: 20 });
    assert.deepEqual(entityIds(page), ['add2']);
    assert.ok(page.items.some(item => item.kind === 'boundary' && item.reason === 'entity_limit'));
    const module = query({ anchors: [{ kind: 'module', id: 'module:0' }], maxEntities: 1 });
    assert.equal(module.traversal.omittedSeeds, 1);
    assert.equal(module.traversal.completeWithinRecordedGraph, false);
  });

  test('module and definition anchors retain instance identity and select every recorded call', () => {
    capture.modules[0].paths = ['shared', 'also_shared'];
    capture.invocations.push({ ...structuredClone(capture.invocations[0]), id: 'call:1', parentId: 'call:0', sequence: 2, endSequence: 4 });
    capture.invocations[0].endSequence = 5;
    capture.operations[1].invocationId = 'call:1';
    capture.operations[1].sequence = 3;
    const page = query({ anchors: [{ kind: 'module', id: 'module:0' }], depth: 0 });
    assert.deepEqual(entityIds(page), ['add1', 'add2']);
    const module = page.items.find(item => item.kind === 'anchor');
    assert.ok(module?.kind === 'anchor');
    assert.deepEqual((module.record as any).paths, ['shared', 'also_shared']);
    assert.deepEqual(entityIds(query({ anchors: [{ kind: 'invocation', id: 'call:1' }], depth: 0 })), ['add2']);
    assert.deepEqual(entityIds(query({ anchors: [{ kind: 'definition', id: 'definition:0' }], depth: 0 })), ['add1', 'add2']);
  });

  test('nested ordered argument paths preserve repeated references without flattening containers', () => {
    capture.operations[1].arguments = { args: [], kwargs: { payload: { kind: 'dict', entries: [
      { key: { kind: 'str', value: 'features' }, value: { kind: 'tuple', items: [{ kind: 'value', id: 'a' }, { kind: 'value', id: 'a' }] } },
    ] } } };
    const uses = query().items.filter(item => item.kind === 'edge').filter(item => item.target.id === 'add2');
    assert.deepEqual(uses.map(item => item.path), [
      ['kwargs', 'payload', 'entries', 0, 'value', 'items', 0], ['kwargs', 'payload', 'entries', 0, 'value', 'items', 1],
    ]);
  });

  test('mutation effects and previous versions never become invented return tensors', () => {
    const op = capture.operations[1], after = capture.values[2];
    op.output = { kind: 'none' };
    op.effects = [{ kind: 'write', argumentPath: ['args', 0], beforeId: 'a', after: { kind: 'value', id: 'b' } }];
    after.producer = { operationId: 'add2', kind: 'effect', path: [0, 'after'] };
    after.previousVersion = 'a';
    const edges = query().items.filter(item => item.kind === 'edge');
    assert.ok(edges.some(edge => edge.relation === 'effect' && edge.target.id === 'b'));
    assert.ok(edges.some(edge => edge.relation === 'previous_version' && edge.source.id === 'a' && edge.target.id === 'b'));
    assert.ok(edges.some(edge => edge.relation === 'effect_input' && edge.target.id === 'add2'));
    assert.ok(!edges.some(edge => edge.relation === 'output' && edge.source.id === 'add2'));
  });

  test('observed state aliases stay on their value without guessing sharing from shapes', () => {
    capture.values[0].origin = 'state';
    capture.values[0].state = [{ kind: 'parameter', name: 'left.weight' }, { kind: 'parameter', name: 'right.weight' }];
    const page = query({ direction: 'upstream' });
    const state = page.items.find(item => item.kind === 'entity' && item.ref.id === 'h');
    assert.ok(state?.kind === 'entity');
    assert.deepEqual((state.record as any).state, capture.values[0].state);
    assert.ok(!page.items.some(item => item.kind === 'edge' && item.source.id === 'h' && item.target.id === 'a'));
  });

  test('a registered parameter name locates every recorded use and preserves tied names', () => {
    capture.values[0].origin = 'state';
    capture.values[0].state = [{ kind: 'parameter', name: 'left.weight' }, { kind: 'parameter', name: 'right.weight' }];
    const page = query({ anchors: [{ kind: 'parameter', id: 'right.weight' }], direction: 'downstream', depth: 20 });
    assert.deepEqual(entityIds(page), ['h', 'add1', 'a', 'add2', 'b']);
    const anchor = page.items.find(item => item.kind === 'anchor');
    assert.ok(anchor?.kind === 'anchor');
    assert.deepEqual(anchor.record, { kind: 'parameter', name: 'right.weight' });
    assert.deepEqual(anchor.inspect.path, ['values', 0, 'state', 1]);
    const inspected = inspectCapture(capture, { ...anchor.inspect, offset: 0, limit: 20 });
    assert.equal(inspected.items.find(item => item.key === 'name')?.value, 'right.weight');
    assert.deepEqual((page.items.find(item => item.kind === 'entity' && item.ref.id === 'h') as any).record.state,
      capture.values[0].state);
    assert.ok(page.items.filter(item => item.kind === 'edge').every(item =>
      ['operation', 'value'].includes(item.source.kind) && ['operation', 'value'].includes(item.target.kind)));
    assert.equal(page.evidence.appliesToCurrentRuntime, 'not_established');
  });

  test('state lookup keeps different objects separate even when shapes and storage match', () => {
    const value = capture.values[0];
    value.state = [{ kind: 'parameter', name: 'left.weight' }]; value.origin = 'state';
    capture.values.push({ ...structuredClone(value), id: 'separate', objectId: 'separate-object',
      state: [{ kind: 'parameter', name: 'right.weight' }] });
    capture.values.push({ ...structuredClone(value), id: 'buffer', objectId: 'buffer-object',
      state: [{ kind: 'buffer', name: 'running_mean' }] });
    assert.deepEqual(entityIds(query({ anchors: [{ kind: 'parameter', id: 'left.weight' }], depth: 0 })), ['h']);
    assert.deepEqual(entityIds(query({ anchors: [{ kind: 'parameter', id: 'right.weight' }], depth: 0 })), ['separate']);
    assert.deepEqual(entityIds(query({ anchors: [{ kind: 'buffer', id: 'running_mean' }], depth: 0 })), ['buffer']);
    assert.throws(() => query({ anchors: [{ kind: 'parameter', id: 'running_mean' }] }),
      (error: any) => error.code === 'STATE_NOT_OBSERVED');
  });

  test('state anchors select distinct logical versions without duplicating aliased seeds', () => {
    const value = capture.values[0];
    value.state = [{ kind: 'parameter', name: 'left.weight' }, { kind: 'parameter', name: 'right.weight' }];
    capture.values.push({ ...structuredClone(value), id: 'later', previousVersion: value.id,
      logicalVersion: 1, version: 1 });
    const anchors = [{ kind: 'parameter', id: 'left.weight' }, { kind: 'parameter', id: 'right.weight' }];
    const page = query({ anchors, depth: 0 });
    assert.deepEqual(entityIds(page), ['h', 'later']);
    assert.ok(page.items.some(item => item.kind === 'edge' && item.relation === 'previous_version'));
    const limited = query({ anchors, maxEntities: 1, depth: 0 });
    assert.equal(limited.traversal.omittedSeeds, 1);
    assert.equal(limited.traversal.completeWithinRecordedGraph, false);
    const items: unknown[] = [];
    let offset = 0;
    for (;;) {
      const part = query({ anchors, depth: 0, offset, limit: 1 });
      items.push(...part.items);
      if (part.nextOffset === null) break;
      assert.ok(part.nextOffset > offset); offset = part.nextOffset;
    }
    assert.deepEqual(items, page.items);
  });

  test('unobserved state is unknown even in a complete recorded path', () => {
    assert.throws(() => query({ anchors: [{ kind: 'parameter', id: 'unreached.weight' }] }),
      (error: any) => error.code === 'STATE_NOT_OBSERVED' && /does not establish absence/.test(error.message));
    const name = 'nested.'.repeat(35) + 'weight';
    capture.values[0].state = [{ kind: 'parameter', name }];
    assert.deepEqual(entityIds(query({ anchors: [{ kind: 'parameter', id: name }], depth: 0 })), ['h']);
    assert.throws(() => query({ anchors: [{ kind: 'parameter', id: 'a'.repeat(4097) }] }));
  });

  test('source bindings retain exact snapshot, digest and ambiguity instead of certifying source', () => {
    capture.sourceBindings = [{ id: 'binding', snapshotId: capture.source.id, kind: 'call', quality: 'ambiguous', anchors: [
      { file: 'model.py', digest: 'a'.repeat(64), startLine: 3, startColumn: 1, endLine: 3, endColumn: 12,
        scope: 'Net.forward', tokenDigest: 'b'.repeat(64) },
    ] }];
    capture.operations[1].sourceBindingIds = ['binding'];
    const source = query().items.find(item => item.kind === 'source');
    assert.ok(source?.kind === 'source');
    assert.deepEqual(source.record, capture.sourceBindings[0]);
    assert.deepEqual(source.inspect.path, ['sourceBindings', 0]);
  });

  test('partial and JAX coverage stay explicit even when the requested slice has no boundaries', () => {
    capture.backend = 'jaxpr'; capture.coverage.scope = 'traced_program'; capture.status = 'partial';
    capture.coverage.omissions = ['opaque_kernel'];
    const page = query({ depth: 20 });
    assert.equal(page.traversal.completeWithinRecordedGraph, true);
    assert.equal(page.evidence.status, 'partial');
    assert.equal(page.evidence.scope, 'traced_program');
    assert.deepEqual(page.evidence.omissions, ['opaque_kernel']);
    assert.equal(page.evidence.appliesToCurrentRuntime, 'not_established');
  });

  test('an operation exposes its call, instance and definition sources without inventing dataflow', () => {
    capture.sourceBindings = ['call', 'constructor', 'definition'].map((kind, i) => ({ id: `binding:${i}`,
      kind: kind as 'call' | 'constructor' | 'definition', snapshotId: capture.source.id, quality: 'exact', anchors: [
        { file: 'model.py', digest: 'a'.repeat(64), startLine: i + 1, startColumn: 0, endLine: i + 1, endColumn: 5,
          scope: 'Net', tokenDigest: 'b'.repeat(64) },
      ] }));
    capture.invocations[0].sourceBindingIds = ['binding:0'];
    capture.modules[0].constructorBindingIds = ['binding:1'];
    capture.definitions[0].sourceBindingIds = ['binding:2'];
    const page = query({ depth: 0 });
    assert.deepEqual(page.items.flatMap(item => item.kind === 'context' ? [item.ref] : []), [
      { kind: 'invocation', id: 'call:0' }, { kind: 'module', id: 'module:0' }, { kind: 'definition', id: 'definition:0' },
    ]);
    assert.deepEqual(page.items.flatMap(item => item.kind === 'source' ? [item.ref.id] : []), ['binding:0', 'binding:1', 'binding:2']);
    assert.ok(!page.items.some(item => item.kind === 'edge'), 'ownership is context, not an operation/value edge');
    capture.sourceBindings.pop();
    assert.throws(() => query(), /source binding has no retained record/);
  });

  test('an unobserved module produces its anchor with no fabricated invocations or operations', () => {
    capture.modules.push({ ...structuredClone(capture.modules[0]), id: 'unused', paths: ['conditional'] });
    const page = query({ anchors: [{ kind: 'module', id: 'unused' }] });
    assert.deepEqual(entityIds(page), []);
    assert.equal(page.items.filter(item => item.kind === 'anchor').length, 1);
    assert.ok(page.items.some(item => item.kind === 'context' && item.ref.kind === 'definition'));
    assert.deepEqual(page.evidence.coverageInspection.path, ['coverage']);
  });

  test('large records are omitted whole with a pointer to the existing exact capture reader', () => {
    capture.operations[1].arguments.args.push(...Array.from({ length: 5 }, () => ({ kind: 'str' as const, value: 'x'.repeat(4000) })));
    const page = query();
    const op = page.items.find(item => item.kind === 'entity' && item.ref.id === 'add2');
    assert.ok(op?.kind === 'entity');
    assert.equal(op.omitted, 'record_bytes');
    assert.equal(op.record, undefined);
    const inspected = inspectCapture(capture, { ...op.inspect, path: [...op.inspect.path, 'arguments', 'args'], offset: 2, limit: 1 });
    assert.equal(inspected.total, 7);
    assert.equal(inspected.items[0].key, 2);
    assert.ok(Buffer.byteLength(JSON.stringify(page)) < 48 * 1024);
  });

  test('pagination reconstructs a bounded slice without dropping relations or producing giant pages', () => {
    for (let i = 0; i < 140; i++) {
      const id = `extra:${i}`;
      capture.operations.push({ ...structuredClone(capture.operations[0]), id,
        arguments: { args: [{ kind: 'value', id: 'h' }], kwargs: {} }, output: { kind: 'none' },
        schemaSignature: 's'.repeat(1500) });
    }
    const index = new ComputationIndex(capture), items: unknown[] = [];
    let offset = 0;
    for (;;) {
      const page = index.query({ captureId: capture.id, sourceId: capture.source.id,
        anchors: [{ kind: 'module', id: 'module:0' }], maxEntities: 500, depth: 20, offset, limit: 100 });
      assert.ok(Buffer.byteLength(JSON.stringify(page)) < 48 * 1024);
      assert.ok(page.items.length);
      items.push(...page.items);
      if (page.nextOffset === null) { assert.equal(items.length, page.total); break; }
      assert.ok(page.nextOffset > offset); offset = page.nextOffset;
    }
    assert.equal(items.filter((item: any) => item.kind === 'entity').length, 145);
    assert.equal(new Set(items.filter((item: any) => item.kind === 'entity').map((item: any) => JSON.stringify(item.ref))).size, 145);
  });

  test('relation expansion has a separate cap with explicit omitted counts', () => {
    capture.operations[1].arguments.args = Array.from({ length: 6000 }, () => ({ kind: 'value', id: 'a' }));
    const page = query({ depth: 20 });
    assert.ok(page.traversal.omittedRelations > 0);
    assert.equal(page.traversal.completeWithinRecordedGraph, false);
    assert.ok(page.nextOffset !== null);
  });

  test('queries do not mutate retained facts and reject invalid or foreign anchors', () => {
    const before = JSON.stringify(capture);
    query({ anchors: [{ kind: 'module', id: 'module:0' }] });
    assert.equal(JSON.stringify(capture), before);
    for (const fields of [
      { captureId: 'foreign' }, { sourceId: 'foreign' }, { anchors: [] }, { anchors: [{ kind: 'value', id: 'missing' }] },
      { anchors: [{ kind: 'value', id: 'h' }, { kind: 'value', id: 'h' }] }, { depth: 21 }, { maxEntities: 0 },
      { limit: 101 }, { offset: -1 }, { unexpected: true },
    ]) assert.throws(() => query(fields));
  });
});
