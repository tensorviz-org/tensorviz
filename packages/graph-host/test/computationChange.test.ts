import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { checkOperationShape } from '@tensorviz/graph-host/computationShapes';
import { analyzeComputationChange } from '@tensorviz/graph-host/computationChange';
import type { CaptureSnapshot } from '@tensorviz/graph-host/semanticCapture';
import { semanticCapture } from '@tensorviz/graph-host/semanticCapture';
import { canonicalFixture } from '../../../tests/fixtures/semantic.cjs';

const oracles = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../../../tests/fixtures/contracts/computation-shape-oracles.json'), 'utf8'));
const width = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../../../tests/fixtures/contracts/computation-width-capture.json'), 'utf8'));
const layerNorm = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../../../tests/fixtures/contracts/computation-layer-norm-capture.json'), 'utf8'));
suite('actual PyTorch dimension oracles', () => {
  for (const item of oracles.cases) test(item.id, () => {
    const op = { ...canonicalFixture().capture.operations[0], target: item.target, arguments: item.arguments };
    if (item.target === 'aten.native_layer_norm.default') {
      op.output = { kind: 'tuple', items: [0, 1, 2].map(i => ({ kind: 'value', id: `output:${i}` })) };
    }
    const check = checkOperationShape(op, id => item.inputs[id] ?? null);
    assert.equal(check.status, item.expected.status, JSON.stringify({ item, check }));
    assert.deepEqual(check.outputShape, item.expected.shape, JSON.stringify({ item, check }));
    if (item.expected.shapes) assert.deepEqual(check.outputShapes,
      item.expected.shapes.map((shape: number[], i: number) => ({ valueId: `output:${i}`, shape })));
  });
});

suite('hypothetical computation changes', () => {
  let capture: CaptureSnapshot;
  setup(() => { capture = canonicalFixture().capture; });
  const analyze = (options: object = {}) => analyzeComputationChange(capture, { captureId: capture.id, sourceId: capture.source.id,
    changes: [{ valueId: 'h', shape: [2, 8] }], depth: 20, limit: 100, ...options });

  test('real dispatch capture predicts the independent valid and invalid width-change forwards', () => {
    const recorded = width.capture as CaptureSnapshot, { plan } = canonicalFixture(recorded.attemptId);
    plan.source = recorded.source; plan.captureLimits = recorded.limits;
    assert.ok(semanticCapture(recorded, recorded.attemptId, plan, recorded.planDigest));
    const input = recorded.values.find(value => value.origin === 'input')!;
    const weight = recorded.values.find(value => value.state.some(state => state.name === 'weight'))!;
    assert.ok(recorded.output?.kind === 'value');
    for (const candidate of width.candidates) {
      const result = analyzeComputationChange(recorded, { captureId: recorded.id, sourceId: recorded.source.id, depth: 20, limit: 100,
        changes: [{ valueId: input.id, shape: [2, candidate.width] }, { valueId: weight.id, shape: [candidate.width, candidate.width] }] });
      assert.deepEqual(result.analysis.values.find(value => value.id === (recorded.output as any).id)?.derivedShape, candidate.shape);
      assert.equal(result.analysis.summary.incompatible, candidate.status === 'passed' ? 0 : 1);
      assert.equal(result.analysis.summary.unknown, candidate.status === 'passed' ? 0 : 1);
    }
    const inconsistent = analyzeComputationChange(recorded, { captureId: recorded.id, sourceId: recorded.source.id,
      changes: [{ valueId: input.id, shape: [2, 15] }], depth: 20, limit: 100 });
    assert.equal(inconsistent.analysis.checks[0].reason, 'contracting_dimension_mismatch');
  });

  test('LayerNorm propagates all three outputs and invalidates downstream shapes on a failed constraint', () => {
    const recorded = layerNorm.capture as CaptureSnapshot, { plan } = canonicalFixture(recorded.attemptId);
    plan.source = recorded.source; plan.captureLimits = recorded.limits;
    assert.ok(semanticCapture(recorded, recorded.attemptId, plan, recorded.planDigest));
    const input = recorded.values.find(value => value.origin === 'input')!;
    assert.equal(recorded.output?.kind, 'tuple');
    const ids = (recorded.output as any).items.map((item: any) => item.id);
    for (const candidate of layerNorm.candidates) {
      const result = analyzeComputationChange(recorded, { captureId: recorded.id, sourceId: recorded.source.id, depth: 20, limit: 100,
        changes: [{ valueId: input.id, shape: candidate.inputShape }] });
      const shapes = ids.map((id: string) => result.analysis.values.find(value => value.id === id)?.derivedShape);
      assert.deepEqual(shapes, candidate.shapes ?? [null, null]);
      assert.equal(result.analysis.summary.incompatible, candidate.status === 'passed' ? 0 : 1);
      assert.equal(result.analysis.summary.unknown, candidate.status === 'passed' ? 0 : 2);
    }
  });

  test('a conflicting or unsupported LayerNorm producer clears every output including a second seed', () => {
    for (const mode of ['conflict', 'unsupported']) {
      const recorded = structuredClone(layerNorm.capture) as CaptureSnapshot;
      const norm = recorded.operations.find(op => op.target === 'aten.native_layer_norm.default')!;
      const outputs = (norm.output as any).items.map((item: any) => item.id);
      const input = recorded.values.find(value => value.origin === 'input')!;
      if (mode === 'unsupported') norm.target = 'custom.opaque.default';
      const result = analyzeComputationChange(recorded, { captureId: recorded.id, sourceId: recorded.source.id, depth: 20, limit: 100,
        changes: [{ valueId: input.id, shape: [3, 4] }, { valueId: outputs[1], shape: [7, 1] }] });
      for (const id of outputs) assert.equal(result.analysis.values.find(value => value.id === id)?.derivedShape, null);
      assert.ok(result.analysis.checks.filter(check => check.operationId !== norm.id).every(check => check.status === 'unknown'));
      if (mode === 'conflict') assert.equal(result.analysis.checks.find(check => check.operationId === norm.id)?.reason,
        'assumed_output_conflicts_with_derived_shape');
    }
  });

  test('LayerNorm requires exact three-value outputs and complete concrete argument shapes', () => {
    const norm = layerNorm.capture.operations.find((op: any) => op.target === 'aten.native_layer_norm.default');
    const values = new Map<string, number[]>(layerNorm.capture.values.map((value: any) => [value.id, value.shape]));
    for (const output of [{ kind: 'value', id: 'one' }, { kind: 'tuple', items: norm.output.items.slice(0, 2) },
      { kind: 'tuple', items: [norm.output.items[0], norm.output.items[0], norm.output.items[2]] }]) {
      assert.equal(checkOperationShape({ ...norm, output }, id => values.get(id) ?? null).status, 'unknown');
    }
    assert.equal(checkOperationShape(norm, () => null).status, 'unknown');
    const args = structuredClone(norm.arguments); args.args[1] = { kind: 'symbolic', type: 'int', expression: 'width' };
    assert.equal(checkOperationShape({ ...norm, arguments: args }, id => values.get(id) ?? null).status, 'unknown');
  });

  test('assumptions and derived shapes are separate from immutable observations and current runtime', () => {
    const before = JSON.stringify(capture), result = analyze();
    assert.equal(result.analysis.kind, 'hypothetical_shape_change');
    assert.equal(result.analysis.scope, 'selected_recorded_path');
    assert.deepEqual(result.analysis.summary, { compatible: 2, incompatible: 0, unknown: 0 });
    assert.deepEqual(result.analysis.values.find(value => value.id === 'b'), { id: 'b', basis: 'derived', derivedShape: [2, 8] });
    assert.equal(result.evidence.appliesToCurrentRuntime, 'not_established');
    assert.equal(JSON.stringify(capture), before);
  });

  test('a hidden-width reshape contradiction prevents downstream use of old captured shapes', () => {
    capture.values[0].shape = [2, 12]; capture.values[1].shape = capture.values[2].shape = [2, 3, 4];
    capture.operations[0].target = 'aten.view.default';
    capture.operations[0].arguments.args[1] = { kind: 'list', items: [2, 3, -1].map(value => ({ kind: 'int', value: String(value) })) };
    const result = analyze({ changes: [{ valueId: 'h', shape: [2, 16] }] });
    assert.equal(result.analysis.checks[0].reason, 'element_count_mismatch');
    assert.equal(result.analysis.checks[1].status, 'unknown');
    assert.equal(result.analysis.values.find(value => value.id === 'b')?.derivedShape, null);
  });

  test('an unchanged residual operand can violate the changed path dimensions', () => {
    capture.values.push({ ...capture.values[0], id: 'skip', shape: [4], objectId: 'skip-object', storageId: null });
    capture.operations[1].arguments.args[1] = { kind: 'value', id: 'skip' };
    const result = analyze();
    assert.equal(result.analysis.checks.find(check => check.operationId === 'add2')?.reason, 'incompatible_broadcast_dimensions');
  });

  test('opaque, failed, mutating and foreign-backend operations keep downstream results unknown', () => {
    for (const mode of ['opaque', 'failed', 'mutation', 'jax']) {
      capture = canonicalFixture().capture;
      if (mode === 'opaque') capture.operations[0].target = 'custom.opaque.default';
      if (mode === 'failed') capture.operations[0].status = 'failed';
      if (mode === 'mutation') capture.operations[0].effects = [{ kind: 'write', argumentPath: ['args', 0], beforeId: 'h', after: { kind: 'value', id: 'a' } }];
      if (mode === 'jax') capture.backend = 'jaxpr';
      const result = analyze();
      assert.deepEqual(result.analysis.summary, { compatible: 0, incompatible: 0, unknown: 2 }, mode);
      assert.equal(result.analysis.values.find(value => value.id === 'b')?.derivedShape, null, mode);
    }
  });

  test('a second assumed output cannot hide an inconsistent producer constraint', () => {
    const result = analyze({ changes: [{ valueId: 'h', shape: [2, 8] }, { valueId: 'a', shape: [2, 9] }] });
    assert.equal(result.analysis.checks.find(check => check.operationId === 'add1')?.reason, 'assumed_output_conflicts_with_derived_shape');
    assert.equal(result.analysis.checks.find(check => check.operationId === 'add2')?.status, 'unknown');
  });

  test('potential aliasing prevents an outside operand from silently retaining an old shape', () => {
    capture.values.push({ ...capture.values[0], id: 'alias', aliasOf: 'h' });
    capture.operations[1].arguments.args[1] = { kind: 'value', id: 'alias' };
    const result = analyze();
    assert.ok(result.analysis.possibleAliases.includes('alias'));
    assert.equal(result.analysis.checks.find(check => check.operationId === 'add2')?.status, 'unknown');
  });

  test('depth, partial coverage and page limits remain visible while analysis entries follow their page', () => {
    capture.status = 'partial'; capture.coverage.omissions = ['operation_limit'];
    const shallow = analyze({ depth: 0 });
    assert.equal(shallow.traversal.completeWithinRecordedGraph, false);
    assert.deepEqual(shallow.analysis.summary, { compatible: 0, incompatible: 0, unknown: 0 });
    const first = analyze({ limit: 1 }), second = analyze({ limit: 1, offset: first.nextOffset });
    assert.equal(first.analysis.values.length, 1); assert.equal(first.analysis.checks.length, 0);
    assert.equal(second.analysis.checks.length, 1); assert.equal(second.analysis.values.length, 0);
    assert.equal(second.evidence.status, 'partial'); assert.deepEqual(second.evidence.omissions, ['operation_limit']);
    assert.deepEqual(second.analysis.summary, { compatible: 2, incompatible: 0, unknown: 0 });
    assert.ok(Buffer.byteLength(JSON.stringify(second.analysis)) < 16 * 1024);
  });

  test('oversized counts remain exact and unsupported shape/argument metadata never becomes a guess', () => {
    const op = capture.operations[0]; op.target = 'aten.reshape.default';
    op.arguments.args[1] = { kind: 'list', items: [Number.MAX_SAFE_INTEGER, 2].map(value => ({ kind: 'int', value: String(value) })) };
    const check = checkOperationShape(op, () => [Number.MAX_SAFE_INTEGER, 2]);
    assert.equal(check.status, 'compatible');
    op.arguments.args[1] = { kind: 'list', items: [{ kind: 'int', value: '-1' }] };
    assert.equal(checkOperationShape(op, () => [Number.MAX_SAFE_INTEGER, 2]).reason, 'shape_outside_supported_range');
    op.arguments.args[1] = { kind: 'symbolic', type: 'int', expression: 'width' };
    assert.equal(checkOperationShape(op, () => [2, 4]).status, 'unknown');
    assert.equal(checkOperationShape(op, () => null).status, 'unknown');
  });

  test('invalid shapes, duplicate assumptions and foreign identities are rejected', () => {
    for (const options of [{ sourceId: 'foreign' }, { changes: [] }, { changes: [{ valueId: 'missing', shape: [2] }] },
      { changes: [{ valueId: 'h', shape: [-1] }] }, { changes: [{ valueId: 'h', shape: [Number.MAX_SAFE_INTEGER + 1] }] },
      { changes: [{ valueId: 'h', shape: [2] }, { valueId: 'h', shape: [3] }] }, { depth: 21 }, { limit: 101 }]) {
      assert.throws(() => analyze(options));
    }
  });

  test('large explicit assumptions leave a bounded response with recoverable omission counts', () => {
    const shape = Array(32).fill(Number.MAX_SAFE_INTEGER), changes: Array<{ valueId: string; shape: number[] }> = [];
    for (let i = 0; i < 16; i++) {
      const id = String(i).padStart(128, 'x'); changes.push({ valueId: id, shape });
      capture.values.push({ ...capture.values[0], id, objectId: `object:${i}`, storageId: null });
      capture.values.push({ ...capture.values[0], id: String(i).padStart(128, 'z'), objectId: `object:${i}`, storageId: null });
    }
    const result = analyze({ changes });
    assert.ok(Buffer.byteLength(JSON.stringify(result.analysis)) < 16 * 1024);
    assert.ok(result.analysis.omittedPageEntries > 0);
    const smaller = analyze({ changes, limit: 1, offset: 15 });
    assert.equal(smaller.analysis.omittedPageEntries, 0);
    assert.equal(smaller.analysis.values[0].id, changes[15].valueId);
    assert.equal(smaller.analysis.values[0].basis, 'assumed');
  });
});
