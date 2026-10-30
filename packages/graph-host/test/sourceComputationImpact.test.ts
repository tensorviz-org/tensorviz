import * as assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as path from 'node:path';
import { sourceContextPage, sourceCaptureContext } from '@tensorviz/graph-host/sourceContext';

const oracle = JSON.parse(readFileSync(path.join(__dirname, '../../../tests/fixtures/contracts/source-change-oracle.json'), 'utf8'));
const variants = JSON.parse(readFileSync(path.join(__dirname, '../../../tests/fixtures/contracts/source-combined-oracles.json'), 'utf8'));
const context = (fixture = oracle, item = fixture.queries[1]) => sourceCaptureContext(
  sourceContextPage(item.page, fixture.sources, item.query), fixture.capture);

suite('combined conditional source and computation dimensions', () => {
  for (const item of oracle.queries) test(`original width ${item.query.assumeValue} agrees with the independent full forward`, () => {
    const result = context(oracle, item), combined = result.computationImpact!;
    const actual = oracle.independentRuns.find((run: any) => run.width === item.query.assumeValue);
    assert.deepEqual(combined.outputs[0].shape, actual.outputShape);
    assert.equal(combined.summary.incompatible, actual.forward === 'passed' ? 0 : 1);
    assert.equal(combined.summary.unknown, actual.forward === 'passed' ? 0 : 2);
    assert.equal(combined.checks.find(check => check.operationId === 'operation:6')?.status,
      actual.forward === 'passed' ? 'compatible' : 'incompatible');
    assert.equal(combined.checks.find(check => check.operationId === 'operation:5')?.substitution, 'standard_layer_norm_constructor');
    assert.equal(combined.forwardCompatibility, 'not_established');
    assert.equal(combined.checkpointCompatibility, 'not_established');
    assert.equal(combined.coverage.sourceBoundaries, 1);
    assert.ok(combined.coverage.sourceDiagnostics);
    if (item.query.assumeValue === 24) {
      assert.ok(result.operationImpact!.effects.some(effect => effect.localCheck?.status === 'incompatible'));
      assert.deepEqual(combined.checks.find(check => check.operationId === 'operation:7')?.outputShape, [2, 24]);
    }
  });

  for (const defect of ['reshape-proof', 'flatten-proof', 'flatten-leaf', 'flatten-before', 'flatten-expansion',
    'layer-norm-state', 'layer-norm-argument', 'constructor', 'baseline', 'opaque']) {
    test(`${defect} prevents stale output shapes from certifying the downstream result`, () => {
      const fixture = structuredClone(oracle), item = fixture.queries[1];
      const reshape = fixture.capture.operations[6], flatten = fixture.capture.operations[7];
      const flattenRecord = item.page.items.find((row: any) => row.kind === 'call' && row.name === 'split.flatten');
      if (defect === 'reshape-proof') delete reshape.sourceShapeCall;
      if (defect === 'flatten-proof') delete flatten.sourceFlattenCall;
      if (defect === 'flatten-leaf') flattenRecord.configurationChange.argumentValues = [];
      if (defect === 'flatten-before') flattenRecord.configurationChange.argumentValues[0].before = 2;
      if (defect === 'flatten-expansion') flattenRecord.configurationChange.hasExpansion = true;
      if (defect === 'layer-norm-state') fixture.capture.operations[5].arguments.args[2].id = 'tensor3';
      if (defect === 'layer-norm-argument') fixture.capture.operations[5].arguments.args[1].items[0].value = '4';
      if (defect === 'constructor') fixture.capture.modules[1].constructorClassVerified = false;
      if (defect === 'baseline') fixture.capture.values.find((value: any) => value.id === 'tensor7').shape = [2, 99];
      if (defect === 'opaque') fixture.capture.operations[4].target = 'custom.opaque.default';
      const result = context(fixture, item).computationImpact!;
      assert.equal(result.outputs[0].shape, null);
      assert.ok(result.summary.unknown, JSON.stringify(result));
    });
  }

  test('observed shape reads use the newly derived tensor dimensions, not the old capture', () => {
    const fixture = variants.cases.find((item: any) => item.name === 'changed_tensor_read');
    for (const item of fixture.queries) {
      const result = context(fixture, item).computationImpact!;
      const actual = fixture.independentRuns.find((run: any) => run.width === item.query.assumeValue);
      assert.deepEqual(result.outputs[0].shape, actual.outputShape);
      assert.deepEqual(result.summary, { compatible: fixture.capture.operations.length, incompatible: 0, unknown: 0 });
      assert.deepEqual(result.checks.find(check => check.substitution === 'source_and_tensor_dimensions')?.outputShape,
        [2, item.query.assumeValue]);
    }
  });

  test('tensor-only reads outside the source page require current exact binding and valid observed axes', () => {
    const original = variants.cases.find((item: any) => item.name === 'changed_tensor_read');
    for (const defect of ['binding', 'snapshot', 'axis', 'proof']) {
      const fixture = structuredClone(original);
      const op = fixture.capture.operations.find((op: any) => op.sourceShapeCall);
      const binding = fixture.capture.sourceBindings.find((binding: any) => binding.id === op.sourceShapeCall.bindingId);
      if (defect === 'binding') binding.anchors[0].digest = 'f'.repeat(64);
      if (defect === 'snapshot') binding.snapshotId = 'other';
      if (defect === 'axis') op.sourceShapeCall.dimensions[1].tensorDimension.axis = 0;
      if (defect === 'proof') delete op.sourceShapeCall.dimensions[1].tensorDimension;
      const result = context(fixture).computationImpact!;
      assert.equal(result.outputs[0].shape, null, defect);
    }
  });

  test('an untracked storage alias cannot reuse its old shape after constructor state changes', () => {
    const fixture = structuredClone(oracle);
    const weight = fixture.capture.values.find((value: any) => value.state.some((state: any) => state.name === 'up.weight'));
    fixture.capture.values.push({ ...weight, id: 'untracked-alias', objectId: 'separate-object', state: [], aliasOf: weight.id });
    fixture.capture.operations[0].arguments.args[0].id = 'untracked-alias';
    const result = context(fixture).computationImpact!;
    assert.ok(result.stateChanges.some(value => value.valueId === 'untracked-alias' && value.reason === 'unresolved_state_alias'));
    assert.equal(result.outputs[0].shape, null);
  });

  test('an unresolved constructor also invalidates nested calls which do not use its registered state', () => {
    const fixture = structuredClone(oracle);
    fixture.capture.modules[1].constructorClassVerified = false;
    fixture.capture.invocations.push({ ...fixture.capture.invocations[1], id: 'nested-call', moduleId: fixture.capture.modules[0].id,
      parentId: fixture.capture.invocations[1].id });
    const op = fixture.capture.operations[4];
    op.invocationId = 'nested-call'; op.arguments.args = [fixture.capture.inputs.args[0], fixture.capture.inputs.args[0]];
    fixture.capture.values.find((value: any) => value.id === op.output.id).shape = [2, 8];
    const result = context(fixture).computationImpact!;
    assert.equal(result.checks.find(check => check.operationId === op.id)?.reason, 'constructor_effect_unresolved');
    assert.equal(result.outputs[0].shape, null);
  });

  test('conflicting shared constructors remain unknown even when later rewiring makes the actual forward pass', () => {
    const fixture = variants.cases.find((item: any) => item.name === 'conflicting_shared_constructors');
    for (const item of fixture.queries) {
      const result = context(fixture, item).computationImpact!;
      const actual = fixture.independentRuns.find((run: any) => run.width === item.query.assumeValue);
      assert.equal(actual.forward, 'passed');
      assert.ok(result.stateChanges.some(value => value.reason === 'conflicting_state_substitutions' && value.shape === null));
      assert.equal(result.outputs[0].shape, null);
      assert.equal(result.forwardCompatibility, 'not_established');
      const broken = structuredClone(fixture);
      broken.capture.modules.find((module: any) => module.paths.includes('other')).constructorClassVerified = false;
      assert.equal(context(broken, item).computationImpact!.outputs[0].shape, null);
    }
  });

  test('source drift, foreign backend and analysis limits cannot yield baseline predictions', () => {
    for (const defect of ['source', 'backend', 'operations', 'values', 'invocations', 'modules']) {
      const fixture = structuredClone(oracle);
      if (defect === 'source') for (const file of fixture.capture.source.files) file.digest = 'f'.repeat(64);
      if (defect === 'backend') fixture.capture.backend = 'jaxpr';
      if (defect === 'operations') fixture.capture.operations = Array.from({ length: 513 }, (_, i) => ({ ...fixture.capture.operations[0], id: `op:${i}` }));
      if (defect === 'values') fixture.capture.values = Array.from({ length: 4097 }, (_, i) => ({ ...fixture.capture.values[0], id: `value:${i}` }));
      if (defect === 'invocations') fixture.capture.invocations = Array.from({ length: 1025 }, (_, i) => ({ ...fixture.capture.invocations[0], id: `call:${i}` }));
      if (defect === 'modules') fixture.capture.modules = Array.from({ length: 1025 }, (_, i) => ({ ...fixture.capture.modules[0], id: `module:${i}` }));
      const result = context(fixture).computationImpact!;
      assert.equal(result.status, 'unknown', defect);
      assert.deepEqual(result.outputs, [], defect);
    }
  });

  test('partial source pages and capture omissions remain visible without claiming full source coverage', () => {
    const fixture = structuredClone(oracle), item = fixture.queries[1];
    fixture.capture.status = 'partial'; fixture.capture.coverage.omissions = ['operation_limit'];
    item.page.items = item.page.items.slice(0, 10); item.page.nextOffset = 10; item.query.limit = 10;
    const result = context(fixture, item).computationImpact!;
    assert.equal(result.coverage.sourcePageComplete, false);
    assert.equal(result.coverage.captureStatus, 'partial');
    assert.deepEqual(result.coverage.captureOmissions, ['operation_limit']);
    assert.equal(result.forwardCompatibility, 'not_established');
  });

  test('analysis and byte budgets preserve omissions and never mutate retained evidence', () => {
    const fixture = structuredClone(oracle), original = JSON.stringify(fixture);
    const first = context(fixture).computationImpact!;
    first.stateChanges[0].shape![0] = 999; first.outputs[0].shape![0] = 999;
    assert.equal(JSON.stringify(fixture), original);
    assert.deepEqual(context(fixture).computationImpact!.outputs[0].shape, [2, 4]);
    const op = fixture.capture.operations[4];
    fixture.capture.operations.push(...Array.from({ length: 300 }, (_, i) => ({ ...op, id: `extra:${i}`, sequence: i + 20 })));
    const result = context(fixture).computationImpact!;
    assert.ok(result.omittedChecks > 0 || result.status === 'unknown');
    assert.ok(Buffer.byteLength(JSON.stringify(result)) <= 16 * 1024);
  });
});
