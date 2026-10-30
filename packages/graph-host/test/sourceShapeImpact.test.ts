import * as assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as path from 'node:path';
import { sourceContextPage, sourceCaptureContext } from '@tensorviz/graph-host/sourceContext';
const oracle = JSON.parse(readFileSync(path.join(__dirname, '../../../tests/fixtures/contracts/source-shape-oracle.json'), 'utf8'));
const first = oracle.cases[0], requested = first.queries[0];
const context = (fixture = first, item = requested) => sourceCaptureContext(sourceContextPage(item.page, fixture.sources, item.query), fixture.capture);

suite('conditional source-to-operation dimensions', () => {
  for (const fixture of oracle.cases) for (const item of fixture.queries) {
    test(`${fixture.name} WIDTH=${item.query.assumeValue} respects CPU observations and uncertainty`, () => {
      const impact = context(fixture, item).operationImpact!;
      assert.equal(impact.evidence, 'conditional_operation_dimensions');
      assert.equal(impact.scope, 'linked_calls_on_this_page_with_fixed_input_shapes');
      assert.ok(impact.effects.length);
      const effect = impact.effects.find(row => row.target === 'aten.view.default')!;
      if (fixture.derivable) {
        assert.equal(effect.status, 'derived');
        assert.ok(effect.dimensions.every(value => value.sourceBefore === value.observedBefore));
        const actual = fixture.independentRuns.find((run: any) => run.width === item.query.assumeValue);
        assert.equal(effect.localCheck!.status, actual.forward === 'passed' ? 'compatible' : 'incompatible');
        assert.deepEqual(effect.localCheck!.outputShape, actual.outputShape);
      } else {
        assert.equal(effect.status, 'unknown'); assert.equal(effect.localCheck, null);
      }
      assert.equal(impact.forwardCompatibility, 'not_established');
      assert.equal(impact.checkpointCompatibility, 'not_established');
    });
  }
  for (const [defect, reason] of [
    ['proof', 'operation_argument_correspondence_unavailable'], ['argument', 'source_argument_values_unavailable'],
    ['expansion', 'expanded_or_omitted_arguments'], ['omitted', 'expanded_or_omitted_arguments'],
    ['before', 'observed_source_value_mismatch'], ['dynamic', 'source_dimension_unknown'],
    ['missing-leaf', 'source_dimension_unknown'], ['baseline', 'recorded_dimension_rule_mismatch'],
    ['backend', 'operation_argument_correspondence_unavailable'],
  ]) test(`refuses ${defect} without freezing unsupported arguments`, () => {
    const fixture = structuredClone(first), item = structuredClone(requested), op = fixture.capture.operations[0];
    const change = item.page.items.find((row: any) => row.kind === 'call').configurationChange;
    if (defect === 'proof') delete op.sourceShapeCall;
    if (defect === 'argument') { delete change.argumentValues; delete change.omittedArgumentValues; }
    if (defect === 'expansion') change.hasExpansion = true;
    if (defect === 'omitted') change.omittedArgumentValues = 1;
    if (defect === 'before') change.argumentValues[0].before++;
    if (defect === 'dynamic') change.argumentValues[0] = { path: ['args', 0], status: 'unknown', before: null, after: null, reason: 'unsupported_expression' };
    if (defect === 'missing-leaf') change.argumentValues.shift();
    if (defect === 'baseline') fixture.capture.values.find((value: any) => value.id === op.output.id).shape = [9];
    if (defect === 'backend') fixture.capture.backend = 'jaxpr';
    const effect = context(fixture, item).operationImpact!.effects[0];
    assert.equal(effect.status, 'unknown'); assert.equal(effect.reason, reason); assert.equal(effect.localCheck, null);
  });
  test('tensor-read dimensions carry exact value identity and keep local checks separate from the full edit', () => {
    const fixture = oracle.cases.find((item: any) => item.name === 'tensor_dimension');
    const effect = context(fixture, fixture.queries[0]).operationImpact!.effects[0];
    assert.deepEqual(effect.dimensions[0].tensorDimension, { valueId: fixture.capture.inputs.args[0].id, axis: 0 });
    assert.equal(effect.dimensions[0].basis, 'captured_tensor_dimension');
    assert.equal(effect.dimensions[0].assumedAfter, 2);
    const width = JSON.parse(readFileSync(path.join(__dirname, '../../../tests/fixtures/contracts/source-change-oracle.json'), 'utf8'));
    const query = width.queries.find((item: any) => item.query.assumeValue === 24);
    const impact = sourceCaptureContext(sourceContextPage(query.page, width.sources, query.query), width.capture).operationImpact!;
    assert.equal(width.independentRuns.find((item: any) => item.width === 24).forward, 'passed');
    assert.ok(impact.effects.some(item => item.localCheck?.status === 'incompatible'), 'Holding the old operand shape fixed is not the full width edit.');
    assert.equal(impact.forwardCompatibility, 'not_established');
  });
  for (const defect of ['duplicate', 'missing-count', 'oversized', 'invalid-path']) test(`host rejects ${defect} argument leaf pages`, () => {
    const item = structuredClone(requested), change = item.page.items.find((row: any) => row.kind === 'call').configurationChange;
    if (defect === 'duplicate') change.argumentValues.push(change.argumentValues[0]);
    if (defect === 'missing-count') delete change.omittedArgumentValues;
    if (defect === 'oversized') change.argumentValues = Array.from({ length: 33 }, (_, i) => ({ ...change.argumentValues[0], path: ['args', i] }));
    if (defect === 'invalid-path') change.argumentValues[0].path = ['args', 0, 'attribute', 'shape'];
    assert.throws(() => sourceContextPage(item.page, first.sources, item.query), /did not match/);
  });
  test('source drift removes correspondence and large call inventories report omissions', () => {
    const fixture = structuredClone(first);
    fixture.capture.source.files[0].digest = 'f'.repeat(64);
    assert.equal(context(fixture).operationImpact!.effects.length, 0);
    assert.ok(context(fixture).operationImpact!.unlinkedCalls);
    fixture.capture.source.files[0].digest = first.capture.source.files[0].digest;
    const operation = fixture.capture.operations[0];
    fixture.capture.operations.push(...Array.from({ length: 100 }, (_, i) => ({ ...operation, id: 'operation:extra:' + i })));
    const impact = context(fixture).operationImpact!;
    assert.ok(impact.omittedEffects && impact.effects.length <= 16);
    assert.ok(Buffer.byteLength(JSON.stringify(impact)) <= 12 * 1024);
    const original = JSON.stringify(context());
    const mutated = context(); mutated.operationImpact!.effects[0].dimensions[0].assumedAfter = 999;
    assert.equal(JSON.stringify(context()), original);
  });
});
