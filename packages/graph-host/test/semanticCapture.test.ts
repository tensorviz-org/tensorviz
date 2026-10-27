import * as assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as path from 'node:path';
import { semanticCapture } from '@tensorviz/graph-host/semanticCapture';
import { canonicalFixture, automaticFixture } from '../../../tests/fixtures/semantic.cjs';
import { initializerSourceArgument } from '@tensorviz/contracts/initializerArguments';
const initializerFixture = JSON.parse(readFileSync(path.join(__dirname, '../../../tests/fixtures/contracts/initializer-arguments.json'), 'utf8'));

suite('canonical capture host boundary', () => {
  test('JAX requires a JAX plan and inferred program coverage', () => {
    const { capture, plan } = canonicalFixture();
    const jaxPlan = { ...plan, execution: { framework: 'jax' } };
    const read = () => semanticCapture(capture, capture.attemptId, jaxPlan, capture.planDigest);
    assert.equal(read(), undefined, 'a Torch capture cannot satisfy a JAX plan');
    Object.assign(capture, { backend: 'jaxpr', torchVersion: null, jaxVersion: '0.11.2' });
    capture.coverage.scope = 'traced_program';
    assert.ok(read());
    assert.equal(semanticCapture(capture, capture.attemptId, plan, capture.planDigest), undefined);
    capture.coverage.scope = 'executed_path';
    assert.equal(read(), undefined);
  });

  test('constructor slots distinguish defaults, variadics and positional-only keyword names', () => {
    const entry = structuredClone(initializerFixture.entry);
    assert.deepEqual(initializerSourceArgument(entry, ['args', 0]), { kind: 'int', value: '4' });
    assert.deepEqual(initializerSourceArgument(entry, ['args', 1]), { kind: 'float', value: '-0' });
    assert.deepEqual(initializerSourceArgument(entry, ['kwargs', 'width']), { kind: 'int', value: '9007199254740993' });
    assert.deepEqual(initializerSourceArgument(entry, ['kwargs', 'bias']), { kind: 'bool', value: true });
    for (const slot of [['args', 2], ['args', -1], ['kwargs', 'toString']]) assert.equal(initializerSourceArgument(entry, slot), undefined);
    entry.varKwargs.toString = { kind: 'none' };
    assert.deepEqual(initializerSourceArgument(entry, ['kwargs', 'toString']), { kind: 'none' });
  });
  function initializerCapture() {
    const { capture, plan } = canonicalFixture();
    const binding = { id: 'constructor:0', snapshotId: capture.source.id, kind: 'constructor' as const, quality: 'exact' as const,
      anchors: [{ file: capture.source.files[0].path, digest: capture.source.files[0].digest,
        startLine: 1, startColumn: 0, endLine: 1, endColumn: 5, scope: 'Model.__init__', tokenDigest: 'a'.repeat(64) }] };
    capture.sourceBindings.push(binding);
    capture.modules[0].constructorClassVerified = true;
    capture.modules[0].constructorBindingIds = [binding.id];
    capture.modules[0].initializerArguments = structuredClone(initializerFixture.entry) as any;
    return { capture, plan };
  }
  for (const item of initializerFixture.cases) {
    test(`initializer agreement with Python: ${item.name}`, () => {
      const { capture, plan } = initializerCapture();
      for (const patch of item.patches) {
        let target: any = capture.modules[0].initializerArguments;
        for (const key of patch.path.slice(0, -1)) target = target[key];
        target[patch.path[patch.path.length - 1]] = patch.value;
      }
      assert.equal(!!semanticCapture(capture, capture.attemptId, plan, capture.planDigest), item.live);
    });
  }
  test('initializer evidence requires class proof and explicit negative zero', () => {
    const { capture, plan } = initializerCapture();
    capture.modules[0].constructorClassVerified = false;
    assert.equal(semanticCapture(capture, capture.attemptId, plan, capture.planDigest), undefined);
    capture.modules[0].constructorClassVerified = true;
    capture.modules[0].initializerArguments!.varArgs = [{ kind: 'float', value: -0 }];
    assert.equal(semanticCapture(capture, capture.attemptId, plan, capture.planDigest), undefined);
  });
  test('accepts the same canonical fixture validated by Python', () => {
    const { capture, plan } = canonicalFixture();
    assert.equal(semanticCapture(capture, capture.attemptId, plan, capture.planDigest), capture);
  });
  test('portable source aliases cannot certify a live execution plan', () => {
    const portable = JSON.parse(readFileSync(path.join(__dirname, '../../../tests/fixtures/contracts/portable-capture.json'), 'utf8'));
    const { capture, plan } = canonicalFixture();
    assert.equal(semanticCapture(portable.capture, capture.attemptId, plan, capture.planDigest), undefined);
  });
  test('automatic evidence needs its exact preparation and a selected candidate actually tried', () => {
    const { capture, preparation } = automaticFixture();
    assert.equal(semanticCapture(capture, capture.attemptId, preparation, capture.preparationDigest), capture);
    const { plan } = canonicalFixture(capture.attemptId);
    assert.equal(semanticCapture(capture, capture.attemptId, plan, capture.preparationDigest), undefined);
    assert.equal(semanticCapture(capture, capture.attemptId, preparation, 'b'.repeat(64)), undefined);
    capture.automatic.candidateIndex = 2;
    assert.equal(semanticCapture(capture, capture.attemptId, preparation, capture.preparationDigest), undefined);
  });
  test('negative zero survives host JSON retention as a typed float literal', () => {
    const { capture, plan } = canonicalFixture();
    capture.operations[0].arguments.args[1] = { kind: 'float', value: '-0' };
    const stored = JSON.parse(JSON.stringify(capture));
    assert.ok(semanticCapture(stored, capture.attemptId, plan, capture.planDigest));
    assert.deepEqual(stored.operations[0].arguments.args[1], { kind: 'float', value: '-0' });
  });
  test('a constructor class claim needs one exact constructor binding', () => {
    const { capture, plan } = canonicalFixture();
    assert.ok(semanticCapture(capture, capture.attemptId, plan, capture.planDigest));
    capture.modules[0].constructorClassVerified = true;
    assert.equal(semanticCapture(capture, capture.attemptId, plan, capture.planDigest), undefined);
    capture.modules[0].constructorBindingIds = ['missing'];
    assert.equal(semanticCapture(capture, capture.attemptId, plan, capture.planDigest), undefined);
    const binding = { id: 'constructor:0', snapshotId: capture.source.id, kind: 'constructor' as const, quality: 'exact' as const,
      anchors: [{ file: capture.source.files[0].path, digest: capture.source.files[0].digest,
        startLine: 1, startColumn: 0, endLine: 1, endColumn: 5, scope: 'Model.__init__', tokenDigest: 'a'.repeat(64) }] };
    capture.sourceBindings.push(binding);
    capture.modules[0].constructorBindingIds = [binding.id];
    assert.ok(semanticCapture(capture, capture.attemptId, plan, capture.planDigest));
    capture.sourceBindings[0].quality = 'ambiguous';
    assert.equal(semanticCapture(capture, capture.attemptId, plan, capture.planDigest), undefined);
  });
  for (const problem of ['attempt', 'plan', 'source', 'limits', 'schema', 'budget', 'opaque', 'missing_value',
    'duplicate_operation', 'count', 'root', 'producer_order', 'unfinished', 'numeric_negative_zero',
    'missing_effect_before', 'missing_effect_after', 'opaque_effect', 'mixed_setup', 'automatic_on_plan']) {
    test(`rejects ${problem} without accepting false completeness`, () => {
      const { capture, plan } = canonicalFixture();
      if (problem === 'attempt') capture.attemptId = 'other';
      if (problem === 'plan') capture.planDigest = 'b'.repeat(64);
      if (problem === 'source') capture.source.files[0].digest = 'b'.repeat(64);
      if (problem === 'limits') capture.limits.values++;
      if (problem === 'schema') (capture.operations[0].arguments.args[1] as any).value = 1;
      if (problem === 'budget') { capture.limits.bytes = plan.captureLimits.bytes = 1024; }
      if (problem === 'opaque') capture.operations[0].arguments.args[1] = { kind: 'opaque', reason: 'unsupported_type', typeName: null };
      if (problem === 'missing_value') capture.operations[1].arguments.args[0] = { kind: 'value', id: 'absent' };
      if (problem === 'duplicate_operation') capture.operations[1].id = capture.operations[0].id;
      if (problem === 'count') capture.coverage.operationsSeen++;
      if (problem === 'root') capture.output = { kind: 'none' };
      if (problem === 'producer_order') capture.operations[0].arguments.args[0] = { kind: 'value', id: 'b' };
      if (problem === 'unfinished') { capture.operations[0].status = 'interrupted'; capture.operations[0].output = null; }
      if (problem === 'numeric_negative_zero') capture.operations[0].arguments.args[1] = { kind: 'float', value: -0 };
      if (problem === 'mixed_setup') (capture as any).preparationDigest = capture.planDigest;
      if (problem === 'automatic_on_plan') (capture as any).automatic = { provenance: 'declared', candidateIndex: 1, candidatesTried: 1 };
      if (problem.includes('effect')) capture.operations[0].effects = [{ kind: 'write', argumentPath: ['args', 0],
        beforeId: problem === 'missing_effect_before' ? 'missing' : 'h',
        after: problem === 'opaque_effect' ? { kind: 'opaque', reason: 'value_limit', typeName: null }
          : { kind: 'value', id: problem === 'missing_effect_after' ? 'missing' : 'a' } }];
      assert.equal(semanticCapture(capture, plan.id, plan, 'a'.repeat(64)), undefined);
    });
  }
  test('unknown plans and null payloads cannot invent semantic captures', () => {
    const { capture, plan } = canonicalFixture();
    assert.equal(semanticCapture(capture, capture.attemptId, undefined, capture.planDigest), undefined);
    assert.equal(semanticCapture(capture, capture.attemptId, plan, null), undefined);
    assert.equal(semanticCapture(null, capture.attemptId, plan, capture.planDigest), undefined);
  });
});
