import * as assert from 'node:assert/strict';
import { executionFailure, profileFailure } from '@tensorviz/graph-host/executionFailure';
import type { components } from '@tensorviz/contracts/semanticContracts.generated';
import { isGraphFailure } from '@tensorviz/contracts/protocol';

type Diagnostic = components['schemas']['Diagnostic'];
const diagnostic = (phase: Diagnostic['phase'], code: Diagnostic['code']): Diagnostic => ({
  id: 'failure', code, phase, remedy: 'inspect_failure', field: [], names: [], sourceBindingId: null,
});
suite('graph failure recovery', () => {
  test('dependency recovery carries bounded module facts, never exception messages or pip arguments', () => {
    const failure = executionFailure({ ...diagnostic('import', 'missing_dependency'), names: ['transformers', '--index-url', 'secret/path'] });
    assert.deepEqual(failure, { kind: 'environment', missingModules: ['transformers'] });
    assert.ok(isGraphFailure(failure));
    assert.equal(isGraphFailure({ ...failure, missingModules: ['--index-url'] }), false);
    assert.equal(isGraphFailure({ ...failure, kind: 'execution' }), false);
  });
  for (const code of ['invalid_shape', 'invalid_dtype', 'assertion_failed', 'execution_failed'] as const) {
    test(`${code} in forward keeps configured inputs`, () => {
      assert.deepEqual(executionFailure(diagnostic('forward', code)), { kind: 'execution' });
    });
  }
  test('binding recovery names only affected fields', () => {
    assert.deepEqual(executionFailure({ ...diagnostic('binding', 'missing_argument'),
      field: ['inputs'], names: ['mask', 'secret/path'] }), { kind: 'setup', fields: ['inputs.mask'] });
  });
  for (const code of ['unresolved_import', 'execution_failed'] as const) test(`${code} during import checks compatibility without inventing a missing module`, () => {
    const failure = executionFailure({ ...diagnostic('import', code), names: ['ValueError'] });
    assert.deepEqual(failure, { kind: 'environment', importFailure: true });
    assert.ok(isGraphFailure(failure));
    assert.equal(isGraphFailure({ ...failure, kind: 'execution' }), false);
    assert.equal(isGraphFailure({ ...failure, importFailure: 'true' }), false);
  });
  test('bad constructor code, missing dependencies and timeout have different recovery', () => {
    assert.equal(executionFailure(diagnostic('construction', 'execution_failed')).kind, 'source');
    assert.equal(executionFailure(diagnostic('construction', 'missing_dependency')).kind, 'environment');
    assert.equal(executionFailure(diagnostic('capture', 'timeout')).kind, 'capture');
    assert.equal(executionFailure().kind, 'capture');
  });
  test('preflight environment and unsaved files never become input errors', () => {
    const requirement = { profile: null, field: [], phase: 'static' as const, reason: 'Unavailable.', remedy: 'Retry.' };
    assert.equal(profileFailure([{ ...requirement, code: 'environment' }]).kind, 'environment');
    assert.equal(profileFailure([{ ...requirement, code: 'unsaved' }]).kind, 'source');
    assert.equal(profileFailure([{ ...requirement, code: 'invalid_source' }]).kind, 'source');
  });
});
