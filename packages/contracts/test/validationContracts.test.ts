import * as assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import Ajv from 'ajv';
import type { RunSpec, ExecutionRecord, ValidationExecutionUpdate } from '@tensorviz/contracts/protocol';

const load = (name: string) => JSON.parse(readFileSync(join(__dirname, '../../../tests/fixtures/contracts', name), 'utf8'));
const schema = load('validation.schema.json');
const corpus = load('validation-contracts.json') as {
  structural: { name: string; schema: string; valid: boolean; value: unknown }[];
};
// Schema comes from Python; the host doesn't implement a second config policy.
const ajv = new Ajv({ strict: false });
ajv.addSchema(schema, 'validation');

suite('validation v1 contracts', () => {
  for (const fixture of corpus.structural) {
    test(fixture.name, () => {
      const validate = ajv.compile({ $ref: `validation#/definitions/${fixture.schema}` });
      assert.equal(validate(fixture.value), fixture.valid, JSON.stringify(validate.errors));
    });
  }
  test('generated types keep process lifetime and invocation identity separate', () => {
    const update = corpus.structural.find(c => c.name === 'stale measurement retains live process')!.value as ValidationExecutionUpdate;
    const record: ExecutionRecord = update.record;
    assert.equal(update.freshness.state, 'stale');
    assert.equal(record.process.state, 'running');
    assert.equal(record.capture.state, 'complete');
    assert.equal(record.invocations[1].moduleId, record.invocations[2].moduleId);
    assert.notEqual(record.invocations[1].id, record.invocations[2].id);
    const spec = corpus.structural.find(c => c.schema === 'RunSpec')!.value as RunSpec;
    assert.equal(spec.profile.kind, 'model');
    // @ts-expect-error capture completion cannot terminate the process contract
    const process: ExecutionRecord['process'] = { state: 'complete' };
    assert.ok(process);
  });
});
