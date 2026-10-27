import * as assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as path from 'node:path';
import { samePreparedExecution, savedExecutionGuard } from '@tensorviz/graph-host/executionGuard';
import type { ValidationPreparation } from '@tensorviz/graph-host/validationController';
import type { RunSpec } from '@tensorviz/contracts/protocol';

const corpus = JSON.parse(readFileSync(path.join(__dirname, '../../../tests/fixtures/contracts/validation-contracts.json'), 'utf8'));
const spec: RunSpec = corpus.structural.find((f: { schema: string }) => f.schema === 'RunSpec').value;
function preparation(): ValidationPreparation {
  return { profiles: [spec.identity.profile], resolution: { status: 'resolved', spec: structuredClone(spec) },
    guards: { '.': ['.', '1:1'], 'tensorviz.toml': ['tensorviz.toml', '1:2', spec.identity.configDigest, null, false] } };
}

suite('saved execution applicability', () => {
  test('new preflight IDs and opening unchanged saved files do not change applicability', () => {
    const a = preparation(), b = structuredClone(a);
    assert.ok(b.resolution.status === 'resolved');
    b.resolution.spec.identity.runId = 'fresh-preflight';
    b.resolution.spec.identity.generation++;
    b.resolution.spec.files[0].documentVersion = 1;
    (b.guards['tensorviz.toml'] as unknown[])[3] = 1;
    assert.equal(samePreparedExecution(a, b), true);
  });
  test('releasing a clean document does not invalidate unchanged saved bytes', () => {
    const a = preparation(), b = structuredClone(a);
    (a.guards['tensorviz.toml'] as unknown[])[3] = 2;
    assert.equal(samePreparedExecution(a, b), true);
    // Once the buffer is released, physical identity, bytes and dirtiness still
    // protect the saved invocation; absence of a buffer revision is not an edit.
    for (const index of [0, 1, 2, 4]) {
      const changed = structuredClone(b);
      (changed.guards['tensorviz.toml'] as unknown[])[index] = index === 4 ? true : 'changed';
      assert.equal(samePreparedExecution(a, changed), false);
    }
  });
  for (const key of ['workspaceSessionId', 'workspaceId', 'profile', 'configDigest', 'profileDigest', 'sourceDigest', 'environmentDigest'] as const) {
    test(`${key} changes cannot certify the resolved invocation`, () => {
      const a = preparation(), b = structuredClone(a);
      assert.ok(b.resolution.status === 'resolved');
      b.resolution.spec.identity[key] = 'changed';
      assert.equal(samePreparedExecution(a, b), false);
    });
  }
  for (const change of ['physical', 'canonical', 'digest', 'dirty', 'revision', 'added', 'missing']) {
    test(`${change} file state invalidates the plan`, () => {
      const a = preparation();
      (a.guards['tensorviz.toml'] as unknown[])[3] = 1;
      const b = structuredClone(a), item = b.guards['tensorviz.toml'] as unknown[];
      if (change === 'physical') item[1] = '1:9';
      if (change === 'canonical') item[0] = 'other.toml';
      if (change === 'digest') item[2] = 'f'.repeat(64);
      if (change === 'dirty') item[4] = true;
      if (change === 'revision') item[3] = 2;
      if (change === 'added') b.guards['helper.py'] = ['helper.py', '1:3', 'f'.repeat(64), null, false];
      if (change === 'missing') delete b.guards['tensorviz.toml'];
      assert.equal(samePreparedExecution(a, b), false);
    });
  }
  test('guard owns an immutable baseline and fails closed on resolution failure', async () => {
    const a = preparation(), now = structuredClone(a);
    const guard = savedExecutionGuard(a, async () => now);
    delete a.guards['tensorviz.toml'];
    assert.equal(await guard.current(new AbortController().signal), true);
    now.resolution = { status: 'needs_setup', requirements: [] };
    assert.equal(await guard.current(new AbortController().signal), false);
  });
  test('exceptions and cancellation cannot allow execution', async () => {
    const controller = new AbortController();
    const guard = savedExecutionGuard(preparation(), async () => { throw new Error('private'); });
    assert.equal(await guard.current(controller.signal), false);
    controller.abort();
    assert.equal(await guard.current(controller.signal), false);
  });
});
