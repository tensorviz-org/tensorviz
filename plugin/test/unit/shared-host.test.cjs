const { test } = require('node:test');
const assert = require('node:assert/strict');
const { TraceWorkerCore, GraphOpeningController, sha256Text } = require('../../out/graph-host.cjs');

test('shared worker refuses an unauthorized model before resolving Python', async () => {
  let resolved = false;
  const worker = new TraceWorkerCore('/unavailable/backend', { appendLine() {} }, {
    trusted: () => false,
    documents: () => ({}),
    interpreter: async () => { resolved = true; throw new Error('must not resolve'); },
  });
  const result = await worker.run({ file: '/unavailable/model.py', cls: 'Model', tier: 'real' });
  assert.equal(result.ok, false);
  assert.match(result.reason, /authorization/);
  assert.equal(resolved, false);
  worker.dispose();
});

test('shared opening invalidates retained evidence when its saved source changes', () => {
  let state;
  const opening = new GraphOpeningController({ current: () => true, publish: value => { state = value; } });
  const target = { workspaceSessionId: 'session', workspaceFolder: '/project', file: 'model.py', cls: 'Model',
    documentVersion: 0, sourceDigest: sha256Text('before'), backendRevision: 1 };
  opening.sourceOnly(target);
  opening.invalidate({ ...target, sourceDigest: sha256Text('after') });
  assert.equal(state.phase, 'cancelled');
  assert.equal(state.cancellationReason, 'source_changed');
  assert.equal(state.hasCapture, false);
  opening.clear();
});
