const { test } = require('node:test');
const assert = require('node:assert/strict');
const { StandaloneHost } = require('../../out/host/host.cjs');
const { ApprovalGate } = require('../../out/host/approval.cjs');

function fixture() {
  const host = new StandaloneHost('/unused', '/unused', new ApprovalGate(async () => false, '/unused'));
  host.checkFreshness = async () => {};
  return host;
}

test('bounded opening waits on one request and returns a short capture directly', async () => {
  const host = fixture(), calls = [];
  host.openGraph.invoke = (_owner, args) => {
    calls.push(args);
    return { requestId: 'request', status: calls.length < 3 ? 'opening' : 'ready', validated: calls.length >= 3 };
  };
  host.progress = () => ({});
  try {
    const result = await host.openAndWait({ file: 'model.py', cls: 'Net' }, new AbortController().signal, 1000);
    assert.equal(result.status, 'ready');
    assert.deepEqual(calls, [{ file: 'model.py', cls: 'Net' }, { requestId: 'request' }, { requestId: 'request' }]);
  } finally { host.dispose(); }
});

test('permission, setup, environment choices and zero waits return without polling', async () => {
  for (const choice of [{ status: 'opening', approval: { method: 'browser' } }, { status: 'needs_setup' }, { status: 'blocked' }, { status: 'opening', waitMs: 0 }]) {
    const host = fixture(); let calls = 0;
    host.openGraph.invoke = () => { calls++; return { requestId: 'request', status: choice.status }; };
    host.progress = () => choice.approval ? { approval: choice.approval } : {};
    try {
      const result = await host.openAndWait({}, new AbortController().signal, choice.waitMs ?? 1000);
      assert.equal(result.status, choice.status); assert.equal(calls, 1);
    } finally { host.dispose(); }
  }
});

test('cancelled tool waiting cancels only its own opening', async () => {
  const host = fixture(), controller = new AbortController(), calls = [];
  host.openGraph.invoke = (_owner, args) => { calls.push(args); return { requestId: 'request', status: 'opening' }; };
  host.progress = () => ({});
  try {
    const waiting = host.openAndWait({}, controller.signal, 1000);
    setTimeout(() => controller.abort(), 10);
    await assert.rejects(waiting, { name: 'AbortError' });
    assert.deepEqual(calls.at(-1), { requestId: 'request', action: 'cancel' });
  } finally { host.dispose(); }
});

test('a cancelled request cannot overwrite the replacement error state', async () => {
  const host = fixture();
  let originalError;
  try { await host.open({ workspaceRoot: '/definitely-not-a-tensorviz-workspace' }, new AbortController().signal); }
  catch (error) { originalError = error; }
  assert.ok(originalError);
  let reject;
  host.open = () => new Promise((_resolve, fail) => { reject = fail; });
  try {
    const first = host.openGraph.invoke(host.id, {});
    host.openGraph.invoke(host.id, { requestId: first.requestId, action: 'cancel' });
    const replacement = { code: 'REPLACEMENT', message: 'current operation' };
    host.lastError = replacement;
    reject(originalError);
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(host.progress().error.code, 'REPLACEMENT');
  } finally { host.dispose(); }
});


test('a file outside the workspace is a plain error and does not inherit a pending approval', async () => {
  const host = fixture();
  host.openGraph.invoke = () => ({ requestId: 'request', status: 'failed',
    error: { code: 'OUTSIDE_WORKSPACE', message: 'Select a file inside this exact workspace.' },
    nextAction: { kind: 'approval_required' }, approval: { id: 'pending' } });
  host.progress = () => ({ approval: { id: 'pending' }, nextAction: { kind: 'approval_required' } });
  try {
    await assert.rejects(() => host.openAndWait({ file: '../secret.py', cls: 'Net' }, new AbortController().signal),
      error => error.code === 'OUTSIDE_WORKSPACE' && !error.approval && error.nextAction === undefined);
  } finally { host.dispose(); }
});

test('explicit cancellation does not inherit the stopped request permission or environment prompt', async () => {
  const host = fixture();
  host.openGraph.invoke = () => ({ requestId: 'request', status: 'cancelled', continuationRequired: false });
  host.progress = () => ({ status: 'blocked', approval: { method: 'browser' } });
  try {
    const result = await host.openAndWait({ requestId: 'request', action: 'cancel' }, new AbortController().signal);
    assert.equal(result.status, 'cancelled'); assert.equal(result.approval, undefined);
  } finally { host.dispose(); }
});

test('no Torch candidate provides an explicit fallback step for the same model instead of an unsupported native chooser', () => {
  const host = fixture();
  host.selected = { root: '/project', file: 'model.py', cls: 'Net' };
  try {
    host.opening.sourceOnly({ workspaceFolder: '/project', file: 'model.py', cls: 'Net', documentVersion: 0,
      sourceDigest: 'a'.repeat(64), backendRevision: 1 });
    host.opening.state.environment = { reason: 'selection_required', detail: 'Choose Python', candidates: [] };
    const result = host.progress();
    assert.deepEqual(result.nextAction.arguments, { workspaceRoot: '/project', file: 'model.py', cls: 'Net' });
    assert.deepEqual(result.nextAction.requiredArguments, ['interpreter or managedRuntime']);
    assert.match(result.nextAction.instruction, /explicitly request managedRuntime/);
    assert.notEqual(result.nextAction.arguments.action, 'environment');
  } finally { host.dispose(); }
});
