const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');
const { randomUUID } = require('node:crypto');

// Exercise the actual registered handler without starting the native host,
// opening a permission page, provisioning Python or using the network.
function fixture() {
  const source = fs.readFileSync(path.join(__dirname, '../../src/main.ts'), 'utf8');
  const begin = source.indexOf('  let environmentTask:');
  const end = source.indexOf('  async function shutdown()', begin);
  assert.ok(begin >= 0 && end > begin, 'environment handler remains discoverable');
  const transformed = ts.transpileModule(source.slice(begin, end), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
    reportDiagnostics: true,
  });
  assert.deepEqual(transformed.diagnostics, []);
  let resolve, reject, handler, calls = 0;
  const pending = new Promise((yes, no) => { resolve = yes; reject = no; });
  // Use the production progress getter and signal scoping method too, rather
  // than duplicating the ownership condition in this test's fake runtime.
  const runtimeSource = fs.readFileSync(path.join(__dirname, '../../src/runtime.ts'), 'utf8');
  const parsed = ts.createSourceFile('runtime.ts', runtimeSource, ts.ScriptTarget.Latest, true);
  const runtimeClass = parsed.statements.find(node => ts.isClassDeclaration(node) && node.name.text === 'ManagedRuntime');
  const members = runtimeClass.members.filter(node => ['progress', 'progressFor'].includes(node.name?.getText(parsed)));
  assert.equal(members.length, 2);
  const scope = {};
  const runtimeCode = ts.transpileModule(`class RuntimeProgress { ${members.map(node => node.getText(parsed)).join('\n')} }; globalThis.RuntimeProgress = RuntimeProgress;`,
    { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText;
  vm.runInNewContext(runtimeCode, scope);
  const runtime = new scope.RuntimeProgress();
  const host = { runtime, inspectEnvironment(_args, signal) { calls++; runtime.activity = { signal }; return pending; } };
  const approvals = {};
  const schema = { optional() { return this; }, uuid() { return this; } };
  class HostError extends Error { constructor(code, message) { super(message); this.code = code; } }
  vm.runInNewContext(transformed.outputText, {
    host, approvals, randomUUID, AbortController, HostError,
    absolute: schema, z: { string: () => schema }, result: action => action(),
    server: { registerTool(name, metadata, callback) {
      assert.equal(name, 'get_execution_environment'); handler = callback;
    } },
  });
  assert.equal(typeof handler, 'function');
  return { host, approvals, resolve, reject, HostError, calls: () => calls,
    invoke: args => handler(args) };
}

test('environment polling reports changing runtime progress without starting another inspection', async () => {
  const f = fixture();
  const first = f.invoke({ workspaceRoot: '/project' });
  assert.equal(first.status, 'checking');
  f.host.runtime.activity.detail = 'Connecting to the pinned Python download…';
  assert.equal(f.invoke({ requestId: first.requestId }).runtimeProgress, f.host.runtime.progress);
  f.host.runtime.activity.detail = 'Download verified (sha256 ok).';
  const poll = f.invoke({ requestId: first.requestId });
  assert.equal(poll.runtimeProgress, f.host.runtime.progress);
  assert.equal(poll.requestId, first.requestId);
  assert.equal(poll.pollAfterMs, 1000);
  assert.equal(f.calls(), 1);
  assert.throws(() => f.invoke({}), { code: 'ENVIRONMENT_BUSY' });
  assert.throws(() => f.invoke({ requestId: randomUUID() }), { code: 'UNKNOWN_REQUEST' });
  f.resolve({ compatible: true }); await Promise.resolve();
  const ready = f.invoke({ requestId: first.requestId });
  assert.equal(ready.status, 'ready');
  assert.equal(ready.environment.compatible, true);
  assert.equal('runtimeProgress' in ready, false, 'terminal states omit stale progress');
});

test('blocked environment requests omit stale progress and retain their actionable error', async () => {
  const f = fixture();
  const first = f.invoke({});
  f.host.runtime.activity.detail = 'Preparing service environment.';
  f.reject(new f.HostError('RUNTIME_DOWNLOAD_FAILED', 'Download could not connect.'));
  await Promise.resolve();
  const blocked = f.invoke({ requestId: first.requestId });
  assert.equal(blocked.status, 'blocked');
  assert.equal(blocked.error.code, 'RUNTIME_DOWNLOAD_FAILED');
  assert.equal(blocked.error.message, 'Download could not connect.');
  assert.equal('runtimeProgress' in blocked, false);
});

test('permission-required and metadata-only checks preserve approval without inventing progress', async () => {
  const f = fixture();
  f.approvals.pending = { id: 'approval-id', method: 'browser' };
  const first = f.invoke({});
  assert.equal(first.status, 'checking');
  assert.equal(first.approval, f.approvals.pending);
  assert.equal('runtimeProgress' in first, false);
  f.host.runtime.activity.detail = '';
  assert.equal('runtimeProgress' in f.invoke({ requestId: first.requestId }), false);
  f.resolve({ compatible: false }); await Promise.resolve();
});

test('unrelated and aborted runtime activity never leaks into an environment request', async () => {
  const f = fixture();
  const first = f.invoke({});
  const own = f.host.runtime.activity;
  own.detail = 'Preparing the exact package installation plan.';
  assert.equal(f.invoke({ requestId: first.requestId }).runtimeProgress, own.detail);
  const other = new AbortController();
  f.host.runtime.activity = { signal: other.signal, detail: 'Unrelated project operation' };
  assert.equal('runtimeProgress' in f.invoke({ requestId: first.requestId }), false);
  assert.equal(f.host.runtime.progressFor(other.signal), 'Unrelated project operation');
  other.abort();
  assert.equal(f.host.runtime.progressFor(other.signal), undefined);
  f.resolve({ compatible: true }); await Promise.resolve();
});
