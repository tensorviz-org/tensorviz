const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { ApprovalGate } = require('../../out/host/approval.cjs');
const { nativePermissionDecision } = require('../../out/host/approval.cjs');
const plan = { operation: 'capture', workspaceRoot: '/exact/worktree', interpreter: '/python', detail: 'One test invocation.', facts: { source: 'source-digest', inputs: [2, 4] } };

async function storage(t) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-permissions-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  return root;
}

test('one permission covers different operations, projects and subsequent hosts', async t => {
  const root = await storage(t);
  let prompts = 0;
  const native = async message => {
    prompts++;
    assert.match(message, /across projects, chats, compatible updates and coding clients/);
    assert.match(message, /build\/install required dependencies/);
    return true;
  };
  const gate = new ApprovalGate(native, root);
  const signal = new AbortController().signal;
  for (const operation of ['inspect_environment', 'prepare_inputs', 'save_setup', 'capture', 'managed_runtime', 'build_dependencies', 'install_dependencies']) {
    await gate.approve({ ...plan, operation }, signal);
  }
  await new ApprovalGate(native, root).approve({ ...plan, workspaceRoot: '/another/project' }, signal);
  assert.equal(prompts, 1);
  assert.equal((await fs.stat(gate.permissions.file)).mode & 0o777, 0o600);
  assert.equal((await fs.stat(gate.permissions.directory)).mode & 0o777, 0o700);
});

test('simultaneous chats share one permission decision', async t => {
  const root = await storage(t);
  let prompts = 0;
  const native = async () => { prompts++; await new Promise(resolve => setTimeout(resolve, 150)); return true; };
  await Promise.all([new ApprovalGate(native, root), new ApprovalGate(native, root)].map(gate => gate.approve(plan, new AbortController().signal)));
  assert.equal(prompts, 1);
});

test('a declined native receipt cannot be replayed or replaced by plan fields', async t => {
  const root = await storage(t);
  let prompts = 0;
  const native = async message => { prompts++; assert.match(message, /exact\/worktree/); return false; };
  const gate = new ApprovalGate(native, root);
  await assert.rejects(gate.approve(plan, new AbortController().signal), { code: 'APPROVAL_DECLINED' });
  await assert.rejects(gate.approve(plan, new AbortController().signal), { code: 'APPROVAL_DECLINED' });
  await assert.rejects(new ApprovalGate(native, root).approve({ ...plan, approved: true }, new AbortController().signal), { code: 'APPROVAL_DECLINED' });
  assert.equal(prompts, 1);
});

test('revocation applies to running hosts and does not re-prompt', async t => {
  const root = await storage(t);
  let prompts = 0;
  const gate = new ApprovalGate(async () => { prompts++; return true; }, root);
  const signal = new AbortController().signal;
  await gate.approve(plan, signal);
  await gate.permissions.save('denied');
  await assert.rejects(gate.approve(plan, signal), { code: 'APPROVAL_DECLINED' });
  assert.equal(prompts, 1);
});

test('native cancellation leaves permission unset and a later opening can prompt', async t => {
  let prompts = 0;
  const gate = new ApprovalGate(async () => nativePermissionDecision(++prompts === 1
    ? { action: 'cancel' } : { action: 'accept', content: { confirm: true } }), await storage(t));
  const signal = new AbortController().signal;
  await assert.rejects(gate.approve(plan, signal), { code: 'APPROVAL_CANCELLED' });
  assert.equal(await gate.permissions.status(), 'missing');
  await gate.approve(plan, signal);
  assert.equal(prompts, 2);
  assert.equal(await gate.permissions.status(), 'granted');
});

test('explicit recovery prompts again without granting through tool arguments', async t => {
  let answer = false, prompts = 0;
  const gate = new ApprovalGate(async () => { prompts++; return answer; }, await storage(t));
  const signal = new AbortController().signal;
  await assert.rejects(gate.approve(plan, signal), { code: 'APPROVAL_DECLINED' });
  await assert.rejects(gate.approve(plan, signal), { code: 'APPROVAL_DECLINED' });
  assert.equal(prompts, 1);
  await assert.rejects(gate.reconsider(signal), { code: 'APPROVAL_DECLINED' });
  assert.equal(await gate.permissions.status(), 'denied');
  answer = true;
  await gate.reconsider(signal);
  assert.equal(prompts, 3);
  assert.equal(await gate.permissions.status(), 'granted');
});

test('a new revocation wins while recovering an already denied permission', async t => {
  let recover = false;
  const gate = new ApprovalGate(async () => {
    if (recover) await gate.permissions.save('denied');
    return recover;
  }, await storage(t));
  const signal = new AbortController().signal;
  await assert.rejects(gate.approve(plan, signal), { code: 'APPROVAL_DECLINED' });
  recover = true;
  await assert.rejects(gate.reconsider(signal), { code: 'APPROVAL_DECLINED' });
  assert.equal(await gate.permissions.status(), 'denied');
});

test('revocation during a pending permission decision cannot be undone by its response', async t => {
  const root = await storage(t);
  const gate = new ApprovalGate(async () => { await gate.permissions.save('denied'); return true; }, root);
  await assert.rejects(gate.approve(plan, new AbortController().signal), { code: 'APPROVAL_DECLINED' });
  assert.equal(await gate.permissions.status(), 'denied');
});

test('cancelled consent is not saved and a changed permission version needs a new decision', async t => {
  const root = await storage(t);
  const controller = new AbortController();
  const gate = new ApprovalGate(async () => { controller.abort(); return true; }, root);
  await assert.rejects(gate.approve(plan, controller.signal), { code: 'APPROVAL_CANCELLED' });
  assert.equal(await gate.permissions.status(), 'missing');
  await gate.permissions.save('granted');
  const value = JSON.parse(await fs.readFile(gate.permissions.file, 'utf8'));
  await fs.writeFile(gate.permissions.file, JSON.stringify({ ...value, version: 0 }));
  let prompts = 0;
  await new ApprovalGate(async () => { prompts++; return true; }, root).approve(plan, new AbortController().signal);
  assert.equal(prompts, 1);
});

test('linked and publicly writable permission records cannot grant access', async t => {
  const root = await storage(t);
  const gate = new ApprovalGate(async () => assert.fail('Unsafe stored permissions must fail closed'), root);
  await gate.permissions.save('granted');
  await fs.chmod(gate.permissions.file, 0o666);
  await assert.rejects(gate.approve(plan, new AbortController().signal), /private, regular file/);
  await fs.chmod(gate.permissions.file, 0o600);
  const target = path.join(root, 'external.json');
  await fs.rename(gate.permissions.file, target);
  await fs.symlink(target, gate.permissions.file);
  await assert.rejects(gate.approve(plan, new AbortController().signal));
});

test('fallback requires the private token and exact origin, and remembers the decision', async t => {
  const gate = new ApprovalGate(async () => undefined, await storage(t));
  const controller = new AbortController();
  const approval = gate.approve(plan, controller.signal);
  let failure;
  void approval.catch(error => { failure = error; });
  // Poll the host-owned pending status, as an MCP continuation would.
  for (let i = 0; !gate.pending?.url && i < 100; i++) await new Promise(resolve => setTimeout(resolve, 10));
  try {
    assert.ok(gate.pending?.url, String(failure ?? 'Approval page did not start'));
    const url = new URL(gate.pending.url), token = url.hash.slice(1);
    const page = await fetch(url);
    assert.equal(page.status, 200);
    const html = await page.text();
    assert.match(html, /Allow TensorViz/);
    assert.doesNotThrow(() => new Function(html.match(/<script[^>]*>([\s\S]*?)<\/script>/)[1]), 'The fallback consent script must parse');
    const call = (origin, credential, value) => fetch(`${url.origin}/decision`, { method: 'POST', headers: {
      Origin: origin, Authorization: `Bearer ${credential}`, 'Content-Type': 'application/json' }, body: JSON.stringify(value) });
    assert.equal((await call('http://attacker.invalid', token, { approve: true })).status, 403);
    assert.equal((await call(url.origin, 'wrong', { approve: true })).status, 403);
    assert.equal((await call(url.origin, token, { approved: true })).status, 400);
    assert.ok(gate.pending);
    assert.equal((await call(url.origin, token, { approve: true })).status, 200);
    await approval;
    assert.equal(gate.pending, undefined);
    assert.equal(await gate.permissions.status(), 'granted');
    await gate.approve({ ...plan, operation: 'save_setup' }, controller.signal);
    await assert.rejects(call(url.origin, token, { approve: true }));
  } finally { controller.abort(); gate.dispose(); await approval.catch(() => {}); }
});
