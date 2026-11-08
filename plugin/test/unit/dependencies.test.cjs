const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { installStandaloneDependencies } = require('../../out/host/dependencies.cjs');
const { ApprovalGate } = require('../../out/host/approval.cjs');

for (const scenario of ['approved', 'declined', 'changed', 'source-declined']) {
  test(`standalone dependency plan: ${scenario}`, async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-dependency-'));
    const calls = [], reviews = [];
    let current = true;
    const gate = new ApprovalGate(async message => {
      reviews.push(message);
      assert.ok(message.includes(root));
      if (scenario === 'changed') current = false;
      return !scenario.includes('declined');
    }, path.join(root, 'state'));
    const options = { python: '/exact/capture/python', managerPython: '/exact/tool/python', backendDir: '/bundled/backend', prefix: root,
      workspaceRoot: '/exact/worktree', storage: path.join(root, 'temporary'), packages: ['humanize==4.12.3'],
      signal: new AbortController().signal, approvals: gate, current: async () => current,
      runner: async (python, _backend, args, _cancelled, isolated) => {
        assert.equal(python, '/exact/tool/python'); assert.equal(isolated, true);
        assert.deepEqual(args.slice(0, 2), ['--target-python', '/exact/capture/python']);
        args = args.slice(2);
        calls.push(args[0]);
        if (args[0] === 'plan-packages') return scenario === 'source-declined'
          ? { ok: false, code: 'NO_COMPATIBLE_WHEEL' }
          : { ok: true, digest: 'frozen-plan', packages: [{ name: 'humanize', version: '4.12.3' }] };
        assert.equal(args[0], 'install'); assert.equal(args[2], 'frozen-plan');
        return { ok: true };
      } };
    try {
      if (scenario === 'approved') await installStandaloneDependencies(options);
      else await assert.rejects(installStandaloneDependencies(options), error =>
        error.code === (scenario === 'changed' ? 'STALE_TARGET' : 'APPROVAL_DECLINED'));
      assert.deepEqual(calls, scenario === 'approved' ? ['plan-packages', 'install'] : ['plan-packages']);
      assert.equal(reviews.length, 1);
      assert.match(reviews[0], scenario === 'source-declined' ? /First requested operation: build dependencies/ : /frozen-plan/);
      assert.deepEqual(await fs.readdir(options.storage), []);
    } finally { gate.dispose(); await fs.rm(root, { recursive: true, force: true }); }
  });
}

test('managed CPU provisioning refuses a missing wheel without source or install approval', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-cpu-wheel-'));
  const calls = [];
  try {
    await assert.rejects(installStandaloneDependencies({ python: '/owned/python', backendDir: '/bundled/backend',
      prefix: root, workspaceRoot: root, storage: path.join(root, 'temporary'), packages: ['torch==2.14.1+cpu'], managedCpu: true,
      signal: new AbortController().signal, current: async () => true,
      approvals: { approve: async () => assert.fail('a missing CPU wheel must not prompt for a source/GPU fallback') },
      runner: async (_python, _backend, args) => {
        calls.push(args[0]); return { ok: false, code: 'NO_COMPATIBLE_WHEEL', message: 'CPU wheel unavailable' };
      },
    }), { code: 'NO_COMPATIBLE_WHEEL' });
    assert.deepEqual(calls, ['plan-packages-managed-cpu']);
    assert.deepEqual(await fs.readdir(path.join(root, 'temporary')), []);
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});
