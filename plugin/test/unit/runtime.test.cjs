const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { ManagedRuntime } = require('../../out/host/runtime.cjs');
const { withResourceLock } = require('../../out/host/locks.cjs');

test('a second host waits for shared provisioning, reuses its result and can cancel without disturbing its owner', async () => {
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-runtime-wait-')));
  let approved;
  const runtime = new ManagedRuntime(path.resolve(__dirname, '../../../plugins/tensorviz/backend'), path.join(root, 'state'),
    { approve: async plan => { approved = plan; } });
  try {
    if (!runtime.availability().available) {
      await assert.rejects(runtime.provisionService(root, new AbortController().signal), { code: 'MANAGED_RUNTIME_UNSUPPORTED' });
      return;
    }
    const location = await runtime.location(root, 'service');
    let releaseOwner, ownerAcquired;
    const acquired = new Promise(resolve => { ownerAcquired = resolve; });
    const owner = withResourceLock(location.directory, new AbortController().signal, async () => {
      ownerAcquired();
      await new Promise(resolve => { releaseOwner = resolve; });
      await fs.mkdir(path.dirname(location.interpreter), { recursive: true });
      await fs.writeFile(location.interpreter, 'readiness must not execute Python');
      await fs.writeFile(location.record, location.owner);
      await fs.writeFile(location.ready, location.owner);
    });
    await acquired;
    const controller = new AbortController();
    const cancelled = runtime.provisionService(root, controller.signal);
    while (!approved) await new Promise(resolve => setTimeout(resolve, 5));
    controller.abort();
    await assert.rejects(cancelled, { name: 'AbortError' });
    await assert.rejects(withResourceLock(location.directory, new AbortController().signal, async () => {}), { code: 'RESOURCE_BUSY' });
    let completed = false;
    const waiter = runtime.provisionService(path.join(root, 'another'), new AbortController().signal).then(result => { completed = true; return result; });
    await new Promise(resolve => setTimeout(resolve, 150));
    assert.equal(completed, false, 'the second host must not fail or install into a locked environment');
    releaseOwner(); await owner;
    assert.equal(await waiter, location.interpreter);
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});

test('managed provisioning writes nothing on decline and trusts only matching completed ownership records', async () => {
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-runtime-contract-')));
  const storage = path.join(root, 'state');
  let reviewed;
  const decline = { approve: async plan => { reviewed = plan; throw new Error('declined'); } };
  const runtime = new ManagedRuntime(path.resolve(__dirname, '../../../plugins/tensorviz/backend'), storage, decline);
  try {
    if (!runtime.availability().available) {
      await assert.rejects(runtime.provision(root, new AbortController().signal), { code: 'MANAGED_RUNTIME_UNSUPPORTED' });
      assert.equal(reviewed, undefined);
    } else {
      await assert.rejects(runtime.provision(root, new AbortController().signal), /declined/);
      await assert.rejects(fs.stat(storage), { code: 'ENOENT' });
      assert.equal(reviewed.facts.sha256.length, 64);
      assert.ok(reviewed.facts.requirements.includes('torch==2.14.1'));
      assert.ok(!reviewed.facts.requirements.some(name => name.startsWith('fastapi')), 'managed capture Python has only worker requirements');
      const directory = path.dirname(path.dirname(path.dirname(reviewed.interpreter)));
      const digest = value => createHash('sha256').update(value).digest('hex');
      const owner = JSON.stringify({ workspaceRoot: root, runtime: reviewed.facts.sha256,
        requirements: digest(JSON.stringify(reviewed.facts.requirements)) });
      await fs.mkdir(path.dirname(reviewed.interpreter), { recursive: true });
      await fs.writeFile(reviewed.interpreter, 'never executed by a readiness read');
      await fs.writeFile(path.join(directory, 'owner.json'), owner);
      await fs.writeFile(path.join(directory, 'ready.json'), owner);
      assert.equal(await runtime.readyCapture(root), reviewed.interpreter);
      assert.equal(await runtime.readyCapture(path.join(root, 'another')), undefined, 'another checkout does not borrow managed project libraries');
      assert.equal(await runtime.provision(root, new AbortController().signal), reviewed.interpreter);
      await fs.writeFile(path.join(directory, 'owner.json'), '{}');
      await assert.rejects(runtime.provision(root, new AbortController().signal), /declined/);
    }
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});

test('the tool service is shared across workspaces and retains server libraries outside capture Python', async () => {
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-service-contract-')));
  let reviewed;
  const runtime = new ManagedRuntime(path.resolve(__dirname, '../../../plugins/tensorviz/backend'), path.join(root, 'state'),
    { approve: async plan => { reviewed = plan; throw new Error('declined'); } });
  try {
    if (!runtime.availability().available) {
      await assert.rejects(runtime.provisionService(root, new AbortController().signal), { code: 'MANAGED_RUNTIME_UNSUPPORTED' });
      return;
    }
    await assert.rejects(runtime.provisionService(root, new AbortController().signal), /declined/);
    await assert.rejects(fs.stat(path.join(root, 'state')), { code: 'ENOENT' });
    assert.ok(reviewed.facts.requirements.some(name => name.startsWith('fastapi')));
    const digest = value => createHash('sha256').update(value).digest('hex');
    const owner = JSON.stringify({ role: 'service', runtime: reviewed.facts.sha256, requirements: digest(JSON.stringify(reviewed.facts.requirements)) });
    const directory = path.dirname(path.dirname(path.dirname(reviewed.interpreter)));
    await fs.mkdir(path.dirname(reviewed.interpreter), { recursive: true });
    await fs.writeFile(reviewed.interpreter, 'readiness must not execute Python');
    await fs.writeFile(path.join(directory, 'owner.json'), owner);
    await fs.writeFile(path.join(directory, 'ready.json'), owner);
    assert.equal(await runtime.provisionService(path.join(root, 'another'), new AbortController().signal), reviewed.interpreter);
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});

test('cancelled runtime progress disappears immediately and cannot replace a newer operation', async () => {
  const https = require('node:https');
  const { EventEmitter } = require('node:events');
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-runtime-progress-')));
  const runtime = new ManagedRuntime(path.resolve(__dirname, '../../../plugins/tensorviz/backend'), path.join(root, 'state'), { approve: async () => {} });
  const original = https.get;
  const waitForDownload = async () => {
    for (let n = 0; n < 100 && !runtime.progress?.includes('Connecting'); n++) await new Promise(resolve => setTimeout(resolve, 5));
    assert.match(runtime.progress, /Connecting/);
  };
  https.get = (_url, options) => {
    const request = new EventEmitter();
    request.setTimeout = () => request;
    request.destroy = error => { if (error) queueMicrotask(() => request.emit('error', error)); return request; };
    options.signal.addEventListener('abort', () => request.destroy(new Error('aborted')), { once: true });
    return request;
  };
  try {
    if (!runtime.availability().available) {
      await assert.rejects(runtime.provision(root, new AbortController().signal), { code: 'MANAGED_RUNTIME_UNSUPPORTED' });
      return;
    }
    const firstController = new AbortController(), secondController = new AbortController();
    const first = assert.rejects(runtime.provision(path.join(root, 'first'), firstController.signal));
    await waitForDownload();
    firstController.abort();
    assert.equal(runtime.progress, undefined);
    const second = assert.rejects(runtime.provision(path.join(root, 'second'), secondController.signal));
    await first;
    await waitForDownload();
    assert.match(runtime.progress, /Connecting/);
    secondController.abort(); await second;
    assert.equal(runtime.progress, undefined);
  } finally { https.get = original; await fs.rm(root, { recursive: true, force: true }); }
});
