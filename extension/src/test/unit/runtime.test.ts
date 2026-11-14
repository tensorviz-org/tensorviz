import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { Readable } from 'node:stream';

import { interpreterCandidates, managedInterpreter } from '../../preflight';
import { ManagedRuntime, PROVISION_STAMP, RUNTIME_STAMP, bundledLayout } from '../../runtime';
import { StockFixture, stockDownloadFixture } from './stockRuntimeFixture';

const REPO = path.join(path.sep, 'repos', 'canvas');

/** The sha256 fetch-runtime.sh would have written for the staged tarball. Any hex string does;
 *  what matters is that the provisioned copy records the same one. */
const RUNTIME_SHA = '81a359f1cfadd4da11766534c5913791cea55f26e1bb902cacd2a531bb1e4b2b';

interface Fixture {
  runtime: ManagedRuntime;
  /** <root>/app — stands in for resources/app in a fork bundle. */
  app: string;
  globalStorage: string;
}

let temp: string;

/** Build the on-disk shape a fork build has: a built-in at <app>/extensions/tensorviz, with
 *  runtime/ and backend/ staged beside it by ide/build.sh. */
function fixture(opts: { bundled?: boolean; remoteName?: string } = {}): Fixture {
  const root = fs.mkdtempSync(path.join(temp, 'fx-'));
  const app = path.join(root, 'app');
  const extensionPath = path.join(app, 'extensions', 'tensorviz');
  const globalStorage = path.join(root, 'globalStorage');
  fs.mkdirSync(extensionPath, { recursive: true });
  fs.mkdirSync(globalStorage, { recursive: true });

  if (opts.bundled !== false) {
    const { runtime, backend } = bundledLayout(extensionPath);
    fs.mkdirSync(path.join(runtime, 'bin'), { recursive: true });
    fs.mkdirSync(backend, { recursive: true });
    fs.writeFileSync(path.join(runtime, RUNTIME_STAMP), JSON.stringify({ sha256: RUNTIME_SHA }));
    fs.writeFileSync(path.join(backend, 'requirements-standalone.txt'), 'websockets>=12\n');
    fs.writeFileSync(path.join(backend, 'requirements.txt'), 'torch==2.7.1\n');
  }

  return {
    app,
    globalStorage,
    runtime: new ManagedRuntime({
      globalStorageDir: globalStorage,
      extensionPath,
      remoteName: opts.remoteName,
    }),
  };
}

/** Put a runnable interpreter where the venv's would be. POSIX only — a shell script cannot
 *  stand in for python.exe, and the tests that need one say so. */
function fakeVenvPython(runtime: ManagedRuntime, exitCode: number): void {
  const interpreter = runtime.interpreter;
  fs.mkdirSync(path.dirname(interpreter), { recursive: true });
  fs.writeFileSync(interpreter, `#!/bin/sh\nexit ${exitCode}\n`, { mode: 0o755 });
}

function markProvisioned(runtime: ManagedRuntime): void {
  // The stamp's SHAPE, not its provenance: remote fixtures get provision-marked only so that
  // an interpreter can appear to exist for the check that must then refuse them on the
  // availability gate alone. pinnedStamp() consults availability for exactly the reason this
  // fallback exists, so the two need different answers in the remote case.
  let stamp: { runtime: string; requirements: string };
  try {
    stamp = runtime.pinnedStamp();
  } catch {
    stamp = { runtime: RUNTIME_SHA, requirements: 'fixture' };
  }
  fs.writeFileSync(runtime.stampPath, JSON.stringify(stamp));
}

/** The managed runtime is the interpreter the IDE built, verified and can repair. Everything
 *  here is about the two ways it must NOT be used: when this build did not ship one, and when
 *  the window is attached to a remote whose server build carries none. Both fall back to the
 *  user's environment, which is the secondary-channel path (task.md F3 / D11). */
suite('runtime.ManagedRuntime', () => {
  setup(() => {
    temp = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'tensorviz-runtime-'));
  });

  teardown(() => {
    fs.rmSync(temp, { recursive: true, force: true });
  });

  test('the bundle is found two levels above the built-in, never inside it', () => {
    const layout = bundledLayout(
      path.join(path.sep, 'Apps', 'resources', 'app', 'extensions', 'tensorviz'),
    );
    assert.equal(layout.runtime, path.join(path.sep, 'Apps', 'resources', 'app', 'runtime'));
    assert.equal(layout.backend, path.join(path.sep, 'Apps', 'resources', 'app', 'backend'));
  });

  test('the venv is created under globalStorage, never inside the bundle', () => {
    const { runtime, globalStorage, app } = fixture();
    assert.equal(runtime.root, path.join(globalStorage, 'runtime-cpu2141-v1'));
    assert.ok(runtime.interpreter.startsWith(path.join(globalStorage, 'runtime-cpu2141-v1', 'venv')));
    // The reason this matters: a pyvenv.cfg home= pointing inside a .app breaks the first time
    // the user drags the app to /Applications.
    assert.ok(!runtime.interpreter.startsWith(app));
  });

  test('the CPU policy preserves the old live managed runtime', async () => {
    const { runtime, globalStorage } = fixture();
    const old = path.join(globalStorage, 'runtime', 'venv', 'bin', 'python');
    fs.mkdirSync(path.dirname(old), { recursive: true });
    fs.writeFileSync(old, 'old live interpreter');
    assert.notEqual(runtime.interpreter, old);
    assert.equal((await runtime.check()).state, 'absent');
    assert.equal(fs.readFileSync(old, 'utf8'), 'old live interpreter');
  });

  test('no bundle and no pinned manifest (a bare dev checkout) is unavailable, not an error', async () => {
    // The fork-shape fixture without the bundle stages nothing: no fetch-runtime.sh output,
    // no staged backend. This is the DEVELOPER-HOST answer, not the packaged one —
    // resolveBackendDir must still be allowed to fail here without crashing the runtime.
    const { runtime } = fixture({ bundled: false });
    const status = await runtime.check();
    assert.equal(status.state, 'unavailable');
    assert.match(status.state === 'unavailable' ? status.why : '', /no pinned runtime manifest/);
  });

  test('an unsupported remote kind is unavailable even with a staged runtime', async () => {
    const { runtime } = fixture({ remoteName: 'dev-container' });
    fakeVenvPython(runtime, 0);
    markProvisioned(runtime);
    const status = await runtime.check();
    assert.equal(status.state, 'unavailable');
    assert.match(status.state === 'unavailable' ? status.why : '', /not supported/);
  });

  test('bundled but never provisioned is absent', async () => {
    const { runtime } = fixture();
    const status = await runtime.check();
    assert.equal(status.state, 'absent');
  });

  test('a provisioned runtime that still imports torch and fastapi is ready', async function () {
    if (process.platform === 'win32') {
      this.skip();
    }
    const { runtime } = fixture();
    fakeVenvPython(runtime, 0);
    markProvisioned(runtime);
    const status = await runtime.check();
    assert.equal(status.state, 'ready');
    assert.equal(status.state === 'ready' ? status.interpreter : '', runtime.interpreter);
  });

  test('a provisioned runtime that can no longer import is broken, not ready', async function () {
    if (process.platform === 'win32') {
      this.skip();
    }
    const { runtime } = fixture();
    fakeVenvPython(runtime, 1);
    markProvisioned(runtime);
    const status = await runtime.check();
    // Every launch re-runs this check (task.md F3), so a venv that lost its site-packages to a
    // disk cleaner has to be caught here rather than at the first canvas action.
    assert.equal(status.state, 'broken');
  });

  test('changed backend pins make the provisioned copy stale, not ready', async function () {
    if (process.platform === 'win32') {
      this.skip();
    }
    const { runtime } = fixture();
    fakeVenvPython(runtime, 0);
    markProvisioned(runtime);
    // New pins must trigger compatibility setup. An incompatible service dependency version must be
    // reported rather than silently treating the old environment as current.
    fs.writeFileSync(path.join(runtime.bundled.backend, 'requirements-standalone.txt'), 'websockets==15.0.1\n');
    const status = await runtime.check();
    assert.equal(status.state, 'absent');
    assert.match(status.state === 'absent' ? status.why : '', /dependencies differ/);
  });
});

/** Order is the contract. The managed runtime is the only candidate the IDE can repair, so it
 *  goes first when it exists — and must not appear at all when it does not, because a
 *  non-existent path in the list costs a spawn and reports "not on PATH" for something the
 *  user never chose. */
suite('preflight.interpreterCandidates with the managed runtime', () => {
  setup(() => {
    temp = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'tensorviz-runtime-'));
  });

  teardown(() => {
    fs.rmSync(temp, { recursive: true, force: true });
  });

  test('puts the managed runtime first when it is present and ready', async function () {
    if (process.platform === 'win32') {
      this.skip();
    }
    const { runtime } = fixture();
    fakeVenvPython(runtime, 0);
    markProvisioned(runtime);

    const managed = await managedInterpreter({ runtime });
    const candidates = await interpreterCandidates(REPO, managed.interpreter);
    assert.equal(candidates[0], runtime.interpreter);
    assert.equal(managed.repairable, true);
  });

  test('omits it for unsupported remote kinds', async function () {
    if (process.platform === 'win32') {
      this.skip();
    }
    const { runtime } = fixture({ remoteName: 'wsl' });
    fakeVenvPython(runtime, 0);
    markProvisioned(runtime);

    const managed = await managedInterpreter({ runtime });
    assert.equal(managed.interpreter, undefined);
    const candidates = await interpreterCandidates(REPO, managed.interpreter);
    assert.ok(!candidates.includes(runtime.interpreter));
    // Unsupported hosts must not offer managed repair.
    assert.equal(managed.repairable, false);
  });

  test('omits it in stock VS Code, where no runtime is bundled', async () => {
    const { runtime } = fixture({ bundled: false });
    const managed = await managedInterpreter({ runtime });
    assert.equal(managed.interpreter, undefined);
    const candidates = await interpreterCandidates(REPO, managed.interpreter);
    assert.ok(!candidates.includes(runtime.interpreter));
    assert.equal(managed.repairable, false);
  });

  test('with no runtime at all the candidate list is exactly what it was before F3', async () => {
    const before = await interpreterCandidates(REPO);
    const managed = await managedInterpreter({});
    assert.deepEqual(await interpreterCandidates(REPO, managed.interpreter), before);
  });
});

// ---------------------------------------------------------------------------
// The stock-install half: nothing is bundled; the pinned standalone interpreter is a
// verified download on first use. The "interpreter" is a POSIX shell script that answers
// exactly what provisioning asks of it (the -c version/probe calls, `-m venv`, `-m pip`), so
// the whole flow — download, extract, swap, venv, pip, verify, stamp, check — really runs
// with everything except 'be an actual Python' out of the fake.
// ---------------------------------------------------------------------------

suite('runtime.ManagedRuntime stock download source', () => {
  setup(() => {
    temp = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'tensorviz-runtime-stock-'));
  });
  teardown(() => {
    fs.rmSync(temp, { recursive: true, force: true });
  });

  test('SSH uses the host manifest without downloading on availability or check', async () => {
    const { runtime, transportCalls } = stockDownloadFixture(temp, undefined, 'ssh-remote');
    const available = runtime.availability();
    assert.equal(available.ok, true);
    if (!available.ok) return;
    assert.equal(available.source.kind, 'download');
    assert.equal((await runtime.check()).state, 'absent');
    assert.equal(transportCalls.length, 0);
  });

  test('SSH provisions and reuses the pinned host runtime through the existing lifecycle', async function () {
    if (process.platform === 'win32') this.skip();
    const { runtime, transportCalls } = stockDownloadFixture(temp, undefined, 'ssh-remote');
    assert.equal((await runtime.provision(() => {})).ok, true);
    assert.equal((await runtime.check()).state, 'ready');
    assert.equal((await managedInterpreter({ runtime })).interpreter, runtime.interpreter);
    assert.equal(transportCalls.length, 1);
  });

  test('a packaged stock install on a pinned host is acquireable and absent', async () => {
    const { runtime } = stockDownloadFixture(temp);
    const available = runtime.availability();
    assert.equal(available.ok, true);
    if (!available.ok) return;
    assert.equal(available.source.kind, 'download');
    const status = await runtime.check();
    assert.equal(status.state, 'absent');
    assert.equal(runtime.previouslyProvisioned, false);
  });

  test('a download source without a resolvable backend is unavailable, with the reason', () => {
    const { extensionPath } = stockDownloadFixture(temp);
    const runtime = new ManagedRuntime({
      globalStorageDir: path.join(temp, 'gs-nobackend'),
      extensionPath,
      remoteName: undefined,
      // No backendDir thunk: resolveBackendDir answered nothing this window.
    });
    const available = runtime.availability();
    assert.equal(available.ok, false);
    if (available.ok) return;
    assert.match(available.why, /no backend directory/);
  });

  test('a clean stock provisioning reaches ready with no user environment at all', async function () {
    if (process.platform === 'win32') {
      this.skip(); // the scripted interpreter is a POSIX shell fixture
    }
    const { runtime, transportCalls } = stockDownloadFixture(temp);
    const result = await runtime.provision(() => {});
    assert.equal(result.ok, true);
    assert.ok(result.timingsMs);
    for (const stage of ['download', 'unpack', 'python-check', 'environment', 'dependencies', 'backend-check', 'total']) {
      assert.equal(typeof result.timingsMs?.[stage], 'number', `cold setup must record ${stage}`);
    }
    const timingRecord = JSON.parse(fs.readFileSync(path.join(runtime.root, 'last-setup.json'), 'utf8'));
    assert.equal(timingRecord.outcome, 'ready');
    assert.equal(JSON.stringify(timingRecord).includes(temp), false, 'timings must not disclose project or storage paths');
    assert.equal(transportCalls.length, 1, 'the archive was fetched exactly once');
    assert.ok(fs.existsSync(runtime.interpreter), 'the venv interpreter landed');
    const status = await runtime.check();
    assert.equal(status.state, 'ready');
    assert.equal(runtime.previouslyProvisioned, true);
  });

  test('a tampered download installs nothing runnable and no stamp', async function () {
    if (process.platform === 'win32') {
      this.skip();
    }
    const { runtime } = stockDownloadFixture(temp, async () => ({
      // Correct headers, wrong bytes: the only protection that matters is the pinned sha256.
      status: 200,
      headers: { 'content-length': '8' },
      stream: Readable.from([Buffer.from('not-the-runtime')]),
    }));
    const result = await runtime.provision(() => {});
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.match(result.message, /checksum/);
    const timingRecord = JSON.parse(fs.readFileSync(path.join(runtime.root, 'last-setup.json'), 'utf8'));
    assert.equal(timingRecord.outcome, 'failed');
    assert.equal(typeof timingRecord.timingsMs.download, 'number');
    assert.equal(timingRecord.timingsMs.dependencies, undefined, 'failed integrity cannot progress to package installation');
    assert.ok(!fs.existsSync(runtime.interpreter), 'no interpreter was promoted into the venv');
    assert.equal(runtime.previouslyProvisioned, false);
  });

  test('a retry after a wiped venv reuses the verified download cache (no second download)', async function () {
    if (process.platform === 'win32') {
      this.skip();
    }
    const { runtime, extensionPath, globalStorageDir, backendDir, transportCalls } =
      stockDownloadFixture(temp);
    assert.equal((await runtime.provision(() => {})).ok, true);
    fs.rmSync(runtime.venv, { recursive: true, force: true });
    fs.rmSync(runtime.stampPath, { force: true });

    // Same globalStorage, same extension — a different, FAILING transport. If the cache were
    // not re-hash-checked and reused this would explode; instead it must never be called.
    let called = 0;
    const retry = new ManagedRuntime({
      globalStorageDir,
      extensionPath,
      remoteName: undefined,
      backendDir: () => backendDir,
      transport: async () => {
        called++;
        return { status: 500, headers: {}, stream: {} as never };
      },
    });
    assert.equal((await retry.provision(() => {})).ok, true);
    assert.equal(called, 0, 'the archive came from the verified cache');
    assert.equal(transportCalls.length, 1, 'and the first download was the only one');
  });

  test('capture pins cannot invalidate the service or cause another install', async function () {
    if (process.platform === 'win32') this.skip();
    const { runtime, backendDir } = stockDownloadFixture(temp);
    assert.equal((await runtime.provision(() => {})).ok, true);
    const original = runtime.pinnedStamp();
    fs.writeFileSync(path.join(backendDir, 'requirements.txt'), 'torch==99.0.0\n');
    assert.deepEqual(runtime.pinnedStamp(), original);
    assert.equal((await runtime.check()).state, 'ready');
  });

  test('changed backend pins in stock mode make the provisioned copy stale', async function () {
    if (process.platform === 'win32') {
      this.skip();
    }
    const { runtime, backendDir } = stockDownloadFixture(temp);
    assert.equal((await runtime.provision(() => {})).ok, true);
    fs.writeFileSync(path.join(backendDir, 'requirements-standalone.txt'), 'websockets==15.0.1\n');
    const status = await runtime.check();
    assert.equal(status.state, 'absent');
    assert.match(status.state === 'absent' ? status.why : '', /dependencies differ/);
  });
});

