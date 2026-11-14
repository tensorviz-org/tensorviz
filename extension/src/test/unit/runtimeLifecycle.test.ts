/** Step 4 behavior: lock arbitration across extension hosts, adoption instead of a second
 *  build, atomic verified replacement (failure/cancellation preserve a working runtime),
 *  additive dependency updates, crash leftovers, and cleanup confined to the runtime root. */
import * as assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { Readable } from 'node:stream';

import { ManagedRuntime, confinedRm } from '../../runtime';
import { Transport } from '@tensorviz/graph-host/runtimeAcquire';
import { stockDownloadFixture } from './stockRuntimeFixture';

let temp: string;

const noReport = (): void => undefined;

function deadPid(): number {
  return spawnSync(process.execPath, ['-e', 'process.exit(0)']).pid ?? 0;
}

function holdLock(runtime: ManagedRuntime, pid: number): string {
  const lockDir = path.join(runtime.root, 'lock');
  fs.mkdirSync(lockDir, { recursive: true });
  fs.writeFileSync(
    path.join(lockDir, 'owner.json'),
    JSON.stringify({ pid, started: new Date().toISOString() }),
  );
  return lockDir;
}

suite('runtime lifecycle', () => {
  setup(function () {
    if (process.platform === 'win32') {
      // The scripted interpreter is a POSIX shell script.
      this.skip();
    }
    temp = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'tensorviz-lifecycle-'));
  });

  teardown(() => {
    fs.rmSync(temp, { recursive: true, force: true });
  });

  test('two extension hosts result in one build, and the loser adopts it', async () => {
    // Host A downloads slowly enough that host B is guaranteed to reach the lock while A holds
    // it. The promise for B starts 100 ms in; A's deliberate 500 ms keeps the window open.
    const hostA = stockDownloadFixture(temp, async (url) => {
      hostA.transportCalls.push(url);
      await new Promise((resolve) => setTimeout(resolve, 500));
      return {
        status: 200,
        headers: { 'content-length': String(fs.statSync(hostA.assetPath).size) },
        stream: fs.createReadStream(hostA.assetPath),
      };
    });
    const hostB: ManagedRuntime = new ManagedRuntime({
      globalStorageDir: hostA.globalStorageDir,
      extensionPath: hostA.extensionPath,
      remoteName: undefined,
      backendDir: () => hostA.backendDir,
      transport: async () => {
        throw new Error('host B must never download: the winner already did');
      },
    });
    const attemptA = hostA.runtime.provision(noReport);
    await new Promise((resolve) => setTimeout(resolve, 100));
    const [resultA, resultB] = await Promise.all([attemptA, hostB.provision(noReport)]);

    assert.ok(resultA.ok, resultA.ok ? '' : `${resultA.message} ${resultA.detail}`);
    assert.ok(resultB.ok, resultB.ok ? '' : `${resultB.message} ${resultB.detail}`);
    assert.equal(resultA.ok ? resultA.interpreter : '', hostA.runtime.interpreter);
    assert.equal(resultB.ok ? resultB.interpreter : '', hostA.runtime.interpreter);
    assert.equal(hostA.transportCalls.length, 1, 'exactly one host downloaded the runtime');
    assert.equal(
      (await hostB.check()).state,
      'ready',
      'the loser must converge on the same ready runtime',
    );
  });

  test('two invocations in one window converge the same way two windows would', async () => {
    const fx = stockDownloadFixture(temp);
    const [first, second] = await Promise.all([
      fx.runtime.provision(noReport),
      fx.runtime.provision(noReport),
    ]);
    assert.ok(first.ok && second.ok);
    assert.equal(
      fx.transportCalls.length,
      1,
      'a second invocation while the first held the lock must not re-download',
    );
  });

  test('a failed repair preserves the previously working runtime', async () => {
    const fx = stockDownloadFixture(temp);
    const first = await fx.runtime.provision(noReport);
    assert.ok(first.ok);

    // Serve bytes that do not hash to the pin: the download cache is wiped for the test by
    // removing the cached archive, so the tampered transport is genuinely consulted.
    const cached = fs
      .readdirSync(fx.runtime.downloads)
      .map((f) => path.join(fx.runtime.downloads, f));
    for (const file of cached) {
      fs.rmSync(file);
    }
    const badTransport: Transport = async () => ({
      status: 200,
      headers: { 'content-length': '16' },
      stream: requireStreamOf(Buffer.from('not the pinned bytes')),
    });
    const repairing = new ManagedRuntime({
      globalStorageDir: fx.globalStorageDir,
      extensionPath: fx.extensionPath,
      remoteName: undefined,
      backendDir: () => fx.backendDir,
      transport: badTransport,
    });
    const attempted = await repairing.provision(noReport, () => false, { force: true });
    assert.equal(attempted.ok, false);
    assert.match(attempted.ok ? '' : attempted.message + attempted.detail, /checksum/i);

    assert.ok(
      fs.existsSync(fx.runtime.interpreter),
      'the failed attempt must not destroy the runtime that was serving',
    );
    assert.equal(
      (await fx.runtime.check()).state,
      'ready',
      'the old runtime must still verify and answer ready after the failed repair',
    );
    assert.ok(fs.existsSync(fx.runtime.stampPath), 'the old stamp survives the failed repair');
  });

  test('cancellation before the swap leaves the working runtime in service', async () => {
    const fx = stockDownloadFixture(temp);
    const first = await fx.runtime.provision(noReport);
    assert.ok(first.ok);

    const cancelled = await fx.runtime.provision(noReport, () => true, { force: true });
    assert.equal(cancelled.ok, false);
    assert.match(cancelled.ok ? '' : cancelled.message, /cancelled/i);
    assert.equal((await fx.runtime.check()).state, 'ready');
  });

  test('a lock whose owner process is gone is recovered, not waited on forever', async () => {
    const fx = stockDownloadFixture(temp);
    const lockDir = holdLock(fx.runtime, deadPid());
    const result = await fx.runtime.provision(noReport, () => false, { lockWaitMs: 5_000 });
    assert.ok(result.ok, 'a dead owner must not block provisioning');
    assert.ok(!fs.existsSync(lockDir), 'the lock is released after provisioning');
  });

  test('a live lock is waited on only for a bounded time, then reported', async () => {
    const fx = stockDownloadFixture(temp);
    holdLock(fx.runtime, process.pid);
    const result = await fx.runtime.provision(noReport, () => false, { lockWaitMs: 500 });
    assert.equal(result.ok, false);
    assert.match(result.ok ? '' : result.message, /another window/i);
    assert.ok(
      !fs.existsSync(fx.runtime.interpreter),
      'a contender that gave up must not leave a half-built runtime',
    );
  });

  test('cancelling while another host holds the lock reports cancellation', async () => {
    const fx = stockDownloadFixture(temp);
    holdLock(fx.runtime, process.pid);
    const result = await fx.runtime.provision(noReport, () => true, { lockWaitMs: 30_000 });
    assert.equal(result.ok, false);
    assert.match(result.ok ? '' : result.message, /cancelled/i);
  });

  test('crash leftovers from a dead host are cleaned up by the next provision', async () => {
    const fx = stockDownloadFixture(temp);
    fs.mkdirSync(path.join(fx.runtime.root, 'staging-python-999999'), { recursive: true });
    fs.mkdirSync(path.join(fx.runtime.root, 'old-venv-999999'), { recursive: true });
    const result = await fx.runtime.provision(noReport);
    assert.ok(result.ok);
    const leftovers = fs
      .readdirSync(fx.runtime.root)
      .filter((entry) => /^(staging|old)-/.test(entry));
    assert.deepEqual(leftovers, [], 'the winner removed the dead host’s staging dirs');
  });

  test('the staged venv is repointed at its final home before promotion', async () => {
    const fx = stockDownloadFixture(temp);
    const result = await fx.runtime.provision(noReport);
    assert.ok(result.ok);
    const pyvenvCfg = fs.readFileSync(path.join(fx.runtime.venv, 'pyvenv.cfg'), 'utf8');
    assert.match(
      pyvenvCfg,
      new RegExp(escapeRegExp(`home = ${fx.runtime.base}/bin`)),
      'pyvenv.cfg must name the PROMOTED base interpreter, not the staging path',
    );
    const pipScript = fs.readFileSync(path.join(fx.runtime.venv, 'bin', 'pip'), 'utf8');
    assert.ok(
      pipScript.startsWith(`#!${fx.runtime.venv}/bin/python`),
      'bin/ entry points must quote the promoted venv path',
    );
    assert.ok(!pyvenvCfg.includes('staging-'), 'no staging path may leak into service');
    assert.ok(!pipScript.includes('staging-'), 'no staging path may leak into service');
  });

  test('the staged venv interpreter shims are repointed before promotion, never left dangling', async () => {
    const fx = stockDownloadFixture(temp);
    const result = await fx.runtime.provision(noReport);
    assert.ok(result.ok);
    // A real venv shims bin/python3 -> <base>/bin/python3 by absolute symlink; renaming the
    // staging tree into service dangles it unless the rewrite re-points the target. Caught on
    // the 2026-09-10 installed-VSIX lane: provision "succeeded", every managed probe ENOENT'd,
    // and the backend silently fell through to the shell's python.
    const shim = path.join(fx.runtime.venv, 'bin', 'python3');
    const target = fs.readlinkSync(shim);
    assert.ok(
      !/staging-/.test(target),
      `bin/python3 re-pointed at the promoted base, was: ${target}`,
    );
    assert.ok(
      fs.existsSync(shim),
      'the promoted interpreter shim must RESOLVE (existsSync follows symlinks)',
    );
  });

  async function existingRuntime(helperCode = 0, probeCode = 0) {
    const fx = stockDownloadFixture(temp);
    assert.ok((await fx.runtime.provision(noReport)).ok);
    const marker = path.join(fx.runtime.venv, 'keep-existing-packages');
    fs.writeFileSync(marker, 'Torch and project extras must survive');
    const stamp = fs.readFileSync(fx.runtime.stampPath, 'utf8');
    fs.appendFileSync(path.join(fx.backendDir, 'requirements-standalone.txt'), 'websockets==15.0.1\n');
    fs.writeFileSync(fx.runtime.interpreter, [
      '#!/bin/sh',
      'printf "%s\\n" "$*" >> "$0.calls"',
      'case "$1" in',
      `  -c) exit ${probeCode} ;;`,
      'esac',
      `if [ "$2" = "update-managed" ]; then exit ${helperCode}; fi`,
      'exit 99',
      '',
    ].join('\n'));
    return { ...fx, marker, stamp };
  }

  test('a dependency update reuses the venv instead of rebuilding or reinstalling Torch', async () => {
    const fx = await existingRuntime();
    assert.equal((await fx.runtime.check()).state, 'absent');
    const [first, second] = await Promise.all([
      fx.runtime.provision(noReport), fx.runtime.provision(noReport),
    ]);
    assert.ok(first.ok && second.ok);
    assert.equal(first.interpreter, fx.runtime.interpreter);
    assert.ok(fs.existsSync(fx.marker), 'existing packages remain in the same venv');
    const calls = fs.readFileSync(fx.runtime.interpreter + '.calls', 'utf8');
    assert.equal(calls.match(/update-managed/g)?.length, 1, 'only the lock holder updates');
    assert.ok(!calls.includes('-m venv') && !calls.includes('-m pip'), 'no full rebuild');
    assert.equal(fx.transportCalls.length, 1, 'no new Python download');
    const stamp = JSON.parse(fs.readFileSync(fx.runtime.stampPath, 'utf8'));
    assert.equal(stamp.requirements, fx.runtime.pinnedStamp().requirements);
    assert.equal((await fx.runtime.check()).state, 'ready');
    assert.ok(!fs.readdirSync(fx.runtime.root).some(name => name.startsWith('dependency-update-')));
  });

  for (const [name, helperCode, probeCode] of [
    ['dependency conflict', 1, 0], ['backend verification failure', 0, 1],
  ] as const) {
    test(`${name} preserves the environment and does not certify or rebuild it`, async () => {
      const fx = await existingRuntime(helperCode, probeCode);
      const result = await fx.runtime.provision(noReport);
      assert.equal(result.ok, false);
      assert.ok(fs.existsSync(fx.marker));
      assert.equal(fs.readFileSync(fx.runtime.stampPath, 'utf8'), fx.stamp);
      assert.equal((await fx.runtime.check()).state, 'absent');
      assert.ok(!fs.readdirSync(fx.runtime.root).some(entry => /^(dependency-update|staging)-/.test(entry)));
    });
  }

  test('cancelling a dependency update starts no installer and leaves the old stamp', async () => {
    const fx = await existingRuntime();
    const result = await fx.runtime.provision(noReport, () => true);
    assert.equal(result.ok, false);
    assert.ok(!fs.existsSync(fx.runtime.interpreter + '.calls'));
    assert.ok(fs.existsSync(fx.marker));
    assert.equal(fs.readFileSync(fx.runtime.stampPath, 'utf8'), fx.stamp);
  });

  test('cancelling an active update stops its installer children', async function () {
    this.timeout(5_000);
    const fx = await existingRuntime();
    fs.writeFileSync(fx.runtime.interpreter, [
      '#!/bin/sh',
      'touch "$0.started"',
      '(sleep 1; touch "$0.child-finished") &',
      'wait',
      '',
    ].join('\n'));
    const result = await fx.runtime.provision(noReport, () => fs.existsSync(fx.runtime.interpreter + '.started'));
    assert.equal(result.ok, false);
    await new Promise(resolve => setTimeout(resolve, 1_100));
    assert.ok(!fs.existsSync(fx.runtime.interpreter + '.child-finished'), 'installer child was stopped');
    assert.equal(fs.readFileSync(fx.runtime.stampPath, 'utf8'), fx.stamp);
    assert.ok(fs.existsSync(fx.marker));
  });

  test('cleanup cannot escape the runtime root', () => {
    const root = path.join(temp, 'root');
    const inside = path.join(root, 'dir');
    fs.mkdirSync(inside, { recursive: true });
    fs.writeFileSync(path.join(inside, 'f'), 'x');
    confinedRm(root, inside, { recursive: true });
    assert.ok(!fs.existsSync(inside));

    assert.throws(() => confinedRm(root, root, { recursive: true }), /not inside/);
    assert.throws(() => confinedRm(root, path.join(root, '..'), { recursive: true }), /not inside/);
    assert.throws(
      () => confinedRm(root, path.join(root, '..', 'root-but-other'), { recursive: true }),
      /not inside/,
    );
  });
});

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function requireStreamOf(data: Buffer): Readable {
  return Readable.from([data]);
}

