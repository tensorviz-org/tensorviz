import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';

import {
  BACKEND_PROBE,
  MIN_BACKEND_PYTHON,
  managedInterpreter,
  runtimeInstallApprovalPending,
  ProbeOutcome,
  ProbeRunner,
  resolveBackendInterpreter,
  withoutTraceback,
} from '../../preflight';
import { ManagedRuntime, PROVISION_STAMP, RUNTIME_STAMP, bundledLayout } from '../../runtime';
import { infoMessages, infoMessageDetails, reset, setInfoResponder, state } from '../vscodeStub';
import { stockDownloadFixture } from './stockRuntimeFixture';

const BACKEND_DIR = path.join(path.sep, 'repos', 'canvas', 'backend');

/** The exact interpreter from the reported launch crash: Conda python 3.7 with torch,
 *  uvicorn and fastapi — and no dotenv. */
const CONDA_37 = path.join(path.sep, 'opt', 'anaconda3', 'envs', 'nlp_course', 'bin', 'python');

/** The probe preflight used before the role separation: three imports and a print. */
const OLD_PROBE = 'import sys, torch, uvicorn, fastapi; print(sys.version.split()[0])';

function outcome(partial: Partial<ProbeOutcome> & { code: number }): ProbeOutcome {
  return { stdout: '', stderr: '', error: undefined, timedOut: false, ...partial };
}

const OK = outcome({ code: 0, stdout: '3.12.14\n' });

/** A runner that fails every candidate with the same stderr body. */
function allFail(stderr: string, code = 1): ProbeRunner {
  return async () => outcome({ code, stderr });
}

const TRACEBACK_DOTENV = [
  'Traceback (most recent call last):',
  '  File "<string>", line 6, in <module>',
  '  File "/repo/backend/server_v2.py", line 50, in <module>',
  '    from dotenv import load_dotenv',
  "ModuleNotFoundError: No module named 'dotenv'",
].join('\n');

/** A transitive import failure, the shape of the ai/skills/curator incident: the app module
 *  loads, then something it imports cannot be found. */
const TRACEBACK_TRANSITIVE = [
  'Traceback (most recent call last):',
  '  File "<string>", line 6, in <module>',
  '  File "/repo/backend/server_v2.py", line 61, in <module>',
  '    from services.ide.catalog import router',
  '  File "/repo/backend/services/ide/catalog.py", line 12, in <module>',
  '    from evals.contracts import TurnContract',
  "ModuleNotFoundError: No module named 'evals'",
].join('\n');

suite('preflight.backendProbe', () => {
  test('checks the application and the WebSocket server configuration used at launch', () => {
    // The contract this whole suite exists to pin: the probe must ask Python to do the same
    // work the launch asks of it. A list of modules is whatever the author remembered; the
    // dotenv crash is what an incomplete list costs.
    assert.match(BACKEND_PROBE, /\bimport server_v2\b/);
    assert.match(BACKEND_PROBE, /Config\(server_v2\.app/);
    assert.match(BACKEND_PROBE, /\.load\(\)/);
    assert.match(BACKEND_PROBE, /ws_protocol_class is None/);
    assert.match(BACKEND_PROBE, /WebSocket support is missing/);
  });

  test('gates on a supported Python version before any third-party import', () => {
    const [major, minor] = MIN_BACKEND_PYTHON;
    assert.match(BACKEND_PROBE, new RegExp(`version_info\\[:2\\] < \\(${major}, ${minor}\\)`));
    assert.match(BACKEND_PROBE, /raise SystemExit/);
  });

  test('refuses the Python 3.7 that passed the old three-module probe', async () => {
    // Reproduction of the reported crash. The conda interpreter answers the OLD probe fine
    // (torch/uvicorn/fastapi all import on it); the new probe must refuse it, both by the
    // version gate and by the application import (dotenv is still missing on it).
    const runner: ProbeRunner = async (_cmd, args) => {
      if (args.join(' ').includes('torch, uvicorn, fastapi')) {
        return OK; // what the old probe saw, and why the crash shipped
      }
      return outcome({
        code: 17,
        stderr: 'the tensorViz backend needs Python 3.11 or newer; this is 3.7.9',
      });
    };
    // Demonstrate the runner really is the 3.7 environment the bug report came from:
    assert.equal((await runner(CONDA_37, ['-c', OLD_PROBE], { timeoutMs: 1 })).code, 0);

    const result = await resolveBackendInterpreter(BACKEND_DIR, { probeRunner: runner });

    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.match(result.detail, /3\.11 or newer/);
  });

  test('refuses a candidate missing dotenv before the backend is spawned', async () => {
    const result = await resolveBackendInterpreter(BACKEND_DIR, {
      probeRunner: allFail(TRACEBACK_DOTENV),
    });

    assert.equal(result.ok, false);
    if (result.ok) return;
    // Names the actual missing module, not whichever one the author listed first.
    assert.match(result.message, /dotenv/);
    assert.match(result.message, /backend/);
  });

  test('refuses a transitive server_v2 import failure before the backend is spawned', async () => {
    const result = await resolveBackendInterpreter(BACKEND_DIR, {
      probeRunner: allFail(TRACEBACK_TRANSITIVE),
    });

    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.match(result.message, /evals/);
  });

  test('failure detail names the backend role and never carries a raw traceback', async () => {
    const result = await resolveBackendInterpreter(BACKEND_DIR, {
      probeRunner: allFail(TRACEBACK_DOTENV),
    });

    assert.equal(result.ok, false);
    if (result.ok) return;
    const visible = `${result.message}\n${result.detail}`;
    assert.ok(!visible.includes('Traceback (most recent call last)'), 'frames leaked into the report');
    // The fact survives; the frames do not.
    assert.match(visible, /No module named 'dotenv'/);
  });

  test('a candidate list with no runtimes reports once, per candidate, in precedence order', async () => {
    state.configuration.set('tensorviz.pythonPath', '');
    const result = await resolveBackendInterpreter(BACKEND_DIR, {
      probeRunner: allFail('not on PATH'),
    });

    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.match(result.detail, /python3\s+->/);
    assert.match(result.detail, /Interpreters tried, in order:/);
  });
  test('service selection never consumes the project Python setting or active editor environment', async () => {
    reset();
    state.configuration.set('tensorviz.pythonPath', '/project/torch-only/python');
    state.extensions.set('ms-python.python', { activate: async () => {
      throw new Error('Service selection must not query project capture Python');
    } });
    const probed: string[] = [];
    const result = await resolveBackendInterpreter(BACKEND_DIR, { probeRunner: async cmd => {
      probed.push(cmd); return cmd === 'python3' ? OK : outcome({ code: 1, stderr: 'not available' });
    } });
    assert.ok(result.ok);
    assert.equal(result.interpreter, 'python3');
    assert.ok(!probed.includes('/project/torch-only/python'));
    reset();
  });
});

suite('preflight.resolveBackendInterpreter with a managed runtime', () => {
  let temp = '';

  setup(() => {
    reset();
    temp = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'tensorviz-preflight-'));
  });

  teardown(() => {
    fs.rmSync(temp, { recursive: true, force: true });
  });

  /** The fork-build fixture from runtime.test.ts, repeated here because what is being tested
   *  is different: not the runtime's own states but how resolution treats a READY one. */
  function managedFixture(): ManagedRuntime {
    const app = path.join(temp, 'app');
    const extensionPath = path.join(app, 'extensions', 'tensorviz');
    const globalStorage = path.join(temp, 'globalStorage');
    fs.mkdirSync(extensionPath, { recursive: true });
    fs.mkdirSync(globalStorage, { recursive: true });
    const { runtime, backend } = bundledLayout(extensionPath);
    fs.mkdirSync(path.join(runtime, 'bin'), { recursive: true });
    fs.mkdirSync(backend, { recursive: true });
    fs.writeFileSync(path.join(runtime, RUNTIME_STAMP), JSON.stringify({ sha256: 'f'.repeat(64) }));
    fs.writeFileSync(path.join(backend, 'requirements-standalone.txt'), 'websockets>=12\n');
    fs.writeFileSync(path.join(backend, 'requirements.txt'), 'torch==2.7.1\n');

    const managed = new ManagedRuntime({ globalStorageDir: globalStorage, extensionPath, remoteName: undefined });
    // A runnable stand-in for the venv interpreter: runtime.check() spawns it for real.
    const interpreter = managed.interpreter;
    fs.mkdirSync(path.dirname(interpreter), { recursive: true });
    fs.writeFileSync(interpreter, '#!/bin/sh\nexit 0\n', { mode: 0o755 });
    fs.writeFileSync(
      path.join(managed.root, PROVISION_STAMP),
      JSON.stringify(managed.pinnedStamp()),
    );
    return managed;
  }

  test('an unusable project interpreter does not block a ready managed backend', async function () {
    if (process.platform === 'win32') {
      this.skip();
    }
    const managed = managedFixture();
    // The broken environment from the bug report, typed into the trace-interpreter setting.
    state.configuration.set('tensorviz.pythonPath', CONDA_37);
    const runner: ProbeRunner = async (cmd) =>
      cmd === managed.interpreter
        ? OK
        : outcome({ code: 1, stderr: TRACEBACK_DOTENV });

    const result = await resolveBackendInterpreter(BACKEND_DIR, {
      runtime: managed,
      probeRunner: runner,
    });

    assert.equal(result.ok, true);
    if (!result.ok) return;
    assert.equal(result.interpreter, managed.interpreter);
    assert.match(result.banner, /managed runtime/);
  });

  test('tensorviz.pythonPath cannot override a ready managed backend in a packaged extension', async function () {
    if (process.platform === 'win32') {
      this.skip();
    }
    const managed = managedFixture();
    // A perfectly GOOD typed path: it probes clean, and must still lose. The setting is the
    // project/trace interpreter; the packaged backend is tensorViz's.
    state.configuration.set('tensorviz.pythonPath', '/opt/explicit/bin/python');

    const result = await resolveBackendInterpreter(BACKEND_DIR, {
      runtime: managed,
      probeRunner: async () => OK,
    });

    assert.equal(result.ok, true);
    if (!result.ok) return;
    assert.equal(result.interpreter, managed.interpreter);
  });

  test('a managed runtime that fails validation is reported with its role named', async function () {
    if (process.platform === 'win32') {
      this.skip();
    }
    const managed = managedFixture();
    const result = await resolveBackendInterpreter(BACKEND_DIR, {
      runtime: managed,
      probeRunner: allFail(TRACEBACK_DOTENV),
    });

    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.match(result.detail, /\(tensorViz managed backend runtime\)/);
    // A managed runtime that exists but does not verify can be rebuilt; the failure must
    // point at repair, not at fixing the user's environments by hand.
    assert.equal(result.repairRuntime, true);
  });
});

suite('preflight.withoutTraceback', () => {
  test('leaves ordinary detail untouched', () => {
    const text = 'the backend exited before becoming ready (code 1)';
    assert.equal(withoutTraceback(text), text);
  });

  test('keeps the lead-in and the actual error, drops the frames', () => {
    const text = `the backend printed:\n\n${TRACEBACK_DOTENV}`;
    const cleaned = withoutTraceback(text);
    assert.match(cleaned, /the backend printed:/);
    assert.match(cleaned, /No module named 'dotenv'/);
    assert.ok(!cleaned.includes('File "/repo/backend/server_v2.py"'));
    assert.match(cleaned, /tensorViz log/);
  });

  test('a message that is ONLY a traceback still answers with the error line', () => {
    const cleaned = withoutTraceback(TRACEBACK_TRANSITIVE);
    assert.match(cleaned, /^ModuleNotFoundError: No module named 'evals'/);
    assert.match(cleaned, /tensorViz log/);
  });
});

/** The first-use contract gate. A stock install must offer the disclosed Install/Cancel
 *  BEFORE one byte of the runtime is downloaded, and a "yes" must be the thing that starts it
 *  — never a dismiss, never activation. */
suite('preflight.managedInterpreter first-install consent', () => {
  let temp = '';

  setup(() => {
    reset();
    temp = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'tensorviz-consent-'));
  });
  teardown(() => {
    fs.rmSync(temp, { recursive: true, force: true });
    setInfoResponder(undefined);
  });

  test('the runtime approval remains observable while the native modal is unanswered', async () => {
    const vscode = require('../vscodeStub'), original = vscode.window.showInformationMessage;
    const fixture = stockDownloadFixture(temp);
    let answer!: (value: string | undefined) => void;
    vscode.window.showInformationMessage = () => new Promise<string | undefined>(resolve => { answer = resolve; });
    let pending: ReturnType<typeof managedInterpreter> | undefined;
    try {
      pending = managedInterpreter({ runtime: fixture.runtime });
      while (!answer) await new Promise<void>(resolve => setImmediate(resolve));
      assert.equal(runtimeInstallApprovalPending(), true);
      assert.equal(fixture.transportCalls.length, 0);
      answer(undefined);
      assert.equal((await pending).cancelled, true);
      assert.equal(runtimeInstallApprovalPending(), false);
    } finally { answer?.(undefined); await pending; vscode.window.showInformationMessage = original; }
  });

  test('declining installs nothing: no download, no interpreter, no stamp', async function () {
    if (process.platform === 'win32') {
      this.skip();
    }
    // The stub default already answers undefined (dismissed), so NOT setting a responder is
    // the most faithful "the user pressed nothing" case exerciseable in a unit host.
    const fixture = stockDownloadFixture(temp);
    let probes = 0;
    const result = await resolveBackendInterpreter(BACKEND_DIR, {
      runtime: fixture.runtime,
      probeRunner: async () => { probes += 1; return OK; },
    });
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.match(result.detail, /declined/);
    assert.equal(probes, 0, 'Declining service setup cannot silently execute another Python');
    assert.equal(fixture.transportCalls.length, 0, 'the refusal never touched the network');
    assert.ok(!fs.existsSync(fixture.runtime.interpreter));
    assert.equal(fixture.runtime.previouslyProvisioned, false);
  });

  test('Install with the disclosure sizes resumes into a ready managed backend', async function () {
    if (process.platform === 'win32') {
      this.skip();
    }
    const fixture = stockDownloadFixture(temp);
    setInfoResponder((_message, actions) => {
      assert.deepEqual(actions, ['Install'], 'native Cancel must not be duplicated by a literal action');
      return 'Install';
    });
    const result = await resolveBackendInterpreter(BACKEND_DIR, {
      runtime: fixture.runtime,
      probeRunner: async (cmd) =>
        cmd === fixture.runtime.interpreter ? OK : outcome({ code: 1, stderr: 'nope' }),
    });
    assert.equal(result.ok, true);
    if (!result.ok) return;
    assert.equal(result.interpreter, fixture.runtime.interpreter);
    assert.equal(fixture.transportCalls.length, 1, 'one download happened after Install');
    // The disclosure itself: what the user read before saying yes. Sizes and the no-touch
    // promise are the contract, so they are asserted, not just presence.
    const notice = [...infoMessages, ...infoMessageDetails].join('\n');
    assert.match(notice, /~128 MB/);
    assert.match(notice, /project environment is never touched/);
  });

  test('managed download failure keeps the actual safe cause ahead of fallback diagnostics', async () => {
    const runtime = {
      check: async () => ({ state: 'broken', why: 'missing runtime' }),
      provision: async () => ({ ok: false,
        message: 'The runtime download could not connect.',
        detail: 'https://user:password@github.com/archive?signature=secret\n\nError: ETIMEDOUT; Bearer private-token',
      }),
    } as unknown as ManagedRuntime;
    const result = await resolveBackendInterpreter(BACKEND_DIR, { runtime, probeRunner: allFail('old Python') });
    assert.ok(!result.ok);
    if (result.ok) return;
    assert.equal(result.message, 'The runtime download could not connect.');
    assert.match(result.detail, /ETIMEDOUT/);
    assert.match(result.detail, /Repair runtime/);
    assert.match(result.detail, /project Python is separate/);
    assert.doesNotMatch(result.detail, /password|signature=|private-token|IDE ships|docs\/local-setup|python3 -m venv/);
    assert.ok(result.detail.indexOf('ETIMEDOUT') < result.detail.indexOf('Interpreters tried'));
  });

  test('cancelled provisioning cannot launch fallback probes or an error recovery flow', async () => {
    const runtime = {
      check: async () => ({ state: 'broken', why: 'missing runtime' }),
      provision: async () => ({ ok: false, cancelled: true, message: 'Setup cancelled.', detail: 'cancelled by user' }),
    } as unknown as ManagedRuntime;
    let probes = 0;
    const result = await resolveBackendInterpreter(BACKEND_DIR, { runtime,
      probeRunner: async () => { probes++; return OK; },
    });
    assert.ok(!result.ok && result.cancelled);
    assert.equal(probes, 0);
  });

  test('an already-provisioned stock runtime never re-asks even without a responder', async function () {
    if (process.platform === 'win32') {
      this.skip();
    }
    const fixture = stockDownloadFixture(temp);
    assert.equal(
      (
        await fixture.runtime.provision(() => {})
      ).ok,
      true,
      'first provisioning (consent bypassed at this level — it is preflight that asks)',
    );
    infoMessages.length = 0;
    const managed = await managedInterpreter({ runtime: fixture.runtime });
    assert.equal(managed.interpreter, fixture.runtime.interpreter);
    assert.equal(infoMessages.length, 0, 'no prompt fired for a ready runtime');
  });
});

