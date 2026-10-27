import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';

import * as vscode from 'vscode';

import { reset, state } from '../vscodeStub';
import {
  INSTALL_HINT,
  TraceWorker,
  WORKER_PROTOCOL_VERSION,
  parseEnvelope,
  runProcess,
  workerArgs,
  workerEnv,
  workspaceTrusted,
} from '../../worker';

/** A valid envelope, as the worker prints it. */
function envelope(overrides: Record<string, unknown> = {}): string {
  return JSON.stringify({
    protocol_version: WORKER_PROTOCOL_VERSION,
    result: { graph: { nodes: [], edges: [] } },
    spans: {},
    error: null,
    worker: { python: '3.11.15', torch: '2.7.1', elapsed_ms: 12 },
    ...overrides,
  });
}

suite('worker.parseEnvelope', () => {
  test('accepts the envelope the worker promises', () => {
    const parsed = parseEnvelope(`${envelope()}\n`);
    assert.ok(parsed.ok);
    assert.equal(parsed.envelope.worker.torch, '2.7.1');
  });

  test('refuses a different protocol version loudly instead of guessing', () => {
    // The whole point of the version: an envelope from another build may be
    // SHAPED right and mean something else, and merging it would put invented
    // shapes on a canvas the user reads as measurement (task.md §6.3).
    const parsed = parseEnvelope(envelope({ protocol_version: WORKER_PROTOCOL_VERSION + 1 }));
    assert.equal(parsed.ok, false);
    assert.ok(!parsed.ok && parsed.reason.includes(String(WORKER_PROTOCOL_VERSION + 1)));
    assert.ok(!parsed.ok && parsed.detail.includes('different builds'));
  });

  test('refuses JSON that carries no protocol version at all', () => {
    const parsed = parseEnvelope('{"result": {}}');
    assert.equal(parsed.ok, false);
  });

  test('reports the first of a corrupted stdout rather than a bare parse error', () => {
    // The realistic corruption is a `print` in the user's own module scope landing
    // on stdout ahead of the envelope. Showing the head of it is what lets someone
    // recognise their own output.
    const parsed = parseEnvelope('loading checkpoint...\n{"protocol_version": 1}');
    assert.equal(parsed.ok, false);
    assert.ok(!parsed.ok && parsed.detail.includes('loading checkpoint'));
  });

  test('an empty stdout is a failure, not an empty graph', () => {
    const parsed = parseEnvelope('   ');
    assert.equal(parsed.ok, false);
  });
});

suite('worker.workerArgs', () => {
  test('names the file and the class, and defaults to the real tier', () => {
    assert.deepEqual(workerArgs({ file: '/ws/models/resid.py', cls: 'Net' }), [
      '-m',
      'services.ingest.worker',
      '--file',
      '/ws/models/resid.py',
      '--cls',
      'Net',
      '--tier',
      'real',
      '--depth',
      '2',
      '--json',
    ]);
  });

  test('passes an explicit input through verbatim', () => {
    // Verbatim matters: the value may be a Python expression, and quoting or
    // splitting it here would turn `torch.randint(0, 100, (1, 16))` into nonsense.
    const args = workerArgs({ file: 'a.py', cls: 'N', input: 'torch.randn(1, 3, 8, 8)' });
    assert.equal(args[args.indexOf('--input') + 1], 'torch.randn(1, 3, 8, 8)');
  });

  test('a Watch carries its step count to the worker', () => {
    // The whole of P4 rests on this one flag arriving. Without it the approval modal
    // names `--steps 20`, the user agrees to twenty steps, and the process takes one
    // — a consent dialog describing a run that does not happen.
    const args = workerArgs({ file: 'a.py', cls: 'N', kind: 'train_step', callable: 'step', steps: 5 });
    assert.equal(args[args.indexOf('--steps') + 1], '5');
  });

  test('--steps is never sent for a kind that would refuse it', () => {
    // `--steps` with any other `--kind` is a usage error the worker exits 2 on. A
    // request that carries a stale step count must not turn a good model trace into
    // one, so the flag follows the kind and not the field.
    assert.ok(!workerArgs({ file: 'a.py', cls: 'N', steps: 5 }).includes('--steps'));
    assert.ok(
      !workerArgs({ file: 'a.py', kind: 'data', callable: 'loader', steps: 5 }).includes('--steps'),
    );
  });
});

suite('worker private input preparation', () => {
  const request = { file: '/ws/net.py', cls: 'Net', workspace: '/ws', prepareInputs: true,
    execution: { constructorBinding: { args: [], kwargs: {} }, inputs: { binding: { args: [], kwargs: {} } },
      device: 'cpu', mode: 'eval' as const, seed: 0, gradEnabled: false, deterministic: false, cwd: '.' } };

  test('only a private plan can prepare inputs', () => {
    const args = workerArgs(request, 'plan');
    assert.ok(args.includes('--prepare-inputs') && args.includes('--resolved-execution'));
    assert.ok(!args.includes('--execution') && !args.includes('--input'));
    assert.throws(() => workerArgs(request), /private construction plan/);
    assert.throws(() => workerArgs(request, 'preparation'), /private construction plan/);
    assert.throws(() => workerArgs(request, 'candidate'), /Invalid candidate request/);
    assert.throws(() => workerArgs({ ...request, execution: undefined }, 'plan'), /private construction plan/);
  });

  test('config defaults need a separately authorized private input probe', () => {
    assert.ok(!workerArgs(request, 'plan').includes('--prepare-config-defaults'));
    assert.ok(workerArgs({ ...request, prepareConfigDefaults: true }, 'plan').includes('--prepare-config-defaults'));
    assert.throws(() => workerArgs({ ...request, prepareInputs: false, prepareConfigDefaults: true }, 'plan'), /private input preparation/);
    assert.throws(() => workerArgs({ ...request, prepareConfigDefaults: true }, 'candidate'), /Invalid candidate request/);
  });

  test('input preparation cannot take over other execution modes or user overrides', () => {
    assert.throws(() => workerArgs({ ...request, overrides: '{}' }, 'plan'), /private construction plan/);
    assert.throws(() => workerArgs({ ...request, input: 'torch.ones(2)' }, 'plan'), /no legacy input/);
    assert.throws(() => workerArgs({ ...request, tier: 'fake' }, 'plan'), /real model tier/);
    assert.throws(() => workerArgs({ ...request, kind: 'train_step' }, 'plan'), /real model tier/);
    assert.throws(() => workerArgs({ ...request, workspace: undefined }, 'plan'), /workspace/);
  });
});

suite('worker.workerEnv', () => {
  // Takes the BACKEND directory since task.md P0, not a checkout root: the backend the
  // worker imports may be bundled with the app or staged in the vsix, and neither has a
  // `backend` to append.
  const BACKEND = path.join('/app', 'backend');

  test('puts the backend on PYTHONPATH without discarding the user own entries', () => {
    // The user's PYTHONPATH is frequently how their repository imports at all, so
    // replacing it would break the very code the worker exists to run.
    const env = workerEnv({ PYTHONPATH: '/user/src' }, BACKEND);
    assert.equal(env.PYTHONPATH, `${BACKEND}${path.delimiter}/user/src`);
    assert.equal(env.KMP_DUPLICATE_LIB_OK, 'TRUE');
  });

  test('sets PYTHONPATH when there was none', () => {
    const env = workerEnv({}, BACKEND);
    assert.equal(env.PYTHONPATH, BACKEND);
  });

  for (const existing of [undefined, '', '/user/src']) {
    test(`private graph bootstraps preserve PYTHONPATH=${String(existing)} without backend packages`, () => {
      const env = workerEnv({ PYTHONPATH: existing, USER_SETTING: 'keep' }, BACKEND, true);
      assert.equal(env.PYTHONPATH, existing);
      assert.equal(env.USER_SETTING, 'keep');
      assert.equal(env.KMP_DUPLICATE_LIB_OK, 'TRUE');
      assert.equal(env.PYTHONUNBUFFERED, '1');
    });
  }
});

suite('worker.workspaceTrusted', () => {
  setup(() => reset());

  /**
   * Replaces the three `confirmTrust` cases (ask once, remember a no, a dismissal is
   * not an answer). Those pinned a tensorViz prompt that no longer exists: task.md D20
   * deletes it because "do you trust the code in this folder enough to execute it" is
   * exactly what VS Code's own Workspace Trust dialog asks on first open, and asking it
   * again in our own words taught people to click through a dialog that matters.
   *
   * What still has to be true is that the answer is READ rather than assumed — an
   * extension that runs a restricted workspace's code is the actual hazard here.
   */
  test('a restricted workspace is not trusted, and a trusted one is', () => {
    state.isTrusted = false;
    assert.equal(workspaceTrusted(), false);

    state.isTrusted = true;
    assert.equal(workspaceTrusted(), true);
  });

  test('it reads the editor every time rather than caching an answer', () => {
    // A workspace that starts restricted becomes trusted without a reload, via
    // onDidGrantWorkspaceTrust. A cached `false` would leave tracing dead until the
    // window was reopened, with nothing on screen saying why.
    state.isTrusted = false;
    assert.equal(workspaceTrusted(), false);
    state.isTrusted = true;
    assert.equal(workspaceTrusted(), true);
  });
});

suite('worker install hint', () => {
  test('names both packages the worker needs and nothing else', () => {
    // torch and pydantic only (task.md §3.2). Naming fastapi here would send a
    // user to install the backend's dependencies into their model environment.
    assert.ok(INSTALL_HINT.includes('torch'));
    assert.ok(INSTALL_HINT.includes('pydantic'));
    assert.ok(!INSTALL_HINT.includes('fastapi'));
  });
});


suite('worker process output bounds', () => {
  for (const pipe of ['stdout', 'stderr'] as const) {
    test(`stops excessive ${pipe} without retaining a partial success`, async () => {
      const result = await runProcess(process.execPath,
        ['-e', `process.${pipe}.write(Buffer.alloc(8192)); setInterval(() => {}, 1000)`],
        { cwd: process.cwd(), env: process.env, timeoutMs: 5000, stdoutLimit: 1024, stderrLimit: 1024 }, () => {});
      assert.equal(result.outputLimit, pipe); assert.equal(result.timedOut, false);
      assert.equal(result.stdout, ''); assert.equal(result.stderr, '');
    });
  }
  test('decodes UTF-8 after collecting complete bounded buffers', async () => {
    const result = await runProcess(process.execPath,
      ['-e', "process.stdout.write(Buffer.from([226])); setTimeout(() => process.stdout.write(Buffer.from([130,172])), 30)"],
      { cwd: process.cwd(), env: process.env, timeoutMs: 5000 }, () => {});
    assert.equal(result.stdout, '€'); assert.equal(result.code, 0);
  });
});

suite('TraceWorker interpreter probe cache', () => {
  /** A stand-in interpreter: ignores its arguments, answers the probe after
   *  `delayMs`, and can therefore be killed mid-probe on purpose. */
  function fakeInterpreter(delayMs: number): string {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tensorviz-probe-'));
    const script = path.join(dir, 'python');
    fs.writeFileSync(
      script,
      `#!/bin/sh\nsleep ${(delayMs / 1000).toFixed(2)}\necho "2.7.1 2.13.5"\n`,
      { mode: 0o755 },
    );
    return script;
  }

  /** `resolveDelays[n]` is how long the nth run waits before it reaches the probe.
   *  The real `userInterpreter` asks the Python extension, so this is not a
   *  contrivance: run() does async work between cancelling and probing, and how
   *  much decides whether it reads the cache the cancelled probe just wrote. */
  function worker(interpreter: string, resolveDelays: number[] = []): TraceWorker {
    const channel = { appendLine() {}, append() {}, dispose() {} } as unknown as vscode.OutputChannel;
    let call = 0;
    return new TraceWorker(process.cwd(), channel, {
      interpreter: async () => {
        const delay = resolveDelays[call] ?? 0;
        call += 1;
        if (delay) {
          await new Promise((resolve) => setTimeout(resolve, delay));
        }
        return interpreter;
      },
      timeoutMs: 10_000,
    });
  }

  test('a different selected interpreter refuses before probing or executing', async () => {
    const trace = worker('/no-longer-the-selected-python');
    try {
      const outcome = await trace.run({ file: '/ws/net.py', cls: 'Net', expectedInterpreter: '/selected-python' });
      assert.ok(!outcome.ok);
      assert.match(outcome.reason, /environment changed/);
    } finally { trace.dispose(); }
  });

  for (const throws of [false, true]) {
    test(`a final host guard ${throws ? 'exception' : 'refusal'} prevents execution after probing`, async () => {
      const interpreter = fakeInterpreter(0), trace = worker(interpreter);
      let checks = 0;
      try {
        const outcome = await trace.run({ file: '/ws/net.py', cls: 'Net', expectedInterpreter: interpreter,
          beforeLaunch: async () => { checks++; if (throws) throw new Error('private'); return false; } });
        assert.ok(!outcome.ok);
        assert.match(outcome.reason, /changed before launch/);
        assert.equal(checks, 1);
      } finally { trace.dispose(); fs.rmSync(path.dirname(interpreter), { recursive: true, force: true }); }
    });
  }

  test('a probe killed by a newer save does not condemn the interpreter forever', async function () {
    this.timeout(20_000);
    // The shape of a cold start: the first save probes, the second save arrives
    // while it is still running and cancels it. The probe registers its child so
    // Cancel can reach it, so the kill looks exactly like a broken interpreter —
    // `exit null`. Cached, that verdict made EVERY later trace refuse with
    // "<python> cannot run the trace worker", with the real interpreter able to
    // import torch the whole time and nothing in the product saying otherwise.
    const interpreter = fakeInterpreter(1500);
    // The second run reaches the probe a beat after the first one's probe was
    // killed, so it reads the verdict that kill produced. That is the ordering
    // the Electron suites hit; it is not the only one, but it is the one where
    // the cache decides the answer.
    const trace = worker(interpreter, [0, 400]);
    try {
      const first = trace.run({ file: '/ws/models/net.py', cls: 'Net' });
      await new Promise((resolve) => setTimeout(resolve, 200));
      const second = trace.run({ file: '/ws/models/net.py', cls: 'Net' });

      const [a, b] = await Promise.all([first, second]);
      assert.ok(!a.ok && a.cancelled, 'a stopped worker is an explicit cancellation, not a worker failure');

      // The superseded run says so; neither run may blame the interpreter.
      for (const outcome of [a, b]) {
        assert.ok(
          outcome.ok || !/cannot run the trace worker/.test(outcome.reason),
          `a cancelled probe was reported as a broken interpreter: ${JSON.stringify(outcome)}`,
        );
      }

      // And the next save still gets a real attempt: the interpreter is probed
      // again rather than refused from cache.
      const third = await trace.run({ file: '/ws/models/net.py', cls: 'Net' });
      assert.ok(
        third.ok || !/cannot run the trace worker/.test(third.reason),
        `the cache stayed poisoned: ${JSON.stringify(third)}`,
      );
    } finally {
      trace.dispose();
    }
  });

  test('a genuinely broken interpreter is still cached, so it is probed once', async function () {
    this.timeout(20_000);
    // The cache is not being removed, only kept honest: an interpreter that really
    // cannot import torch must still be remembered, or every save pays the probe.
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tensorviz-probe-bad-'));
    const script = path.join(dir, 'python');
    fs.writeFileSync(script, '#!/bin/sh\necho "No module named torch" >&2\nexit 1\n', { mode: 0o755 });
    const trace = worker(script);
    try {
      const first = await trace.run({ file: '/ws/models/net.py', cls: 'Net' });
      assert.ok(!first.ok && /cannot run the trace worker/.test(first.reason), JSON.stringify(first));

      fs.rmSync(dir, { recursive: true, force: true });
      // Answered from cache: the script is gone, so a second probe could not succeed.
      const second = await trace.run({ file: '/ws/models/net.py', cls: 'Net' });
      assert.ok(!second.ok && /cannot run the trace worker/.test(second.reason), JSON.stringify(second));
      assert.ok(!second.ok && second.detail.includes('No module named torch'), second.detail);
    } finally {
      trace.dispose();
    }
  });
});
