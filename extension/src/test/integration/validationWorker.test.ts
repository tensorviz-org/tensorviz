import * as assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import { existsSync, readFileSync } from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { reset, state } from '../vscodeStub';
import { ValidationFrames, ValidationWorker } from '../../validationWorker';
import type { RunSpec, ExecutionRecord } from '@tensorviz/contracts/protocol';

const root = path.resolve(__dirname, '../../../..');
/** The gate hands this in as TENSORVIZ_TEST_PYTHON (a torch-capable interpreter); a dev
 *  machine's ./venv is the fallback. Missing altogether is a failure, not a skip. */
const python = process.env.TENSORVIZ_TEST_PYTHON || path.join(root, 'venv/bin/python');
suiteSetup(() => {
  assert.ok(existsSync(python), `these suites spawn a real interpreter: set TENSORVIZ_TEST_PYTHON or create ./venv (looked at ${python})`);
});
const backend = path.join(root, 'backend');
const corpus = JSON.parse(readFileSync(path.join(root, 'tests/fixtures/contracts/validation-contracts.json'), 'utf8'));
function spec(): RunSpec {
  const value = structuredClone(corpus.structural.find((f: { name: string }) => f.name === 'host resolved run').value);
  value.interpreter.executable = python;
  value.profile.model = 'models:Model';
  value.profile.init = { args: [], kwargs: {} };
  value.profile.inputs.kwargs = {};
  // The real resolver materializes this optional capture budget before launch.
  value.profile.limits.max_capture_bytes = null;
  return value;
}
function frame(value: unknown): Buffer {
  const body = Buffer.from(JSON.stringify(value));
  const header = Buffer.alloc(4);
  header.writeUInt32BE(body.length);
  return Buffer.concat([header, body]);
}

suite('validation worker framing', () => {
  test('fragmented and coalesced frames preserve exact messages', () => {
    const values: unknown[] = [];
    const parser = new ValidationFrames(() => 1024, value => values.push(value));
    const bytes = Buffer.concat([frame({ one: 1 }), frame({ two: 'two' })]);
    for (const byte of bytes) { parser.push(Buffer.from([byte])); }
    parser.push(bytes);
    assert.deepEqual(values, [{ one: 1 }, { two: 'two' }, { one: 1 }, { two: 'two' }]);
    assert.equal(parser.partial, false);
  });
  test('oversize lengths are rejected before receiving their bodies', () => {
    const parser = new ValidationFrames(() => 1024, () => assert.fail());
    const header = Buffer.alloc(4);
    header.writeUInt32BE(0xffffffff);
    assert.throws(() => parser.push(header), /frame_limit/);
  });
  test('invalid UTF-8 and JSON cannot masquerade as records', () => {
    const parser = () => new ValidationFrames(() => 1024, () => assert.fail());
    assert.throws(() => parser().push(Buffer.from([0, 0, 0, 1, 0xff])));
    assert.throws(() => parser().push(Buffer.from([0, 0, 0, 1, 0x7b])));
  });
});

suite('validation worker host', function () {
  this.timeout(25000);
  let directory: string;
  let worker: ValidationWorker;
  setup(async () => {
    reset();
    state.isTrusted = true;
    directory = await fs.mkdtemp(path.join(os.tmpdir(), 'validation-host-test-'));
    worker = new ValidationWorker(backend);
  });
  teardown(async () => { worker.dispose(); await fs.rm(directory, { recursive: true, force: true }); });
  async function model(body: string): Promise<void> {
    await fs.writeFile(path.join(directory, 'models.py'), body);
  }
  async function waitFile(file: string): Promise<void> {
    const until = Date.now() + 15000;
    while (Date.now() < until) {
      try { await fs.access(path.join(directory, file)); return; } catch { /* pending */ }
      await new Promise(resolve => setTimeout(resolve, 20));
    }
    assert.fail(`worker did not create ${file}`);
  }
  test('requires explicit authorization and live workspace trust', async () => {
    await assert.rejects(worker.run(spec(), directory, false), /authorized/);
    state.isTrusted = false;
    await assert.rejects(worker.run(spec(), directory, true), /trusted/);
  });
  test('trust revocation during launch preparation prevents execution', async () => {
    await model("from pathlib import Path\nPath('called').touch()\n");
    const pending = worker.run(spec(), directory, true);
    state.isTrusted = false;
    assert.equal((await pending).record.process.state, 'failed');
    await assert.rejects(fs.access(path.join(directory, 'called')));
  });
  test('launches the selected interpreter and isolates/drains bounded logs', async () => {
    await model(`import torch, sys, os
assert 'TENSORVIZ_VALIDATION_BOOTSTRAP' not in os.environ
print('x' * 200000)
print('y' * 200000, file=sys.stderr)
class Model(torch.nn.Module):
    def forward(self, x): return x + 2
`);
    const result = await worker.run(spec(), directory, true);
    assert.deepEqual(result.record.process, { state: 'exited', exitCode: 0 });
    assert.equal(result.record.invocations[0].outcome, 'succeeded');
    assert.equal(result.stdout.length, 65536);
    assert.equal(result.stderr.length, 65536);
    assert.equal(result.record.capture.state, 'complete');
  });
  test('last saved-state check can refuse after private channel preparation without executing', async () => {
    await model("from pathlib import Path\nPath('called').touch()\n");
    let checked = false;
    const result = await worker.run(spec(), directory, true, undefined, async () => { checked = true; return false; });
    assert.equal(checked, true);
    assert.equal(result.record.process.state, 'failed');
    await assert.rejects(fs.access(path.join(directory, 'called')));
  });
  test('no fallback when the selected executable is missing; bootstrap is removed', async () => {
    const before = (await fs.readdir(os.tmpdir())).filter(p => p.startsWith('tensorviz-validation-'));
    const request = spec();
    request.interpreter.executable = path.join(directory, 'missing-python');
    const result = await worker.run(request, directory, true);
    assert.equal(result.record.process.state, 'failed');
    if (result.record.process.state === 'failed') { assert.equal(result.record.process.failure.phase, 'setup'); }
    assert.deepEqual((await fs.readdir(os.tmpdir())).filter(p => p.startsWith('tensorviz-validation-')), before);
  });
  test('timeout terminates a sleeping worker', async () => {
    await model('import time\ntime.sleep(30)\n');
    const request = spec();
    request.profile.limits!.timeout_ms = 200;
    const result = await worker.run(request, directory, true);
    assert.equal(result.record.process.state, 'timed_out');
  });
  test('OS spawn publishes waiting evidence before slow imports and remains cancellable', async () => {
    await model("from pathlib import Path\nPath('entered-import').touch()\nimport time\ntime.sleep(30)\n");
    const updates: ExecutionRecord[] = [];
    const request = spec();
    const pending = worker.run(request, directory, true, record => updates.push(record));
    await waitFile('entered-import');
    assert.equal(updates.length, 1, 'running state must arrive before any captured forward');
    assert.deepEqual(updates[0], { version: 1, identity: request.identity, sequence: 0, limits: request.profile.limits,
      process: { state: 'running' }, capture: { state: 'waiting' }, context: null,
      modules: [], invocations: [], tensors: [], edges: [], requirements: [] });
    worker.cancel();
    const result = await pending;
    assert.equal(result.record.process.state, 'cancelled');
    assert.ok(result.record.sequence > updates[0].sequence);
  });
  test('cancel before spawn cleans bootstrap and does not execute source', async () => {
    await model("from pathlib import Path\nPath('called').touch()\n");
    const promise = worker.run(spec(), directory, true);
    worker.cancel();
    assert.equal((await promise).record.process.state, 'cancelled');
    await assert.rejects(fs.access(path.join(directory, 'called')));
  });
  test('cancellation terminates descendants, including a child ignoring SIGTERM', async () => {
    await model(`import torch, subprocess, sys, time
from pathlib import Path
class Model(torch.nn.Module):
    def forward(self, x):
        child = subprocess.Popen([sys.executable, '-c', "import signal,time; from pathlib import Path; signal.signal(signal.SIGTERM, signal.SIG_IGN); Path('child-ready').touch(); time.sleep(30)"])
        Path('child-pid').write_text(str(child.pid))
        time.sleep(30)
`);
    const promise = worker.run(spec(), directory, true);
    await waitFile('child-ready');
    const pid = Number(await fs.readFile(path.join(directory, 'child-pid'), 'utf8'));
    worker.cancel();
    const result = await promise;
    assert.equal(result.record.process.state, 'cancelled');
    await new Promise(resolve => setTimeout(resolve, 400));
    assert.throws(() => process.kill(pid, 0), (error: NodeJS.ErrnoException) => error.code === 'ESRCH');
  });
  test('cwd escapes are refused before importing code', async () => {
    const request = spec();
    request.profile.cwd = '..';
    const result = await worker.run(request, directory, true);
    assert.equal(result.record.process.state, 'failed');
  });

  function scriptSpec(): RunSpec {
    const request = spec();
    request.profile = { kind: 'script', script: 'models.py', args: ['--flag', 'two words'], cwd: '.',
      tracked_files: [], limits: request.profile.limits, capture: { model: '__main__:Model', call: 1 } };
    request.files[1].path = 'models.py';
    return request;
  }
  const scriptModel = `import torch, sys
class Model(torch.nn.Module):
    def forward(self, x): return x + 1
Model()(torch.ones(2))
print('selected', sys.argv[1:], flush=True)
`;
  test('script late failure retains the selected call and nonzero exit', async () => {
    await model(scriptModel + "raise RuntimeError('late script failure')\n");
    const result = await worker.run(scriptSpec(), directory, true);
    assert.deepEqual(result.record.process, { state: 'exited', exitCode: 1 });
    assert.equal(result.record.invocations[0].outcome, 'succeeded');
    assert.match(result.stdout, /selected \['--flag', 'two words'\]/);
    assert.match(result.stderr, /RuntimeError: late script failure/);
  });
  test('script no-match is incomplete despite zero process exit', async () => {
    await model("print('ordinary program')\n");
    const result = await worker.run(scriptSpec(), directory, true);
    assert.deepEqual(result.record.process, { state: 'exited', exitCode: 0 });
    assert.deepEqual(result.record.capture, { state: 'incomplete', rootInvocationId: null, reason: 'target_not_observed' });
  });
  test('script timeout after capture preserves evidence and terminates the program', async () => {
    await model(scriptModel + "import time\ntime.sleep(30)\n");
    const request = scriptSpec();
    // Cold Torch startup on an SSH host needs room to produce evidence before
    // the deadline. The script still sleeps longer than this entire run budget.
    request.profile.limits!.timeout_ms = 15000;
    const result = await worker.run(request, directory, true);
    assert.equal(result.record.process.state, 'timed_out');
    assert.equal(result.record.invocations[0].outcome, 'succeeded');
    assert.deepEqual(result.record.capture, { state: 'incomplete', rootInvocationId: 'call0', reason: 'timeout' });
  });
  test('script cancellation after capture still kills descendants and retains evidence', async () => {
    await model(scriptModel + `import subprocess, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', "import signal,time; from pathlib import Path; signal.signal(signal.SIGTERM, signal.SIG_IGN); Path('script-child-ready').touch(); time.sleep(30)"])
Path('script-child-pid').write_text(str(child.pid))
time.sleep(30)
`);
    const pending = worker.run(scriptSpec(), directory, true);
    await waitFile('script-child-ready');
    const pid = Number(await fs.readFile(path.join(directory, 'script-child-pid'), 'utf8'));
    worker.cancel();
    const result = await pending;
    assert.equal(result.record.process.state, 'cancelled');
    assert.equal(result.record.invocations[0].outcome, 'succeeded');
    await new Promise(resolve => setTimeout(resolve, 400));
    assert.throws(() => process.kill(pid, 0), (error: NodeJS.ErrnoException) => error.code === 'ESRCH');
  });
  test('script path escapes are refused before execution', async () => {
    const request = scriptSpec();
    if (request.profile.kind !== 'script') { assert.fail(); }
    request.profile.script = '../outside.py';
    const result = await worker.run(request, directory, true);
    assert.equal(result.record.process.state, 'failed');
    assert.equal(result.stdout, '');
  });

  for (const problem of ['wrong-token', 'wrong-run', 'duplicate-hello', 'oversize', 'truncated', 'bad-sequence', 'foreign-identity', 'too-many']) {
    test(`rejects ${problem} on private worker channel`, async () => {
      const executable = path.join(directory, 'fake-worker');
      const peer = path.join(directory, 'faulty peer.py');
      const request = spec();
      request.interpreter.executable = executable;
      // Deliberately faulty peer, not a model substitute. It uses the real private
      // bootstrap and TCP channel to exercise host rejection and cleanup.
      await fs.writeFile(peer, `import json, os, socket, struct
p = os.environ.pop('TENSORVIZ_VALIDATION_BOOTSTRAP')
b = json.load(open(p))
os.unlink(p)
c = socket.create_connection(('127.0.0.1', b['resultChannel']['port']))
def send(v):
 data = json.dumps(v).encode()
 c.sendall(struct.pack('!I',len(data))+data)
h = dict(version=1,type='hello',runId=b['spec']['identity']['runId'],token=b['resultChannel']['token'])
problem = '${problem}'
if problem == 'wrong-token': h['token'] = '0'*64
if problem == 'wrong-run': h['runId'] = 'foreign'
send(h)
if problem == 'duplicate-hello': send(h)
if problem == 'oversize': c.sendall(struct.pack('!I', 0xffffffff))
if problem == 'truncated': c.sendall(struct.pack('!I', 100) + b'{')
r = dict(version=1, identity=b['spec']['identity'],sequence=1,limits=b['spec']['profile']['limits'],process=dict(state='running'),capture=dict(state='incomplete',rootInvocationId=None,reason='unsupported'),context=None,modules=[],invocations=[],tensors=[],edges=[],requirements=[])
if problem == 'foreign-identity': r['identity']['workspaceSessionId'] = 'foreign'
if problem in ('bad-sequence', 'foreign-identity', 'too-many'):
 for i in range(9 if problem == 'too-many' else 2):
  if problem == 'too-many': r['sequence'] = i
  send(dict(version=1,type='record',record=r,dependenciesCurrent=True))
c.close()
`);
      // A shebang splits interpreter paths at spaces. Quote both paths in the
      // launcher so these cases actually reach the malformed-channel peer.
      const quote = (value: string) => `'${value.replace(/'/g, `'\\''`)}'`;
      await fs.writeFile(executable, `#!/bin/sh\nexec ${quote(python)} ${quote(peer)} "$@"\n`, { mode: 0o700 });
      const result = await worker.run(request, directory, true);
      // The OS process can exit zero while transport capture has failed.
      assert.equal(result.record.process.state, 'exited');
      assert.equal(result.record.capture.state, 'incomplete');
      if (result.record.capture.state === 'incomplete') { assert.equal(result.record.capture.reason, 'transport_error'); }
    });
  }
});
