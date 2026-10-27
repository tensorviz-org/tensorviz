import * as assert from 'node:assert/strict';
import { AttemptLedger } from '@tensorviz/graph-host/attemptLedger';
import { existsSync } from 'node:fs';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import * as os from 'node:os';
import { execFileSync } from 'node:child_process';
import { readValidationSnapshot } from '../../validationHost';
import { ValidationWorker } from '../../validationWorker';
import { ValidationController } from '@tensorviz/graph-host/validationController';
import { revealExecutionNode } from '../../validationNavigation';
import type { ValidationExecutionUpdate } from '@tensorviz/contracts/protocol';
import { editorState, openStubEditor, resetEditors, reset, state } from '../vscodeStub';
import { runModelProfileFixture } from '../modelProfileFixture';
import { savedExecutionGuard } from '@tensorviz/graph-host/executionGuard';
import { TraceWorker } from '../../worker';
import { executionSpecSchema } from '@tensorviz/graph-host/executionSpec';
import { translateProfile } from '@tensorviz/graph-host/validationProfileBridge';
import { selectedExecutionInterpreter } from '../../projectEnvironment';
import * as vscode from 'vscode';

const repo = path.resolve(__dirname, '../../../..');
const backend = path.join(repo, 'backend');
/** The gate hands this in as TENSORVIZ_TEST_PYTHON (a torch-capable interpreter); a dev
 *  machine's ./venv is the fallback. Missing altogether is a failure, not a skip. */
const python = process.env.TENSORVIZ_TEST_PYTHON || path.join(repo, 'venv/bin/python');
suiteSetup(() => {
  assert.ok(existsSync(python), `these suites spawn a real interpreter: set TENSORVIZ_TEST_PYTHON or create ./venv (looked at ${python})`);
});
const config = `version = 1
[profiles.test]
kind = "script"
script = "main.py"
args = []
[profiles.test.capture]
model = "__main__:Model"
call = 1
`;

suite('canvas saved model profile lifecycle with real Python snapshots', function () {
  this.timeout(20000);
  for (const scenario of ['invalid', 'syntax', 'limit', 'dirty', 'selected_failure', 'snapshot_failure', 'unreadable', 'removed']) {
    test(`${scenario} refuses before any automatic or explicit execution`, async () => {
      const run = await runModelProfileFixture(scenario);
      assert.equal(run.result.type, 'validateResult');
      assert.equal((run.result as { status: string }).status, 'refused');
      assert.deepEqual(run.executions, [], 'no execution under substituted settings');
      assert.equal(run.sourceExecuted, false, 'snapshot is static');
      const detail = (run.result as { detail: string }).detail;
      assert.doesNotMatch(detail, /private-source|model-profiles-|Traceback|ENOENT/);
      if (scenario === 'invalid') assert.match(detail, /profiles.audit_small.timeout_ms.*limits/);
      if (scenario === 'limit') assert.match(detail, /limits.timeout_ms.*100.*86400000/);
    });
  }
  for (const scenario of ['dismiss', 'cancel_listing', 'cancel_choice', 'cancel_resolution']) {
    test(`${scenario} cancels without a late automatic or explicit run`, async () => {
      const run = await runModelProfileFixture(scenario);
      assert.equal((run.result as { status: string }).status, 'cancelled');
      assert.deepEqual(run.executions, []);
      assert.equal(run.sourceExecuted, false);
      if (scenario === 'dismiss' || scenario === 'cancel_choice') assert.deepEqual(run.snapshots, [null]);
    });
  }
  for (const scenario of ['absent', 'unrelated']) {
    test(`${scenario} requires explicit setup before class validation`, async () => {
      const run = await runModelProfileFixture(scenario);
      assert.equal((run.result as { status: string }).status, 'refused');
      assert.match((run.result as { detail: string }).detail, /Choose model setup and inputs with Open as Graph/);
      assert.deepEqual(run.executions, [], 'missing setup must not launch an input-guessing run');
      assert.deepEqual(run.snapshots, scenario === 'absent' ? [] : [null]);
      assert.deepEqual(run.choices, []);
      assert.equal(run.sourceExecuted, false);
    });
  }
  test('the installed two-profile fixture preserves the selected large input and limit', async () => {
    const run = await runModelProfileFixture('valid');
    assert.equal((run.result as { status: string }).status, 'passed');
    assert.deepEqual(run.choices, [['audit_small', 'audit_large']]);
    assert.deepEqual(run.snapshots, [null, 'audit_large']);
    assert.equal(run.executions.length, 1);
    assert.equal(run.executions[0].timeoutMs, 30000);
    const inputs = run.executions[0].execution!.inputs;
    assert.ok('binding' in inputs);
    assert.deepEqual((inputs.binding.args[0] as { shape: number[] }).shape, [3, 4]);
    assert.equal(run.sourceExecuted, false);
  });
});

suite('host resolver and worker integration', function () {
  this.timeout(20000);
  let root: string;
  setup(async () => {
    reset(); state.isTrusted = true;
    root = await fs.mkdtemp(path.join(os.tmpdir(), 'validation-lifecycle-'));
    await fs.writeFile(path.join(root, 'tensorviz.toml'), config);
  });
  teardown(async () => { resetEditors(); await fs.rm(root, { recursive: true, force: true }); });
  const request = (root: string) => ({ root, documents: {}, executable: python, environmentRevision: 0,
    session: 'session', workspaceId: 'workspace', runId: 'run', generation: 1, profile: 'test' });
  for (const layout of ['flat', 'namespace']) {
    test(`both isolated workers load ${layout} object factories and user services with the same import-time seed`, async () => {
      await fs.mkdir(path.join(root, 'services'));
      await fs.writeFile(path.join(root, 'services/__init__.py'), 'WIDTH = 4\n');
      const module = layout === 'namespace' ? 'models.factory_net' : 'factory_net';
      if (layout === 'namespace') await fs.mkdir(path.join(root, 'models'));
      const file = path.join(root, ...module.split('.')) + '.py';
      const seedFile = file.replace(/\.py$/, '.seed');
      await fs.writeFile(file, `from pathlib import Path
import torch
Path(__file__).with_suffix('.seed').write_text(str(torch.rand(1).item()))
class Model(torch.nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.fc = torch.nn.Linear(dimensions['width'], 2)
    def forward(self, x): return self.fc(x)
`);
      await fs.writeFile(path.join(root, 'recipe.py'), `from ${module} import Model\nfrom services import WIDTH\ndef build(): return Model({"width": WIDTH})\n`);
      await fs.writeFile(path.join(root, 'tensorviz.toml'), `version = 1
[profiles.test]
kind = "model"
model = "${module}:Model"
mode = "eval"
device = "cpu"
seed = 7
grad_enabled = false
factory = {file = "recipe.py", callable = "build", kwargs = {}}
inputs = {args = [{kind="tensor", shape=[2,4], dtype="float32", requires_grad=false, generator={kind="full", value=1}}], kwargs = {}}
`);
      const prepared = await readValidationSnapshot(python, backend, request(root), new AbortController().signal);
      assert.ok(prepared.resolution.status === 'resolved');
      const spec = prepared.resolution.spec;
      const direct = new ValidationWorker(backend);
      const graph = new TraceWorker(backend, { appendLine() {} } as unknown as vscode.OutputChannel, { interpreter: async () => python });
      try {
        const run = await direct.run(spec, root, true);
        assert.equal(run.record.capture.state, 'complete', JSON.stringify(run));
        assert.equal(run.record.invocations[0].outcome, 'succeeded');
        const directSeed = await fs.readFile(seedFile, 'utf8');
        await fs.unlink(seedFile);
        const translated = translateProfile(spec.profile);
        assert.ok(translated.ok);
        const captured = await graph.run({ file, cls: 'Model', workspace: root, profileSpec: spec,
          expectedInterpreter: python, execution: translated.run.execution, timeoutMs: 15000 });
        assert.ok(captured.ok, JSON.stringify(captured));
        assert.equal(captured.executionCurrent, true);
        assert.equal(await fs.readFile(seedFile, 'utf8'), directSeed);
      } finally { direct.dispose(); graph.dispose(); }
    });
  }
  for (const mode of ['current', 'saved_profile', 'managed_profile', 'changed_before_import', 'changed_during_forward']) {
    test(`graph worker resolves and preserves exact execution plans (${mode})`, async () => {
      await fs.unlink(path.join(root, 'tensorviz.toml'));
      const file = path.join(root, 'model.py');
      await fs.writeFile(file, `from pathlib import Path
import torch
Path(__file__).with_suffix('.executed').touch()
class Model(torch.nn.Module):
    def forward(self, x):
        ${mode === 'changed_during_forward' ? "Path(__file__).with_name('tensorviz.toml').write_text('# changed setup')" : 'pass'}
        return torch.sin(x) + 1
`);
      const logs: string[] = [];
      const channel = { appendLine: (line: string) => { logs.push(line); } } as unknown as vscode.OutputChannel;
      const managedContext = { workspaceState: { get: () => ({ kind: 'managed' }) },
        globalState: { get: () => undefined } } as unknown as vscode.ExtensionContext;
      const interpreter = mode === 'managed_profile'
        ? (resource: vscode.Uri) => selectedExecutionInterpreter(managedContext, resource, async () => python)
        : async () => python;
      const worker = new TraceWorker(backend, channel, { interpreter });
      const planTransport = require('@tensorviz/graph-host/executionPlanCore') as typeof import('@tensorviz/graph-host/executionPlanCore');
      const original = planTransport.executionPlanFile;
      const temporary: string[] = [];
      planTransport.executionPlanFile = async snapshot => {
        const result = await original(snapshot); temporary.push(result.file); return result;
      };
      try {
        let execution = executionSpecSchema.parse({ inputs: { args: [{ shape: [2, 4], dtype: 'float32', fill: 'ones' }] } });
        let profileSpec;
        if (mode === 'saved_profile' || mode === 'managed_profile') {
          await fs.writeFile(path.join(root, 'tensorviz.toml'), `version = 1
[profiles.test]
kind = "model"
model = "model:Model"
mode = "eval"
device = "cpu"
grad_enabled = false
init = {args = [], kwargs = {}}
inputs = {args = [{kind="tensor", shape=[2,4], dtype="float32", requires_grad=false, generator={kind="full", value=1}}], kwargs = {}}
`);
          const resolvedPython = await interpreter(vscode.Uri.file(root));
          assert.equal(resolvedPython, python);
          const prepared = await readValidationSnapshot(resolvedPython!, backend,
            { ...request(root), executable: resolvedPython }, new AbortController().signal);
          assert.ok(prepared.resolution.status === 'resolved');
          profileSpec = prepared.resolution.spec;
          const translated = translateProfile(profileSpec.profile, profileSpec.identity.profile);
          assert.ok(translated.ok);
          execution = translated.run.execution;
        }
        let requestedPlan: any;
        const ledger = new AttemptLedger();
        const target = { workspaceFolder: root, file: 'model.py', cls: 'Model', sourceDigest: 'a'.repeat(64),
          documentVersion: 1, backendRevision: 1 };
        const attempt = ledger.begin(`attempt:${mode}`, target, 'manual', null);
        const outcome = await worker.run({ file, cls: 'Model', workspace: root, timeoutMs: 15000,
          execution, profileSpec, attemptId: `attempt:${mode}`,
          onResolvedPlan: snapshot => { requestedPlan = snapshot; attempt.resolved(snapshot); },
          beforeLaunch: async () => {
            if (mode === 'changed_before_import') await fs.appendFile(file, '# changed after resolution\n');
            return true;
          } });
        attempt.captured(outcome);
        attempt.finish(outcome.ok && outcome.executionCurrent ? 'passed' : 'refused');
        assert.equal(outcome.ok, mode !== 'changed_before_import', JSON.stringify(outcome));
        assert.equal(requestedPlan.plan.id, `attempt:${mode}`, 'worker plans retain the host attempt identity');
        assert.equal(requestedPlan.plan.source.modelFile, 'model.py', 'even a failed worker retains the exact requested plan');
        assert.equal(existsSync(path.join(root, 'model.executed')), mode !== 'changed_before_import');
        if (outcome.ok) {
          assert.equal(outcome.executionCurrent, mode !== 'changed_during_forward');
          const evidence = (outcome.envelope.result as any).diagnostics.execution;
          assert.equal(evidence.planDigest, requestedPlan.applicabilityDigest);
          assert.equal(evidence.resolvedPlan.source.modelFile, 'model.py');
          assert.equal(evidence.resolvedPlan.interpreter.executable, python);
          assert.equal(evidence.planCurrent, mode !== 'changed_during_forward');
          assert.equal(evidence.resolvedPlan.configuration, mode === 'saved_profile' || mode === 'managed_profile' ? 'selected' : 'absent');
          const diagnostics = (outcome.envelope.result as any).diagnostics;
          const capture = diagnostics.semantic_capture;
          assert.equal(diagnostics.semantic_capture_error, null);
          assert.equal(capture.status, 'complete');
          assert.equal(capture.attemptId, requestedPlan.plan.id);
          assert.deepEqual(capture.source, requestedPlan.plan.source);
          assert.equal(capture.planDigest, requestedPlan.applicabilityDigest);
          assert.equal(capture.inputDigest, evidence.inputs.digest);
          assert.equal(capture.stateDigest, evidence.state.digest);
          assert.deepEqual(capture.operations.map((op: any) => op.target), ['aten.sin.default', 'aten.add.Tensor']);
          assert.deepEqual(capture.operations[1].arguments.args[1], { kind: 'int', value: '1' });
          assert.equal(capture.output.id, capture.operations[1].output.id);
          assert.equal(ledger.state(target).lastCompleteCaptureId, capture.id);
          assert.equal(ledger.capture(capture.id)!.record.format, 'semantic');
          assert.equal(ledger.capture(capture.id)!.record.semanticRejection, undefined);
          assert.deepEqual((ledger.capture(capture.id)!.envelope!.result as any).diagnostics.semantic_capture, capture);
        } else {
          assert.equal(ledger.state(target).lastCompleteCaptureId, null);
          assert.deepEqual(ledger.state(target).latestAttempt!.captureIds, []);
        }
        assert.equal(temporary.length, 1);
        for (const planFile of temporary) assert.equal(existsSync(planFile), false);
        assert.ok(!logs.some(line => line.includes('plan.json')), 'private plan path stays out of logs');
      } finally { planTransport.executionPlanFile = original; worker.dispose(); }
    });
  }
  for (const mode of ['seeded', 'bad_input', 'changed_before_import', 'changed_during_forward',
    'changed_config', 'changed_environment', 'new_import', 'cancelled', 'timeout', 'invalid_config', 'dirty_helper']) {
    test(`prepared automatic worker: ${mode}`, async () => {
      await fs.unlink(path.join(root, 'tensorviz.toml'));
      const file = path.join(root, 'model.py');
      const mutation = mode === 'changed_during_forward' ? "Path(__file__).write_text(Path(__file__).read_text() + '# changed\\n')"
        : mode === 'changed_config' ? "Path(__file__).with_name('tensorviz.toml').write_text('version=1\\n')"
        : mode === 'new_import' ? "Path(__file__).with_name('late.py').write_text('VALUE=1\\n'); import late"
        : mode === 'timeout' ? 'time.sleep(30)' : 'pass';
      await fs.writeFile(file, `from pathlib import Path
import torch, time
Path(__file__).with_suffix('.executed').touch()
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)
    def forward(self, x):
        ${mutation}
        return self.fc(x) + 1
`);
      if (mode === 'invalid_config') await fs.writeFile(path.join(root, 'tensorviz.toml'), 'broken = [');
      if (mode === 'dirty_helper') {
        const helper = path.join(root, 'helper.py');
        await fs.writeFile(helper, 'WIDTH=4\n');
        openStubEditor(helper, 'WIDTH=5\n');
        editorState.dirtyDocuments.add(helper);
      }
      const logs: string[] = [];
      let resolutions = 0;
      const worker = new TraceWorker(backend, { appendLine: (line: string) => logs.push(line) } as unknown as vscode.OutputChannel,
        { interpreter: async () => ++resolutions > 1 && mode === 'changed_environment' ? '/changed/python' : python });
      const transport = require('@tensorviz/graph-host/executionPlanCore') as typeof import('@tensorviz/graph-host/executionPlanCore');
      const original = transport.executionPlanFile, temporary: string[] = [];
      transport.executionPlanFile = async snapshot => {
        const result = await original(snapshot); temporary.push(result.file); return result;
      };
      const ledger = new AttemptLedger(), id = `automatic:${mode}`;
      const target = { workspaceFolder: root, file: 'model.py', cls: 'Model', sourceDigest: 'a'.repeat(64),
        documentVersion: 1, backendRevision: 1 };
      const attempt = ledger.begin(id, target, 'manual', null);
      let prepared: any;
      try {
        const outcome = await worker.run({ file, cls: 'Model', workspace: root, attemptId: id,
          timeoutMs: mode === 'timeout' ? 100 : 15000, input: mode === 'bad_input' ? '1,3' : undefined,
          onCapturePrepared: snapshot => { prepared = snapshot; attempt.prepared(snapshot); },
          onResolvedPlan: () => assert.fail('input search must not invent an executable plan'),
          beforeLaunch: async () => {
            if (mode === 'changed_before_import') await fs.appendFile(file, '# changed before import\n');
            if (mode === 'cancelled') worker.cancel();
            return true;
          } });
        attempt.captured(outcome);
        attempt.finish(outcome.ok && outcome.executionCurrent ? 'passed' : 'refused');
        const noCapture = ['changed_before_import', 'cancelled', 'timeout', 'invalid_config', 'dirty_helper'].includes(mode);
        const beforePreparation = ['invalid_config', 'dirty_helper'].includes(mode);
        if (beforePreparation) assert.equal(prepared, undefined);
        else {
          assert.equal(prepared.preparation.id, id);
          assert.equal(prepared.preparation.policy, 'legacy-input-search-v1');
          assert.equal(ledger.state(target).latestAttempt!.sourceId, prepared.preparation.source.id);
        }
        assert.equal(ledger.state(target).latestAttempt!.planDigest, null);
        if (noCapture) {
          assert.equal(outcome.ok, false);
          assert.deepEqual(ledger.state(target).latestAttempt!.captureIds, []);
          if (mode !== 'timeout') assert.equal(existsSync(path.join(root, 'model.executed')), false);
          if (mode === 'cancelled') assert.equal(ledger.state(target).latestAttempt!.status, 'cancelled');
          if (mode === 'timeout') assert.equal(ledger.state(target).latestAttempt!.status, 'timed_out');
        } else {
          assert.equal(outcome.ok, mode !== 'bad_input', JSON.stringify(outcome));
          assert.equal(outcome.executionCurrent, ['seeded', 'bad_input'].includes(mode));
          const diagnostics = (outcome.envelope!.result as any).diagnostics;
          const capture = diagnostics.semantic_capture;
          if (mode === 'new_import') {
            assert.equal(capture, null);
            assert.equal(diagnostics.semantic_capture_error, 'source_mismatch');
            assert.equal(ledger.state(target).lastCompleteCaptureId, null);
          } else {
            assert.equal(capture.attemptId, id);
            assert.equal(capture.planDigest, null);
            assert.equal(capture.preparationDigest, prepared.applicabilityDigest);
            assert.deepEqual(capture.source, prepared.preparation.source);
            assert.equal(capture.automatic.candidateIndex, 1);
            assert.equal(capture.automatic.candidatesTried, 1);
            assert.equal(capture.automatic.provenance, mode === 'bad_input' ? 'explicit' : 'seed-from-first-layer');
            const retained = ledger.capture(capture.id)!;
            assert.equal(retained.record.format, 'semantic');
            assert.deepEqual(retained.preparation, prepared.preparation);
            assert.equal(mode === 'bad_input' ? ledger.state(target).lastPartialCaptureId : ledger.state(target).lastCompleteCaptureId, capture.id);
            if (mode === 'seeded') {
              assert.equal(await outcome.recheckExecution!(new AbortController().signal), true);
              await fs.writeFile(path.join(root, 'tensorviz.toml'), 'version=1\n');
              assert.equal(await outcome.recheckExecution!(new AbortController().signal), false, 'configuration edits during projection cannot certify the current setup');
            }
          }
        }
        assert.equal(temporary.length, beforePreparation ? 0 : 1);
        for (const file of temporary) assert.equal(existsSync(file), false);
        assert.ok(!logs.some(line => line.includes('plan.json')), 'private preparation path stays out of logs');
      } finally { transport.executionPlanFile = original; worker.dispose(); }
    });
  }
  test('planned custom writes without returned tensors survive host retention', async () => {
    await fs.unlink(path.join(root, 'tensorviz.toml'));
    const file = path.join(root, 'model.py');
    await fs.writeFile(file, `import torch
library = torch.library.Library('tensorviz_host_effect', 'DEF')
library.define('bump(Tensor(a!) x) -> ()')
def bump(x):
    x.add_(1)
library.impl('bump', bump, 'CPU')
class Model(torch.nn.Module):
    def forward(self, x):
        torch.ops.tensorviz_host_effect.bump(x)
        return x + 2
`);
    const worker = new TraceWorker(backend, { appendLine() {} } as unknown as vscode.OutputChannel,
      { interpreter: async () => python });
    const target = { workspaceFolder: root, file: 'model.py', cls: 'Model', sourceDigest: 'a'.repeat(64),
      documentVersion: 1, backendRevision: 1 };
    const ledger = new AttemptLedger(), attempt = ledger.begin('effects', target, 'manual', null);
    try {
      const outcome = await worker.run({ file, cls: 'Model', workspace: root, timeoutMs: 15000, attemptId: 'effects',
        execution: executionSpecSchema.parse({ inputs: { args: [{ shape: [2, 4], dtype: 'float32', fill: 'ones' }] } }),
        onResolvedPlan: snapshot => attempt.resolved(snapshot) });
      assert.equal(outcome.ok, true, JSON.stringify(outcome));
      attempt.captured(outcome); attempt.finish('passed');
      const id = ledger.state(target).lastCompleteCaptureId;
      assert.ok(id, JSON.stringify(ledger.state(target)));
      const retained = ledger.capture(id)!;
      assert.equal(retained.record.format, 'semantic');
      const capture = (retained.envelope!.result as any).diagnostics.semantic_capture;
      assert.equal(capture.operations[0].output.kind, 'none');
      const effect = capture.operations[0].effects[0];
      assert.deepEqual(effect.argumentPath, ['args', 0]);
      assert.equal(effect.beforeId, capture.inputs.args[0].id);
      assert.deepEqual(effect.after, capture.operations[1].arguments.args[0]);
      const changed = capture.values.find((v: any) => v.id === effect.after.id);
      assert.equal(changed.logicalVersion, 1);
      assert.deepEqual(changed.producer, { operationId: capture.operations[0].id, kind: 'effect', path: [0, 'after'] });
    } finally { worker.dispose(); }
  });
  for (const changed of ['config', 'helper', 'native', 'environment', 'dirty']) {
    test(`saved invocation guard rejects real ${changed} changes without importing source`, async () => {
      const source = "from pathlib import Path\nPath('must-not-run').touch()\n";
      await fs.writeFile(path.join(root, 'main.py'), source);
      await fs.writeFile(path.join(root, 'helper.py'), 'WIDTH = 8\n');
      await fs.writeFile(path.join(root, 'data.bin'), Buffer.from([1, 2]));
      await fs.writeFile(path.join(root, 'tensorviz.toml'), config.replace('args = []', 'args = []\ntracked_files = ["data.bin"]'));
      const input = request(root), signal = new AbortController().signal;
      const baseline = await readValidationSnapshot(python, backend, input, signal);
      assert.equal(baseline.resolution.status, 'resolved');
      const guard = savedExecutionGuard(baseline, next => readValidationSnapshot(python, backend, input, next));
      input.runId = 'next-snapshot';
      input.documents = { 'main.py': { version: 1, dirty: false } };
      assert.equal(await guard.current(signal), true, 'opening a saved file does not change the plan');
      if (changed === 'config') await fs.appendFile(path.join(root, 'tensorviz.toml'), '# new revision\n');
      if (changed === 'helper') await fs.appendFile(path.join(root, 'helper.py'), 'WIDTH = 9\n');
      if (changed === 'native') await fs.writeFile(path.join(root, 'data.bin'), Buffer.from([1, 3]));
      if (changed === 'environment') input.environmentRevision++;
      if (changed === 'dirty') input.documents = { 'main.py': { version: 2, dirty: true } };
      assert.equal(await guard.current(signal), false);
      assert.equal(existsSync(path.join(root, 'must-not-run')), false);
    });
  }
  for (const mode of ['model', 'script']) {
    for (const fails of [false, true]) {
      test(`${mode} real records navigate exact saved module/function/failure lines and refuse unsafe targets (${fails})`, async () => {
        root = await fs.realpath(root);
        const source = `import torch
def operation(x):
    ${fails ? 'return x.reshape(19)' : 'return torch.sin(x)'}
class Child(torch.nn.Module):
    def forward(self, x):
        return operation(x)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.child = Child()
    def forward(self, x):
        return self.child(x)
`;
        const file = path.join(root, 'models.py');
        await fs.writeFile(file, source);
        await fs.writeFile(path.join(root, 'main.py'), 'import torch\nfrom models import Model\nModel()(torch.ones(2,3))\n');
        await fs.writeFile(path.join(root, 'tensorviz.toml'), mode === 'script'
          ? config.replace('__main__:Model', 'models:Model') : `version = 1
[profiles.test]
kind = "model"
model = "models:Model"
mode = "eval"
device = "cpu"
grad_enabled = false
[profiles.test.init]
args = []
kwargs = {}
[profiles.test.inputs]
args = [{kind="tensor", shape=[2,3], dtype="float32", requires_grad=false, generator={kind="full", value=1}}]
kwargs = {}
`);
        const worker = new ValidationWorker(backend);
        let current = true;
        const controller = new ValidationController({
          trusted: () => true, choose: async () => 'test', approve: async () => true,
          prepare: (target, runId, generation, profile, signal) => readValidationSnapshot(python, backend,
            { ...request(root), session: target.session, runId, generation, profile }, signal),
          run: (spec, directory, receive, beforeLaunch) => worker.run(spec, directory, true, receive, beforeLaunch),
          cancel: () => worker.cancel(), publish: () => {},
        });
        try {
          await controller.run({ root, session: 'session', current: () => current });
          const update = controller.snapshot('session');
          assert.ok(update);
          assert.equal(update.freshness.state, 'current');
          assert.equal(update.record.capture.state, 'complete');
          assert.equal(update.record.invocations[0].outcome, fails ? 'failed' : 'succeeded');
          const editor = openStubEditor(file, source);
          const calls = update.record.invocations;
          assert.equal(calls.length, 3); // root, nested module and actual operation
          for (const call of calls) {
            const node: string = `execution:${update.record.identity.runId}:${call.id}`;
            const evidence = controller.sourceForNode('session', node);
            assert.ok(evidence, call.name);
            const line = fails ? 3 : call.kind === 'function' ? 3 : call.parentId ? 5 : 11;
            assert.deepEqual(evidence.source, { file: 'models.py', line, column: 0 });
            assert.equal((await revealExecutionNode(() => controller.sourceForNode('session', node))).ok, true);
            assert.equal(editor.selection.active.line, line - 1);
            assert.equal(editor.selection.active.character, 0);
          }
          const node = `execution:${update.record.identity.runId}:${calls[0].id}`;
          const evidence = () => controller.sourceForNode('session', node);
          assert.equal((await revealExecutionNode(() => controller.sourceForNode('foreign', node))).ok, false);
          assert.equal((await revealExecutionNode(() => controller.sourceForNode('session', 'execution:old:call0'))).ok, false);
          await fs.writeFile(file, source + '# changed\n');
          assert.equal((await revealExecutionNode(evidence)).ok, false);
          await fs.writeFile(file, source);
          // The same bytes can be dirty (e.g. after editing then undoing).
          editorState.dirtyDocuments.add(file);
          assert.equal((await revealExecutionNode(evidence)).ok, false);
          editorState.dirtyDocuments.delete(file);
          openStubEditor(file, source + '# unsaved\n');
          assert.equal((await revealExecutionNode(evidence)).ok, false);
          openStubEditor(file, source);
          const outside = await fs.mkdtemp(path.join(os.tmpdir(), 'validation-outside-'));
          try {
            const external = path.join(outside, 'models.py');
            await fs.writeFile(external, source);
            await fs.unlink(file);
            await fs.symlink(external, file);
            assert.equal((await revealExecutionNode(evidence)).ok, false);
            await fs.unlink(file);
            await fs.writeFile(file, source);
          } finally { await fs.rm(outside, { recursive: true, force: true }); }
          current = false;
          assert.equal((await revealExecutionNode(evidence)).ok, false);
          current = true;
          controller.invalidate('source');
          assert.equal((await revealExecutionNode(evidence)).ok, false);
        } finally { controller.dispose(); worker.dispose(); }
      });
    }
  }
  test('isolated helper resolves saved files without running or importing source', async () => {
    await fs.writeFile(path.join(root, 'main.py'), "from pathlib import Path\nPath('must-not-run').touch()\nraise RuntimeError('source must not execute during preparation')\n");
    // A fresh helper tree proves -I does not defeat the no-bytecode-write policy.
    const helper = path.join(root, 'helper');
    await fs.mkdir(path.join(helper, 'services/validation'), { recursive: true });
    await fs.writeFile(path.join(helper, 'services/__init__.py'), '');
    for (const name of ['__init__', 'host_snapshot', 'contracts', 'config', 'signatures']) {
      await fs.copyFile(path.join(backend, `services/validation/${name}.py`), path.join(helper, `services/validation/${name}.py`));
    }
    const result = await readValidationSnapshot(python, helper, request(root), new AbortController().signal);
    assert.equal(result.resolution.status, 'resolved');
    const dirty = await readValidationSnapshot(python, helper, { ...request(root),
      documents: { 'main.py': { version: 2, dirty: true } } }, new AbortController().signal);
    assert.equal(dirty.resolution.status, 'needs_setup');
    assert.deepEqual(await caches(root), {});
    await assert.rejects(fs.access(path.join(root, 'must-not-run')));
  });

  async function caches(directory: string): Promise<Record<string, string>> {
    const found: Record<string, string> = {};
    for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
      const file = path.join(directory, entry.name);
      if (entry.isDirectory()) Object.assign(found, await caches(file));
      else if (entry.name.endsWith('.pyc')) found[file] = (await fs.readFile(file)).toString('hex');
    }
    return found;
  }

  for (const mode of ['model', 'script']) {
    for (const stale of ['model', 'dependency']) {
      test(`${mode} production launch uses saved ${stale} source and preserves user caches/imports`, async () => {
        const packageDir = path.join(root, 'services');
        await fs.mkdir(packageDir);
        await fs.writeFile(path.join(packageDir, '__init__.py'), 'from .values import width\n');
        await fs.writeFile(path.join(packageDir, 'values.py'), 'def width(): return 2\n');
        await fs.writeFile(path.join(packageDir, 'models.py'), `import torch, sys, os, services
from pathlib import Path
from .values import width
assert services.width() == 2
assert __name__ == 'services.models'
assert sys.path[0] == os.getcwd()
Path('private-cache').write_text(sys.pycache_prefix)
assert sys.dont_write_bytecode
class Model(torch.nn.Module):
    def forward(self, x):
        with open('calls', 'a') as stream: stream.write('call\\n')
        return x[:, :2] if width() == 2 else x[:, :1]
`);
        const source = path.join(packageDir, stale === 'model' ? 'models.py' : 'values.py');
        const current = await fs.readFile(source, 'utf8');
        const old = stale === 'model' ? current.replace('return x[:, :2]', 'return x[:, :1]') : current.replace('return 2', 'return 1');
        assert.equal(old.length, current.length);
        await fs.writeFile(source, old);
        const fixed = new Date(1726000000000);
        await fs.utimes(source, fixed, fixed);
        // Explicit compilation is limited to this disposable fixture, including an
        // inherited alternate cache. Both must remain byte-for-byte untouched.
        const userCache = path.join(root, 'user-cache');
        for (const options of [[], ['-X', `pycache_prefix=${userCache}`]]) {
          execFileSync(python, ['-B', ...options, '-c', 'import py_compile, sys; py_compile.compile(sys.argv[1], doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)', source],
            { env: { ...process.env, PYTHONPYCACHEPREFIX: '', PYTHONDONTWRITEBYTECODE: '1' } });
        }
        await fs.writeFile(source, current);
        await fs.utimes(source, fixed, fixed);
        const before = await caches(root);
        assert.equal(Object.keys(before).length, 2);
        const body = mode === 'script' ? `version = 1
[profiles.test]
kind = "script"
script = "main.py"
args = ["two words"]
[profiles.test.capture]
model = "services.models:Model"
call = 1
` : `version = 1
[profiles.test]
kind = "model"
model = "services.models:Model"
mode = "eval"
device = "cpu"
grad_enabled = false
[profiles.test.init]
args = []
kwargs = {}
[profiles.test.inputs]
args = [{kind="tensor", shape=[2,2], dtype="float32", requires_grad=false, generator={kind="full", value=1}}]
kwargs = {}
`;
        await fs.writeFile(path.join(root, 'tensorviz.toml'), body);
        await fs.writeFile(path.join(root, 'main.py'), `import torch, sys, atexit
from services.models import Model
assert sys.argv[1:] == ['two words']
assert __name__ == '__main__' and __spec__ is None
@atexit.register
def end(): print('shutdown')
print(Model()(torch.ones(2,2)).tolist())
`);
        const preparation = await readValidationSnapshot(python, backend, request(root), new AbortController().signal);
        if (preparation.resolution.status !== 'resolved') assert.fail(JSON.stringify(preparation));
        const savedEnv = process.env.PYTHONPYCACHEPREFIX;
        const worker = new ValidationWorker(backend);
        try {
          process.env.PYTHONPYCACHEPREFIX = userCache;
          const result = await worker.run(preparation.resolution.spec, root, true);
          assert.deepEqual(result.record.process, { state: 'exited', exitCode: 0 }, result.stderr);
          assert.equal(result.record.capture.state, 'complete');
          assert.equal(result.dependenciesCurrent, true);
          const output = result.record.invocations[0].output;
          assert.ok(output?.kind === 'tensor');
          assert.deepEqual(result.record.tensors.find(t => t.id === output.tensorId)?.shape, [2, 2]);
          assert.equal(await fs.readFile(path.join(root, 'calls'), 'utf8'), 'call\n');
          const privateCache = await fs.readFile(path.join(root, 'private-cache'), 'utf8');
          assert.notEqual(privateCache, userCache);
          await assert.rejects(fs.access(path.dirname(privateCache))); // host cleanup after exit
          assert.deepEqual(await caches(root), before);
          assert.equal(await fs.readFile(source, 'utf8'), current);
          if (mode === 'script') {
            // Compare with a fresh-source ordinary Python run; default CPython
            // intentionally trusts the stale user cache, so give this oracle an empty prefix.
            const plain = execFileSync(python, ['-B', '-X', `pycache_prefix=${path.join(root, 'oracle-cache')}`,
              path.join(root, 'main.py'), 'two words'], { cwd: root, env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1' } });
            assert.equal(result.stdout, plain.toString());
            assert.deepEqual(await caches(root), before);
          }
        } finally {
          worker.dispose();
          if (savedEnv === undefined) delete process.env.PYTHONPYCACHEPREFIX;
          else process.env.PYTHONPYCACHEPREFIX = savedEnv;
        }
      });
    }
  }
  test('sourceless workspace imports keep Python semantics but cannot certify saved source', async () => {
    const source = path.join(root, 'legacy.py');
    const bytecode = path.join(root, 'legacy.pyc');
    await fs.writeFile(source, 'VALUE = 2\n');
    execFileSync(python, ['-B', '-c', 'import py_compile, sys; py_compile.compile(sys.argv[1], cfile=sys.argv[2], doraise=True)', source, bytecode],
      { env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1' } });
    await fs.unlink(source);
    const before = await caches(root);
    await fs.writeFile(path.join(root, 'main.py'), `import torch, legacy
assert legacy.VALUE == 2
class Model(torch.nn.Module):
    def forward(self, x): return x + legacy.VALUE
Model()(torch.ones(2))
`);
    const preparation = await readValidationSnapshot(python, backend, request(root), new AbortController().signal);
    if (preparation.resolution.status !== 'resolved') assert.fail(JSON.stringify(preparation));
    const worker = new ValidationWorker(backend);
    try {
      const result = await worker.run(preparation.resolution.spec, root, true);
      assert.deepEqual(result.record.process, { state: 'exited', exitCode: 0 }, result.stderr);
      assert.equal(result.record.capture.state, 'complete');
      assert.equal(result.dependenciesCurrent, false);
      assert.deepEqual(await caches(root), before);
    } finally { worker.dispose(); }
  });
  test('capture publishes while script lives; config edit stales evidence and late exit survives', async () => {
    await fs.writeFile(path.join(root, 'main.py'), `import torch, time
class Model(torch.nn.Module):
    def forward(self, x): return x + 2
Model()(torch.ones(2))
time.sleep(2)
raise SystemExit(7)
`);
    const worker = new ValidationWorker(backend);
    const updates: ValidationExecutionUpdate[] = [];
    let workerError = '';
    let captured!: () => void;
    const capture = new Promise<void>(resolve => { captured = resolve; });
    const controller = new ValidationController({
      trusted: () => true, choose: async () => 'test', approve: async () => true,
      prepare: (target, runId, generation, profile, signal) => readValidationSnapshot(python, backend,
        { ...request(root), session: target.session, runId, generation, profile }, signal),
      run: async (spec, directory, receive, beforeLaunch) => {
        const result = await worker.run(spec, directory, true, receive, beforeLaunch);
        workerError = result.stderr;
        return result;
      }, cancel: () => worker.cancel(),
      publish: update => { updates.push(update); if (update.record.capture.state === 'complete') captured(); },
    });
    try {
      const pending = controller.run({ root, session: 'session', current: () => true });
      await Promise.race([capture, pending.then(outcome => { throw new Error(JSON.stringify({ outcome, updates, workerError })); })]);
      assert.equal(controller.running, true);
      assert.equal(updates[0].freshness.state, 'current');
      assert.equal(updates[0].record.process.state, 'running');
      await fs.appendFile(path.join(root, 'tensorviz.toml'), '\n# edited after capture\n');
      await pending;
      assert.equal(updates.at(-1)?.freshness.state, 'stale');
      assert.deepEqual(updates.at(-1)?.record.process, { state: 'exited', exitCode: 7 });
      assert.equal(controller.running, false);
    } finally { controller.dispose(); worker.dispose(); }
  });
});
