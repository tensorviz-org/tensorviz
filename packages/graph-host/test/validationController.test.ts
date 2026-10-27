import * as assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as path from 'node:path';
import { ValidationController, ValidationControllerOptions, ValidationPreparation, ValidationTarget } from '@tensorviz/graph-host/validationController';
import type { ExecutionRecord, RunSpec, ValidationExecutionUpdate } from '@tensorviz/contracts/protocol';

const corpus = JSON.parse(readFileSync(path.join(__dirname, '../../../tests/fixtures/contracts/validation-contracts.json'), 'utf8'));
const savedSpec: RunSpec = corpus.structural.find((f: { schema: string }) => f.schema === 'RunSpec').value;
const captured: ExecutionRecord = corpus.structural.find((f: { name: string }) => f.name === 'complete capture with live process').value.record;
const tick = () => new Promise<void>(resolve => setImmediate(resolve));

suite('single validation lifecycle', () => {
  let controller: ValidationController;
  let options: ValidationControllerOptions;
  let receive: (record: ExecutionRecord, dependenciesCurrent?: boolean) => void;
  let finish: (result: { record: ExecutionRecord }) => void;
  let spec: RunSpec;
  let current: boolean, trusted: boolean, runs: number, cancels: number;
  let target: ValidationTarget;
  let published: ValidationExecutionUpdate[];
  let preparation: ValidationPreparation;
  setup(() => {
    current = trusted = true; runs = cancels = 0; published = [];
    target = { root: '/workspace', session: 'session-1', current: () => current };
    preparation = { profiles: ['encoder'], guards: {}, resolution: { status: 'resolved', spec: structuredClone(savedSpec) } };
    options = {
      trusted: () => trusted,
      prepare: async (_target, id, generation) => {
        const result = structuredClone(preparation);
        if (result.resolution.status === 'resolved') {
          result.resolution.spec.identity.runId = id;
          result.resolution.spec.identity.generation = generation;
        }
        return result;
      },
      choose: async () => 'encoder', approve: async () => true,
      run: async (value, _root, callback) => { runs++; spec = value; receive = callback; return new Promise(resolve => { finish = resolve; }); },
      cancel: () => { cancels++; }, publish: update => published.push(update),
    };
    controller = new ValidationController(options);
  });
  teardown(() => controller.dispose());
  async function start() {
    const pending = controller.run(target);
    await tick();
    assert.equal(runs, 1);
    return { pending };
  }
  function record(sequence: number, exitCode?: number): ExecutionRecord {
    return { ...structuredClone(captured), identity: spec.identity, sequence,
      process: exitCode === undefined ? { state: 'running' } : { state: 'exited', exitCode } };
  }
  test('one controller refuses competing surface runs while keeping the original process', async () => {
    const { pending } = await start();
    assert.equal((await controller.run({ ...target, session: 'another-surface' })).status, 'refused');
    receive(record(0)); await tick();
    assert.equal(controller.running, true);
    assert.equal(published[0].record.process.state, 'running');
    finish({ record: record(1, 7) }); await pending;
    assert.deepEqual(published.at(-1)?.record.process, { state: 'exited', exitCode: 7 });
    assert.equal(runs, 1);
  });
  test('a completed process keeps its exit record while the compatibility result stays refused', async () => {
    const { pending } = await start();
    receive(record(1)); receive(record(1)); await tick();
    assert.equal(published.length, 1);
    finish({ record: record(2, 0) });
    assert.equal((await pending).status, 'refused');
    assert.deepEqual(controller.snapshot(target.session)?.record.process, { state: 'exited', exitCode: 0 });
  });
  test('cancelled profile selection does not launch a worker', async () => {
    options.choose = async () => undefined;
    const outcome = await controller.run(target);
    assert.equal(outcome.status, 'refused');
    assert.equal(runs, 0);
  });
  test('foreign identity, duplicate and out-of-order snapshots cannot replace evidence', async () => {
    const { pending } = await start();
    receive({ ...record(100), identity: { ...spec.identity, runId: 'foreign' } });
    receive(record(2)); receive(record(1)); receive(record(2)); await tick();
    assert.equal(published.length, 1);
    assert.equal(published[0].record.sequence, 2);
    finish({ record: record(3, 0) }); await pending;
  });
  for (const key of ['workspaceSessionId', 'workspaceId', 'profile', 'configDigest', 'profileDigest', 'sourceDigest', 'environmentDigest'] as const) {
    test(`rejects foreign ${key}`, async () => {
      const { pending } = await start();
      receive({ ...record(0), identity: { ...spec.identity, [key]: 'foreign' } }); await tick();
      assert.equal(published.length, 0);
      finish({ record: record(1, 0) }); await pending;
    });
  }
  for (const [key, reason] of [['configDigest', 'config'], ['sourceDigest', 'source'], ['environmentDigest', 'environment'], ['profileDigest', 'profile']] as const) {
    test(`${reason} changes stale measurement but preserve exit tracking`, async () => {
      const { pending } = await start();
      if (preparation.resolution.status === 'resolved') preparation.resolution.spec.identity[key] = 'f'.repeat(64);
      receive(record(0)); await tick();
      assert.ok(published[0].freshness.reasons.includes(reason));
      assert.equal(controller.running, true);
      finish({ record: record(1, 9) }); await pending;
      assert.equal(published.at(-1)?.freshness.state, 'stale');
      assert.deepEqual(published.at(-1)?.record.process, { state: 'exited', exitCode: 9 });
    });
  }
  test('dirty buffers and saved versions are rechecked after capture', async () => {
    const { pending } = await start();
    preparation.guards['model.py'] = ['model.py', 'inode', 'digest', 4, true];
    receive(record(0)); await tick();
    assert.equal(published[0].freshness.state, 'stale');
    finish({ record: record(1, 0) }); await pending;
  });
  test('retarget/reconnect does not cancel the live script or replay current success', async () => {
    const { pending } = await start();
    receive(record(0)); await tick();
    current = false;
    controller.invalidate('session', target.session);
    assert.equal(cancels, 0);
    assert.equal(controller.snapshot(target.session)?.freshness.state, 'stale');
    assert.equal(controller.snapshot('other-session'), undefined);
    finish({ record: record(1, 0) }); await pending;
    assert.equal(published.at(-1)?.freshness.state, 'stale');
  });
  test('snapshot immediately guards a closed target before the polling tick', async () => {
    const { pending } = await start();
    receive(record(0)); await tick();
    current = false;
    assert.equal(controller.snapshot(target.session)?.freshness.state, 'stale');
    assert.equal(cancels, 0);
    finish({ record: record(1, 0) }); await pending;
  });
  test('cancel ownership survives staleness and late evidence stays stale', async () => {
    const { pending } = await start();
    controller.cancel('foreign'); assert.equal(cancels, 0);
    controller.cancel(target.session, 'previous-run'); assert.equal(cancels, 0);
    controller.invalidate('source'); controller.cancel(target.session);
    assert.equal(cancels, 1); assert.equal(controller.running, true);
    receive(record(0)); await tick();
    finish({ record: record(1, 0) });
    assert.equal((await pending).status, 'cancelled');
    assert.ok(published.at(-1)?.freshness.reasons.includes('cancelled'));
  });
  test('new untracked imports cannot become current measured evidence', async () => {
    const { pending } = await start();
    receive(record(0), false); await tick();
    assert.ok(published[0].freshness.reasons.includes('dependency_scope'));
    finish({ record: record(1, 0) }); await pending;
  });
  test('navigation resolves only current run invocation ids and tracked source locations', async () => {
    const { pending } = await start();
    const evidence = record(0);
    const file = spec.files.find(file => file.role === 'source')!;
    evidence.invocations[0].source = { file: file.path, line: 2, column: 4 };
    receive(evidence); await tick();
    const id = `execution:${spec.identity.runId}:${evidence.invocations[0].id}`;
    assert.deepEqual(controller.sourceForNode(target.session, id), { root: target.root, source: evidence.invocations[0].source, file });
    assert.equal(controller.sourceForNode('foreign', id), undefined);
    assert.equal(controller.sourceForNode(target.session, 'execution:previous:i0'), undefined);
    assert.equal(controller.sourceForNode(target.session, '../../secrets'), undefined);
    controller.invalidate('source');
    assert.equal(controller.sourceForNode(target.session, id), undefined);
    finish({ record: record(1, 0) }); await pending;
  });
  test('setup requirements never invoke a worker', async () => {
    preparation.resolution = { status: 'needs_setup', requirements: [{ code: 'unsaved', phase: 'static', profile: 'encoder', field: ['profiles', 'encoder', 'inputs'], reason: 'Save the file.', remedy: 'Save.' }] };
    let shown = false;
    options.setup = async (where, requirements) => {
      shown = true; assert.equal(where.current(), true);
      assert.deepEqual(requirements[0].field, ['profiles', 'encoder', 'inputs']);
      current = false; assert.equal(where.current(), false);
    };
    const outcome = await controller.run(target);
    assert.equal(outcome.status, 'refused'); assert.equal(runs, 0); assert.equal(shown, true);
    assert.match(outcome.detail, /profiles.encoder.inputs/);
  });
  test('missing configuration offers setup without choosing a profile or launching', async () => {
    preparation.profiles = [];
    const requirement = { code: 'missing' as const, phase: 'static' as const, profile: null,
      field: ['tensorviz.toml'], reason: 'Configuration is missing.', remedy: 'Create and save tensorviz.toml.' };
    preparation.resolution = { status: 'needs_setup', requirements: [requirement] };
    options.choose = async () => assert.fail('no profile can be selected yet');
    let shown = 0;
    options.setup = async (_target, requirements) => { shown++; assert.deepEqual(requirements, [requirement]); };
    await controller.run(target);
    assert.equal(shown, 1); assert.equal(runs, 0);
  });
  test('terminal runtime requirements reach field setup without a second run', async () => {
    let shown = 0;
    options.setup = async (_target, requirements) => { shown++; assert.equal(requirements[0].phase, 'runtime'); };
    const { pending } = await start();
    const terminal = record(1, 1);
    terminal.requirements = [{ code: 'signature', phase: 'runtime', profile: 'encoder',
      field: ['profiles', 'encoder', 'inputs'], reason: 'Required argument missing.', remedy: 'Set the argument.' }];
    finish({ record: terminal });
    assert.match((await pending).detail, /profiles.encoder.inputs/);
    assert.equal(shown, 1); assert.equal(runs, 1);
  });
  test('trust and host target are checked before any preparation', async () => {
    options.prepare = async () => { assert.fail('must not prepare'); };
    trusted = false; await controller.run(target);
    trusted = true; current = false; await controller.run(target);
    assert.equal(runs, 0);
  });
  test('an editor invalidation during approval blocks launch even before disk changes', async () => {
    options.approve = async () => { controller.invalidate('source'); return true; };
    assert.equal((await controller.run(target)).status, 'refused'); assert.equal(runs, 0);
  });
  test('edits during native approval refuse launch without rerunning with newer values', async () => {
    options.approve = async () => { preparation.guards.changed = true; return true; };
    assert.equal((await controller.run(target)).status, 'refused'); assert.equal(runs, 0);
  });
  test('only declared native files invalidate and native edits remain cancellable', async () => {
    const { pending } = await start(); receive(record(0)); await tick();
    controller.fileChanged('/workspace/log.txt');
    assert.equal(published.length, 1);
    controller.fileChanged('/workspace/' + spec.files[0].path);
    assert.equal(published.at(-1)?.freshness.state, 'stale');
    finish({ record: record(1, 0) }); await pending;
  });
});
