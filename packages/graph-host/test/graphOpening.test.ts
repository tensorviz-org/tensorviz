import * as assert from 'node:assert/strict';
import { GraphOpeningController, GraphOpeningError, type GraphOpeningDependencies } from '@tensorviz/graph-host/graphOpening';
import { isGraphOpeningState, type GraphEnvironmentRecovery, type GraphOpeningState, type GraphSetupDiscovery, type LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';
import type { ProfileLookup } from '@tensorviz/graph-host/profileValidation';

const target: LocalBrowserTargetIdentity = { workspaceFolder: '/workspace', file: 'model.py', cls: 'Net',
  documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 };
const discovery: GraphSetupDiscovery = { model: 'model:Net', sourceDigest: target.sourceDigest,
  init: [{ name: 'width', kind: 'keyword', required: true }], inputs: [{ name: 'x', kind: 'keyword', required: true }],
  examples: [], factories: [], configs: [], parents: [], constraints: [], truncated: false };
const draft = { profile: { kind: 'model', model: 'model:Net', init: { args: [], kwargs: { width: 8 } } } };
const resolved = { ok: true, profile: 'tensorviz_Net', run: {}, guard: { spec: { profile: draft.profile } } } as unknown as ProfileLookup;
const deferred = <T>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(accept => { resolve = accept; });
  return { resolve, promise };
};
function rig(initial: ProfileLookup = { ok: false, kind: 'none' }) {
  const states: GraphOpeningState[] = [];
  let found = initial, captures = 0, saves = 0, valid = true;
  const deps: GraphOpeningDependencies = {
    current: () => valid,
    lookup: async () => found,
    discover: async () => ({ discovery }),
    save: async () => { saves++; found = resolved; return {}; },
    capture: async (target, _profile, _signal, progress) => {
      captures++; progress('Arranging captured operations.');
      return { target: { ...target, backendRevision: target.backendRevision + 1 }, hasCapture: true,
        outcome: { status: 'passed', detail: 'Captured the selected input.' } };
    },
    recover: async () => target,
    parent: async () => {},
    publish: state => states.push(JSON.parse(JSON.stringify(state))),
  };
  const opening = new GraphOpeningController(deps);
  return { deps, opening, states, get captures() { return captures; }, get saves() { return saves; },
    set valid(value: boolean) { valid = value; } };
}
const action = (opening: GraphOpeningController, requestId: string, value: Parameters<GraphOpeningController['action']>[3]) =>
  opening.action(opening.state!.id, requestId, opening.state!.target, value);

const environment = (reason: GraphEnvironmentRecovery['reason'] = 'missing_libraries'): GraphEnvironmentRecovery => ({
  reason, detail: reason === 'ready' ? 'Project Python is ready.' : 'This model needs torchmetrics, OpenCV and SciPy.',
  candidates: [{ id: 'project', interpreter: '/workspace/.venv/bin/python', source: 'Saved', selected: true,
    status: reason === 'ready' ? 'libraries_present' : 'missing_libraries', python: '3.12.13', torch: '2.6.0+cu124' }],
  selectedId: 'project', requirements: ['torchmetrics', 'opencv-python', 'scipy'], missingModules: ['torchmetrics', 'cv2', 'scipy'],
});

suite('environment recovery in the same graph opening', () => {
  test('Browse checks the selected Python in the same opening before requesting library repair', async () => {
    const r = rig(); let selected = false;
    r.deps.environment = async () => ({ ...environment(), selectedId: selected ? 'browsed' : 'project',
      candidates: [{ ...environment().candidates[0], id: selected ? 'browsed' : 'project' }] });
    r.deps.recover = async (_target, requested) => { assert.equal(requested, 'environment'); selected = true; return target; };
    await r.opening.open(target); const id = r.opening.state!.id;
    await action(r.opening, 'browse', { kind: 'environment' });
    assert.equal(r.opening.state!.id, id); assert.equal(r.opening.state!.phase, 'needs_setup');
    assert.equal(r.opening.state!.environment!.selectedId, 'browsed'); assert.equal(r.captures, 0);
    assert.ok(r.states.some(state => state.phase === 'resolving' && state.detail === 'Checking the selected Python environment…'));
  });
  test('a library repair failure and its progress survive reconnect without a silent retry', async () => {
    const r = rig(); let calls = 0; const blocked = environment();
    r.deps.environment = async () => blocked;
    r.deps.recoverEnvironment = async (_target, _env, _action, _signal, progress) => {
      calls++; progress('Checking availability of triton>=3.5.0…');
      throw new GraphOpeningError('triton>=3.5.0: No distribution for this platform.', { kind: 'environment', code: 'PACKAGE_UNAVAILABLE' }, blocked);
    };
    await r.opening.open(target); const id = r.opening.state!.id;
    assert.equal(await action(r.opening, 'install', { kind: 'install_dependencies' }), false);
    assert.equal(r.opening.state!.phase, 'failed'); assert.equal(r.opening.state!.id, id);
    assert.equal(r.opening.state!.failure?.code, 'PACKAGE_UNAVAILABLE'); assert.ok(isGraphOpeningState(r.opening.state));
    assert.ok(r.states.some(state => state.detail === 'Checking availability of triton>=3.5.0…'));
    assert.equal(await action(r.opening, 'install', { kind: 'install_dependencies' }), false); assert.equal(calls, 1);
    assert.equal(r.captures, 0);
  });
  test('an explicit library retry after Stop uses a fresh operation, and changed source still refuses repair', async () => {
    const r = rig(); let calls = 0; r.deps.environment = async () => environment();
    r.deps.recoverEnvironment = async (_target, _env, _action, signal) => { assert.equal(signal.aborted, false); calls++; return false; };
    await r.opening.open(target); r.opening.cancel();
    await action(r.opening, 'retry-libraries', { kind: 'install_dependencies' }); assert.equal(calls, 1);
    r.opening.cancel(); r.valid = false;
    await action(r.opening, 'stale-retry', { kind: 'install_dependencies' }); assert.equal(calls, 1);
    assert.equal(r.opening.state!.cancellationReason, 'source_changed');
  });
  test('creation failure stays a failure with its package diagnostic, not another normal Prepare step', async () => {
    const r = rig(); let calls = 0;
    const blocked = { ...environment(), canCreate: true as const };
    r.deps.environment = async () => blocked;
    r.deps.recoverEnvironment = async () => { calls++; throw new GraphOpeningError('helper==1 is unavailable for this Python.', { kind: 'environment' }, blocked); };
    await r.opening.open(target); const id = r.opening.state!.id;
    assert.equal(await action(r.opening, 'create', { kind: 'create_environment' }), false);
    assert.equal(r.opening.state!.phase, 'failed'); assert.equal(r.opening.state!.id, id);
    assert.match(r.opening.state!.detail, /helper==1/); assert.equal(r.captures, 0); assert.equal(calls, 1);
    assert.equal(await action(r.opening, 'create', { kind: 'create_environment' }), false);
    assert.equal(calls, 1, 'a failed request replay cannot silently rebuild');
  });
  test('fresh environment creation publishes progress and resumes the same model without agent help', async () => {
    const r = rig(); let ready = false;
    r.deps.environment = async () => ({ ...environment(ready ? 'ready' : 'missing_libraries'), canCreate: true });
    r.deps.recoverEnvironment = async (selected, _env, requested, _signal, progress) => {
      assert.deepEqual(selected, target); assert.equal(requested.kind, 'create_environment');
      progress('Downloading and installing model libraries…'); ready = true; return true;
    };
    await r.opening.open(target); const id = r.opening.state!.id;
    assert.equal(await action(r.opening, 'create', { kind: 'create_environment', name: 'my-model' }), true);
    assert.equal(r.opening.state!.id, id); assert.equal(r.opening.state!.phase, 'needs_setup');
    assert.ok(r.opening.state!.discovery); assert.equal(r.opening.state!.environment?.reason, 'ready');
    assert.equal(r.opening.state!.environmentName, 'my-model');
    assert.ok(isGraphOpeningState(r.opening.state), 'the name survives checked snapshots for reconnects');
    assert.ok(r.states.some(state => state.detail === 'Downloading and installing model libraries…'));
    assert.equal(r.captures, 0, 'continue to Prepare; never invent a constructor or input');
  });
  test('creation needs the native capability and cannot resume after cancellation', async () => {
    const r = rig(); let calls = 0; const pending = deferred<boolean>(); let progress!: (detail: string) => void;
    r.deps.environment = async () => environment();
    r.deps.recoverEnvironment = async (_target, _env, _action, _signal, update) => { calls++; progress = update; return pending.promise; };
    await r.opening.open(target);
    assert.equal(await action(r.opening, 'unavailable', { kind: 'create_environment' }), false); assert.equal(calls, 0);
    r.deps.environment = async () => ({ ...environment(), canCreate: true }); await r.opening.open(target);
    const creating = action(r.opening, 'create', { kind: 'create_environment' });
    r.opening.cancel(); progress('Must not publish'); pending.resolve(true); await creating;
    assert.equal(r.opening.state!.phase, 'cancelled'); assert.ok(!r.states.some(state => state.detail === 'Must not publish'));
  });
  test('an explicit creation after cancellation starts a new operation in the retained opening', async () => {
    const r = rig(); let calls = 0, ready = false;
    r.deps.environment = async () => ({ ...environment(ready ? 'ready' : 'missing_libraries'), canCreate: true });
    r.deps.recoverEnvironment = async (_target, _environment, requested, signal, progress) => {
      assert.equal(requested.kind, 'create_environment'); assert.equal(signal.aborted, false);
      calls++; progress('Creating a separate Python environment for this model…'); ready = true; return true;
    };
    await r.opening.open(target); const id = r.opening.state!.id;
    r.opening.cancel('Agent stopped.');
    assert.equal(await action(r.opening, 'create-after-stop', { kind: 'create_environment' }), true);
    assert.equal(calls, 1); assert.equal(r.opening.state!.id, id);
    assert.equal(r.opening.state!.environment?.reason, 'ready'); assert.ok(r.opening.state!.discovery);
    assert.ok(r.states.some(state => state.detail === 'Creating a separate Python environment for this model…'));
    assert.equal(await action(r.opening, 'create-after-stop', { kind: 'create_environment' }), false);
    assert.equal(calls, 1, 'a replay must not create another environment');
  });
  test('creation after cancellation still refuses changed source before starting a helper', async () => {
    const r = rig(); let creations = 0;
    r.deps.environment = async () => ({ ...environment(), canCreate: true });
    r.deps.recoverEnvironment = async () => { creations++; return true; };
    await r.opening.open(target); r.opening.cancel(); r.valid = false;
    assert.equal(await action(r.opening, 'stale-create', { kind: 'create_environment' }), false);
    assert.equal(creations, 0); assert.equal(r.opening.state!.cancellationReason, 'source_changed');
  });
  test('source inspection escapes a missing dependency without declaring capture ready or importing the model', async () => {
    const r = rig(resolved); let lookups = 0;
    r.deps.environment = async () => ({ ...environment(), requirements: [], missingModules: ['torch_remat'] });
    r.deps.lookup = async () => { lookups++; return resolved; };
    await r.opening.open(target);
    const id = r.opening.state!.id;
    assert.equal(r.opening.state!.phase, 'needs_setup');
    assert.equal(await action(r.opening, 'source', { kind: 'source_only' }), true);
    assert.equal(r.opening.state!.id, id);
    assert.equal(r.opening.state!.phase, 'source_only');
    assert.equal(r.opening.state!.hasCapture, false);
    assert.equal(r.opening.state!.environment!.reason, 'missing_libraries');
    assert.equal(lookups, 0);
    assert.equal(r.captures, 0);
    assert.equal(r.saves, 0);
  });

  test('missing libraries pause before profile/model imports and explicit repair resumes the same opening once', async () => {
    const r = rig(resolved); let ready = false, lookups = 0, repairs = 0;
    r.deps.environment = async () => environment(ready ? 'ready' : 'missing_libraries');
    r.deps.lookup = async () => { lookups++; return resolved; };
    r.deps.recoverEnvironment = async (_target, recovery, choice) => {
      assert.deepEqual(recovery.requirements, ['torchmetrics', 'opencv-python', 'scipy']);
      assert.equal(choice.kind, 'install_dependencies'); repairs++; ready = true; return true;
    };
    await r.opening.open(target);
    const id = r.opening.state!.id;
    assert.equal(r.opening.state?.phase, 'needs_setup'); assert.equal(lookups, 0); assert.equal(r.captures, 0);
    assert.equal(await action(r.opening, 'install', { kind: 'install_dependencies' }), true);
    assert.equal(r.opening.state?.id, id); assert.equal(r.opening.state?.phase, 'ready');
    assert.equal(repairs, 1); assert.equal(lookups, 1); assert.equal(r.captures, 1);
    assert.ok(r.states.every(isGraphOpeningState));
  });

  test('declined and unsuccessful repairs stay recoverable without an automatic install/retry loop', async () => {
    const r = rig(resolved); let repairs = 0, accepted = false;
    r.deps.environment = async () => environment();
    r.deps.recoverEnvironment = async () => { repairs++; return accepted; };
    await r.opening.open(target);
    assert.equal(await action(r.opening, 'declined', { kind: 'install_dependencies' }), false);
    assert.equal(r.opening.state?.phase, 'needs_setup'); assert.equal(repairs, 1);
    accepted = true;
    await action(r.opening, 'retry-explicitly', { kind: 'install_dependencies' });
    assert.equal(r.opening.state?.phase, 'needs_setup'); assert.equal(repairs, 2); assert.equal(r.captures, 0);
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(repairs, 2, 'remaining missing libraries must require another explicit recovery action');
  });

  test('unlisted environment choices and constructor actions cannot bypass environment recovery', async () => {
    const r = rig(resolved); let changes = 0;
    r.deps.environment = async () => environment('selection_required');
    r.deps.recoverEnvironment = async () => { changes++; return true; };
    await r.opening.open(target);
    assert.equal(await action(r.opening, 'forged', { kind: 'select_environment', candidateId: 'not-listed' }), false);
    assert.equal(await action(r.opening, 'form', { kind: 'submit', draft }), false);
    assert.equal(await action(r.opening, 'factory', { kind: 'factory_template' }), false);
    assert.equal(changes, 0); assert.equal(r.saves, 0); assert.equal(r.captures, 0);
  });

  test('trying installed versions needs host-offered pins, then captures in the same opening', async () => {
    const r = rig(resolved); let accepted = false, choices = 0;
    const pinned: GraphEnvironmentRecovery = { ...environment('incompatible'), requirements: undefined, missingModules: undefined,
      detail: 'Installed libraries do not match this model: torch 2.7.1 (needs ==2.0.0). Choose a compatible environment.' };
    r.deps.environment = async () => accepted ? environment('ready') : pinned;
    r.deps.recoverEnvironment = async (_target, recovery, choice) => {
      assert.equal(choice.kind, 'use_installed_versions'); assert.deepEqual(recovery.declaredPins, ['torch==2.0.0']);
      choices++; accepted = true; return environment('ready');
    };
    await r.opening.open(target);
    // Without pins offered by the host there is nothing to try.
    assert.equal(await action(r.opening, 'no-pins', { kind: 'use_installed_versions' }), false);
    assert.equal(choices, 0); assert.equal(r.captures, 0);
    r.deps.environment = async () => accepted ? environment('ready') : { ...pinned, declaredPins: ['torch==2.0.0'] };
    await r.opening.open(target); const id = r.opening.state!.id;
    assert.equal(await action(r.opening, 'try', { kind: 'use_installed_versions' }), true);
    assert.equal(r.opening.state?.id, id);
    assert.equal(r.opening.state?.phase, 'ready'); assert.equal(choices, 1); assert.equal(r.captures, 1);
    assert.ok(r.states.every(isGraphOpeningState));
  });

  test('replacing pinned versions needs the host to mark them replaceable', async () => {
    const r = rig(resolved); let replaced = false, replacements = 0;
    const pinned: GraphEnvironmentRecovery = { ...environment('incompatible'), requirements: undefined, missingModules: undefined,
      declaredPins: ['torch==2.0.0'], detail: 'Installed libraries do not match this model: torch 2.7.1 (needs ==2.0.0). Choose a compatible environment.' };
    r.deps.environment = async () => replaced ? environment('ready') : pinned;
    r.deps.recoverEnvironment = async (_target, _recovery, choice) => {
      assert.equal(choice.kind, 'replace_versions'); replacements++; replaced = true; return environment('ready');
    };
    await r.opening.open(target);
    assert.equal(await action(r.opening, 'not-offered', { kind: 'replace_versions' }), false);
    assert.equal(replacements, 0);
    r.deps.environment = async () => replaced ? environment('ready') : { ...pinned, replaceable: true };
    await r.opening.open(target);
    assert.equal(await action(r.opening, 'replace', { kind: 'replace_versions' }), true);
    assert.equal(replacements, 1); assert.equal(r.opening.state?.phase, 'ready'); assert.equal(r.captures, 1);
    assert.ok(r.states.every(isGraphOpeningState));
  });

  test('selecting a listed candidate retains opening identity and checks fresh setup before capture', async () => {
    const r = rig(resolved); let selected = false;
    r.deps.environment = async () => environment(selected ? 'ready' : 'selection_required');
    r.deps.recoverEnvironment = async (_target, _recovery, choice) => {
      assert.deepEqual(choice, { kind: 'select_environment', candidateId: 'project' }); selected = true; return true;
    };
    await r.opening.open(target); const id = r.opening.state!.id;
    await action(r.opening, 'choose', { kind: 'select_environment', candidateId: 'project' });
    assert.equal(r.opening.state?.id, id); assert.equal(r.opening.state?.phase, 'ready'); assert.equal(r.captures, 1);
  });

  test('changed source cancels a pending environment check and discards its old reply', async () => {
    const r = rig(resolved), pending = deferred<GraphEnvironmentRecovery>();
    r.deps.environment = () => pending.promise;
    const opening = r.opening.open(target);
    await new Promise(resolve => setImmediate(resolve));
    r.opening.invalidate({ ...target, sourceDigest: 'b'.repeat(64), documentVersion: 2 });
    pending.resolve(environment('ready')); await opening;
    assert.equal(r.opening.state?.phase, 'cancelled'); assert.equal(r.opening.state?.environment, undefined);
    assert.equal(r.captures, 0); assert.ok(r.states.every(isGraphOpeningState));
  });

  test('a runtime import failure can return to environment recovery without reopening input setup', async () => {
    const r = rig(resolved);
    r.deps.environment = async () => environment('ready');
    r.deps.capture = async () => { throw new GraphOpeningError('A deferred model import needs SciPy.', { kind: 'environment' }, environment()); };
    await r.opening.open(target);
    assert.equal(r.opening.state?.phase, 'needs_setup'); assert.equal(r.opening.state?.environment?.reason, 'missing_libraries');
    assert.equal(r.saves, 0); assert.ok(r.states.every(isGraphOpeningState));
  });

  test('refreshing environment metadata never captures even when a previous selection is ready', async () => {
    const r = rig(resolved);
    r.deps.environment = async () => environment('ready');
    r.deps.recoverEnvironment = async () => environment('ready');
    await r.opening.open(target); assert.equal(r.captures, 1);
    const id = r.opening.state!.id;
    await action(r.opening, 'refresh-choices', { kind: 'refresh_environments' });
    assert.equal(r.opening.state?.id, id); assert.equal(r.opening.state?.phase, 'needs_setup');
    assert.equal(r.opening.state?.environment?.reason, 'selection_required'); assert.equal(r.captures, 1);
    assert.ok(r.states.every(isGraphOpeningState));
  });
});

suite('captured graph opening', () => {
  test('a declined dependency install remains an environment failure with a useful retry instruction', async () => {
    const r = rig(resolved);
    r.deps.lookup = async () => { throw new GraphOpeningError('Installation cancelled. Refresh to review it again.', { kind: 'environment' }); };
    await r.opening.open(target);
    assert.equal(r.opening.state?.phase, 'failed');
    assert.equal(r.opening.state?.failure?.kind, 'environment');
    assert.match(r.opening.state!.detail, /Refresh/);
    assert.equal(r.captures, 0);
  });
  test('installing a discovered dependency resolves a fresh profile before retrying capture', async () => {
    const r = rig(resolved);
    let lookups = 0, captures = 0;
    r.deps.lookup = async () => { lookups++; return resolved; };
    r.deps.capture = async target => {
      captures++;
      assert.equal(lookups, captures, 'do not reuse a pre-installation execution/environment guard');
      return captures === 1
        ? { target, hasCapture: false, dependenciesInstalled: true, outcome: { status: 'failed', detail: 'Missing library.' } }
        : { target, hasCapture: true, outcome: { status: 'passed', detail: 'Captured with the installed library.' } };
    };
    await r.opening.open(target);
    assert.equal(r.opening.state?.phase, 'ready');
    assert.equal(captures, 2);
    assert.ok(r.states.every(isGraphOpeningState));
  });
  test('source invalidation is refreshable, but explicit cancellation stays stopped across later edits', async () => {
    const r = rig(resolved);
    await r.opening.open(target);
    const changed = { ...target, sourceDigest: 'b'.repeat(64), documentVersion: 2 };
    r.opening.invalidate(changed);
    assert.equal(r.opening.state?.cancellationReason, 'source_changed');
    assert.ok(isGraphOpeningState(r.opening.state));
    r.opening.cancel();
    r.opening.invalidate({ ...changed, sourceDigest: 'c'.repeat(64), documentVersion: 3 });
    assert.equal(r.opening.state?.phase, 'cancelled');
    assert.equal(r.opening.state?.cancellationReason, undefined);
    assert.equal(r.captures, 1, 'neither invalidation nor cancellation executes a model');
  });
  test('a profile can follow an automatic recapture without reopening setup or recapturing', async () => {
    const r = rig(resolved);
    await r.opening.open(target);
    const saved = { ...target, sourceDigest: 'b'.repeat(64), documentVersion: 2, backendRevision: 4 };
    r.opening.invalidate(saved);
    r.opening.acceptAutomaticCapture(saved, 'Captured after save.');
    const profiled = { ...saved, backendRevision: 5 };
    r.opening.acceptMeasuredProjection(profiled);
    assert.deepEqual(r.opening.state?.target, profiled);
    assert.equal(r.opening.state?.phase, 'ready');
    assert.equal(r.opening.state?.hasCapture, true);
    assert.equal(r.opening.state?.cancellationReason, undefined);
    assert.equal(r.captures, 1, 'adopting either completed run must not execute another capture');
  });

  test('a later profiled projection advances ready setup without erasing edits or source freshness', async () => {
    const r = rig(resolved);
    await r.opening.open(target);
    const next = { ...r.opening.state!.target, backendRevision: 4 };
    r.opening.acceptMeasuredProjection(next);
    assert.deepEqual(r.opening.state?.target, next);
    assert.equal(r.opening.state?.phase, 'ready');
    assert.equal(r.captures, 1, 'adopting the measured projection never executes again');
    r.opening.acceptMeasuredProjection({ ...next, backendRevision: 3 });
    assert.equal(r.opening.state?.target.backendRevision, 4);
    await r.opening.open(next, true);
    r.opening.acceptMeasuredProjection({ ...next, backendRevision: 5 });
    assert.equal(r.opening.state?.phase, 'needs_setup');
    assert.equal(r.opening.state?.target.backendRevision, 4);
    const changed = { ...next, sourceDigest: 'b'.repeat(64), backendRevision: 6 };
    r.opening.invalidate(changed);
    r.opening.acceptMeasuredProjection(changed);
    assert.equal(r.opening.state?.phase, 'cancelled');
    assert.equal(r.opening.state?.hasCapture, false);
  });

  test('save-driven capture cannot promote stale source, another model or an explicit opening in flight', async () => {
    const r = rig(resolved);
    await r.opening.open(target);
    r.opening.invalidate(target, true);
    r.valid = false;
    r.opening.acceptAutomaticCapture(target, 'Late capture');
    assert.equal(r.opening.state!.phase, 'cancelled');
    r.valid = true;
    r.opening.acceptAutomaticCapture({ ...target, cls: 'Other' }, 'Other model');
    assert.equal(r.opening.state!.phase, 'cancelled');
    const pending = deferred<Awaited<ReturnType<GraphOpeningDependencies['capture']>>>();
    r.deps.capture = () => pending.promise;
    const opening = r.opening.open(target);
    for (let i = 0; i < 10; i++) await Promise.resolve();
    assert.equal(r.opening.state!.phase, 'capturing');
    r.opening.acceptAutomaticCapture(target, 'Older automatic run');
    assert.equal(r.opening.state!.phase, 'capturing');
    pending.resolve({ target, hasCapture: false, outcome: { status: 'failed', detail: 'Explicit run failed.' } });
    await opening;
    assert.equal(r.opening.state!.phase, 'failed');
  });

  test('file presence is independent of missing profiles; stat errors are unknown', async () => {
    const r = rig();
    r.deps.hasConfig = async () => true;
    await r.opening.open(target);
    assert.equal(r.opening.state?.phase, 'needs_setup');
    assert.equal(r.opening.state?.configFileExists, true);
    r.deps.hasConfig = async () => false;
    await r.opening.refreshConfigFile();
    assert.equal(r.opening.state?.configFileExists, false);
    r.deps.hasConfig = async () => { throw new Error('permission denied'); };
    await r.opening.refreshConfigFile();
    assert.equal(r.opening.state?.configFileExists, undefined);
    assert.equal(r.captures, 0); assert.equal(r.saves, 0);
    assert.ok(r.states.every(isGraphOpeningState));
  });
  test('saving checks presence again; later deletion retains the capture without running', async () => {
    const r = rig();
    r.deps.hasConfig = async () => r.saves > 0;
    await r.opening.open(target);
    assert.equal(r.opening.state?.configFileExists, false);
    await action(r.opening, 'save-config', { kind: 'submit', draft });
    assert.equal(r.opening.state?.configFileExists, true);
    r.deps.hasConfig = async () => false;
    await r.opening.refreshConfigFile();
    assert.equal(r.opening.state?.configFileExists, false);
    assert.equal(r.opening.state?.phase, 'ready');
    assert.equal(r.opening.state?.hasCapture, true);
    assert.equal(r.captures, 1); assert.equal(r.saves, 1);
  });
  test('late presence checks cannot overwrite a newer check or another model', async () => {
    const r = rig();
    await r.opening.open(target);
    const old = deferred<boolean>();
    r.deps.hasConfig = () => old.promise;
    const pending = r.opening.refreshConfigFile();
    r.deps.hasConfig = async () => true;
    await r.opening.refreshConfigFile();
    old.resolve(false); await pending;
    assert.equal(r.opening.state?.configFileExists, true);
    const other = deferred<boolean>();
    r.deps.hasConfig = () => other.promise;
    const stale = r.opening.refreshConfigFile();
    r.deps.hasConfig = async () => true;
    await r.opening.open({ ...target, workspaceFolder: '/other', cls: 'Other' });
    other.resolve(false); await stale;
    assert.equal(r.opening.state?.configFileExists, true);
  });
  test('source-only warning actions open help without resuming setup or execution', async () => {
    const r = rig();
    r.deps.hasConfig = async () => false;
    const recovered: string[] = [];
    r.deps.recover = async (_target, kind) => { recovered.push(kind); return undefined; };
    await r.opening.open(target);
    await action(r.opening, 'source', { kind: 'source_only' });
    await action(r.opening, 'guide', { kind: 'agent_help' });
    await action(r.opening, 'manual', { kind: 'edit_config' });
    assert.deepEqual(recovered, ['agent_help', 'edit_config']);
    assert.equal(r.opening.state?.phase, 'source_only');
    assert.equal(r.captures, 0); assert.equal(r.saves, 0);
    await r.opening.open(target);
    r.opening.cancel();
    await action(r.opening, 'cancelled-guide', { kind: 'agent_help' });
    assert.deepEqual(recovered, ['agent_help', 'edit_config', 'agent_help']);
    assert.equal(r.opening.state?.phase, 'cancelled');
    assert.equal(r.captures, 0); assert.equal(r.saves, 0);
  });
  test('agent help preserves the opening and never saves or captures; manual submission still works', async () => {
    const r = rig();
    const actions: string[] = [];
    r.deps.recover = async (_target, kind) => { actions.push(kind); return undefined; };
    await r.opening.open(target);
    const before = JSON.stringify(r.opening.state);
    assert.equal(await action(r.opening, 'help', { kind: 'agent_help' }), true);
    assert.deepEqual(actions, ['agent_help']);
    assert.equal(JSON.stringify(r.opening.state), before);
    assert.equal(r.captures, 0); assert.equal(r.saves, 0);
    await action(r.opening, 'manual-submit', { kind: 'submit', draft });
    assert.equal(r.captures, 1); assert.equal(r.saves, 1);
    assert.equal(r.opening.state?.phase, 'ready');
  });
  test('retry can reread edited source while stale setup submissions remain refused', async () => {
    const r = rig(resolved);
    let restarts = 0;
    r.deps.restart = async requested => { assert.equal(requested.file, target.file); restarts++; };
    await r.opening.open(target);
    r.valid = false;
    assert.equal(await action(r.opening, 'retry-current-source', { kind: 'retry' }), true);
    assert.equal(restarts, 1);
    const oldTarget = r.opening.state!.target;
    r.opening.invalidate({ ...oldTarget, documentVersion: 2, sourceDigest: 'b'.repeat(64) });
    assert.equal(await r.opening.action(r.opening.state!.id, 'retry-in-flight', oldTarget, { kind: 'retry' }), true);
    assert.equal(restarts, 2);
    assert.equal(await action(r.opening, 'stale-form', { kind: 'submit', draft }), false);
    assert.equal(r.saves, 0);
    assert.equal(restarts, 2);
    assert.equal(await r.opening.action('old-opening', 'old-retry', target, { kind: 'retry' }), false);
    assert.equal(await r.opening.action(r.opening.state!.id, 'foreign-retry', { ...target, cls: 'Other' }, { kind: 'retry' }), false);
  });
  for (const kind of ['execution', 'source', 'environment', 'capture'] as const) {
    test(`${kind} failure preserves the exact recipe without setup discovery`, async () => {
      const r = rig(resolved);
      r.deps.discover = async () => { assert.fail('This failure cannot be repaired by new inputs.'); };
      r.deps.capture = async () => ({ target, hasCapture: kind === 'execution',
        outcome: { status: 'failed', detail: 'Failed.', failure: { kind } } });
      await r.opening.open(target);
      assert.equal(r.opening.state?.phase, 'failed');
      assert.equal(r.opening.state?.failure?.kind, kind);
      assert.deepEqual(r.opening.state?.draft?.profile, draft.profile);
      assert.equal(r.saves, 0);
      assert.ok(r.states.every(isGraphOpeningState));
    });
  }
  test('a new required input reopens only setup and retains existing bindings', async () => {
    const r = rig(resolved);
    r.deps.capture = async () => ({ target, hasCapture: false, outcome: { status: 'refused',
      detail: 'Missing mask.', failure: { kind: 'setup', fields: ['inputs.mask'] } } });
    await r.opening.open(target);
    assert.equal(r.opening.state?.phase, 'needs_setup');
    assert.deepEqual(r.opening.state?.draft?.profile, draft.profile);
    assert.deepEqual(r.opening.state?.requirements, ['inputs.mask: Check this binding.']);
    assert.equal(r.saves, 0);
  });
  const automatic = (r: ReturnType<typeof rig>, status: 'complete' | 'partial' = 'complete', requirements?: string[]) => {
    r.deps.discover = async () => ({ discovery: { ...discovery, resolution: { status, draft,
      evidence: [{ field: 'inputs.x', kind: 'example', detail: 'Synthetic input; batch 1.' }],
      unresolved: status === 'partial' ? ['inputs.context: provide a value.'] : [] } }, draft, requirements });
  };
  test('a complete source recipe saves and captures once without displaying a form', async () => {
    const r = rig(); automatic(r);
    await r.opening.open(target);
    assert.equal(r.saves, 1);
    assert.equal(r.captures, 1);
    assert.equal(r.opening.state?.phase, 'ready');
    assert.equal(r.opening.state?.evidence?.[0].kind, 'example');
    assert.ok(r.states.every(state => state.phase !== 'needs_setup' && isGraphOpeningState(state)));
  });
  test('a partial recipe prefills the form without executing or writing', async () => {
    const r = rig(); automatic(r, 'partial');
    await r.opening.open(target);
    assert.equal(r.saves, 0); assert.equal(r.captures, 0);
    assert.deepEqual(r.opening.state?.draft, draft);
    assert.equal(r.opening.state?.phase, 'needs_setup');
  });
  test('dirty sources block even a complete recipe', async () => {
    const r = rig(); automatic(r, 'complete', ['Save changed source.']);
    await r.opening.open(target);
    assert.equal(r.saves, 0); assert.equal(r.captures, 0);
    assert.deepEqual(r.opening.state?.requirements, ['Save changed source.']);
  });
  test('invalid saved setup and Change inputs never use automatic recipes', async () => {
    for (const blocked of [true, false]) {
      const r = rig(blocked ? { ok: false, kind: 'blocked', detail: 'Invalid saved profile.' } : { ok: false, kind: 'none' });
      automatic(r);
      await r.opening.open(target, !blocked);
      assert.equal(r.saves, 0); assert.equal(r.captures, 0);
      assert.equal(r.opening.state?.phase, 'needs_setup');
    }
  });
  test('cancelling an automatic save prevents late execution', async () => {
    const r = rig(); automatic(r);
    const started = deferred<void>(), pending = deferred<{}>();
    r.deps.save = async () => { started.resolve(); return pending.promise; };
    const opening = r.opening.open(target);
    await started.promise;
    r.opening.cancel(); pending.resolve({}); await opening;
    assert.equal(r.captures, 0);
    assert.equal(r.opening.state?.phase, 'cancelled');
  });
  test('post-save lookup cannot cause an automatic discovery loop', async () => {
    const r = rig(); automatic(r);
    let saves = 0;
    r.deps.save = async () => { saves++; return {}; };
    await r.opening.open(target);
    assert.equal(saves, 1); assert.equal(r.captures, 0);
    assert.equal(r.opening.state?.phase, 'needs_setup');
  });
  test('a failed automatic example keeps its exact attempt and does not retry', async () => {
    const r = rig(); automatic(r);
    let captures = 0;
    r.deps.capture = async () => { captures++; return { target, hasCapture: true,
      outcome: { status: 'failed', detail: 'Example dimensions do not fit the later operation.' } }; };
    await r.opening.open(target);
    assert.equal(r.saves, 1); assert.equal(captures, 1);
    assert.equal(r.opening.state?.phase, 'failed');
    assert.equal(r.opening.state?.draft?.profileName, 'tensorviz_Net');
    assert.equal(r.opening.state?.evidence?.[0].kind, 'example');
  });

  test('factory scaffold retains the form and does not execute', async () => {
    const r = rig();
    r.deps.factoryTemplate = async (_target, draft) => ({ ...draft!, configObject: undefined, profile: { ...draft!.profile, init: undefined, factory: { file: '.tensorviz/setup_Net_0123456789ab.py', callable: 'make_model' } } });
    await r.opening.open(target);
    await action(r.opening, 'helper', { kind: 'factory_template', draft });
    assert.equal(r.opening.state?.draft?.profile.init, undefined);
    assert.equal(isGraphOpeningState(r.opening.state), true, 'the host publishes a valid wire state after removing constructor fields');
    assert.ok(r.opening.state?.draft?.profile.factory);
    assert.equal(r.opening.state?.phase, 'needs_setup');
    assert.equal(r.captures, 0);
  });
  test('a configured model captures once without a second user action', async () => {
    const r = rig(resolved);
    await r.opening.open(target);
    assert.equal(r.captures, 1);
    assert.equal(r.saves, 0);
    assert.equal(r.opening.state?.phase, 'ready');
    assert.equal(r.opening.state?.hasCapture, true);
    assert.equal(r.opening.state?.target.backendRevision, 2);
    assert.ok(!r.states.some(state => state.phase === 'needs_setup'));
  });
  test('missing setup waits without searching inputs; submit resumes the same request', async () => {
    const r = rig();
    await r.opening.open(target);
    assert.equal(r.opening.state?.phase, 'needs_setup');
    assert.equal(r.captures, 0);
    const id = r.opening.state!.id;
    await action(r.opening, 'submit-1', { kind: 'submit', draft });
    assert.equal(r.saves, 1);
    assert.equal(r.captures, 1);
    assert.equal(r.opening.state?.id, id);
    assert.equal(r.opening.state?.phase, 'ready');
  });
  test('invalid explicit settings never fall back to automatic execution', async () => {
    const r = rig({ ok: false, kind: 'blocked', detail: 'The configured checkpoint is missing.' });
    await r.opening.open(target);
    assert.equal(r.captures, 0);
    assert.equal(r.opening.state?.phase, 'needs_setup');
    assert.match(r.opening.state!.detail, /checkpoint/);
  });
  test('all remaining requirements are shown and entered values are retained', async () => {
    const r = rig();
    r.deps.save = async () => ({ requirements: ['init.width is required', 'inputs.x is required'] });
    await r.opening.open(target);
    await action(r.opening, 'submit', { kind: 'submit', draft });
    assert.deepEqual(r.opening.state?.requirements, ['init.width is required', 'inputs.x is required']);
    assert.deepEqual(r.opening.state?.draft, draft);
    assert.equal(r.captures, 0);
  });
  test('a real forward failure retains its captured prefix and chosen setup', async () => {
    const r = rig(resolved);
    r.deps.capture = async () => ({ target, hasCapture: true, outcome: { status: 'failed', detail: 'Dimension 8 does not match 4.' } });
    await r.opening.open(target);
    assert.equal(r.opening.state?.phase, 'failed');
    assert.equal(r.opening.state?.hasCapture, true);
    assert.equal(r.opening.state?.draft?.profileName, 'tensorviz_Net');
    assert.equal(r.opening.state?.discovery, undefined, 'a forward failure never rediscovers or rewrites setup');
  });
  test('execution success without a measured projection cannot mark the opening ready', async () => {
    const r = rig(resolved);
    r.deps.capture = async () => ({ target, hasCapture: false, outcome: { status: 'passed', detail: 'Worker returned.' } });
    await r.opening.open(target);
    assert.equal(r.opening.state?.phase, 'failed');
    assert.match(r.opening.state!.detail, /without a captured graph/);
    assert.equal(r.opening.state?.hasCapture, false);
  });
  test('failure to read recovery fields preserves the original failure and attempted profile', async () => {
    const r = rig(resolved);
    r.deps.capture = async () => ({ target, hasCapture: false, outcome: { status: 'failed', detail: 'Missing input.', failure: { kind: 'setup' } } });
    r.deps.discover = async () => { throw new Error('discovery unavailable'); };
    await r.opening.open(target);
    assert.equal(r.opening.state?.phase, 'failed');
    assert.match(r.opening.state!.detail, /Missing input.*Setup fields could not be read/);
    assert.equal(r.opening.state?.draft?.profileName, 'tensorviz_Net');
  });
  test('change inputs uses the selected profile and does not run until submit', async () => {
    const r = rig(resolved);
    await r.opening.open(target, true);
    assert.equal(r.captures, 0);
    assert.deepEqual(r.opening.state?.draft?.profile, draft.profile);
    assert.equal(r.opening.state?.phase, 'needs_setup');
  });
  for (const failed of [false, true]) {
    test(`saved configuration fields remain editable after ${failed ? 'a failed capture' : 'Change inputs'}`, async () => {
      const r = rig(resolved);
      const restored = { profileName: 'tensorviz_Net', profile: draft.profile,
        configObject: { index: 0, parameter: 'config', values: { depth: 3 } } };
      r.deps.discover = async (_target, _signal, saved) => {
        assert.deepEqual(saved?.profile, draft.profile);
        return { discovery, restoredDraft: restored };
      };
      if (failed) r.deps.capture = async () => ({ target, hasCapture: false,
        outcome: { status: 'failed', detail: 'Missing argument.', failure: { kind: 'setup' } } });
      await r.opening.open(target, !failed);
      assert.deepEqual(r.opening.state?.draft, restored);
      assert.equal(r.opening.state?.phase, 'needs_setup');
      assert.equal(r.saves, 0, 'restoring the form never writes or runs a new example');
    });
  }
  test('source-only is explicit and cancels pending setup', async () => {
    const r = rig();
    await r.opening.open(target);
    await action(r.opening, 'source', { kind: 'source_only' });
    assert.equal(r.opening.state?.phase, 'source_only');
    assert.equal(r.captures, 0);
  });
  test('late save completion after cancellation cannot launch a model', async () => {
    const r = rig();
    const pending = deferred<{}>();
    r.deps.save = async () => pending.promise;
    await r.opening.open(target);
    const submit = action(r.opening, 'submit', { kind: 'submit', draft });
    await action(r.opening, 'cancel', { kind: 'cancel' });
    pending.resolve({});
    await submit;
    assert.equal(r.captures, 0);
    assert.equal(r.opening.state?.phase, 'cancelled');
  });
  test('retargeting rejects an old form and ignores a late discovery', async () => {
    const r = rig();
    const pending = deferred<{ discovery: GraphSetupDiscovery }>();
    r.deps.discover = async () => pending.promise;
    const first = r.opening.open(target);
    await Promise.resolve();
    const oldId = r.opening.state!.id;
    r.deps.discover = async () => ({ discovery: { ...discovery, model: 'model:Other' } });
    await r.opening.open({ ...target, cls: 'Other' });
    pending.resolve({ discovery });
    await first;
    assert.equal(r.opening.state?.target.cls, 'Other');
    assert.equal(await r.opening.action(oldId, 'old-submit', target, { kind: 'submit', draft }), false);
    assert.equal(r.saves, 0);
  });
  test('source changes during lookup stop the request even without another UI event', async () => {
    const r = rig(resolved);
    r.deps.lookup = async () => { r.valid = false; return resolved; };
    await r.opening.open(target);
    assert.equal(r.captures, 0);
    assert.equal(r.opening.state?.phase, 'cancelled');
  });
  test('duplicate form submission cannot save or capture twice', async () => {
    const r = rig();
    await r.opening.open(target);
    const id = r.opening.state!.id;
    await action(r.opening, 'submit', { kind: 'submit', draft });
    assert.equal(await r.opening.action(id, 'submit', target, { kind: 'submit', draft }), false);
    assert.equal(r.saves, 1);
    assert.equal(r.captures, 1);
  });
});

suite('constructed-model input preparation', () => {
  const inputDraft = { profile: { kind: 'model', model: 'model:Net', init: { args: [], kwargs: {} },
    inputs: { args: [], kwargs: {} } } };
  const inferred = { profile: { ...inputDraft.profile, inputs: { args: [], kwargs: {
    x: { kind: 'tensor', shape: [1, 8], dtype: 'float32' } } } } };
  const prepare = (r: ReturnType<typeof rig>, requirements?: string[]) => {
    r.deps.discover = async () => ({ discovery: { ...discovery, resolution: { status: 'partial',
      runtimeInputs: true, draft: inputDraft, evidence: [], unresolved: ['inputs.x: provide a value.'] } },
      draft: inputDraft, requirements });
  };
  test('a runtime proposal is saved and then captured once through normal preflight', async () => {
    const r = rig(); prepare(r); let probes = 0;
    r.deps.inferInputs = async () => { probes++; return { draft: inferred, detail: 'Example from constructed Linear.' }; };
    const save = r.deps.save;
    r.deps.save = async (target, draft, signal) => { assert.deepEqual(draft, inferred); return save(target, draft, signal); };
    await r.opening.open(target);
    assert.equal(probes, 1); assert.equal(r.saves, 1); assert.equal(r.captures, 1);
    assert.equal(r.opening.state?.phase, 'ready');
    assert.ok(r.states.every(isGraphOpeningState));
  });
  test('unsupported input preparation keeps setup open and starts no forward', async () => {
    const r = rig(); prepare(r);
    r.deps.inferInputs = async () => ({ detail: 'No input boundary found.' });
    await r.opening.open(target);
    assert.equal(r.saves, 0); assert.equal(r.captures, 0);
    assert.equal(r.opening.state?.phase, 'needs_setup');
    assert.ok(r.opening.state?.requirements?.includes('No input boundary found.'));
  });
  test('cancellation discards a late proposal without saving or capturing', async () => {
    const r = rig(); prepare(r); const pending = deferred<{ draft: typeof inferred; detail: string }>();
    r.deps.inferInputs = () => pending.promise;
    const opening = r.opening.open(target);
    await new Promise(resolve => setImmediate(resolve));
    r.opening.cancel(); pending.resolve({ draft: inferred, detail: 'Late proposal.' }); await opening;
    assert.equal(r.saves, 0); assert.equal(r.captures, 0); assert.equal(r.opening.state?.phase, 'cancelled');
  });
  test('a failed forward remains failed and does not launch another input probe', async () => {
    const r = rig(); prepare(r); let probes = 0, forwards = 0;
    r.deps.inferInputs = async () => { probes++; return { draft: inferred, detail: 'Example.' }; };
    r.deps.capture = async () => { forwards++; return { target, hasCapture: true,
      outcome: { status: 'failed', detail: 'The model rejected this input.' } }; };
    await r.opening.open(target);
    assert.equal(probes, 1); assert.equal(forwards, 1); assert.equal(r.opening.state?.phase, 'failed');
  });
  for (const reason of ['saved', 'change_inputs', 'dirty']) {
    test(`${reason} setup is never replaced by an automatic probe`, async () => {
      const r = rig(reason === 'saved' ? resolved : { ok: false, kind: 'none' });
      prepare(r, reason === 'dirty' ? ['Save changed files.'] : undefined);
      r.deps.inferInputs = async () => { assert.fail('No constructor probe is authorized for this setup.'); };
      await r.opening.open(target, reason === 'change_inputs');
      assert.equal(r.saves, 0);
    });
  }
});


suite('agent setup draft handoff', () => {
  test('current manual edits become agent context without saving or running', async () => {
    const r = rig(); let received: unknown;
    r.deps.recover = async (_target, kind, _signal, request) => {
      assert.equal(kind, 'agent_help'); received = request;
      assert.deepEqual(r.opening.state?.draft, edited); return undefined;
    };
    const edited = { profile: { kind: 'model', inputs: { args: [], kwargs: { batch: 7 } } } };
    await r.opening.open(target, true);
    assert.equal(await action(r.opening, 'current-draft', { kind: 'agent_help', draft: edited }), true);
    assert.deepEqual(received, { kind: 'agent_help', draft: edited });
    assert.equal(r.saves, 0); assert.equal(r.captures, 0);
    assert.equal(await r.opening.action('old-opening', 'old', target, { kind: 'agent_help', draft: edited }), false);
  });
});
