/** Saved settings must be honored or refused; only confirmed absence permits automatic inputs. */
import * as assert from 'node:assert/strict';
import { lookupProfile, setupDetail } from '@tensorviz/graph-host/profileValidation';
import { translateProfile, targetsFromModelSymbol } from '@tensorviz/graph-host/validationProfileBridge';
import type { ValidationPreparation } from '@tensorviz/graph-host/validationController';

const LIMITS = { timeout_ms: 60000, max_events: 10, max_tensors: 10, max_record_bytes: 1024 };
const tensor = (shape: number[]) => ({ kind: 'tensor', shape, dtype: 'float32', requires_grad: false,
  generator: { kind: 'normal', mean: 0, std: 1, seed: 0 } });
function modelProfile(symbol: string, extra: Record<string, unknown> = {}) {
  return { cwd: '.', tracked_files: [], limits: LIMITS, kind: 'model', model: symbol, mode: 'eval',
    device: 'cpu', grad_enabled: false, seed: null, init: { args: [], kwargs: {} },
    inputs: { args: [tensor([2, 8])], kwargs: {} }, ...extra } as never;
}
function scriptProfile() {
  return { cwd: '.', tracked_files: [], limits: LIMITS, kind: 'script', script: 'train.py', args: [],
    capture: { model: 'models.tiny_lm:TinyLM', call: 1 } } as never;
}
const resolved = (profile: unknown, name = 'net'): ValidationPreparation => ({ profiles: [], guards: {},
  resolution: { status: 'resolved', spec: { profile, identity: { profile: name } } } as never });

function rig(profiles: string[], byName: Record<string, unknown>, picked?: string) {
  const asked: string[][] = [];
  return {
    asked,
    options: {
      snapshot: async (_root: string, profile: string | undefined) =>
        profile === undefined
          ? ({ profiles, profileDefinitions: byName, guards: {}, resolution: { status: 'needs_setup', requirements: [
            { profile: null, field: ['profiles'], code: 'ambiguous', phase: 'static', reason: 'Select a profile.', remedy: 'Choose in the host.' },
          ] } } as never)
          : (byName[profile] ? resolved(byName[profile], profile) : ({ profiles, guards: {},
              resolution: { status: 'needs_setup', requirements: [] } } as never)),
      choose: async (names: string[]) => { asked.push(names); return picked; },
    },
  };
}
const signal = () => new AbortController().signal;

suite('saved profile lookup', () => {
  test('a saved JAX profile retains its framework through worker translation', () => {
    const result = translateProfile(modelProfile('net:net', { framework: 'jax' }));
    assert.ok(result.ok);
    assert.equal(result.run.execution.framework, 'jax');
    const torch = translateProfile(modelProfile('net:Net'));
    assert.ok(torch.ok);
    assert.equal(torch.run.execution.framework, undefined);
  });
  test('a remembered matching profile avoids repeated variant selection', async () => {
    const { options, asked } = rig(['small', 'large'], { small: modelProfile('net:Net'), large: modelProfile('net:Net') });
    const remembered: string[] = [];
    const result = await lookupProfile({ ...options, preferred: async () => 'large',
      remember: async (_root, _target, name) => { remembered.push(name); } }, '/ws', { file: 'net.py', cls: 'Net' }, signal());
    assert.ok(result.ok);
    assert.equal(result.profile, 'large');
    assert.deepEqual(asked, []);
    assert.deepEqual(remembered, ['large']);
  });
  test('a removed remembered profile cannot silently select a different invocation', async () => {
    const { options } = rig(['small'], { small: modelProfile('net:Net') });
    const result = await lookupProfile({ ...options, preferred: async () => 'large' }, '/ws', { file: 'net.py', cls: 'Net' }, signal());
    assert.ok(!result.ok && result.kind === 'blocked');
    assert.match(result.detail, /no longer matches/);
  });
  test('an explicit empty input binding is retained for zero-argument forwards', () => {
    const result = translateProfile(modelProfile('net:Net', { inputs: { args: [], kwargs: {} } }));
    assert.ok(result.ok);
    assert.deepEqual(result.run.execution.inputs, { binding: { args: [], kwargs: {} } });
  });
  test('automatic tracing refuses ambiguous profiles without opening a picker', async () => {
    const { options, asked } = rig(['small', 'large'], { small: modelProfile('net:Net'), large: modelProfile('net:Net') }, 'large');
    const found = await lookupProfile(options, '/ws', { file: 'net.py', cls: 'Net' }, signal(), false);
    assert.ok(!found.ok && found.kind === 'blocked');
    assert.match(found.detail, /Open as Graph/);
    assert.deepEqual(asked, []);
  });
  for (const timeoutMs of [100, 300_000, 86_400_000]) {
    test(`preserves the resolved profile deadline ${timeoutMs} without a second cap`, async () => {
      const { options } = rig(['net'], { net: modelProfile('net:Net', { limits: { ...LIMITS, timeout_ms: timeoutMs } }) });
      const result = await lookupProfile(options, '/ws', { file: 'net.py', cls: 'Net' }, signal());
      assert.ok(result.ok);
      assert.equal(result.run.timeoutMs, timeoutMs);
    });
  }

  test('one profile naming this class runs with no prompt', async () => {
    const { options, asked } = rig(['residual'], { residual: modelProfile('models.resid:Net') });
    const found = await lookupProfile(options, '/ws', { file: 'models/resid.py', cls: 'Net' }, signal());
    assert.equal(found.ok, true);
    assert.equal(found.ok && found.profile, 'residual');
    assert.deepEqual(found.ok && found.run.target, { file: 'models/resid.py', cls: 'Net', packageModule: false });
    assert.equal(found.ok && found.run.timeoutMs, 60000, "the profile's own limit bounds the run");
    assert.equal(asked.length, 0, 'a single match must not prompt');
  });

  test("a profile for another class is not borrowed, and does not block this one", async () => {
    const { options } = rig(['other'], { other: modelProfile('models.other:Other') });
    const found = await lookupProfile(options, '/ws', { file: 'models/resid.py', cls: 'Net' }, signal());
    assert.equal(found.ok, false);
    assert.equal(found.ok ? '' : found.kind, 'none', 'the caller validates automatically, as it always did');
  });

  test('a project with no tensorviz.toml at all still validates automatically', async () => {
    // The overwhelmingly common case, and the one that must not regress: a profile is an
    // override. Demanding one before anything runs is a contract this product never had.
    const options = {
      snapshot: async () => ({ profiles: [], guards: {}, resolution: { status: 'needs_setup', requirements: [
        { profile: null, field: ['tensorviz.toml'], phase: 'static', code: 'missing',
          reason: 'The project has no saved validation configuration.', remedy: 'Create tensorviz.toml.' },
      ] } } as never),
      choose: async () => undefined,
    };
    const found = await lookupProfile(options, '/ws', { file: 'a.py', cls: 'A' }, signal());
    assert.equal(found.ok, false);
    assert.equal(found.ok ? '' : found.kind, 'none');
  });

  test('a failed snapshot refuses without guessing or leaking exception text', async () => {
    const options = {
      snapshot: async () => { throw new Error('private-token /private/workspace source-body'); },
      choose: async () => undefined,
    };
    const found = await lookupProfile(options, '/ws', { file: 'model.py', cls: 'F5AECStftXAttn' }, signal());
    assert.equal(found.ok, false);
    assert.equal(found.ok ? '' : found.kind, 'blocked');
    assert.match(!found.ok && found.kind === 'blocked' ? found.detail : '', /Python environment/);
    assert.doesNotMatch(JSON.stringify(found), /private-token|private\/workspace|source-body/);
  });


  test('invalid saved fields cannot be mistaken for an absent config', async () => {
    const { options } = rig([], {});
    options.snapshot = async () => ({ profiles: [], guards: {}, resolution: { status: 'needs_setup', requirements: [
      { profile: 'audit_small', field: ['profiles', 'audit_small', 'timeout_ms'], phase: 'static', code: 'unknown_key',
        reason: 'This setting is not supported.', remedy: 'Put timeout_ms under limits.' },
    ] } } as never);
    const result = await lookupProfile(options, '/ws', { file: 'audit_models.py', cls: 'GoodNet' }, signal());
    assert.ok(!result.ok && result.kind === 'blocked');
    assert.match(result.detail, /profiles.audit_small.timeout_ms/);
    assert.match(result.detail, /limits/);
  });


  test('confirmed absence does not need a snapshot or selected interpreter', async () => {
    const result = await lookupProfile({ hasConfig: async () => false,
      snapshot: async () => { throw new Error('must not probe Python'); },
      choose: async () => { throw new Error('must not ask'); } }, '/ws', { file: 'net.py', cls: 'Net' }, signal());
    assert.deepEqual(result, { ok: false, kind: 'none' });
  });

  test('an unreadable config check refuses rather than treating an I/O error as absence', async () => {
    const result = await lookupProfile({ hasConfig: async () => { throw new Error('EACCES private path'); },
      snapshot: async () => { throw new Error('must not snapshot'); }, choose: async () => undefined },
    '/ws', { file: 'net.py', cls: 'Net' }, signal());
    assert.ok(!result.ok && result.kind === 'blocked');
    assert.doesNotMatch(result.detail, /EACCES|private path/);
  });

  for (const failure of ['needs_setup', 'throw', 'wrong_target', 'wrong_profile', 'removed']) {
    test(`selected profile ${failure} cannot fall back to another invocation`, async () => {
      const { options } = rig(['small', 'large'], { small: modelProfile('net:Net'), large: modelProfile('net:Net') }, 'large');
      const snapshot = options.snapshot;
      options.snapshot = async (root, name) => {
        if (!name) return snapshot(root, name);
        assert.equal(name, 'large', 'resolve only the chosen profile');
        if (failure === 'throw') throw new Error('private exception');
        if (failure === 'wrong_target') return resolved(modelProfile('other:Other'), name) as never;
        if (failure === 'wrong_profile') return resolved(modelProfile('net:Net'), 'small') as never;
        return { profiles: [], guards: {}, resolution: { status: 'needs_setup', requirements: [
          { profile: name, field: failure === 'removed' ? ['profiles'] : ['profiles', name, 'inputs'], phase: 'static', code: 'missing',
            reason: 'A required setting is missing.', remedy: 'Save the selected profile and retry.' },
        ] } } as never;
      };
      const result = await lookupProfile(options, '/ws', { file: 'net.py', cls: 'Net' }, signal());
      assert.ok(!result.ok && result.kind === 'blocked');
      assert.doesNotMatch(result.detail, /private exception/);
    });
  }

  test('an unrelated valid definition is not resolved even if its own preflight would fail', async () => {
    const { options } = rig(['other'], { other: modelProfile('other:Other') });
    const snapshot = options.snapshot;
    options.snapshot = async (root, name) => {
      assert.equal(name, undefined, 'do not preflight another class');
      return snapshot(root, name);
    };
    assert.deepEqual(await lookupProfile(options, '/ws', { file: 'net.py', cls: 'Net' }, signal()), { ok: false, kind: 'none' });
  });

  for (const stage of ['listing', 'choice', 'resolution']) {
    test(`cancellation during ${stage} wins over late success`, async () => {
      const controller = new AbortController();
      const { options } = rig(['small', 'large'], { small: modelProfile('net:Net'), large: modelProfile('net:Net') }, 'large');
      const snapshot = options.snapshot;
      options.snapshot = async (root, name) => {
        if (stage === (name ? 'resolution' : 'listing')) controller.abort();
        return snapshot(root, name);
      };
      options.choose = async () => { if (stage === 'choice') controller.abort(); return 'large'; };
      const result = await lookupProfile(options, '/ws', { file: 'net.py', cls: 'Net' }, controller.signal);
      assert.ok(!result.ok && result.kind === 'cancelled');
    });
  }

  test('a stalled picker settles on cancellation and a late choice never resolves or runs', async () => {
    const controller = new AbortController();
    const { options } = rig(['small', 'large'], { small: modelProfile('net:Net'), large: modelProfile('net:Net') });
    let choose!: (name: string) => void;
    const snapshot = options.snapshot;
    let resolutions = 0;
    options.snapshot = async (root, name) => { if (name) resolutions++; return snapshot(root, name); };
    options.choose = () => new Promise<string>(resolve => { choose = resolve; });
    const pending = lookupProfile(options, '/ws', { file: 'net.py', cls: 'Net' }, controller.signal);
    await new Promise(resolve => setImmediate(resolve));
    controller.abort();
    const result = await pending;
    assert.ok(!result.ok && result.kind === 'cancelled');
    choose('large');
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(resolutions, 0);
  });

  test('two profiles for one class ask which, and honour the answer', async () => {
    const { options, asked } = rig(['fast', 'slow'],
      { fast: modelProfile('models.resid:Net'), slow: modelProfile('models.resid:Net') }, 'slow');
    const found = await lookupProfile(options, '/ws', { file: 'models/resid.py', cls: 'Net' }, signal());
    assert.deepEqual(asked, [['fast', 'slow']]);
    assert.equal(found.ok && found.profile, 'slow');
  });

  test('dismissing that question runs nothing', async () => {
    const { options } = rig(['fast', 'slow'],
      { fast: modelProfile('models.resid:Net'), slow: modelProfile('models.resid:Net') }, undefined);
    const found = await lookupProfile(options, '/ws', { file: 'models/resid.py', cls: 'Net' }, signal());
    assert.equal(found.ok, false);
    assert.equal(found.ok ? '' : found.kind, 'cancelled');
    assert.match(found.ok || found.kind === 'none' ? '' : found.detail, /No profile was selected/);
  });

  test('a script profile for this class is refused by name, not silently skipped', async () => {
    const { options } = rig(['training'], { training: scriptProfile() });
    const found = await lookupProfile(options, '/ws', { file: 'models/tiny_lm.py', cls: 'TinyLM' }, signal());
    assert.equal(found.ok, false);
    assert.equal(found.ok ? '' : found.kind, 'blocked', 'a profile that names this class must not be silently replaced by a guess');
    const detail = found.ok || found.kind !== 'blocked' ? '' : found.detail;
    assert.match(detail, /"training" is a script profile/);
    assert.match(detail, /runs train\.py and captures call 1 of models\.tiny_lm:TinyLM/);
    assert.match(detail, /add a model profile/);
  });
});

suite('profile translation', () => {
  test('a saved binding travels verbatim rather than being mapped to a fill', () => {
    const negative = { kind: 'tensor', shape: [2, 8], dtype: 'float32', requires_grad: false,
      generator: { kind: 'full', value: -1.0 } };
    const translated = translateProfile(modelProfile('models.gate:Gate', { inputs: { args: [negative], kwargs: {} } }), 'gate');
    assert.equal(translated.ok, true);
    const inputs = translated.ok ? translated.run.execution.inputs as { binding: { args: unknown[] } } : { binding: { args: [] } };
    assert.deepEqual(inputs.binding.args[0], negative, 'full = -1.0 has no zeros/ones/randn equivalent and must not be approximated');
  });

  test('init is always carried, so a retired init comment cannot reach the constructor', () => {
    const translated = translateProfile(modelProfile('models.resid:Net'), 'residual');
    assert.equal(translated.ok, true);
    assert.ok(translated.ok && translated.run.execution.constructorBinding, 'an absent init is what let --overrides win');
  });

  test('a factory reaches the same explicit setup path as MCP without a competing init', () => {
    const factory = { file: 'recipes/whisper.py', callable: 'make_model', kwargs: { width: 32 } };
    const translated = translateProfile(modelProfile('whisper.model:Whisper', { init: null, factory }), 'whisper');
    assert.ok(translated.ok);
    assert.deepEqual(translated.run.execution.setup, factory);
    assert.equal(translated.run.execution.constructorBinding, undefined);
    assert.equal(translated.run.execution.constructorArgs, undefined);
  });

  test('declared execution context survives translation without a hidden seed reset', () => {
    const translated = translateProfile(modelProfile('net:Net', {
      seed: null, cwd: 'experiments', grad_enabled: true, deterministic: false,
    }));
    assert.ok(translated.ok);
    assert.equal(translated.run.execution.seed, null);
    assert.equal(translated.run.execution.gradEnabled, true);
    assert.equal(translated.run.execution.deterministic, false);
    assert.equal(translated.run.execution.cwd, 'experiments');
  });

  test('missing or competing construction is refused instead of substituting defaults', () => {
    const factory = { file: 'recipe.py', callable: 'make_model', kwargs: {} };
    for (const extra of [{ init: null, factory: null }, { factory }]) {
      const translated = translateProfile(modelProfile('model:Model', extra), 'model');
      assert.ok(!translated.ok);
      assert.match(translated.reason, /exactly one/);
    }
  });

  test('a package module offers its __init__ spelling too', () => {
    assert.deepEqual(targetsFromModelSymbol('pkg.sub:Net').map(t => t.file),
      ['pkg/sub.py', 'pkg/sub/__init__.py']);
    assert.equal(targetsFromModelSymbol('pkg:Net', 'src')[0].file, 'src/pkg.py');
  });

  test('a profile with no declared input is refused rather than given one', () => {
    const translated = translateProfile(modelProfile('models.resid:Net', { inputs: undefined }), 'empty');
    assert.equal(translated.ok, false);
    assert.match(translated.ok ? '' : translated.reason, /must declare its input binding/);
  });
});

suite('setup detail', () => {
  test('names every field the user has to fix', () => {
    const detail = setupDetail([
      { profile: 'p', field: ['profiles', 'p', 'device'], phase: 'static', code: 'malformed',
        reason: 'The setting is malformed.', remedy: 'Set device to cpu.' } as never,
    ]);
    assert.match(detail, /profiles\.p\.device: The setting is malformed\. Set device to cpu\./);
  });
});
