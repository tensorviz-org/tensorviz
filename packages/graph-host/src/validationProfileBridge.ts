/** A saved tensorviz.toml profile, expressed as the explicit execution this build's
 *  trace worker already accepts.
 *
 * Why this exists: the validation redesign replaced guessed inputs — a shape read off the
 * first Linear, a brute-force search, a per-library default — with one explicit profile
 * the user saves and the host resolves. That is worth keeping. What it also did was move
 * measured results off the graph into a separate view. This build keeps the graph as the
 * place measurements appear, so a resolved profile becomes the `ExecutionSpec` the
 * existing worker takes and the existing projection merge renders.
 *
 * The profile's own bindings are passed through, never translated. The profile grammar
 * can say `full = -1.0`, `randint`, literal `values` and per-tensor seeds; this build's
 * older vocabulary is zeros/ones/randn. Mapping between them would substitute an input
 * the user did not write, which is precisely what explicit profiles exist to prevent —
 * so Python's own materialiser builds them instead.
 */
import type { RunSpec } from '@tensorviz/contracts/protocol';
import { executionSpecSchema, type ExecutionSpec } from './executionSpec';

export interface ProfileTarget {
  /** Workspace-relative POSIX path of the file defining `cls`. */
  file: string;
  cls: string;
  /** True when the symbol resolves to a package, so the caller looks for `__init__.py`. */
  packageModule: boolean;
}
export interface ProfileRun {
  target: ProfileTarget;
  execution: ExecutionSpec;
  /** The profile's own limit, so a run is bounded the way its author asked. */
  timeoutMs?: number;
}
export type ProfileTranslation = { ok: true; run: ProfileRun } | { ok: false; reason: string };

type Profile = RunSpec['profile'];

/** `"models.resid:Net"` -> `models/resid.py` + `Net`, under the profile's own `cwd`.
 *  A dotted module can also be a package, in which case the class lives in its
 *  `__init__.py`; both spellings are offered and the caller picks the one on disk. */
export function targetsFromModelSymbol(symbol: string, cwd = '.'): ProfileTarget[] {
  const [module, cls] = symbol.split(':');
  if (!module || !cls || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(cls)) return [];
  const parts = module.split('.');
  if (!parts.length || parts.some(part => !/^[A-Za-z_][A-Za-z0-9_]*$/.test(part))) return [];
  const base = cwd && cwd !== '.' ? `${cwd.replace(/\/+$/, '')}/` : '';
  return [
    { file: `${base}${parts.join('/')}.py`, cls, packageModule: false },
    { file: `${base}${parts.join('/')}/__init__.py`, cls, packageModule: true },
  ];
}

/** What a script profile is, and the model profile that would measure the same class. */
function scriptRefusal(name: string, raw: Record<string, any>): string {
  const capture = raw.capture ?? {};
  const symbol = String(capture.model ?? 'the captured class');
  const cls = symbol.includes(':') ? symbol.slice(symbol.indexOf(':') + 1) : symbol;
  return `Profile "${name}" is a script profile: it runs ${String(raw.script)} and captures call ${String(capture.call ?? 1)} of ${symbol}. `
    + `This build measures kind = "model" profiles only. To measure ${cls} here, add a model profile to tensorviz.toml that names model = "${symbol}", `
    + `its constructor keyword arguments under init, and one explicit input tensor under inputs, then run that profile. `
    + `If ${cls} needs a config object, use a workspace factory with factory.file, factory.callable and factory.kwargs instead of init; it must return the exact selected class.`;
}

/**
 * Translate one resolved profile. `name` is `spec.identity.profile` — the profile has no
 * name of its own, and a refusal that cannot say which profile it is about is not useful.
 */
export function translateProfile(profile: Profile, name = 'selected'): ProfileTranslation {
  const raw = profile as unknown as Record<string, any>;
  if (raw.kind !== 'model') return { ok: false, reason: scriptRefusal(name, raw) };

  const targets = typeof raw.model === 'string' ? targetsFromModelSymbol(raw.model, raw.cwd) : [];
  if (!targets.length) {
    return { ok: false, reason: `Profile "${name}" must name its model as "package.module:ClassName". Saved value: ${JSON.stringify(raw.model)}.` };
  }

  if (!raw.inputs || typeof raw.inputs !== 'object') {
    return { ok: false, reason: `Profile "${name}" must declare its input binding, including an explicit empty binding for a zero-argument forward.` };
  }
  const declared = raw.inputs;
  if (typeof declared.fixture === 'string') {
    return { ok: false, reason: `Profile "${name}" loads its inputs from the fixture file ${declared.fixture}. This build takes inputs declared in the profile; describe them under inputs instead.` };
  }
  const binding = { args: (declared.args ?? []) as unknown[], kwargs: (declared.kwargs ?? {}) as Record<string, unknown> };

  // Always carried, including when empty: an absent init is what lets a retired
  // `# tensorviz: init=` comment reach the constructor as a fallback.
  if ((raw.init == null) === (raw.factory == null)) {
    return { ok: false, reason: `Profile "${name}" must provide exactly one of init or factory.` };
  }
  const init = raw.init ?? {};
  const parsed = executionSpecSchema.safeParse({
    ...(raw.framework === 'jax' ? { framework: 'jax' } : {}),
    inputs: declared.factory ? { factory: declared.factory } : { binding },
    ...(raw.factory != null ? { setup: raw.factory }
      : { constructorBinding: { args: init.args ?? [], kwargs: init.kwargs ?? {} } }),
    device: typeof raw.device === 'string' ? raw.device : 'cpu',
    mode: raw.mode === 'train' || raw.mode === 'preserve' ? raw.mode : 'eval',
    seed: raw.seed ?? null,
    gradEnabled: raw.grad_enabled,
    deterministic: raw.deterministic ?? true,
    cwd: raw.cwd ?? '.',
  });
  if (!parsed.success) {
    return { ok: false, reason: `Profile "${name}" does not describe a runnable execution: ${parsed.error.issues.map(i => i.message).join('; ').slice(0, 300)}` };
  }
  const timeout = raw.limits?.timeout_ms;
  return { ok: true, run: { target: targets[0], execution: parsed.data, ...(typeof timeout === 'number' ? { timeoutMs: timeout } : {}) } };
}

/** Both spellings of the profile's target, so a caller can pick the file that exists.
 *
 * A script profile has no `model` of its own — the class it is about is its capture
 * selector. It is reported here so that a user looking at that class is told their script
 * profile cannot run in this build, rather than the much less useful "no saved profile
 * measures this class". */
export function candidateTargets(profile: Profile): ProfileTarget[] {
  const raw = profile as unknown as Record<string, any>;
  const symbol = typeof raw.model === 'string' ? raw.model
    : typeof raw.capture?.model === 'string' ? raw.capture.model : undefined;
  return symbol ? targetsFromModelSymbol(symbol, raw.cwd) : [];
}
