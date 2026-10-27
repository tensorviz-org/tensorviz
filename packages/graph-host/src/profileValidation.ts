/** Resolve explicit saved settings before the canvas can execute a model. */
import type { ValidationPreparation } from './validationController';
import type { ValidationRequirement } from '@tensorviz/contracts/protocol';
import { candidateTargets, translateProfile, type ProfileRun } from './validationProfileBridge';
import { savedExecutionGuard, type ExecutionGuard } from './executionGuard';
import { profileFailure } from './executionFailure';
import type { GraphFailure } from '@tensorviz/contracts/protocol';

export interface ProfileSnapshot {
  (root: string, profile: string | undefined, signal: AbortSignal): Promise<ValidationPreparation>;
}
export interface ProfileLookupOptions {
  snapshot: ProfileSnapshot;
  /** Host stat before interpreter selection; only a genuinely absent path returns false. */
  hasConfig?(root: string): Promise<boolean>;
  choose(names: string[], signal: AbortSignal): Promise<string | undefined>;
  preferred?(root: string, target: { file: string; cls: string }): Promise<string | undefined>;
  remember?(root: string, target: { file: string; cls: string }, name: string): Promise<void>;
}
export type ProfileLookup =
  | { ok: true; profile: string; run: ProfileRun; guard: ExecutionGuard }
  | { ok: false; kind: 'none' }
  | { ok: false; kind: 'cancelled'; detail: string }
  | { ok: false; kind: 'blocked'; detail: string; requirements?: ValidationRequirement[];
      failure?: GraphFailure;
      profileName?: string; definition?: Record<string, unknown> };

/** Python supplies fixed diagnostic prose, never exception text or saved values.
 * Structural locations are bounded; file paths and arbitrary quoted keys stay private. */
export function setupDetail(requirements: readonly ValidationRequirement[]): string {
  if (!requirements.length) return 'Saved profile resolution was incomplete. Check tensorviz.toml and retry.';
  const kind = profileFailure(requirements).kind;
  return (kind === 'source' ? 'Source needs attention. ' : kind === 'environment' ? 'Execution environment needs attention. '
    : 'Needs setup in tensorviz.toml. ') + requirements.map(entry => {
    const field = entry.field.map(part => typeof part === 'number' || part === 'tensorviz.toml'
      || /^[A-Za-z_][A-Za-z0-9_]{0,63}$/.test(part) ? part : '<field>').join('.');
    return `${field}: ${entry.reason} ${entry.remedy}`;
  }).join(' ').slice(0, 460);
}

function matches(profile: Parameters<typeof candidateTargets>[0], file: string, cls: string): boolean {
  return candidateTargets(profile).some(target => target.cls === cls && target.file === file);
}
const cancelled = (): ProfileLookup => ({ ok: false, kind: 'cancelled', detail: 'No profile was selected or validation was cancelled, so nothing was run.' });
const snapshotFailure = (): ProfileLookup => ({ ok: false, kind: 'blocked',
  detail: 'Saved model settings could not be checked. Check the selected Python environment and project-root tensorviz.toml in VS Code, then retry. Nothing was run.' });
const needsSetup = (requirements: ValidationRequirement[]): ProfileLookup =>
  ({ ok: false, kind: 'blocked', detail: setupDetail(requirements), requirements, failure: profileFailure(requirements) });

async function resolveProfile(options: ProfileLookupOptions, root: string,
  target: { file: string; cls: string }, signal: AbortSignal, interactive: boolean): Promise<ProfileLookup> {
  try {
    if (signal.aborted) return cancelled();
    if (options.hasConfig) {
      const present = await options.hasConfig(root);
      if (signal.aborted) return cancelled();
      if (!present) return { ok: false, kind: 'none' };
    }
    const listing = await options.snapshot(root, undefined, signal);
    if (signal.aborted) return cancelled();
    const resolution = listing.resolution;
    // This exact canonical requirement means absence. An empty profile list alone does not.
    if (!listing.profiles.length && resolution.status === 'needs_setup'
      && resolution.requirements.length === 1) {
      const [requirement] = resolution.requirements;
      if (requirement.profile === null && requirement.code === 'missing'
        && requirement.field.length === 1 && requirement.field[0] === 'tensorviz.toml') {
        // A config seen by the host and then removed must not become an automatic run.
        return options.hasConfig ? snapshotFailure() : { ok: false, kind: 'none' };
      }
    }
    // Listing a valid config needs an explicit selection. All other requirements block.
    if (resolution.status !== 'needs_setup') return snapshotFailure();
    if (resolution.requirements.length !== 1 || resolution.requirements[0].code !== 'ambiguous'
      || resolution.requirements[0].field.join('.') !== 'profiles') return needsSetup(resolution.requirements);
    const definitions = listing.profileDefinitions;
    if (!definitions || !listing.profiles.length || listing.profiles.some(name => !Object.hasOwn(definitions, name))) {
      return snapshotFailure();
    }
    const candidates = listing.profiles.filter(name => matches(definitions[name], target.file, target.cls));
    if (!candidates.length) return { ok: false, kind: 'none' };
    const preferred = await options.preferred?.(root, target);
    if (signal.aborted) return cancelled();
    if (preferred && !candidates.includes(preferred)) return { ok: false, kind: 'blocked',
      detail: 'The selected profile no longer matches this model. Use Change inputs to save a new setup.' };
    if (candidates.length > 1 && !preferred && !interactive) return { ok: false, kind: 'blocked',
      detail: 'Several saved profiles match this model. Open as Graph to choose one; capture on save ran nothing.' };
    const name = preferred ?? (candidates.length === 1 ? candidates[0] : await options.choose(candidates, signal));
    if (signal.aborted || name === undefined) return cancelled();
    if (!candidates.includes(name)) return snapshotFailure();
    // Resolve AFTER selection: unrelated profiles cannot fail this class's preflight,
    // and changes while the picker was open cannot launch a stale saved invocation.
    const prepared = await options.snapshot(root, name, signal);
    if (signal.aborted) return cancelled();
    if (prepared.resolution.status !== 'resolved') return { ...needsSetup(prepared.resolution.requirements),
      profileName: name, definition: definitions[name] as unknown as Record<string, unknown> } as ProfileLookup;
    const { spec } = prepared.resolution;
    if (spec.identity.profile !== name || !matches(spec.profile, target.file, target.cls)) return snapshotFailure();
    const translated = translateProfile(spec.profile, name);
    if (translated.ok) await options.remember?.(root, target, name);
    if (signal.aborted) return cancelled();
    return translated.ok ? { ok: true, profile: name, run: translated.run,
      guard: savedExecutionGuard(prepared, nextSignal => options.snapshot(root, name, nextSignal)) }
      : { ok: false, kind: 'blocked', detail: translated.reason };
  } catch {
    return signal.aborted ? cancelled() : snapshotFailure();
  }
}

/** Release request ownership promptly even if an environment probe or picker ignores abort. */
export async function lookupProfile(options: ProfileLookupOptions, root: string,
  target: { file: string; cls: string }, signal: AbortSignal, interactive = true): Promise<ProfileLookup> {
  if (signal.aborted) return cancelled();
  let abort!: () => void;
  const stopped = new Promise<ProfileLookup>(resolve => { abort = () => resolve(cancelled()); });
  signal.addEventListener('abort', abort, { once: true });
  try { return await Promise.race([resolveProfile(options, root, target, signal, interactive), stopped]); }
  finally { signal.removeEventListener('abort', abort); }
}
