/** One execution owner for every surface. Static projections never call Run. */
import * as path from 'node:path';
import { randomUUID } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
import type { ExecutionRecord, RunSpec, ValidationExecutionUpdate, LocalBrowserValidationOutcome, ValidationResolution, ValidationRequirement } from '@tensorviz/contracts/protocol';

export interface ValidationTarget {
  root: string;
  session: string;
  /** Re-evaluated after every asynchronous boundary; resolved only from host state. */
  current(): boolean;
}
export interface ValidationPreparation {
  profiles: string[];
  /** Canonical parsed profiles for target matching only; resolution still gates execution. */
  profileDefinitions?: Record<string, RunSpec['profile']>;
  resolution: ValidationResolution;
  guards: Record<string, unknown>;
}
export interface ValidationControllerOptions {
  trusted(): boolean;
  prepare(target: ValidationTarget, runId: string, generation: number, profile: string | undefined, signal: AbortSignal): Promise<ValidationPreparation>;
  setup?(target: ValidationTarget, requirements: ValidationRequirement[]): Promise<void>;
  choose(profiles: string[]): Promise<string | undefined>;
  approve(spec: RunSpec): Promise<boolean>;
  run(spec: RunSpec, root: string, receive: (record: ExecutionRecord, dependenciesCurrent?: boolean) => void, beforeLaunch: () => Promise<boolean>): Promise<{ record: ExecutionRecord }>;
  cancel(): void;
  publish(update: ValidationExecutionUpdate): void;
}
type Reason = ValidationExecutionUpdate['freshness']['reasons'][number];

export class ValidationController {
  private generation = 0;
  private active?: { abort: AbortController; target: ValidationTarget; reasons: Set<Reason>; spec?: RunSpec };
  private latestTarget?: ValidationTarget;
  private readonly poll: ReturnType<typeof setInterval>;
  private latest?: ValidationExecutionUpdate;
  constructor(private readonly options: ValidationControllerOptions) {
    this.poll = setInterval(() => {
      const target = this.active?.target ?? this.latestTarget;
      if (target && (!target.current() || !this.options.trusted())) this.invalidate('session', target.session);
    }, 250);
    this.poll.unref();
  }
  fileChanged(file: string): void {
    const target = this.active?.target ?? this.latestTarget;
    if (!target) return;
    const relative = path.relative(target.root, file).split(path.sep).join('/');
    const spec = this.active?.spec ?? this.lastSpec;
    const tracked = spec?.files.find(entry => entry.path === relative);
    if (tracked) this.invalidate(tracked.role === 'config' ? 'config' : tracked.role === 'native' ? 'native_file' : 'source');
    else if (relative && !relative.startsWith('../') && !path.isAbsolute(relative) && relative.endsWith('.py')) this.invalidate('source');
  }
  private lastSpec?: RunSpec;
  get running(): boolean { return !!this.active; }
  get root(): string | undefined { return (this.active?.target ?? this.latestTarget)?.root; }
  snapshot(session: string): ValidationExecutionUpdate | undefined {
    if (this.latest?.workspaceSessionId !== session) return;
    if (!this.latestTarget?.current() || !this.options.trusted()) this.invalidate('session', session);
    return structuredClone(this.latest);
  }
  /** A view names an opaque invocation id, never a path. Only current host evidence
   * can supply navigation and the saved-file identity checked by the opener. */
  sourceForNode(session: string, nodeId: string): { root: string; source: NonNullable<ExecutionRecord['invocations'][number]['source']>; file: RunSpec['files'][number] } | undefined {
    const update = this.latest;
    if (!update || update.workspaceSessionId !== session || update.freshness.state !== 'current'
      || !this.latestTarget?.current() || !this.options.trusted() || !this.lastSpec) return;
    const call = update.record.invocations.find(call => `execution:${update.record.identity.runId}:${call.id}` === nodeId);
    if (!call) return;
    const source = call.failure?.source ?? call.source ?? update.record.modules.find(module => module.id === call.moduleId)?.source;
    const file = source && this.lastSpec.files.find(file => file.path === source.file && file.role === 'source');
    if (!source || !file) return;
    return structuredClone({ root: this.latestTarget.root, source, file });
  }

  invalidate(reason: Reason, session?: string): void {
    if (this.active && (!session || this.active.target.session === session)) this.active.reasons.add(reason);
    if (this.latest && (!session || this.latest.workspaceSessionId === session) && !this.latest.freshness.reasons.includes(reason)) {
      this.latest.freshness = { state: 'stale', reasons: [...new Set([...this.latest.freshness.reasons, reason])] };
      this.options.publish(structuredClone(this.latest));
    }
  }
  cancel(session?: string, runId?: string): void {
    if (!this.active || (session && this.active.target.session !== session)
      || (runId && this.active.spec?.identity.runId !== runId)) return;
    this.invalidate('cancelled', session);
    this.active.abort.abort();
    this.options.cancel();
  }
  dispose(): void { this.cancel(); clearInterval(this.poll); }

  async run(target: ValidationTarget, signal?: AbortSignal): Promise<LocalBrowserValidationOutcome> {
    const runId = randomUUID();
    const refused = (detail: string): LocalBrowserValidationOutcome => ({ status: 'refused', detail });
    if (this.active) return refused('Another validation process is still running. Cancel it or wait for exit.');
    if (!this.options.trusted() || !target.current()) return refused('Run requires a current target in a trusted workspace.');
    this.invalidate('profile');
    const active: NonNullable<ValidationController['active']> = { abort: new AbortController(), target, reasons: new Set<Reason>() };
    this.active = active;
    const onAbort = () => this.cancel(target.session);
    signal?.addEventListener('abort', onAbort, { once: true });
    if (signal?.aborted) onAbort();
    const generation = ++this.generation;
    const valid = () => !active.abort.signal.aborted && this.options.trusted() && target.current() && active.reasons.size === 0;
    let queue = Promise.resolve();
    try {
      let prepared = await this.options.prepare(target, runId, generation, undefined, active.abort.signal);
      if (!valid()) return refused('The request changed or was cancelled before launch.');
      const needsSetup = async (requirements: ValidationRequirement[]) => {
        if (valid()) await this.options.setup?.({ ...target, current: valid }, requirements);
        return refused('Needs setup in tensorviz.toml. ' + requirements.map(r => `${r.field.join('.')}: ${r.reason} ${r.remedy}`).join(' ').slice(0, 460));
      };
      if (!prepared.profiles.length) return prepared.resolution.status === 'needs_setup'
        ? await needsSetup(prepared.resolution.requirements) : refused('Create a saved tensorviz.toml profile.');
      const profile = await this.options.choose(prepared.profiles);
      if (!profile || !valid()) return { status: 'refused', detail: 'Profile selection cancelled or target changed.' };
      prepared = await this.options.prepare(target, runId, generation, profile, active.abort.signal);
      if (prepared.resolution.status !== 'resolved') return await needsSetup(prepared.resolution.requirements);
      const spec = prepared.resolution.spec;
      active.spec = spec;
      if (!valid() || !await this.options.approve(spec) || !valid()) return refused('Run was not approved for the current saved target.');
      const launch = await this.options.prepare(target, runId, generation, profile, active.abort.signal);
      if (!valid() || !isDeepStrictEqual(launch, prepared)) return refused('Saved files or environment changed before launch. Run again.');
      let sequence = -1;
      const receive = (record: ExecutionRecord, dependenciesCurrent = true): void => {
        // Refuse before queuing or publishing. Late records never acquire another run's identity.
        if (this.active !== active || !isDeepStrictEqual(record.identity, spec.identity)
          || !isDeepStrictEqual(record.limits, spec.profile.limits)
          || !Number.isSafeInteger(record.sequence) || record.sequence < 0 || record.sequence <= sequence) return;
        sequence = record.sequence;
        if (!dependenciesCurrent) active.reasons.add('dependency_scope');
        const copy = structuredClone(record);
        queue = queue.then(async () => {
          if (!target.current() || !this.options.trusted()) active.reasons.add('session');
          if (!active.abort.signal.aborted) {
            try {
              const now = await this.options.prepare(target, runId, generation, profile, active.abort.signal);
              if (now.resolution.status !== 'resolved') active.reasons.add('source');
              else {
                const identity = now.resolution.spec.identity;
                for (const [key, reason] of [['configDigest', 'config'], ['sourceDigest', 'source'],
                  ['environmentDigest', 'environment'], ['profileDigest', 'profile']] as const) {
                  if (identity[key] !== spec.identity[key]) active.reasons.add(reason);
                }
              }
              if (!isDeepStrictEqual(now.guards, prepared.guards)) active.reasons.add('source');
            } catch { active.reasons.add('source'); }
          }
          if (this.active !== active) return;
          if (!target.current() || !this.options.trusted()) active.reasons.add('session');
          this.latestTarget = target;
          this.lastSpec = spec;
          this.latest = { version: 1, type: 'executionUpdate', workspaceSessionId: target.session,
            record: copy, freshness: { state: active.reasons.size ? 'stale' : 'current', reasons: [...active.reasons] } };
          this.options.publish(structuredClone(this.latest));
        }).catch(() => {
          // A disconnected sink must not abandon the live process or its terminal record.
          active.reasons.add('session');
          if (this.latest?.record.identity.runId === runId) {
            this.latest.freshness = { state: 'stale', reasons: [...active.reasons] };
          }
        });
      };
      const result = await this.options.run(spec, target.root, receive, async () => {
        if (!valid()) return false;
        const final = await this.options.prepare(target, runId, generation, profile, active.abort.signal);
        const ready = valid() && isDeepStrictEqual(final, prepared);
        return ready;
      });
      receive(result.record);
      await queue;
      if (active.abort.signal.aborted) return { status: 'cancelled', detail: 'Validation process cancelled.' };
      if (!this.latest || this.latest.record.identity.runId !== runId || this.latest.freshness.state === 'stale') return refused('Measurements are stale. Process outcome was retained.');
      if (this.latest.record.requirements.length) return await needsSetup(this.latest.record.requirements);
      // U08 owns the presentation state machine. Do not attach a legacy Passed badge to a static graph.
      return { status: 'refused', detail: 'Execution recorded. The measured execution view shows the capture and process outcome.' };
    } catch {
      return refused('Validation could not prepare or launch. Check saved configuration and the selected Python environment.');
    } finally {
      await queue;
      signal?.removeEventListener('abort', onAbort);
      if (this.active === active) this.active = undefined;
    }
  }
}
