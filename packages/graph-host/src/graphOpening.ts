import { randomUUID } from 'node:crypto';
import type { ExecutionReviewState, GraphEnvironmentRecovery, GraphFailure, GraphOpeningState, GraphSetupAction, GraphSetupDiscovery, GraphSetupDraft,
  LocalBrowserTargetIdentity, LocalBrowserValidationOutcome } from '@tensorviz/contracts/protocol';
import type { ProfileLookup } from './profileValidation';

type ResolvedProfile = Extract<ProfileLookup, { ok: true }>;
export interface PreparedGraphSetup { profile: string; config: string; factory: { file: string; source: string } | null }

/** Setup actions the host's environment recovery handles. */
export type EnvironmentRecoveryAction = Extract<GraphSetupAction, { kind: 'select_environment' | 'install_dependencies' | 'refresh_environments'
  | 'use_installed_versions' | 'replace_versions' | 'create_environment' }>;

export class GraphOpeningError extends Error {
  constructor(message: string, readonly failure?: GraphFailure, readonly environment?: GraphEnvironmentRecovery) { super(message); }
}
export interface GraphSetupPreparation {
  discovery: GraphSetupDiscovery;
  draft?: GraphSetupDraft;
  restoredDraft?: GraphSetupDraft;
  requirements?: string[];
}
export interface GraphOpeningDependencies {
  current(target: LocalBrowserTargetIdentity): boolean;
  hasConfig?(root: string): Promise<boolean>;
  environment?(target: LocalBrowserTargetIdentity, signal: AbortSignal, progress: (detail: string) => void): Promise<GraphEnvironmentRecovery>;
  recoverEnvironment?(target: LocalBrowserTargetIdentity, environment: GraphEnvironmentRecovery,
    action: EnvironmentRecoveryAction, signal: AbortSignal, progress: (detail: string) => void): Promise<boolean | GraphEnvironmentRecovery>;
  lookup(target: LocalBrowserTargetIdentity, signal: AbortSignal): Promise<ProfileLookup>;
  discover(target: LocalBrowserTargetIdentity, signal: AbortSignal, saved?: GraphSetupDraft): Promise<GraphSetupPreparation>;
  save(target: LocalBrowserTargetIdentity, draft: GraphSetupDraft, signal: AbortSignal): Promise<{ requirements?: string[] }>;
  inferInputs?(target: LocalBrowserTargetIdentity, draft: GraphSetupDraft, signal: AbortSignal): Promise<{ draft?: GraphSetupDraft; detail: string }>;
  factoryTemplate?(target: LocalBrowserTargetIdentity, draft: GraphSetupDraft | undefined, signal: AbortSignal): Promise<GraphSetupDraft>;
  capture(target: LocalBrowserTargetIdentity, profile: ResolvedProfile, signal: AbortSignal,
    progress: (detail: string) => void): Promise<{ outcome: LocalBrowserValidationOutcome; target: LocalBrowserTargetIdentity; hasCapture: boolean; dependenciesInstalled?: boolean }>;
  recover(target: LocalBrowserTargetIdentity, action: GraphSetupAction['kind'], signal: AbortSignal, request?: GraphSetupAction): Promise<LocalBrowserTargetIdentity | undefined>;
  restart?(target: LocalBrowserTargetIdentity): Promise<void>;
  parent(target: LocalBrowserTargetIdentity, choice: GraphSetupDiscovery['parents'][number]): Promise<void>;
  publish(state: GraphOpeningState): void;
}

const sameModel = (left: LocalBrowserTargetIdentity, right: LocalBrowserTargetIdentity) =>
  left.workspaceFolder === right.workspaceFolder && left.file === right.file && left.cls === right.cls;
const sameSource = (left: LocalBrowserTargetIdentity, right: LocalBrowserTargetIdentity) =>
  sameModel(left, right) && left.sourceDigest === right.sourceDigest && left.documentVersion === right.documentVersion;

/** One explicit opening owns setup, persistence and one captured invocation. */
export class GraphOpeningController {
  private controller?: AbortController;
  private value?: GraphOpeningState;
  private busy = false;
  private configCheck = 0;
  private readonly seen = new Set<string>();
  constructor(private readonly dependencies: GraphOpeningDependencies) {}

  get state(): GraphOpeningState | undefined { return this.value; }

  private executionPhase?: GraphOpeningState['phase'];
  /** Shared with manual capture: one host-owned panel, replayed with the opening. */
  execution(state: ExecutionReviewState): void {
    if (!this.value) return;
    if (state.phase === 'review' && this.value.phase !== 'awaiting_execution') this.executionPhase = this.value.phase;
    const phase = state.phase === 'review' ? 'awaiting_execution'
      : this.value.phase === 'awaiting_execution' ? this.executionPhase ?? 'resolving' : this.value.phase;
    this.set({ execution: state, phase, ...(phase !== 'cancelled' ? { cancellationReason: undefined } : {}) });
  }

  /** File presence is independent of profile validity and never starts a run. */
  async refreshConfigFile(): Promise<void> {
    const state = this.value, check = ++this.configCheck;
    if (!state || !this.dependencies.hasConfig) return;
    let configFileExists: boolean | undefined;
    try { configFileExists = await this.dependencies.hasConfig(state.target.workspaceFolder); }
    catch { /* A read error is unknown, not evidence of absence. */ }
    if (this.value?.id === state.id && check === this.configCheck) this.set({ configFileExists });
  }

  sourceOnly(target: LocalBrowserTargetIdentity): void {
    this.clear();
    this.controller = new AbortController();
    this.value = { id: randomUUID(), target: { ...target }, phase: 'source_only', hasCapture: false,
      detail: 'Source-only view. Shapes are not measurements from a new execution.' };
    this.dependencies.publish(this.value);
    void this.refreshConfigFile();
  }

  cancel(detail = 'Opening cancelled. Your saved setup is available for the next opening.', cancellationReason?: 'source_changed'): void {
    this.controller?.abort();
    this.busy = false;
    if (this.value && !['ready', 'source_only'].includes(this.value.phase)) this.set({ phase: 'cancelled', detail, cancellationReason });
  }

  clear(): void {
    this.controller?.abort();
    this.controller = undefined;
    this.value = undefined;
    this.busy = false;
  }

  invalidate(target: LocalBrowserTargetIdentity, dependencyChanged = false): void {
    if (this.value?.phase === 'cancelled' && !this.value.cancellationReason) return;
    // A version alone is not a source change: VS Code restarts it when it reloads an
    // unchanged buffer. Follow it so later checks compare against the live document.
    if (this.value && !dependencyChanged && sameModel(this.value.target, target) && this.value.target.sourceDigest === target.sourceDigest) {
      if (this.value.target.documentVersion !== target.documentVersion) this.set({ target: { ...this.value.target, documentVersion: target.documentVersion } });
      return;
    }
    if (this.value && (dependencyChanged || !sameSource(this.value.target, target))) {
      this.cancel();
      this.set({ target: { ...target }, phase: 'cancelled', cancellationReason: 'source_changed', hasCapture: false, detail: 'Source changed. Refresh the saved model to capture it.' });
    }
  }

  /** A later measured run can replace a ready graph without reopening setup. */
  acceptMeasuredProjection(target: LocalBrowserTargetIdentity): void {
    if (this.value && ['ready', 'source_only'].includes(this.value.phase) && sameSource(this.value.target, target)
      && target.backendRevision > this.value.target.backendRevision) this.set({ target: { ...target }, phase: 'ready', hasCapture: true });
  }

  /** A completed save-driven capture supersedes the stale opening, never an
   * in-flight explicit opening or measurements for another source/model. */
  acceptAutomaticCapture(target: LocalBrowserTargetIdentity, detail: string): void {
    if (!this.value || this.busy || !sameModel(this.value.target, target) || !this.dependencies.current(target)) return;
    this.controller?.abort();
    this.controller = new AbortController();
    this.set({ target: { ...target }, phase: 'ready', hasCapture: true, detail,
      failure: undefined, requirements: undefined, cancellationReason: undefined });
  }

  async open(target: LocalBrowserTargetIdentity, changeInputs = false): Promise<void> {
    this.clear();
    const controller = new AbortController();
    this.controller = controller;
    this.value = { id: randomUUID(), target: { ...target }, phase: 'resolving',
      detail: 'Checking saved setup and the selected Python environment.', hasCapture: false };
    this.dependencies.publish(this.value);
    await this.resolve(controller, changeInputs);
  }

  private current(controller: AbortController): boolean {
    if (this.controller !== controller || controller.signal.aborted || !this.value) return false;
    if (!this.dependencies.current(this.value.target)) {
      this.cancel('Source changed. Open the current saved model again to capture it.', 'source_changed');
      return false;
    }
    return true;
  }

  private set(update: Partial<GraphOpeningState>): void {
    if (!this.value) return;
    // Optional form fields must be absent on the JSON wire, never undefined values.
    this.value = JSON.parse(JSON.stringify({ ...this.value, ...update })) as GraphOpeningState;
    this.dependencies.publish(this.value);
  }

  private async setup(controller: AbortController, detail: string, draft?: GraphSetupDraft, automatic = false): Promise<void> {
    const prepared = await this.dependencies.discover(this.value!.target, controller.signal, draft);
    if (!this.current(controller)) return;
    const resolution = prepared.discovery.resolution;
    let selected = draft ? prepared.restoredDraft ?? draft : prepared.draft ?? this.value?.draft;
    const requirements = [...(prepared.requirements ?? []), ...(draft ? [] : resolution?.unresolved ?? [])]
      .filter(item => !item.endsWith(': provide a value.')).slice(0, 64);
    let inferred = false;
    let evidence = draft ? undefined : resolution?.evidence;
    if (automatic && !draft && resolution?.status === 'partial' && resolution.runtimeInputs
      && selected && !prepared.requirements?.length && this.dependencies.inferInputs) {
      this.set({ phase: 'resolving', detail: 'Constructing an example model to determine its input dimensions.',
        discovery: prepared.discovery, draft: selected, evidence });
      const proposal = await this.dependencies.inferInputs(this.value!.target, selected, controller.signal);
      if (!this.current(controller)) return;
      if (proposal.draft) {
        selected = proposal.draft;
        inferred = true;
        evidence = [...(evidence ?? []).slice(0, 63), { field: 'inputs', kind: 'example', detail: proposal.detail }];
      } else requirements.push(proposal.detail);
    }
    const canOpen = automatic && (resolution?.status === 'complete' || inferred) && selected && !prepared.requirements?.length;
    this.set({ phase: canOpen ? 'saving' : 'needs_setup', detail: canOpen
      ? 'Saving an example setup. Then capturing the saved call.' : detail, discovery: prepared.discovery,
      draft: selected, requirements: requirements.length ? requirements : undefined,
      evidence });
    if (canOpen) {
      const saved = await this.dependencies.save(this.value!.target, selected!, controller.signal);
      if (!this.current(controller)) return;
      if (saved.requirements?.length) {
        this.set({ phase: 'needs_setup', detail: 'Complete the remaining setup fields.', requirements: saved.requirements });
        return;
      }
      // A saved recipe must pass normal preflight. Never discover/save in a loop.
      await this.resolve(controller, false, false);
    }
  }

  private async resolve(controller: AbortController, changeInputs = false, automatic = true): Promise<void> {
    if (!this.current(controller)) return;
    this.busy = true;
    try {
      const target = this.value!.target;
      await this.refreshConfigFile();
      if (!this.current(controller)) return;
      if (this.dependencies.environment) {
        const environment = await this.dependencies.environment(target, controller.signal,
          detail => { if (this.current(controller)) this.set({ detail: detail.slice(0, 1024) }); });
        if (!this.current(controller)) return;
        this.set({ environment, failure: undefined });
        if (environment.reason !== 'ready') {
          this.set({ phase: 'needs_setup', detail: environment.detail, failure: { kind: 'environment' }, requirements: undefined });
          return;
        }
      }
      const found = await this.dependencies.lookup(target, controller.signal);
      if (!this.current(controller)) return;
      if (changeInputs) {
        const draft = found.ok ? { profileName: found.profile, profile: found.guard.spec.profile as unknown as Record<string, unknown> }
          : found.kind === 'blocked' && found.definition ? { profileName: found.profileName, profile: found.definition } : undefined;
        await this.setup(controller, 'Choose the model configuration and example inputs. Saving continues this opening.', draft);
      } else if (found.ok) {
        this.set({ phase: 'capturing', detail: 'Constructing the model and capturing the selected example input.', failure: undefined,
          draft: { profileName: found.profile, profile: found.guard.spec.profile as unknown as Record<string, unknown> }, requirements: undefined });
        const result = await this.dependencies.capture(target, found, controller.signal,
          detail => { if (this.current(controller)) this.set({ detail }); });
        if (!this.current(controller)) return;
        if (result.dependenciesInstalled) {
          this.set({ target: result.target, phase: 'resolving', detail: 'Model libraries installed. Checking setup in the updated environment.' });
          await this.resolve(controller, false, false);
          return;
        }
        const missingCapture = result.outcome.status === 'passed' && !result.hasCapture;
        const phase = result.outcome.status === 'passed' && !missingCapture ? 'ready' : result.outcome.status === 'cancelled' ? 'cancelled' : 'failed';
        const detail = missingCapture ? 'Execution finished without a captured graph. Retry to capture this invocation; the source graph is not a measured result.' : result.outcome.detail;
        this.set({ target: result.target, hasCapture: result.hasCapture, phase, detail,
          failure: phase === 'failed' ? result.outcome.failure ?? { kind: 'capture' } : undefined });
        if (phase === 'failed' && result.outcome.failure?.kind === 'setup') {
          // Only an invalid invocation needs a form. Preserve every saved value;
          // ordinary forward failures stay on the graph that explains them.
          try {
            await this.setup(controller, detail, this.value?.draft);
            if (this.current(controller) && result.outcome.failure.fields?.length) this.set({
              requirements: [...(this.value?.requirements ?? []),
                ...result.outcome.failure.fields.map(field => `${field}: Check this binding.`)].slice(0, 64),
            });
          } catch {
            if (this.current(controller)) this.set({ phase: 'failed',
              detail: `${detail} Setup fields could not be read. Edit tensorviz.toml, then retry.`.slice(0, 1024) });
          }
        }
      } else if (found.kind === 'cancelled') {
        this.set({ phase: 'cancelled', detail: found.detail });
      } else if (found.kind === 'blocked' && found.failure && found.failure.kind !== 'setup') {
        this.set({ phase: 'failed', detail: found.detail, failure: found.failure });
      } else {
        await this.setup(controller, found.kind === 'blocked' ? found.detail
          : 'Complete the missing values for one example call. Values found in source are already filled in.',
          found.kind === 'blocked' && found.definition ? { profileName: found.profileName, profile: found.definition } : undefined,
          automatic && found.kind === 'none');
      }
    } catch (error) {
      if (this.current(controller)) this.set({ phase: error instanceof Error && error.name === 'AbortError' ? 'cancelled' : error instanceof GraphOpeningError && error.environment ? 'needs_setup' : 'failed',
        failure: error instanceof GraphOpeningError ? error.failure : undefined,
        ...(error instanceof GraphOpeningError && error.environment ? { environment: error.environment } : {}),
        detail: error instanceof Error && error.name === 'AbortError' ? 'Execution cancelled. Nothing further was run.' : error instanceof GraphOpeningError ? error.message.slice(0, 1024)
          : 'Opening could not complete. Check the selected Python environment or edit setup, then retry.' });
    } finally {
      if (this.controller === controller) this.busy = false;
    }
  }

  /** The opening id and source identity reject replayed forms from another model/version. */
  async action(openingId: string, requestId: string, target: LocalBrowserTargetIdentity, action: GraphSetupAction): Promise<boolean> {
    const state = this.value;
    let controller = this.controller;
    const reread = action.kind === 'retry' && this.dependencies.restart;
    if (!state || !controller || openingId !== state.id || this.seen.has(requestId)
      || !(reread ? sameModel(state.target, target) : sameSource(state.target, target))) return false;
    this.seen.add(requestId);
    if (this.seen.size > 256) this.seen.delete(this.seen.values().next().value!);
    if (action.kind === 'cancel') { this.cancel(); return true; }
    if (action.kind === 'source_only') {
      this.controller?.abort();
      this.controller = new AbortController();
      this.busy = false;
      this.set({ phase: 'source_only', cancellationReason: undefined, hasCapture: false, detail: 'Source-only view. Shapes are not measurements from a new execution.' });
      return true;
    }
    if (this.busy) return false;
    if (state.environment && state.environment.reason !== 'ready' && (action.kind === 'submit' || action.kind === 'factory_template')) return false;
    // Refresh requests the latest source of this opening's model. Parsing can
    // finish while the request is in flight; that must not turn Refresh into a
    // no-op. Other actions still require the exact source they were prepared for.
    if (reread) { await reread(state.target); return true; }
    if (!this.dependencies.current(state.target)) {
      this.cancel('Source changed while setup was open. Reopen the current model before saving setup.', 'source_changed');
      return false;
    }
    if (action.kind === 'parent') {
      const choice = state.discovery?.parents[action.index];
      if (!choice) return false;
      this.clear();
      await this.dependencies.parent(target, choice);
      return true;
    }
    if (['retry', 'change_inputs'].includes(action.kind)) {
      await this.open(target, action.kind === 'change_inputs');
      return true;
    }
    if (controller.signal.aborted) {
      // A new explicit repair is independent of the cancelled agent/run.
      // Keep the opening identity; never revive the old asynchronous operation.
      if (action.kind === 'edit_config' || action.kind === 'agent_help'
        || (action.kind === 'create_environment' && state.environment?.canCreate || action.kind === 'install_dependencies') && this.dependencies.recoverEnvironment
          && !state.cancellationReason) this.controller = controller = new AbortController();
      else {
        await this.open(target, true);
        return false;
      }
    }
    this.busy = true;
    try {
      if (action.kind === 'select_environment' || action.kind === 'install_dependencies' || action.kind === 'refresh_environments'
        || action.kind === 'use_installed_versions' || action.kind === 'replace_versions' || action.kind === 'create_environment') {
        const environment = state.environment;
        if (!environment || !this.dependencies.recoverEnvironment) return false;
        if (action.kind === 'create_environment' && !environment.canCreate) return false;
        if (action.kind === 'select_environment' && !environment.candidates.some(candidate => candidate.id === action.candidateId && candidate.status !== 'unavailable')) return false;
        if (action.kind === 'install_dependencies' && !(environment.reason === 'missing_libraries' && environment.requirements?.length)
          && !(environment.reason === 'incompatible' && environment.compatibilityPackages?.length)) return false;
        if (action.kind === 'use_installed_versions' && !(environment.reason === 'incompatible' && environment.declaredPins?.length)) return false;
        if (action.kind === 'replace_versions' && !(environment.reason === 'incompatible' && environment.declaredPins?.length && environment.replaceable)) return false;
        this.set({ phase: 'resolving', detail: action.kind === 'install_dependencies' ? 'Preparing the reviewed library repair.'
          : action.kind === 'use_installed_versions' ? 'Continuing with the installed library versions.'
            : action.kind === 'replace_versions' ? 'Checking that the pinned versions have a build for this Python.'
              : action.kind === 'create_environment' ? 'Creating a separate Python environment for this model…' : 'Checking Python environments…',
          cancellationReason: undefined, failure: undefined,
          environmentName: action.kind === 'create_environment' ? action.name ?? 'environment' : undefined });
        const accepted = await this.dependencies.recoverEnvironment(target, environment, action, controller.signal,
          detail => { if (this.current(controller)) this.set({ detail: detail.slice(0, 1024) }); });
        if (!this.current(controller)) return false;
        if (!accepted) { this.set({ phase: 'needs_setup', detail: environment.detail }); return false; }
        if (action.kind === 'refresh_environments') {
          const refreshed = typeof accepted === 'object' ? accepted : await this.dependencies.environment?.(target, controller.signal, () => {});
          if (!this.current(controller) || !refreshed) return false;
          this.set({ phase: 'needs_setup', environment: { ...refreshed, reason: refreshed.reason === 'ready' ? 'selection_required' : refreshed.reason },
            detail: 'Environments refreshed. Choose Python to continue opening this model.' });
          return true;
        }
        if (typeof accepted === 'object' && accepted.reason !== 'ready') {
          this.set({ phase: 'needs_setup', environment: accepted, detail: accepted.detail }); return true;
        }
        await this.resolve(controller);
      } else if (action.kind === 'submit') {
        this.set({ phase: 'saving', detail: 'Checking and saving this execution setup.', draft: action.draft, requirements: undefined, evidence: undefined });
        const result = await this.dependencies.save(target, action.draft, controller.signal);
        if (!this.current(controller)) return false;
        if (result.requirements?.length) {
          this.set({ phase: 'needs_setup', detail: 'Complete the highlighted setup fields.', requirements: result.requirements });
          return true;
        }
        this.set({ phase: 'resolving', detail: 'Setup saved. Checking the explicit invocation.' });
        await this.resolve(controller, false, false);
      } else if (action.kind === 'factory_template' && this.dependencies.factoryTemplate) {
        const draft = await this.dependencies.factoryTemplate(target, action.draft ?? state.draft, controller.signal);
        if (!this.current(controller)) return false;
        this.set({ phase: 'needs_setup', draft, detail: 'Complete and save the Python factory in VS Code, then save the input setup here.' });
      } else {
        if (action.kind === 'agent_help' && action.draft) {
          if (state.phase !== 'needs_setup' && !(state.phase === 'failed' && state.failure?.kind === 'setup')) return false;
          this.set({ draft: action.draft, evidence: undefined });
        }
        const fresh = await this.dependencies.recover(target, action.kind, controller.signal, action);
        if (this.controller !== controller || controller.signal.aborted) return false;
        if (fresh) {
          this.set({ target: fresh, phase: 'resolving', detail: action.kind === 'environment' ? 'Checking the selected Python environment…' : 'Continuing with saved source.' });
          await this.resolve(controller);
        }
      }
      return true;
    } catch (error) {
      if (this.current(controller)) this.set({ phase: action.kind === 'create_environment' || action.kind === 'install_dependencies' || error instanceof GraphOpeningError && error.failure && !error.environment ? 'failed' : 'needs_setup',
        failure: error instanceof GraphOpeningError ? error.failure : undefined,
        ...(error instanceof GraphOpeningError && error.environment ? { environment: error.environment } : {}),
        detail: error instanceof GraphOpeningError ? error.message.slice(0, 1024)
          : 'Setup could not be saved or resumed. Check unsaved files and tensorviz.toml in VS Code; your form values are retained.' });
      return false;
    } finally {
      if (this.controller === controller) this.busy = false;
    }
  }
}
