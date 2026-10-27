import * as path from 'node:path';
import * as vscode from 'vscode';
import type { GraphEnvironmentCandidate, GraphEnvironmentRecovery, LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';
import { GraphOpeningError, type EnvironmentRecoveryAction } from '@tensorviz/graph-host/graphOpening';
import { inspectEnvironmentCandidates, type EnvironmentRunner } from '@tensorviz/graph-host/projectEnvironmentCore';
import { inspectModelEnvironment, startModelInspection } from '@tensorviz/graph-host/modelEnvironment';
import { GraphToolError } from '@tensorviz/graph-host/mcpGraph';
import { automaticProjectCandidate, discoverEnvironments, environmentCost, environmentExecutable, rankEnvironmentCandidates, readyForModel,
  rememberEnvironment, runEnvironmentHelper, savedEnvironment } from './projectEnvironment';
import { ensureModelDependencies, installMcpDependencies, replaceModelVersions, type DependencyInstallOptions } from './mcpDependencies';
import { createModelEnvironment } from './modelEnvironmentCreation';

type Context = Pick<vscode.ExtensionContext, 'workspaceState' | 'globalState' | 'globalStorageUri'>;
export interface OpeningEnvironmentOptions {
  context: Context; backendDir: string; managerPython?: string; target: LocalBrowserTargetIdentity; signal: AbortSignal;
  readyManaged(): Promise<string | undefined>;
  workerCache: Set<string>;
  runner?: EnvironmentRunner;
  diagnostic?(message: string): void;
}
const resourceFor = (target: LocalBrowserTargetIdentity) => vscode.Uri.file(path.join(target.workspaceFolder, target.file));
/** Project pins the user chose to run without, per workspace and interpreter, for this session.
 * Re-selecting the environment starts a fresh check. */
const acceptedPins = new Map<string, string[]>();
const pinKey = (target: LocalBrowserTargetIdentity, interpreter: string) => `${target.workspaceFolder}\0${interpreter}`;

/** Explicit model opening uses a Python that can already run the model when one exists:
 * the best-ranked ready environment when nothing is saved, or in place of a choice tensorViz
 * saved itself. A choice the user made stays authoritative, even when broken. Every
 * adoption still passes the full model and worker checks below. */
export async function checkOpeningEnvironment(options: OpeningEnvironmentOptions, evidence: {
  refresh?: boolean; missingModules?: string[]; importFailure?: boolean;
  /** Set after one automatic adoption, so a check never switches environments twice. */
  adopted?: boolean;
} = {}): Promise<GraphEnvironmentRecovery> {
  if (!vscode.workspace.isTrusted) throw new GraphToolError('WORKSPACE_UNTRUSTED', 'Trust this workspace before checking model environments.');
  const resource = resourceFor(options.target), choice = savedEnvironment(options.context, resource);
  const configured = vscode.workspace.getConfiguration('tensorviz', resource).get<string>('pythonPath', '').trim();
  const selected = choice?.kind === 'project' ? choice.interpreter : choice?.kind === 'managed' ? await options.readyManaged()
    : configured ? environmentExecutable(configured) ?? configured : undefined;
  options.signal.throwIfAborted();
  const runner = options.runner ?? runEnvironmentHelper;
  const collect = async () => rankEnvironmentCandidates(await discoverEnvironments(options.context, options.backendDir, { ...options,
    resource, workspaceRoot: options.target.workspaceFolder, runner,
    model: { workspaceFolder: options.target.workspaceFolder, file: options.target.file, missingModules: evidence.missingModules } }));
  const adopt = async (candidate: GraphEnvironmentCandidate) => {
    options.signal.throwIfAborted();
    await rememberEnvironment(options.context, { kind: 'project', interpreter: candidate.interpreter }, resource);
    return checkOpeningEnvironment(options, { ...evidence, adopted: true });
  };
  // The user's own choice, or the tensorViz Python setting, is never second-guessed by a recommendation.
  const userChoice = choice ? !!choice.explicit : !!configured;
  const recommended = (candidates: GraphEnvironmentCandidate[], failed?: string) => {
    const best = userChoice ? undefined : candidates.find(item => item.id !== failed && environmentCost(item) !== undefined);
    return best ? { recommendedId: best.id } : {};
  };
  if (!selected || evidence.refresh) {
    const candidates = await collect();
    if (!evidence.refresh && !choice && !configured && !evidence.adopted) {
      // An environment that can already run this model comes first. Otherwise the VS Code
      // selection or the only local environment is adopted so its repair can be offered.
      const automatic = candidates.find(candidate => readyForModel(candidate) && !candidate.managed) ?? automaticProjectCandidate(candidates);
      if (automatic?.status === 'libraries_present' && !automatic.managed) return adopt(automatic);
    }
    const saved = candidates.find(candidate => candidate.selected);
    const suggestion = recommended(candidates);
    return { reason: !evidence.refresh && (choice || configured) ? 'unavailable' : 'selection_required', candidates,
      ...(options.managerPython ? { canCreate: true as const } : {}),
      ...(saved ? { selectedId: saved.id } : {}), ...suggestion,
      detail: choice || configured ? 'Choose Python for this model. The saved choice and detected alternatives are shown below.'
        : suggestion.recommendedId ? 'Choose the Python environment for this model. The recommended one needs the fewest changes.'
          : 'Choose the Python environment for this model. The VS Code selection is highlighted as a suggestion.' };
  }
  const capture = inspectEnvironmentCandidates([{ interpreter: selected,
    source: choice?.kind === 'managed' ? 'tensorViz-managed fallback' : configured && !choice ? 'tensorViz Python setting' : 'Saved for this folder',
    selected: true, ...(choice?.kind === 'managed' ? { managed: true } : {}) }], { ...options, runner });
  // The model check needs only the interpreter, so it runs alongside the capture
  // check instead of after it. Both are read-only; if the environment turns out to
  // be unavailable, its result is simply not used.
  const inspection = startModelInspection(selected, { ...options, missingModules: evidence.missingModules, runner });
  const [candidate] = await capture;
  const recovery = await inspectModelEnvironment(candidate, { ...options, ...evidence, reviewMissingTorch: true,
    offerPins: true, acceptedPins: acceptedPins.get(pinKey(options.target, candidate.interpreter)),
    runner, inspection });
  if (recovery.reason === 'ready') return recovery;
  const alternatives = await collect();
  const same = (item: GraphEnvironmentCandidate) => item.id === candidate.id || (!!candidate.prefix && item.prefix === candidate.prefix);
  const others = alternatives.filter(item => !same(item));
  // A choice tensorViz saved, or one saved before choices were marked, gives way once to an
  // environment that can already run this model when it only lacks libraries or has other
  // versions. A missing or failing Python, or an observed import failure, stays visible instead.
  if (choice && !choice.explicit && !evidence.adopted && !evidence.importFailure
    && (recovery.reason === 'missing_libraries' || recovery.reason === 'incompatible')) {
    const ready = others.find(item => readyForModel(item) && !item.managed);
    if (ready) return adopt(ready);
  }
  // The saved row shows discovery's model check; its full check above stays authoritative.
  const discovered = alternatives.find(same);
  const saved: GraphEnvironmentCandidate = { ...candidate, ...(discovered?.modelChecked ? { modelChecked: true as const,
    ...(discovered.modelMissing ? { modelMissing: discovered.modelMissing } : {}),
    ...(discovered.modelIncompatible ? { modelIncompatible: discovered.modelIncompatible } : {}) } : {}) };
  const candidates = rankEnvironmentCandidates([saved, ...others]).slice(0, 16);
  return { ...recovery, ...(recovery.declaredPins && !candidate.managed ? { replaceable: true as const } : {}),
    ...(options.managerPython ? { canCreate: true as const } : {}), candidates,
    ...recommended(candidates, ['worker_failed', 'unavailable'].includes(recovery.reason) ? saved.id : undefined) };
}

export async function recoverOpeningEnvironment(options: OpeningEnvironmentOptions, environment: GraphEnvironmentRecovery,
  action: EnvironmentRecoveryAction,
  host: { current(): Promise<boolean>; dependencies(): Promise<Omit<DependencyInstallOptions, 'packages'>>;
    changed(): void; progress?(detail: string): void }): Promise<GraphEnvironmentRecovery | boolean> {
  const check = async () => {
    options.signal.throwIfAborted();
    if (!vscode.workspace.isTrusted || !await host.current()) throw new GraphToolError('STALE_TARGET', 'The workspace or model changed. Open the current saved model again.');
  };
  await check();
  if (action.kind === 'refresh_environments') return checkOpeningEnvironment(options, { refresh: true });
  if (action.kind === 'select_environment') {
    const candidate = environment.candidates.find(item => item.id === action.candidateId);
    if (!candidate || candidate.status === 'unavailable') return false;
    const [fresh] = await inspectEnvironmentCandidates([candidate], { ...options, runner: options.runner ?? runEnvironmentHelper });
    await check();
    if (fresh.status === 'unavailable' || candidate.fingerprint && candidate.fingerprint !== fresh.fingerprint) {
      return { ...environment, reason: 'selection_required', detail: 'This environment changed while the chooser was open. Refresh the list, then select it again.' };
    }
    await rememberEnvironment(options.context, candidate.managed ? { kind: 'managed', explicit: true }
      : { kind: 'project', interpreter: candidate.interpreter, explicit: true }, resourceFor(options.target));
    acceptedPins.delete(pinKey(options.target, candidate.interpreter));
    host.changed();
    return true;
  }
  const saved = environment.candidates.find(item => item.id === environment.selectedId);
  if (action.kind === 'use_installed_versions') {
    if (!saved || environment.reason !== 'incompatible' || !environment.declaredPins?.length) return false;
    acceptedPins.set(pinKey(options.target, saved.interpreter), environment.declaredPins);
    return checkOpeningEnvironment(options);
  }
  // The tensorViz-managed runtime is shared by every model; it is never changed in place.
  if (action.kind === 'replace_versions' && (!saved || saved.managed || environment.reason !== 'incompatible'
    || !environment.replaceable || !environment.declaredPins?.length)) return false;
  try {
    if (action.kind === 'create_environment') {
      if (!environment.canCreate || !options.managerPython) return false;
      await createModelEnvironment({ managerPython: options.managerPython, backendDir: options.backendDir,
        storage: options.context.globalStorageUri.fsPath, workspace: options.target.workspaceFolder,
        name: action.name,
        file: path.join(options.target.workspaceFolder, options.target.file), signal: options.signal,
        current: host.current, progress: detail => { options.diagnostic?.(`[model environment] ${detail}`); host.progress?.(detail); }, runner: options.runner,
        selectEnvironment: python => rememberEnvironment(options.context, { kind: 'project', interpreter: python, explicit: true }, resourceFor(options.target)) });
      await check();
      host.changed();
      return checkOpeningEnvironment(options);
    }
    const dependencies = await host.dependencies();
    if (action.kind === 'replace_versions' && saved) {
      if (dependencies.python !== saved.interpreter) throw new GraphToolError('STALE_TARGET', 'The selected environment changed. Check it again before replacing versions.');
      await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification,
        title: 'tensorViz: replacing pinned library versions', cancellable: true }, async (_progress, token) => {
        const controller = new AbortController(), abort = () => controller.abort();
        options.signal.addEventListener('abort', abort, { once: true });
        const subscription = token.onCancellationRequested(abort);
        if (options.signal.aborted || token.isCancellationRequested) abort();
        try { await replaceModelVersions({ ...dependencies, packages: environment.declaredPins!, project: options.target.workspaceFolder, signal: controller.signal }); }
        finally { subscription.dispose(); options.signal.removeEventListener('abort', abort); }
      });
      acceptedPins.delete(pinKey(options.target, saved.interpreter));
    } else if (environment.reason === 'incompatible' && environment.compatibilityPackages?.length) {
      const repaired = await ensureModelDependencies({ ...dependencies, progress: host.progress, workspace: options.target.workspaceFolder,
        file: path.join(options.target.workspaceFolder, options.target.file), importFailure: true,
        selectEnvironment: python => rememberEnvironment(options.context, { kind: 'project', interpreter: python, explicit: true }, resourceFor(options.target)) });
      if (!repaired) return false;
    } else if (environment.reason === 'missing_libraries' && environment.requirements?.length) {
      await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification,
        title: 'tensorViz: installing reviewed model libraries', cancellable: true }, async (_progress, token) => {
        const controller = new AbortController(), abort = () => controller.abort();
        options.signal.addEventListener('abort', abort, { once: true });
        const subscription = token.onCancellationRequested(abort);
        if (options.signal.aborted || token.isCancellationRequested) abort();
        try { await installMcpDependencies({ ...dependencies, packages: environment.requirements!, signal: controller.signal, progress: host.progress }); }
        finally { subscription.dispose(); options.signal.removeEventListener('abort', abort); }
      });
    } else return false;
    await check();
    host.changed();
    host.progress?.('Checking capture tools and required model libraries…');
    const verified = await checkOpeningEnvironment(options);
    if (action.kind === 'install_dependencies' && verified.reason !== 'ready') throw new GraphToolError('LIBRARY_CHECK_FAILED', verified.detail);
    return verified;
  } catch (error) {
    options.diagnostic?.(`Environment repair: ${error instanceof Error ? error.stack ?? error.message : String(error)}`);
    await check();
    const observed = await checkOpeningEnvironment(options);
    if (action.kind === 'create_environment' || action.kind === 'install_dependencies') throw new GraphOpeningError(
      error instanceof GraphToolError ? error.message : action.kind === 'create_environment'
        ? 'Environment creation failed. Choose another Python environment or inspect the setup details.' : 'Library setup failed. Retry or choose another Python environment.',
      { kind: 'environment', ...(action.kind === 'install_dependencies' ? { code: error instanceof GraphToolError ? error.code : 'DEPENDENCY_SETUP_FAILED' } : {}) },
      { ...observed, reason: observed.reason === 'ready' ? 'selection_required' : observed.reason });
    return { ...observed, reason: observed.reason === 'ready' ? 'selection_required' : observed.reason,
      detail: ((error instanceof GraphToolError ? error.message : 'Library repair did not complete. Refresh or choose another environment.')
        + (action.kind === 'replace_versions' ? ' The environment was checked again.' : ' The environment was checked again; existing packages were preserved.')).slice(0, 1024) };
  }
}
