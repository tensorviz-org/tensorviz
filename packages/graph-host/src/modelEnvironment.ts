import * as path from 'node:path';
import type { GraphEnvironmentCandidate, GraphEnvironmentRecovery, LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';
import { runEnvironmentHelper, type EnvironmentRunner, type HelperResult } from './projectEnvironmentCore';

interface ModelInspectionOptions {
  target: LocalBrowserTargetIdentity; backendDir: string; managerPython?: string; signal: AbortSignal;
  runner?: EnvironmentRunner; missingModules?: string[];
}

/** The `inspect-model` check for one interpreter. It needs nothing from the capture
 *  check, which is why an opening can start it while that check is still running. */
function runModelInspection(interpreter: string, options: ModelInspectionOptions): Promise<HelperResult> {
  return (options.runner ?? runEnvironmentHelper)(options.managerPython ?? interpreter, options.backendDir,
    [...(options.managerPython ? ['--target-python', interpreter] : []), 'inspect-model', options.target.workspaceFolder,
      path.join(options.target.workspaceFolder, options.target.file), JSON.stringify(options.missingModules ?? [])],
    () => options.signal.aborted, true, true);
}

/** A model check already started for `interpreter`. Its outcome is captured as a
 *  value, so a check that is never awaited (an unavailable environment) cannot
 *  become an unhandled rejection. */
export interface ModelInspection {
  interpreter: string;
  settled: Promise<{ result: HelperResult } | { error: unknown }>;
}

export function startModelInspection(interpreter: string, options: ModelInspectionOptions): ModelInspection {
  const settled = Promise.resolve().then(() => runModelInspection(interpreter, options))
    .then(result => ({ result }), error => ({ error }));
  return { interpreter, settled };
}


/** Check only the chosen project Python; candidate discovery never imports models. */
export async function inspectModelEnvironment(candidate: GraphEnvironmentCandidate, options: {
  target: LocalBrowserTargetIdentity; backendDir: string; managerPython?: string; signal: AbortSignal;
  runner?: EnvironmentRunner; missingModules?: string[]; importFailure?: boolean;
  workerCache?: Set<string>;
  /** Native hosts review exact resolved wheels before approving a missing Torch build. */
  reviewMissingTorch?: boolean;
  /** Native hosts can offer running with installed versions despite the project's own pins. */
  offerPins?: boolean;
  /** Project pins the user chose to run without in this environment. */
  acceptedPins?: string[];
  /** The same model check, already started for this interpreter (see startModelInspection). */
  inspection?: ModelInspection;
}): Promise<GraphEnvironmentRecovery> {
  const base = { candidates: [candidate], selectedId: candidate.id };
  if (candidate.status === 'unavailable') return { ...base, reason: 'unavailable',
    detail: 'The selected Python environment is no longer available. Choose an existing environment.' };
  const run = async (args: string[]) => {
    options.signal.throwIfAborted();
    const result = await (options.runner ?? runEnvironmentHelper)(options.managerPython ?? candidate.interpreter, options.backendDir,
      [...(options.managerPython ? ['--target-python', candidate.interpreter] : []), ...args], () => options.signal.aborted, true, true);
    options.signal.throwIfAborted();
    return result;
  };
  const started = options.inspection?.interpreter === candidate.interpreter ? options.inspection : undefined;
  options.signal.throwIfAborted();
  const outcome = started ? await started.settled : { result: await runModelInspection(candidate.interpreter, options) };
  options.signal.throwIfAborted();
  if ('error' in outcome) throw outcome.error;
  const model = outcome.result;
  if (!model.ok) return { ...base, reason: 'worker_failed', detail: (model.message ?? 'Model libraries could not be checked. Use MCP to investigate this environment.').slice(0, 1024) };
  const missing = (model.missingModules ?? []).filter(item => !item.optional);
  const requirements = [...new Set([...(candidate.missing ?? []), ...missing.flatMap(item => item.requirement ? [item.requirement] : [])])];
  const missingModules = missing.map(item => item.module).slice(0, 32);
  const unsupportedDeclared = missingModules.filter(module => model.unsupportedDeclaredModules?.includes(module));
  // Only the project's own pins are optional. Capture-library conflicts are never
  // overridable, and an observed import failure brings the choice back.
  const pins = options.offerPins && !candidate.incompatible?.length ? model.incompatibleRequirements ?? [] : [];
  const accepted = pins.length > 0 && !options.importFailure && pins.every(pin => options.acceptedPins?.includes(pin));
  const conflicts = [...(candidate.incompatible ?? []), ...(accepted ? [] : model.incompatible ?? [])];
  if (conflicts.length || options.importFailure && model.compatibilityPackages?.length) return { ...base, reason: 'incompatible',
    detail: (conflicts.length ? `Installed libraries do not match this model: ${conflicts.join(', ')}. ${pins.length
      ? 'Use “Try with installed versions” or open_graph action: use_installed_versions to continue without changing packages. Capture-worker checks still apply.'
      : 'Choose a compatible environment.'}`
      : 'An installed model library failed to import. A reviewed compatibility copy may resolve this; the original environment is preserved.').slice(0, 1024),
    ...(model.compatibilityPackages?.length && !candidate.incompatible?.length ? { compatibilityPackages: model.compatibilityPackages.slice(0, 32) } : {}),
    ...(pins.length ? { declaredPins: pins.slice(0, 32) } : {}) };
  if (requirements.length || missing.length) {
    const noTorch = !candidate.torch;
    return { ...base, reason: 'missing_libraries', missingModules,
      ...((!noTorch || options.reviewMissingTorch) && requirements.length <= 32 ? { requirements } : {}),
      detail: noTorch && !options.reviewMissingTorch ? 'PyTorch is not installed here. Choose an environment with the Torch build your model uses, or explicitly select the tensorViz-managed environment.'
        : requirements.length > 32 ? 'This model needs more than 32 libraries. Prepare its project requirements in a compatible environment, then refresh.'
        : unsupportedDeclared.length ? `This project already declares ${unsupportedDeclared.join(', ')}, but its dependency source or condition is not supported by the guided installer. Use an environment prepared with this project's installation instructions, or use agent help. Refresh after preparing it.`.slice(0, 1024)
        : missing.some(item => !item.requirement) ? `Missing model libraries: ${missingModules.join(', ')}. Add the correct distribution names to requirements.txt or pyproject.toml, or use MCP to investigate.`.slice(0, 1024)
        : noTorch ? `PyTorch and model libraries are missing in the selected environment: ${missingModules.join(', ')}. Review the exact PyPI wheel versions before installing; PyTorch can be a large download. Choose another environment if your model needs a specific Torch/CUDA build.`.slice(0, 1024)
        : `This model needs ${[...(candidate.missing ?? []), ...missingModules.map(name => name === 'cv2' ? 'OpenCV' : name === 'scipy' ? 'SciPy' : name)].join(', ')} in the selected environment. Review the missing libraries, then continue opening this model.`.slice(0, 1024) };
  }
  const key = candidate.fingerprint ? `${options.backendDir}:${candidate.fingerprint}` : undefined;
  if (!key || !options.workerCache?.has(key)) {
    const probe = await run(['probe-capture']);
    if (!probe.ok) return { ...base, reason: 'worker_failed',
      detail: (probe.message ?? 'The capture worker could not load in this Python environment. Choose another environment or use MCP to investigate.').slice(0, 1024) };
    if (key && options.workerCache) {
      if (options.workerCache.size >= 32) options.workerCache.delete(options.workerCache.values().next().value!);
      options.workerCache.add(key);
    }
  }
  if (options.importFailure) return { ...base, reason: 'worker_failed',
    detail: 'The model failed to import despite its libraries being present. Use MCP to investigate the observed import failure, or choose another environment.' };
  return { ...base, reason: 'ready', detail: (`${accepted ? `Using installed ${model.incompatible!.join(', ')} instead of the project pins. ` : ''}`
    + 'Selected Python and capture libraries are ready. Model execution still needs the saved setup.').slice(0, 1024) };
}
