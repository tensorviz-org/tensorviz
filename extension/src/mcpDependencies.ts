import * as vscode from 'vscode';
import { installDependencies, replaceDependencies, replacementSummary, type DependencyInstallOptions as CoreDependencyInstallOptions } from '@tensorviz/graph-host/dependencyInstallCore';
import type { CanvasAgentPermissionRequest } from '@tensorviz/contracts/protocol';
import { runEnvironmentHelper } from '@tensorviz/graph-host/projectEnvironmentCore';
import { GraphToolError } from '@tensorviz/graph-host/mcpGraph';
import { repairModelCompatibility } from './modelCompatibility';
export interface DependencyInstallOptions extends CoreDependencyInstallOptions {
  approve?(request: CanvasAgentPermissionRequest, signal: AbortSignal): Promise<boolean | undefined>;
}

/** An active Canvas turn owns its permission UI. A denial never opens a second prompt. */
async function review(options: DependencyInstallOptions, request: CanvasAgentPermissionRequest, native: () => PromiseLike<boolean>): Promise<boolean> {
  options.signal.throwIfAborted();
  if (options.approve) options.progress?.('Waiting for approval in chat…');
  const approved = await options.approve?.(request, options.signal);
  options.signal.throwIfAborted();
  if (approved !== undefined) return approved;
  if (options.approve) options.progress?.('Waiting for approval in VS Code…');
  return native();
}

/** Canvas and agent requests review the same frozen plan in the originating surface. */
export async function installMcpDependencies(options: DependencyInstallOptions): Promise<void> {
  // The shared installer names its native reviewer. Until routing is resolved, keep progress neutral.
  const progress = options.approve && options.progress ? (detail: string) => options.progress!(detail.replace(' in VS Code', '')) : options.progress;
  return installDependencies({ ...options, progress }, {
    trusted: () => vscode.workspace.isTrusted,
    reviewSource: async () => {
      const details = `${options.python}\n\n${options.packages.join('\n')}\n\n` +
        'Compatible wheels were not found. Resolving and building source distributions runs package-provided Python code, which can access your machine. pip uses separate build environments; this is not a security sandbox. You will review the resulting package versions before installation in the TensorViz capture environment.';
      return review(options, { operation: 'python.packages.build', scope: 'existing_environment', effects: ['execute_code'],
        facts: { python: options.python, requestedPackages: options.packages, access: { filesystem: true, network: true } } },
        async () => await vscode.window.showWarningMessage('Build model dependencies from source?',
        { modal: true, detail: details }, 'Build from source') === 'Build from source');
    },
    reviewInstall: async plan => {
      const details = `${options.python}\n\n${plan.packages!.map(item => `${item.name} ${item.version}`).join('\n')}\n\n` +
        (plan.builtFromSource?.length ? `Built from source: ${plan.builtFromSource.join(', ')}.\n\n` : '') +
        'This installs the reviewed, hash-verified wheels in the TensorViz capture environment. Existing packages, including Torch/CUDA, will not be replaced. Cancellation may leave some new packages installed.';
      return review(options, { operation: 'python.packages.install', scope: 'existing_environment', effects: ['install'],
        facts: { python: options.python, planDigest: plan.digest!, packages: plan.packages!.map(({ name, version }) => ({ name, version })),
          builtFromSource: plan.builtFromSource ?? [], preservesExistingPackages: true, preservesTorch: true,
          cancellationMayLeaveChanges: true } }, async () => await vscode.window.showInformationMessage('Install model dependencies for this graph?',
        { modal: true, detail: details }, 'Install dependencies') === 'Install dependencies');
    },
  });
}

/** In-place replacement asks twice when the Torch/CUDA build changes; nothing runs before both answers. */
export async function replaceModelVersions(options: DependencyInstallOptions & { project: string }): Promise<void> {
  return replaceDependencies(options, {
    trusted: () => vscode.workspace.isTrusted,
    reviewReplacement: async plan => {
      const details = `${options.python}\n\n${replacementSummary(plan).join('\n')}\n\n` +
        'These hash-verified wheels replace the installed versions in this environment, not in a copy. Other projects and models that use it get these versions too. tensorViz keeps no backup; reinstall the previous versions to undo.';
      return review(options, { operation: 'python.packages.replace', scope: 'existing_environment', effects: ['install', 'replace'],
        facts: { project: options.project, python: options.python, planDigest: plan.digest!,
          packages: plan.packages!.map(({ name, version, previous }) => ({ name, version, ...(previous ? { previous } : {}) })),
          replacesTorch: plan.replacesTorch ?? false, affectsSharedEnvironment: true, backupAvailable: false,
          cancellationMayLeaveChanges: true } }, async () => await vscode.window.showWarningMessage('Replace library versions in this environment?',
        { modal: true, detail: details }, 'Replace versions', 'Cancel') === 'Replace versions');
    },
    confirmTorch: async plan => {
      const details = `${options.python}\n\n${replacementSummary(plan).filter(line => /^\+? ?(?:torch|torchvision|torchaudio|triton|xformers|nvidia-)/.test(line)).join('\n')}\n\n` +
        'This changes the PyTorch build, including any CUDA libraries, for everything that uses this environment. A different Torch can change device support and results.';
      return review(options, { operation: 'python.packages.replace_torch', scope: 'existing_environment', effects: ['install', 'replace'],
        facts: { project: options.project, python: options.python, planDigest: plan.digest!,
          packages: plan.packages!.filter(item => /^(?:torch|torchvision|torchaudio|triton|xformers|nvidia-)/.test(item.name))
            .map(({ name, version, previous }) => ({ name, version, ...(previous ? { previous } : {}) })),
          replacesTorch: true, affectsSharedEnvironment: true, backupAvailable: false, deviceSupportMayChange: true } },
        async () => await vscode.window.showWarningMessage('Replace PyTorch in this environment?',
        { modal: true, detail: details }, 'Replace PyTorch', 'Cancel') === 'Replace PyTorch');
    },
  });
}

/** Inspect saved imports before graph execution; runtime diagnostics can confirm
 * a deferred import. Never turn an arbitrary import name into a pip package. */
export async function ensureModelDependencies(options: Omit<DependencyInstallOptions, 'packages'> & {
  workspace: string; file: string; missingModules?: string[];
  importFailure?: boolean; selectEnvironment?(python: string): Promise<void>;
}): Promise<boolean> {
  if (!vscode.workspace.isTrusted) throw new GraphToolError('WORKSPACE_UNTRUSTED', 'Trust the workspace before checking model dependencies.');
  const check = async () => {
    if (options.signal.aborted || !await options.current()) throw new GraphToolError('STALE_TARGET', 'The model or environment changed. Refresh the graph.');
  };
  await check();
  const inspect = () => (options.runner ?? runEnvironmentHelper)(options.managerPython ?? options.python, options.backendDir,
    [...(options.managerPython ? ['--target-python', options.python] : []),
      'inspect-model', options.workspace, options.file, JSON.stringify(options.missingModules ?? [])], () => options.signal.aborted);
  const report = await inspect();
  await check();
  if (!report.ok) throw new GraphToolError(report.code ?? 'DEPENDENCY_CHECK_FAILED', report.message ?? 'Could not check model dependencies in the selected Python environment.');
  if ((options.importFailure || report.incompatible?.length) && report.compatibilityPackages?.length && options.selectEnvironment) {
    return vscode.window.withProgress({ location: vscode.ProgressLocation.Notification,
      title: 'tensorViz: preparing compatible model libraries', cancellable: true }, async (_progress, token) => {
      const controller = new AbortController();
      const abort = () => controller.abort();
      options.signal.addEventListener('abort', abort, { once: true });
      const subscription = token.onCancellationRequested(abort);
      if (options.signal.aborted || token.isCancellationRequested) abort();
      try { return await repairModelCompatibility({ ...options, selectEnvironment: options.selectEnvironment!, signal: controller.signal }, report.compatibilityPackages!); }
      finally { subscription.dispose(); options.signal.removeEventListener('abort', abort); }
    });
  }
  if (report.incompatible?.length) throw new GraphToolError('DEPENDENCY_CONFLICT',
    `Installed libraries do not match project requirements: ${report.incompatible.join(', ')}. Select a compatible Python environment. Existing packages were preserved.`);
  const missing = (report.missingModules ?? []).filter(item => !item.optional);
  if (!missing.length) return false;
  const packages = [...new Set(missing.flatMap(item => item.requirement ? [item.requirement] : []))];
  if (packages.length > 32) throw new GraphToolError('DEPENDENCY_LIMIT', 'This model needs more than 32 libraries. Install its project requirements in the selected Python environment, then Refresh.');
  if (packages.length) {
    await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification,
      title: `tensorViz: installing missing model libraries (${missing.filter(item => item.requirement).map(item => item.module).join(', ')})`, cancellable: true },
    async (_progress, token) => {
      const controller = new AbortController();
      const abort = () => controller.abort();
      options.signal.addEventListener('abort', abort, { once: true });
      const subscription = token.onCancellationRequested(abort);
      if (options.signal.aborted || token.isCancellationRequested) abort();
      try { await installMcpDependencies({ ...options, packages, signal: controller.signal }); }
      finally { subscription.dispose(); options.signal.removeEventListener('abort', abort); }
    });
  }
  await check();
  const verified = await inspect();
  await check();
  if (!verified.ok) throw new GraphToolError('DEPENDENCY_CHECK_FAILED', verified.message ?? 'Could not verify the installed libraries. Refresh to check again.');
  const remaining = (verified.missingModules ?? []).filter(item => !item.optional);
  if (remaining.length) throw new GraphToolError('MISSING_DEPENDENCIES',
    `Missing model libraries: ${remaining.map(item => item.module).join(', ')}. ` +
    (remaining.some(item => !item.requirement)
      ? 'Add their correct package names to requirements.txt or pyproject.toml, then Refresh to review installation. Import names may differ from package names.'
      : 'The selected environment still cannot find these imports. Check its project requirements, then Refresh.') + ` Python: ${options.python}`);
  return packages.length > 0;
}
