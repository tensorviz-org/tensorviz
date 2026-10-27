import * as fs from 'node:fs/promises';
import { GraphToolError } from './mcpGraph';
import { runEnvironmentHelper, type EnvironmentRunner, type HelperResult } from './projectEnvironmentCore';
import { ownedTemporaryDirectory } from './ownedTemporary';

export interface DependencyInstallOptions {
  python: string;
  /** Tool-owned installer; the project target does not need pip or packaging. */
  managerPython?: string;
  backendDir: string;
  storage: string;
  packages: string[];
  signal: AbortSignal;
  current(): Promise<boolean>;
  progress?(detail: string): void;
  runner?: EnvironmentRunner;
}
export interface DependencyInstallHost {
  trusted(): boolean;
  reviewSource(): Promise<boolean>;
  reviewInstall(plan: HelperResult): Promise<boolean>;
  isolated?: boolean;
  parentPipe?: boolean;
}

/** Both hosts execute the same frozen additive plan; only their approval UI differs. */
export async function installDependencies(options: DependencyInstallOptions, host: DependencyInstallHost): Promise<void> {
  if (!host.trusted()) throw new GraphToolError('WORKSPACE_UNTRUSTED', 'Trust the workspace before installing model dependencies.');
  const cancelled = () => options.signal.aborted;
  if (cancelled() || !await options.current()) throw new GraphToolError('STALE_TARGET', 'The model or environment changed. Start graph opening again.');
  await fs.mkdir(options.storage, { recursive: true });
  const { directory, dispose } = await ownedTemporaryDirectory(await fs.realpath(options.storage), 'model-dependencies-');
  const run = (args: string[]) => (options.runner ?? runEnvironmentHelper)(options.managerPython ?? options.python,
    options.backendDir, options.managerPython ? ['--target-python', options.python, ...args] : args,
    cancelled, host.isolated, host.parentPipe);
  try {
    options.progress?.('Resolving and downloading compatible model libraries…');
    let plan = await run(['plan-packages', directory, JSON.stringify(options.packages)]);
    if (!plan.ok && plan.code === 'NO_COMPATIBLE_WHEEL') {
      if (cancelled() || !host.trusted() || !await options.current()) throw new GraphToolError('STALE_TARGET', 'The model or environment changed before source-build review.');
      if (plan.requirement) {
        options.progress?.(`Checking availability of ${plan.requirement}…`);
        const available = await run(['check-package-availability', plan.requirement]);
        if (!available.ok) throw new GraphToolError(available.code ?? 'DEPENDENCY_SETUP_FAILED',
          `${available.requirement ?? plan.requirement}: ${available.message ?? 'Package availability could not be checked.'}`);
        if (cancelled() || !host.trusted() || !await options.current()) throw new GraphToolError('STALE_TARGET', 'The model or environment changed before source-build review.');
      }
      options.progress?.('Waiting for source-build approval in VS Code…');
      if (!await host.reviewSource() || cancelled()) throw new GraphToolError('SOURCE_BUILD_DECLINED', 'Building dependencies from source was not approved. No source build was started.');
      if (!host.trusted() || !await options.current()) throw new GraphToolError('STALE_TARGET', 'The model or environment changed after source-build review.');
      options.progress?.('Building approved model libraries and resolving their downloads…');
      plan = await run(['plan-packages-source', directory, JSON.stringify(options.packages)]);
    }
    if (!plan.ok || !plan.digest || !plan.packages) throw new GraphToolError(plan.code ?? 'DEPENDENCY_CONFLICT',
      (plan.requirement ? `${plan.requirement}: ` : '') + (plan.message ?? 'No additive installation can satisfy these packages in the capture environment. Inspect the requirements and installed versions.'));
    if (cancelled() || !await options.current()) throw new GraphToolError('STALE_TARGET', 'The model or environment changed before dependency review.');
    if (!plan.packages.length) return;
    options.progress?.('Waiting for package approval in VS Code…');
    if (!await host.reviewInstall(plan) || cancelled()) throw new GraphToolError('INSTALL_DECLINED', 'Dependency installation was not approved. No installation was started.');
    if (!host.trusted() || !await options.current()) throw new GraphToolError('STALE_TARGET', 'The model or selected environment changed after review.');
    options.progress?.('Downloading and installing the reviewed libraries…');
    const result = await run(['install', directory, plan.digest]);
    if (!result.ok || cancelled()) throw new GraphToolError(result.code ?? 'INSTALL_FAILED', result.message ?? 'Dependency installation did not complete. Some new packages may have been installed.');
  } finally { await dispose(); }
}

export interface DependencyReplacementHost {
  trusted(): boolean;
  reviewReplacement(plan: HelperResult): Promise<boolean>;
  /** A second, separate confirmation whenever the plan changes the Torch/CUDA build. */
  confirmTorch(plan: HelperResult): Promise<boolean>;
}

/** One line per reviewed change: `torch 2.7.1 → 2.0.0`, or `+ sympy 1.13.1` for an addition. */
export function replacementSummary(plan: HelperResult): string[] {
  return (plan.packages ?? []).map(item => item.previous ? `${item.name} ${item.previous} → ${item.version}` : `+ ${item.name} ${item.version}`);
}

/** Replace the project's pinned versions in the selected environment itself, after review.
 * Only the project's declarations are requested; the plan lists every change pip needs. */
export async function replaceDependencies(options: DependencyInstallOptions & { project: string }, host: DependencyReplacementHost): Promise<void> {
  if (!host.trusted()) throw new GraphToolError('WORKSPACE_UNTRUSTED', 'Trust the workspace before replacing model library versions.');
  const cancelled = () => options.signal.aborted;
  if (cancelled() || !await options.current()) throw new GraphToolError('STALE_TARGET', 'The model or environment changed. Start graph opening again.');
  await fs.mkdir(options.storage, { recursive: true });
  const { directory, dispose } = await ownedTemporaryDirectory(await fs.realpath(options.storage), 'model-replacement-');
  const run = (args: string[]) => (options.runner ?? runEnvironmentHelper)(options.managerPython ?? options.python,
    options.backendDir, options.managerPython ? ['--target-python', options.python, ...args] : args, cancelled);
  try {
    const plan = await run(['plan-replacement', directory, JSON.stringify(options.packages), options.project]);
    if (!plan.ok || !plan.digest || !plan.packages?.length) throw new GraphToolError(plan.code ?? 'DEPENDENCY_CONFLICT',
      plan.message ?? 'No replacement plan satisfies the project pins in this environment. Choose another environment.');
    if (cancelled() || !await options.current()) throw new GraphToolError('STALE_TARGET', 'The model or environment changed before review.');
    if (!await host.reviewReplacement(plan) || cancelled()) throw new GraphToolError('INSTALL_DECLINED', 'Replacing library versions was not approved. Nothing was changed.');
    if (plan.replacesTorch && (!await host.confirmTorch(plan) || cancelled())) {
      throw new GraphToolError('INSTALL_DECLINED', 'Replacing PyTorch was not approved. Nothing was changed.');
    }
    if (!host.trusted() || !await options.current()) throw new GraphToolError('STALE_TARGET', 'The model or selected environment changed after review.');
    const result = await run(['install', directory, plan.digest]);
    if (!result.ok || cancelled()) throw new GraphToolError(result.code ?? 'INSTALL_FAILED',
      result.message ?? 'Replacing library versions did not complete. Some versions may already have changed; check the environment again.');
  } finally { await dispose(); }
}
