import { promises as fs } from 'node:fs';
import path from 'node:path';
import { installDependencies, type DependencyInstallOptions } from '@tensorviz/graph-host/dependencyInstallCore';
import { ApprovalGate } from './approval';
import { withResourceLock } from './locks';

export async function installStandaloneDependencies(options: DependencyInstallOptions & {
  workspaceRoot: string; prefix: string; approvals: ApprovalGate;
}): Promise<void> {
  const resource = await fs.realpath(options.prefix);
  await withResourceLock(path.join(resource, '.tensorviz-environment'), options.signal, async check => {
    await installDependencies({ ...options, current: async () => { check(); return options.current(); } }, {
      trusted: () => !options.signal.aborted, isolated: true, parentPipe: true,
      reviewSource: async () => {
        await options.approvals.approve({ operation: 'build_dependencies', workspaceRoot: options.workspaceRoot,
          interpreter: options.python, detail: 'Compatible wheels were not found. Resolve and build these requirements from source, then install verified wheels under your TensorViz permission. Package build code can access your machine; the build environment is not a security sandbox.',
          facts: { packages: options.packages, prefix: resource } }, options.signal);
        return true;
      },
      reviewInstall: async plan => {
        await options.approvals.approve({ operation: 'install_dependencies', workspaceRoot: options.workspaceRoot,
          interpreter: options.python, detail: 'Install exactly these hash-verified wheels in the selected capture environment. Existing package versions, including Torch/CUDA, are frozen and will not be replaced. Cancellation can leave some new packages installed; inspect this environment before retrying.',
          facts: { digest: plan.digest, packages: plan.packages, builtFromSource: plan.builtFromSource ?? [], prefix: resource } }, options.signal);
        return true;
      },
    });
  });
}
