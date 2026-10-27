import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import * as vscode from 'vscode';
import { GraphToolError } from '@tensorviz/graph-host/mcpGraph';
import { ownedTemporaryDirectory } from '@tensorviz/graph-host/ownedTemporary';
import { runEnvironmentHelper, type HelperResult } from '@tensorviz/graph-host/projectEnvironmentCore';
import type { DependencyInstallOptions } from '@tensorviz/graph-host/dependencyInstallCore';

export interface ModelCompatibilityOptions extends Omit<DependencyInstallOptions, 'packages'> {
  workspace: string;
  file: string;
  selectEnvironment(python: string): Promise<void>;
}

/** Try tested, project-compatible versions in a copy. Only a successful model
 * import can promote that environment; the original remains untouched. */
export async function repairModelCompatibility(options: ModelCompatibilityOptions, packages: string[]): Promise<boolean> {
  const check = async () => {
    if (options.signal.aborted || !vscode.workspace.isTrusted || !await options.current()) {
      throw new GraphToolError('STALE_TARGET', 'The model or selected environment changed during compatibility repair.');
    }
  };
  await check();
  const storage = path.join(options.storage, 'model-environments');
  await fs.mkdir(storage, { recursive: true });
  const owned = await ownedTemporaryDirectory(await fs.realpath(storage), 'compatible-');
  let selected = false;
  const run = async (python: string, args: string[]): Promise<HelperResult> => {
    await check();
    const result = await (options.runner ?? runEnvironmentHelper)(options.managerPython ?? python, options.backendDir,
      options.managerPython ? ['--target-python', python, ...args] : args, () => options.signal.aborted);
    await check();
    if (!result.ok) throw new GraphToolError(result.code ?? 'COMPATIBILITY_FAILED', result.message ?? 'The compatible environment could not be prepared. The original environment was preserved.');
    return result;
  };
  try {
    const cloned = await run(options.python, ['clone-compatibility', path.join(owned.directory, 'environment'), JSON.stringify(packages)]);
    const python = cloned.interpreter;
    if (!python || !path.resolve(python).startsWith(path.join(owned.directory, 'environment') + path.sep)) {
      throw new GraphToolError('COMPATIBILITY_FAILED', 'The compatible interpreter was not created in its owned directory.');
    }
    const plan = await run(python, ['plan-compatibility', owned.directory, JSON.stringify(packages), options.workspace]);
    if (!plan.digest || !plan.packages?.length) throw new GraphToolError('COMPATIBILITY_FAILED', 'No compatible version change was resolved.');
    const approved = await vscode.window.showInformationMessage('Use compatible model library versions?', { modal: true,
      detail: `${plan.packages.map(item => `${item.name} ${item.version}`).join('\n')}\n\n` +
        `TensorViz will install these versions in a separate environment for this workspace, then test the model import before using it. Project requirements are preserved. Your existing environment and Torch/CUDA remain unchanged.\n\n${python}` },
    'Install compatible versions', 'Cancel');
    if (approved !== 'Install compatible versions') throw new GraphToolError('INSTALL_DECLINED', 'Compatibility repair was not approved.');
    await run(python, ['install', owned.directory, plan.digest]);
    await run(python, ['probe-model', options.workspace, options.file]);
    await check();
    await owned.retain();
    await options.selectEnvironment(python);
    selected = true;
    return true;
  } finally {
    if (!selected) await owned.dispose();
  }
}
