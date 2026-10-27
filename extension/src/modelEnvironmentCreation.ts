import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import * as vscode from 'vscode';
import { isModelEnvironmentName } from '@tensorviz/contracts/protocol';
import { GraphToolError } from '@tensorviz/graph-host/mcpGraph';
import { ownedTemporaryDirectory } from '@tensorviz/graph-host/ownedTemporary';
import { runEnvironmentHelper, type EnvironmentRunner } from '@tensorviz/graph-host/projectEnvironmentCore';

export interface ModelEnvironmentCreationOptions {
  managerPython: string; backendDir: string; storage: string; workspace: string; file: string;
  name?: string;
  signal: AbortSignal; current(): Promise<boolean>; progress(detail: string): void;
  selectEnvironment(python: string): Promise<void>; runner?: EnvironmentRunner;
}

/** A model owns this new venv. The service runtime and existing project Python are never install targets. */
export async function createModelEnvironment(options: ModelEnvironmentCreationOptions): Promise<void> {
  const check = async () => {
    options.signal.throwIfAborted();
    if (!vscode.workspace.isTrusted || !await options.current()) {
      throw new GraphToolError('STALE_TARGET', 'The workspace or model changed. Open the current saved model again.');
    }
  };
  await check();
  const name = options.name ?? 'environment';
  if (!isModelEnvironmentName(name)) throw new GraphToolError('INVALID_ENVIRONMENT_NAME', 'Use an environment name of 1–48 letters, numbers, hyphens or underscores.');
  const storage = path.join(options.storage, 'model-environments');
  await fs.mkdir(storage, { recursive: true });
  const owned = await ownedTemporaryDirectory(await fs.realpath(storage), 'model-');
  let selected = false;
  const destination = path.join(owned.directory, name);
  const python = path.join(destination, process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
  let unavailableWheel: string | undefined;
  const run = async (args: string[], target = true) => {
    await check();
    const result = await (options.runner ?? runEnvironmentHelper)(options.managerPython, options.backendDir,
      [...(target ? ['--target-python', python] : []), ...args], () => options.signal.aborted, true, true);
    await check();
    if (!result.ok && result.code === 'NO_COMPATIBLE_WHEEL') unavailableWheel = result.requirement;
    if (!result.ok) throw new GraphToolError(result.code ?? 'MODEL_ENVIRONMENT_FAILED',
      (result.requirement ? `${result.requirement}: ` : '') + (result.message ?? 'Model environment setup did not complete. Retry or choose another environment.'));
    return result;
  };
  try {
    options.progress(`Creating “${name}”…`);
    await run(['create-environment', destination], false);
    options.progress('Reading this model’s required libraries and project declarations…');
    const recipe = await run(['inspect-model-setup', options.workspace, options.file]);
    if (!recipe.manifestDigest) throw new GraphToolError('MODEL_ENVIRONMENT_FAILED', 'The project requirements could not be inspected. Retry environment setup.');
    const reviewSource = async () => {
      await check();
      const approved = await vscode.window.showWarningMessage('Build model libraries from source?', { modal: true,
        detail: `${recipe.sourcePackages?.length ? recipe.sourcePackages.join('\n') : unavailableWheel ?? 'Required model libraries'}\n\nBuild code can access files and the network. Libraries will be built for “${name}”; your current Python stays unchanged.` }, 'Build libraries');
      await check();
      if (approved !== 'Build libraries') throw new GraphToolError('SOURCE_BUILD_DECLINED', 'Source builds were not approved. No model environment was selected.');
    };
    let sourceApproved = !!recipe.sourcePackages?.length;
    if (sourceApproved) await reviewSource();
    options.progress(sourceApproved ? 'Building pinned model libraries and resolving their downloads…' : 'Resolving and downloading compatible model libraries…');
    let plan;
    try { plan = await run([sourceApproved ? 'plan-model-environment-source' : 'plan-model-environment', owned.directory, options.workspace, options.file, recipe.manifestDigest, name]); }
    catch (error) {
      if (sourceApproved || !(error instanceof GraphToolError) || error.code !== 'NO_COMPATIBLE_WHEEL') throw error;
      if (unavailableWheel) {
        options.progress(`Checking availability of ${unavailableWheel}…`);
        await run(['check-package-availability', unavailableWheel]);
      }
      await reviewSource(); sourceApproved = true;
      options.progress('Building model libraries and resolving their downloads…');
      plan = await run(['plan-model-environment-source', owned.directory, options.workspace, options.file, recipe.manifestDigest, name]);
    }
    if (!plan.digest || !plan.packages?.length) throw new GraphToolError('MODEL_ENVIRONMENT_FAILED', 'No compatible model-library installation was resolved.');
    options.progress('Ready to install. Review the package list in VS Code…');
    const approved = await vscode.window.showInformationMessage(`Install ${plan.packages.length} libraries in “${name}”?`, { modal: true,
      detail: `${plan.packages.map(item => `${item.name} ${item.version}`).join('\n')}\n\nIncludes PyTorch; downloads may be large. Your current Python stays unchanged. Installation continues opening this model.` }, 'Install and continue');
    await check();
    if (approved !== 'Install and continue') throw new GraphToolError('INSTALL_DECLINED', 'Installation was not approved. No model environment was selected.');
    options.progress('Downloading and installing the reviewed libraries…');
    await run(['install', owned.directory, plan.digest]);
    options.progress('Checking capture libraries and the model import…');
    await run(['probe-capture']);
    await run(['probe-model', options.workspace, options.file]);
    await check();
    await owned.retain();
    await options.selectEnvironment(python);
    selected = true;
    options.progress('Model environment ready. Continuing with this model…');
  } finally { if (!selected) await owned.dispose(); }
}
