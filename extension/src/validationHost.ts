import { isHostFile, matchesHostFile } from './workspaceFiles';
/** VS Code adapter for the single execution lifecycle. No browser paths enter here. */
import { lstat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import * as path from 'node:path';
import * as vscode from 'vscode';
import { ValidationController, ValidationPreparation, ValidationTarget } from '@tensorviz/graph-host/validationController';
import { ValidationSetup } from './validationSetup';
import { ValidationWorker } from './validationWorker';
import { savedEnvironment } from './projectEnvironment';
import type { ValidationExecutionUpdate } from '@tensorviz/contracts/protocol';
import { readHostSnapshot } from '@tensorviz/graph-host/hostSnapshot';
export { readHostSnapshot } from '@tensorviz/graph-host/hostSnapshot';

/** Mirror the interpreter resolver's inputs, scoped to the run's workspace.
 * Python's active environment is obtained from its API, not Python settings;
 * executable/package fingerprints remain checked by every saved preparation. */
export function watchValidationEnvironmentConfiguration(context: Pick<vscode.ExtensionContext, 'workspaceState' | 'globalState'>,
  controller: ValidationController, changed: () => void): vscode.Disposable {
  return vscode.workspace.onDidChangeConfiguration(event => {
    const selected = savedEnvironment(context);
    const resource = controller.root ? vscode.Uri.file(controller.root) : undefined;
    if ((!selected && event.affectsConfiguration('tensorviz.pythonPath', resource))
      || (selected?.kind === 'managed' && event.affectsConfiguration('tensorviz.repoPath'))) changed();
  });
}

/** Missing config needs no interpreter. Unreadable paths and dangling symlinks are not absence. */
export async function hasValidationConfig(root: string): Promise<boolean> {
  if (vscode.workspace.textDocuments.some(doc => doc.isDirty && matchesHostFile(doc.uri, path.join(root, 'tensorviz.toml')))) return true;
  try { await lstat(path.join(root, 'tensorviz.toml')); return true; }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return false;
    throw error;
  }
}

export async function readValidationSnapshot(python: string, backendDir: string, request: object, signal: AbortSignal): Promise<ValidationPreparation> {
  const result = await readHostSnapshot(python, backendDir, request, signal) as ValidationPreparation;
  if (!result.resolution || !Array.isArray(result.profiles) || !result.guards) throw new Error('snapshot_failed');
  return result;
}

export function createValidationHost(backendDir: string, interpreter: (root: string) => Promise<string | undefined>,
  publish: (update: ValidationExecutionUpdate) => void, environmentRevision: () => number = () => 0, setup?: ValidationSetup): ValidationController {
  const worker = new ValidationWorker(backendDir);
  return new ValidationController({
    trusted: () => vscode.workspace.isTrusted,
    prepare: async (target, runId, generation, profile, signal) => {
      if (!ownsValidationRoot(target)) throw new Error('workspace_removed');
      const python = await interpreter(target.root);
      if (!python) throw new Error('environment_required');
      const documents: Record<string, { version: number; dirty: boolean }> = {};
      for (const doc of vscode.workspace.textDocuments) {
        if (!isHostFile(doc.uri)) continue;
        const relative = path.relative(target.root, doc.uri.fsPath).split(path.sep).join('/');
        if (relative && !relative.startsWith('../') && !path.isAbsolute(relative)) {
          documents[relative] = { version: doc.version, dirty: doc.isDirty };
        }
      }
      const preparation = await readValidationSnapshot(python, backendDir, { root: target.root, documents, executable: python,
        environmentRevision: environmentRevision(), session: target.session, workspaceId: createHash('sha256').update(target.root).digest('hex'),
        runId, generation, profile }, signal);
      if (preparation.resolution.status === 'resolved') setup?.clear(target.root);
      return preparation;
    },
    setup: (target, requirements) => setup?.show(target, requirements) ?? Promise.resolve(),
    choose: async profiles => vscode.window.showQuickPick(profiles, { title: 'Run a saved tensorviz.toml profile', ignoreFocusOut: true }),
    approve: async spec => await vscode.window.showWarningMessage(
      `Run profile ${spec.identity.profile} with the selected Python? This executes project code (${spec.profile.kind}).`,
      { modal: true, detail: `Python: ${spec.interpreter.executable}\nProfile: ${JSON.stringify(spec.profile)}\nDependency scope: ${spec.dependencyScope}` }, 'Run') === 'Run',
    run: (spec, root, receive, beforeLaunch) => worker.run(spec, root, true, receive, beforeLaunch),
    cancel: () => worker.cancel(),
    publish,
  });
}

/** Host-owned root must still belong to this window. */
export function ownsValidationRoot(target: ValidationTarget): boolean {
  return !!vscode.workspace.workspaceFolders?.some(folder => folder.uri.fsPath === target.root);
}
