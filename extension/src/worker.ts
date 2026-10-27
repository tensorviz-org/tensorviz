/** VS Code policy adapter for the shared semantic capture worker. */
import * as vscode from 'vscode';
import { TraceWorkerCore } from '@tensorviz/graph-host/traceWorkerCore';
import type { ExecutionCacheSettings } from '@tensorviz/graph-host/executionCache';
import { executionDocumentStates } from './executionPlan';
export * from '@tensorviz/graph-host/traceWorkerCore';

/** Whether the editor's own Workspace Trust covers running this workspace's code.
 *
 * This used to be a tensorViz prompt of our own, asked once per workspace and remembered
 * in a memento, with a reset command. All three are deleted
 * (task.md D20): "do you trust the code in this folder enough to execute it" is exactly
 * the question VS Code's Workspace Trust dialog already asks, on first open, for every
 * extension at once. Asking it a second time in our own words taught users to click
 * through a dialog that matters, and it was the last prompt standing between opening a
 * folder and seeing measured shapes.
 *
 * `onDidGrantWorkspaceTrust` is how a workspace that starts restricted becomes traceable
 * without a reload — callers that cache the answer must re-read it there.
 */
export function workspaceTrusted(): boolean {
  return vscode.workspace.isTrusted;
}

interface PythonExtensionApi {
  environments: {
    getActiveEnvironmentPath(resource?: vscode.Uri): { id: string; path: string };
    resolveEnvironment(p: { id: string; path: string }): Promise<{ executable?: { uri?: vscode.Uri } } | undefined>;
  };
}

/** The interpreter the user's own project is configured with.
 *
 * Deliberately ms-python's answer and not the backend's interpreter (task.md D11):
 * the backend needs fastapi and uvicorn and may be the IDE's own managed runtime,
 * while the worker must be the environment the user's imports resolve in. They are
 * frequently not the same interpreter, and tracing in the wrong one produces a
 * graph of a model the user does not have.
 */
export async function userInterpreter(resource: vscode.Uri): Promise<string | undefined> {
  const extension = vscode.extensions.getExtension('ms-python.python');
  if (!extension) {
    return undefined;
  }
  try {
    const api = (await extension.activate()) as PythonExtensionApi;
    const active = api.environments.getActiveEnvironmentPath(resource);
    const resolved = await api.environments.resolveEnvironment(active);
    // resolveEnvironment matters: for some conda environments the active path is the
    // environment DIRECTORY, and spawning a directory fails with EACCES, which reads
    // as a permissions problem rather than a wrong path.
    return resolved?.executable?.uri?.fsPath ?? active.path;
  } catch {
    return undefined;
  }
}

export interface TraceWorkerOptions {
  timeoutMs?: number;
  interpreter?: (resource: vscode.Uri) => Promise<string | undefined>;
  /** Read `tensorviz.executionCache.*` on every run (default). Tests pass their own. */
  executionCache?: () => ExecutionCacheSettings | undefined;
}

/** `tensorviz.executionCache.*`, read per run so a change applies to the next Run. */
export function executionCacheSettings(): ExecutionCacheSettings {
  const config = vscode.workspace.getConfiguration('tensorviz');
  const mode = config.get<string>('executionCache.mode', 'off');
  return { mode: mode === 'shadow' || mode === 'on' ? mode : 'off', maxSizeMB: config.get<number>('executionCache.maxSizeMB', 96) };
}

export class TraceWorker extends TraceWorkerCore {
  constructor(backendDir: string, channel: vscode.OutputChannel, options: TraceWorkerOptions = {}) {
    super(backendDir, channel, {
      timeoutMs: options.timeoutMs,
      interpreter: file => (options.interpreter ?? userInterpreter)(vscode.Uri.file(file)),
      trusted: workspaceTrusted,
      documents: executionDocumentStates,
      executionCache: options.executionCache ?? executionCacheSettings,
    });
  }
}
