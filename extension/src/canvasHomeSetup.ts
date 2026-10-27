import * as vscode from 'vscode';
import type { RuntimeStatus } from './runtime';

export interface CanvasHomeSetup {
  phase: 'checking' | 'ready' | 'needs_setup' | 'untrusted' | 'unavailable';
  detail: string;
}

/** Home checks only tensorViz's owned runtime. Project Python and capture
 * dependencies belong to explicit model opening, never welcome-screen readiness. */
export async function checkCanvasHomeSetup(options: {
  signal: AbortSignal; runtime(): Promise<RuntimeStatus>;
}): Promise<CanvasHomeSetup> {
  options.signal.throwIfAborted();
  if (!vscode.workspace.isTrusted) return { phase: 'untrusted', detail: 'Trust this workspace to check the tensorViz runtime.' };
  const runtime = await options.runtime();
  options.signal.throwIfAborted();
  if (runtime.state === 'ready') return { phase: 'ready', detail: 'Choose a model or open a pull request. Model dependencies are checked after you select a model.' };
  if (runtime.state === 'unavailable') return { phase: 'unavailable', detail: 'The tensorViz runtime is unavailable on this host. Inspect the log for setup details.' };
  return { phase: 'needs_setup', detail: runtime.state === 'broken'
    ? 'The tensorViz runtime needs repair. Set up tensorViz again or inspect the log.'
    : 'Set up tensorViz’s own Python runtime once to open graphs.' };
}
