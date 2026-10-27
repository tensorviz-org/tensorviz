/** VS Code saved-buffer adapter for the shared execution planner. */
import * as path from 'node:path';
import * as vscode from 'vscode';
import { isHostFile } from './workspaceFiles';
import * as core from '@tensorviz/graph-host/executionPlanCore';
export * from '@tensorviz/graph-host/executionPlanCore';

export function executionDocumentStates(workspace: string): Record<string, { version: number; dirty: boolean }> {
  const documents: Record<string, { version: number; dirty: boolean }> = {};
  for (const doc of vscode.workspace.textDocuments) {
    if (!isHostFile(doc.uri)) continue;
    const relative = path.relative(workspace, doc.uri.fsPath).split(path.sep).join('/');
    if (relative && !relative.startsWith('../') && !path.isAbsolute(relative)) {
      documents[relative] = { version: doc.version, dirty: doc.isDirty };
    }
  }
  return documents;
}

type HostRequest = Omit<core.SnapshotRequest, 'documents'>;
export function resolveExecutionPlan(python: string, backendDir: string,
  request: HostRequest & Pick<Parameters<typeof core.resolveExecutionPlan>[2], 'execution' | 'job'>, signal: AbortSignal) {
  return core.resolveExecutionPlan(python, backendDir, { ...request, documents: executionDocumentStates(request.workspace) }, signal);
}
export function resolveCapturePreparation(python: string, backendDir: string,
  request: HostRequest & Pick<Parameters<typeof core.resolveCapturePreparation>[2], 'captureRequest'>, signal: AbortSignal) {
  return core.resolveCapturePreparation(python, backendDir, { ...request, documents: executionDocumentStates(request.workspace) }, signal);
}
export function resolveRetainedPlan(python: string, backendDir: string, workspace: string,
  plan: core.ResolvedExecutionPlan, signal: AbortSignal) {
  return core.resolveRetainedPlan(python, backendDir, workspace, plan, signal, executionDocumentStates(workspace));
}
