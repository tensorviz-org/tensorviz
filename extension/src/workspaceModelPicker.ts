import * as vscode from 'vscode';
import type { WorkspaceClass } from './workspaceClassIndex';

/** Explicit model selection when Canvas has no selected Python file. Discovery never imports models. */
export async function pickWorkspaceModel(folders: readonly vscode.WorkspaceFolder[],
  discover: (root: string) => Promise<WorkspaceClass[]>): Promise<{ uri: string; cls?: string } | undefined> {
  if (folders.length) {
    const models = (await Promise.all(folders.map(async folder => (await discover(folder.uri.fsPath)).map(model => ({
      label: model.name,
      description: folders.length > 1 ? `${folder.name}/${model.file}` : model.file,
      detail: `Class declared on line ${model.line}`,
      target: { uri: vscode.Uri.joinPath(folder.uri, model.file).toString(), cls: model.name },
    }))))).flat();
    const chosen = await vscode.window.showQuickPick([...models, {
      label: 'Choose a Python file…', description: 'Browse for a model not listed here', detail: '', target: undefined,
    }], { title: 'Open as Graph', placeHolder: models.length ? 'Search models by class, file or folder' : 'No models discovered. Choose a Python file to inspect',
      matchOnDescription: true, matchOnDetail: true });
    if (!chosen) return undefined;
    if (chosen.target) return chosen.target;
  }
  const files = await vscode.window.showOpenDialog({ title: 'Choose a Python model file', openLabel: 'Select model file',
    canSelectMany: false, canSelectFolders: false, filters: { Python: ['py', 'pyw'] }, defaultUri: folders[0]?.uri });
  return files?.[0] ? { uri: files[0].toString() } : undefined;
}
