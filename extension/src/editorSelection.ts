import * as vscode from 'vscode';
import { isHostFile } from './workspaceFiles';

/** Keep the user's Python file available while a webview or palette owns focus. */
export function selectedTextUri(): string | undefined {
  const pythonUri = (uri: vscode.Uri | undefined): string | undefined =>
    uri && isHostFile(uri) && /\.pyw?$/i.test(uri.fsPath) ? uri.toString() : undefined;
  const active = pythonUri(vscode.window.activeTextEditor?.document.uri);
  if (active) return active;
  for (const editor of vscode.window.visibleTextEditors) {
    const uri = pythonUri(editor.document.uri);
    if (uri) return uri;
  }
  const groups = vscode.window.tabGroups;
  for (const group of [groups.activeTabGroup, ...groups.all]) {
    const input = group.activeTab?.input;
    const uri = input instanceof vscode.TabInputText ? pythonUri(input.uri) : undefined;
    if (uri) return uri;
  }
  return undefined;
}
