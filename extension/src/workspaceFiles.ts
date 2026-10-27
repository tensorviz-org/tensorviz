import type * as vscode from 'vscode';

/** In a workspace extension host, VS Code transforms the owning remote files to
 * file: and desktop-local files to vscode-local:. Never compare paths alone:
 * a virtual document or another authority can have the same fsPath. */
export function isHostFile(uri: vscode.Uri): boolean {
  return uri.scheme === 'file' && !uri.query && !uri.fragment;
}

export function matchesHostFile(uri: vscode.Uri, fsPath: string): boolean {
  return isHostFile(uri) && uri.fsPath === fsPath;
}
