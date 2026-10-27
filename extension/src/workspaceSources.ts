/** VS Code source adapter; admission, budgets and digests stay shared. */
import * as path from 'node:path';
import * as vscode from 'vscode';
import { isHostFile, matchesHostFile } from './workspaceFiles';
import { ENUMERATION_ASK, type SourceIO } from '@tensorviz/graph-host/workspaceSourceCore';
export * from '@tensorviz/graph-host/workspaceSourceCore';

/** The production `SourceIO`: VS Code's own enumeration (its file-search
 *  excludes apply) over the target's folder, with the dev-environment noise a
 *  class graph can never mean hard-excluded. */
export function vscodeSourceIO(folder: string): SourceIO {
  const root = vscode.Uri.file(folder);
  const fsPathOf = (relativePath: string) => path.join(folder, relativePath);
  return {
    async list() {
      const found = await vscode.workspace.findFiles(
        new vscode.RelativePattern(root, '**/*.py'),
        '{**/__pycache__/**,**/.git/**,**/.venv/**,**/venv/**,**/node_modules/**,**/.tox/**}',
        ENUMERATION_ASK,
      );
      return found.filter(isHostFile).map((uri) => path.relative(folder, uri.fsPath).split(path.sep).join('/'));
    },
    async read(relativePath) {
      return Buffer.from(
        await vscode.workspace.fs.readFile(vscode.Uri.file(fsPathOf(relativePath))),
      ).toString('utf8');
    },
    openDocument(relativePath) {
      const fsPath = fsPathOf(relativePath);
      return vscode.workspace.textDocuments.find(
        (document) => matchesHostFile(document.uri, fsPath),
      );
    },
  };
}
