/** Source navigation uses only the host's current execution record and saved identities. */
import * as path from 'node:path';
import { realpath, readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import * as vscode from 'vscode';
import type { ValidationController } from '@tensorviz/graph-host/validationController';
import type { LocalBrowserRevealResult } from './localBrowserSync';

type Evidence = ReturnType<ValidationController['sourceForNode']>;
export async function revealExecutionNode(current: () => Evidence): Promise<LocalBrowserRevealResult> {
  const refused: LocalBrowserRevealResult = { ok: false, errorCode: 'STALE_TARGET',
    detail: 'This source location is unavailable or no longer matches the saved execution.' };
  const evidence = current();
  if (!evidence) return refused;
  const { source, file } = evidence;
  if (!source.file || source.file.includes('\\') || source.file.split('/').some(part => !part || part === '.' || part === '..')
    || path.isAbsolute(source.file) || !Number.isSafeInteger(source.line) || source.line < 1
    || !Number.isSafeInteger(source.column) || source.column < 0) return refused;
  const stillCurrent = () => JSON.stringify(current()) === JSON.stringify(evidence);
  try {
    const root = await realpath(evidence.root);
    const resolved = await realpath(path.join(root, source.file));
    const relative = path.relative(root, resolved);
    if (relative.startsWith('..' + path.sep) || relative === '..' || path.isAbsolute(relative)) return refused;
    if (createHash('sha256').update(await readFile(resolved)).digest('hex') !== file.digest || !stillCurrent()) return refused;
    const document = await vscode.workspace.openTextDocument(vscode.Uri.file(resolved));
    if (document.isDirty || (file.documentVersion !== null && document.version !== file.documentVersion)
      || createHash('sha256').update(document.getText(), 'utf8').digest('hex') !== file.digest || !stillCurrent()) return refused;
    const editor = await vscode.window.showTextDocument(document, { viewColumn: vscode.ViewColumn.One, preserveFocus: false, preview: false });
    if (!stillCurrent()) return refused;
    const position = new vscode.Position(source.line - 1, source.column);
    editor.selection = new vscode.Selection(position, position);
    editor.revealRange(new vscode.Range(position, position), vscode.TextEditorRevealType.InCenterIfOutsideViewport);
    return { ok: true };
  } catch { return refused; }
}
