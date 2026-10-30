/** VS Code buffers adapt the shared immutable review snapshots. */
import * as vscode from 'vscode';
import * as path from 'node:path';
import { realpathSync } from 'node:fs';
import { workingSnapshotCore, type ReviewSnapshot, type ReviewSession as SharedReviewSession } from '@tensorviz/graph-host/reviewSnapshotsCore';
export * from '@tensorviz/graph-host/reviewSnapshotsCore';

/** Keep missing/deleted descendants while resolving existing directory aliases. */
export function reviewDirectory(directory: string): string {
  try { return realpathSync.native(directory); }
  catch (error) {
    const parent = path.dirname(directory);
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT' || parent === directory) throw error;
    return path.join(reviewDirectory(parent), path.basename(directory));
  }
}
/** Preserve the leaf so snapshot admission can still reject source-file symlinks. */
export function reviewFilePath(file: string): string {
  return path.join(reviewDirectory(path.dirname(file)), path.basename(file));
}

export function workingSnapshot(folder: string): Promise<ReviewSnapshot> {
  folder = reviewDirectory(folder);
  const open = new Map<string, { file: string; isDirty: boolean; getText(): string }>();
  for (const doc of vscode.workspace.textDocuments.filter(doc => doc.uri.scheme === 'file')) {
    const file = reviewFilePath(doc.uri.fsPath), relative = path.relative(folder, file);
    if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) continue;
    const previous = open.get(file);
    if (previous && previous.getText() !== doc.getText()) throw new Error('Open buffers for the same source file disagree. Close one alias before comparing changes.');
    if (!previous || doc.isDirty) open.set(file, { file, isDirty: doc.isDirty, getText: () => doc.getText() });
  }
  return workingSnapshotCore(folder, [...open.values()]);
}

/** Native editor locations are private to the VS Code review adapter. */
export interface ReviewSession extends SharedReviewSession {
  editorDiff?: { original: vscode.Uri; modified: vscode.Uri; beforeFile: string; afterFile: string };
  refresh(): Promise<ReviewSession>;
  locate?(uri: vscode.Uri): { file: string; side: 'before' | 'after' } | undefined;
}
