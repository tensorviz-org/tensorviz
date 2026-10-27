/** VS Code buffers adapt the shared immutable review snapshots. */
import * as vscode from 'vscode';
import * as path from 'node:path';
import { workingSnapshotCore, type ReviewSnapshot, type ReviewSession as SharedReviewSession } from '@tensorviz/graph-host/reviewSnapshotsCore';
export * from '@tensorviz/graph-host/reviewSnapshotsCore';

export function workingSnapshot(folder: string): Promise<ReviewSnapshot> {
  return workingSnapshotCore(folder, vscode.workspace.textDocuments.filter(doc => doc.uri.scheme === 'file'
    && !path.relative(folder, doc.uri.fsPath).startsWith('..')).map(doc => ({ file: doc.uri.fsPath,
      isDirty: doc.isDirty, getText: () => doc.getText() })));
}

/** Native editor locations are private to the VS Code review adapter. */
export interface ReviewSession extends SharedReviewSession {
  editorDiff?: { original: vscode.Uri; modified: vscode.Uri; beforeFile: string; afterFile: string };
  refresh(): Promise<ReviewSession>;
  locate?(uri: vscode.Uri): { file: string; side: 'before' | 'after' } | undefined;
}
