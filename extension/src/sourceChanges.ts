import { createHash } from 'node:crypto';
import * as vscode from 'vscode';
import { toRange } from './apply';

/** One replacement inside one file, as `AgentEditCodeOperation` carries it. */
export interface CodeEdit {
  op?: 'edit_code';
  op_id: string;
  path: string;
  range: { start: number[]; end: number[] };
  before_digest: string;
  after: string;
}

/** The whole reviewed result for one file: what it was, and what it becomes. */
export interface CodeArtifact {
  path: string;
  before: string;
  before_digest: string;
  after: string;
  after_digest: string;
  // Optional because the contract that crosses the bridge has them optional.
  // `write` refuses an artifact with no edits rather than writing nothing and
  // calling it a success.
  diff?: string;
  edits?: CodeEdit[];
}

/** What the host observed and did, as `commit/confirm` wants it. */
export interface ApplyReport {
  outcome: 'applied' | 'failed' | 'timed_out';
  observedDigests: Record<string, string>;
  reason?: string;
}

export function sha256(text: string): string {
  return createHash('sha256').update(text, 'utf8').digest('hex');
}

/** Absolute uri for a workspace-relative artifact path. */
export function artifactUri(root: vscode.Uri, relativePath: string): vscode.Uri {
  return vscode.Uri.joinPath(root, ...relativePath.split('/').filter(Boolean));
}

/** Source writes shared by deterministic forms and external tool proposals. */
export class SourceWriter {
  constructor(
    private readonly root: vscode.Uri,
    private readonly channel: Pick<vscode.OutputChannel, 'appendLine'>,
  ) {}

  /** Apply reviewed ranges to the matching live buffers as one undoable edit. */
  async write(artifacts: CodeArtifact[], stillCurrent?: () => boolean): Promise<ApplyReport> {
    const observedDigests: Record<string, string> = {};
    const documents: Array<{ artifact: CodeArtifact; document: vscode.TextDocument }> = [];

    for (const artifact of artifacts) {
      let document: vscode.TextDocument;
      try {
        document = await vscode.workspace.openTextDocument(
          artifactUri(this.root, artifact.path),
        );
      } catch (error) {
        return {
          outcome: 'failed',
          observedDigests,
          reason: `${artifact.path} could not be opened: ${String(error)}`,
        };
      }
      const observed = sha256(document.getText());
      observedDigests[artifact.path] = observed;
      if (observed !== artifact.before_digest) {
        this.channel.appendLine(
          `[source] ${artifact.path}: digest ${observed.slice(0, 12)} is not the reviewed ${artifact.before_digest.slice(0, 12)}`,
        );
        return {
          outcome: 'failed',
          observedDigests,
          reason: `${artifact.path} changed after the change was reviewed`,
        };
      }
      documents.push({ artifact, document });
    }

    // One WorkspaceEdit for every file, so the whole change is one undo.
    const edit = new vscode.WorkspaceEdit();
    for (const { artifact, document } of documents) {
      // An artifact with no edits describes no change. Writing nothing and
      // reporting success would report a successful change over a file that
      // was never touched, so it is refused instead.
      if (!artifact.edits || artifact.edits.length === 0) {
        return {
          outcome: 'failed',
          observedDigests,
          reason: `${artifact.path} arrived with no edits to apply`,
        };
      }
      for (const change of artifact.edits) {
        // A range that is not a line/column pair cannot be placed in the file.
        // Guessing one would write the patch somewhere, which is worse than not
        // writing it: the file would compile and mean something else.
        if (change.range.start.length < 2 || change.range.end.length < 2) {
          return {
            outcome: 'failed',
            observedDigests,
            reason: `${artifact.path}: an edit arrived without a line and column`,
          };
        }
        edit.replace(
          document.uri,
          toRange({
            start: [change.range.start[0], change.range.start[1]],
            end: [change.range.end[0], change.range.end[1]],
          }),
          change.after,
        );
      }
    }
    // Counted before the edit is applied, while `documents` still describes what is about
    // to change: after applyEdit the ranges have moved.
    const receipt = documents.map(({ artifact }) => ({
      path: artifact.path,
      lines: (artifact.edits ?? []).reduce(
        (total, change) => total + change.after.split('\n').length,
        0,
      ),
      line: (artifact.edits ?? [])[0]?.range.start[0] ?? 0,
    }));

    // Callers that review source also guard document versions, dependencies,
    // target changes and cancellation after the asynchronous document reads.
    if (stillCurrent && !stillCurrent()) {
      return { outcome: 'failed', observedDigests, reason: 'the reviewed source is no longer current' };
    }
    const ok = await vscode.workspace.applyEdit(edit);
    if (!ok) {
      return {
        outcome: 'failed',
        observedDigests,
        reason: 'the editor rejected the workspace edit',
      };
    }
    // A receipt, not a log line (ide-plan.html §15 item 8). A write the user did not watch
    // happen is a write they cannot check, and "your file changed" with no location is the
    // thing that makes an agent feel like it is operating on your repository behind your
    // back. Naming the file and the line makes it a fact they can go and look at.
    void vscode.window.showInformationMessage(
      receipt.length === 1
        ? `Wrote ${receipt[0].lines} line${receipt[0].lines === 1 ? '' : 's'} to ${receipt[0].path}:${receipt[0].line}`
        : `Wrote ${receipt.reduce((total, entry) => total + entry.lines, 0)} lines across ${receipt.length} files`,
    );
    this.channel.appendLine(
      `[source] wrote ${artifacts.length} file(s): ${artifacts.map((item) => item.path).join(', ')}`,
    );
    return { outcome: 'applied', observedDigests };
  }

}
