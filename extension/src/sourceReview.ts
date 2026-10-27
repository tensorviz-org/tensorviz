import { randomUUID } from 'node:crypto';
import * as vscode from 'vscode';
import { isHostFile } from './workspaceFiles';
import type { CodeArtifact } from './sourceChanges';
import { CandidateReview, candidateReviewText } from './candidateReview';

/** Read-only virtual documents; a proposal never creates a workspace file. */
export async function reviewSource(
  title: string,
  artifact: CodeArtifact,
  signal: AbortSignal,
  candidate?: CandidateReview,
  onShown?: (reveal: () => Promise<void>, decide: (approved: boolean) => void) => void,
  onDecision?: () => void,
): Promise<boolean> {
  if (signal.aborted) return false;
  const scheme = `tensorviz-review-${randomUUID()}`;
  const before = vscode.Uri.from({ scheme, path: `/before/${artifact.path}` });
  const after = vscode.Uri.from({ scheme, path: `/after/${artifact.path}` });
  const result = vscode.Uri.from({ scheme, path: '/candidate-validation.md' });
  const changed = new vscode.EventEmitter<vscode.Uri>();
  const provider = vscode.workspace.registerTextDocumentContentProvider(scheme, {
    onDidChange: changed.event,
    provideTextDocumentContent(uri) {
      if (candidate && uri.toString() === result.toString()) return candidateReviewText(candidate.snapshot());
      return uri.toString() === before.toString() ? artifact.before : artifact.after;
    },
  });
  let onAbort: (() => void) | undefined;
  let closing = false;
  let resultColumn: vscode.ViewColumn | undefined;
  let decide!: (approved: boolean) => void;
  let decided = false;
  const decision = new Promise<boolean>(resolve => { decide = approved => {
    if (decided) return;
    decided = true;
    onDecision?.();
    resolve(approved);
  }; });
  const controls = [
    vscode.commands.registerCommand('tensorviz.applySourceProposal', () => {
      if (vscode.window.activeTextEditor?.document.uri.scheme === scheme) decide(!signal.aborted);
    }),
    vscode.commands.registerCommand('tensorviz.discardSourceProposal', () => decide(false)),
    vscode.window.tabGroups.onDidChangeTabs(event => {
      if (event.closed.some(tab => tab.input instanceof vscode.TabInputTextDiff && tab.input.modified.scheme === scheme)) decide(false);
    }),
  ];
  if (candidate) {
    const active = () => !closing && vscode.window.activeTextEditor?.document.uri.scheme === scheme && !signal.aborted;
    const inspect = async () => {
      try {
        await candidate.refresh();
        if (!active()) return;
        const document = await vscode.workspace.openTextDocument(result);
        if (active()) {
          // A preview result could replace and close the review diff, which is
          // correctly treated as Discard. Keep the full diff open beside it.
          const editor = await vscode.window.showTextDocument(document,
            { preview: false, viewColumn: resultColumn ?? vscode.ViewColumn.Beside });
          resultColumn = editor.viewColumn;
        }
      } catch {
        if (active()) void vscode.window.showWarningMessage('The candidate result could not be opened. The source review is still available.');
      }
    };
    controls.push(
      candidate.onDidChange(value => {
        void vscode.commands.executeCommand('setContext', 'tensorviz.sourceCandidateRunning', value.runtime === 'running');
        changed.fire(result);
      }),
      vscode.workspace.onDidChangeTextDocument(event => { if (isHostFile(event.document.uri)) changed.fire(result); }),
      vscode.commands.registerCommand('tensorviz.runSourceCandidate', async () => {
        if (!active()) return;
        await candidate.run();
        if (active()) await inspect();
      }),
      vscode.commands.registerCommand('tensorviz.cancelSourceCandidate', () => { if (active()) candidate.cancel(); }),
      vscode.commands.registerCommand('tensorviz.inspectSourceCandidate', async () => {
        if (!active()) return;
        await inspect();
      }),
    );
  }
  try {
    if (candidate) await vscode.commands.executeCommand('setContext', 'tensorviz.sourceCandidateAvailable', true);
    await vscode.commands.executeCommand('setContext', 'tensorviz.sourceReviewActive', true);
    await vscode.commands.executeCommand('vscode.diff', before, after, title, { preview: !candidate });
    if (signal.aborted) return false;
    try { onShown?.(async () => {
      const group = vscode.window.tabGroups.all.find(group => group.tabs.some(tab =>
        tab.input instanceof vscode.TabInputTextDiff && tab.input.modified.toString() === after.toString()));
      if (group && !closing && !signal.aborted) await vscode.commands.executeCommand('vscode.diff', before, after, title,
        { preview: !candidate, viewColumn: group.viewColumn });
    }, approved => { if (!closing && !signal.aborted) decide(approved); }); } catch { /* An optional observer cannot interrupt source review. */ }
    const aborted = new Promise<boolean>(resolve => {
      onAbort = () => resolve(false);
      signal.addEventListener('abort', onAbort, { once: true });
    });
    void vscode.window.showInformationMessage(`${title}. Use Apply source change or Discard proposal in this diff's editor toolbar.`);
    const approved = await Promise.race([decision, aborted]);
    return !signal.aborted && approved;
  } finally {
    closing = true;
    candidate?.cancel();
    if (onAbort) signal.removeEventListener('abort', onAbort);
    for (const control of controls) control.dispose();
    await vscode.commands.executeCommand('setContext', 'tensorviz.sourceReviewActive', false);
    if (candidate) {
      await vscode.commands.executeCommand('setContext', 'tensorviz.sourceCandidateAvailable', false);
      await vscode.commands.executeCommand('setContext', 'tensorviz.sourceCandidateRunning', false);
    }
    const tabs = vscode.window.tabGroups.all.flatMap(group => group.tabs).filter(tab => {
      const input = tab.input;
      return input instanceof vscode.TabInputTextDiff && input.modified.scheme === scheme
        || !!candidate && input instanceof vscode.TabInputText && input.uri.scheme === scheme;
    });
    if (tabs.length) await vscode.window.tabGroups.close(tabs, true);
    provider.dispose();
    changed.dispose();
  }
}
