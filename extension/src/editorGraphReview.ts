import * as vscode from 'vscode';
import * as path from 'node:path';
import { createHash } from 'node:crypto';
import { reviewGit } from '@tensorviz/graph-host/gitReviewSources';
import { gitSnapshot, indexSnapshot, ReviewSession, ReviewSnapshot, workingSnapshot } from './reviewSnapshots';
import { githubRepository, GitHubRead, loadPullRequest, parsePullRequestUrl } from './githubReview';
import { sha256Text } from './workspaceSources';

function query(uri: vscode.Uri): Record<string, unknown> {
  try { return JSON.parse(uri.query || '{}'); } catch { return {}; }
}
function filename(uri: vscode.Uri): string {
  const value = query(uri);
  return typeof value.path === 'string' ? value.path : uri.fsPath;
}
export async function workspaceGitFolder(uri?: vscode.Uri): Promise<string | undefined> {
  const selected = uri && vscode.workspace.getWorkspaceFolder(vscode.Uri.file(filename(uri)));
  const folder = selected ?? (vscode.workspace.workspaceFolders?.length === 1 ? vscode.workspace.workspaceFolders[0] : undefined);
  const directory = folder?.uri.fsPath ?? (uri && path.isAbsolute(filename(uri)) ? path.dirname(filename(uri)) : undefined);
  if (!directory) return undefined;
  try { return (await reviewGit(directory, ['rev-parse', '--show-toplevel'])).trim(); } catch { return undefined; }
}
export async function repositoryForFolder(folder: string, remote = 'origin'): Promise<string | undefined> {
  if (!/^[\w.-]+$/.test(remote)) return undefined;
  try { return githubRepository(await reviewGit(folder, ['config', '--get', `remote.${remote}.url`])); } catch { return undefined; }
}

/** SCM actions identify their side explicitly; an unrelated active diff is never consulted. */
export async function reviewFileChanges(uri: vscode.Uri, kind: 'working' | 'staged' | 'head'): Promise<ReviewSession> {
  const folder = await workspaceGitFolder(uri);
  if (!folder) throw new Error('Open a Python file in a Git repository to compare changes.');
  const file = path.relative(folder, filename(uri)).split(path.sep).join('/');
  const before = kind === 'working' ? await indexSnapshot(folder) : await gitSnapshot(folder, 'HEAD');
  const after = kind === 'staged' ? await indexSnapshot(folder) : await workingSnapshot(folder);
  const active = vscode.window.activeTextEditor;
  return {
    title: kind === 'staged' ? 'Staged changes' : kind === 'working' ? 'Unstaged changes' : 'Changes since HEAD',
    folder, before, after, preferredFile: file,
    preferredLocation: { file, side: 'after', line: active?.document.uri.toString() === uri.toString() ? active.selection.active.line + 1 : 1 },
    refresh: () => reviewFileChanges(uri, kind),
    locate: document => {
      const relative = path.relative(folder, filename(document)).split(path.sep).join('/');
      return after.working && ['file', 'vscode-remote'].includes(document.scheme) && after.files.has(relative)
        ? { file: relative, side: 'after' } : undefined;
    },
  };
}

/** Capture the tab before showing Canvas, which can change the active editor. */
export async function reviewEditorDiff(input: vscode.TabInputTextDiff, read: GitHubRead, cache: string): Promise<ReviewSession> {
  const active = vscode.window.activeTextEditor;
  const cursor = active && { uri: active.document.uri.toString(), line: active.selection.active.line + 1 };
  const preferCursor = (session: ReviewSession) => {
    const location = active && session.locate?.(active.document.uri);
    if (location && cursor) session.preferredLocation = { ...location, line: cursor.line };
    return session;
  };
  const folder = await workspaceGitFolder(input.modified) ?? await workspaceGitFolder(input.original);
  const pr = query(input.original.scheme === 'pr' ? input.original : input.modified);
  if ((input.original.scheme === 'pr' || input.modified.scheme === 'pr') && folder
    && typeof pr.prNumber === 'number' && typeof pr.remoteName === 'string'
    && typeof pr.baseCommit === 'string' && typeof pr.headCommit === 'string') {
    const repository = await repositoryForFolder(folder, pr.remoteName);
    if (repository) {
      const session = await loadPullRequest(parsePullRequestUrl(`https://github.com/${repository}/pull/${pr.prNumber}`), read, cache,
        { before: pr.baseCommit, after: pr.headCommit });
      session.preferredFile = typeof pr.fileName === 'string' ? pr.fileName : undefined;
      session.editorDiff = { original: input.original, modified: input.modified,
        beforeFile: typeof pr.previousFileName === 'string' ? pr.previousFileName : String(pr.fileName), afterFile: String(pr.fileName) };
      for (const [side, uri, file] of [['before', input.original, session.editorDiff.beforeFile], ['after', input.modified, session.editorDiff.afterFile]] as const) {
        const source = session[side].files.has(file) ? await session[side].read(file) : '';
        if ((await vscode.workspace.openTextDocument(uri)).getText() !== source) throw new Error('The PR code diff is stale. Refresh it and try again.');
      }
      return preferCursor(session);
    }
  }
  const relative = (uri: vscode.Uri) => folder ? path.relative(folder, filename(uri)).split(path.sep).join('/') : path.basename(filename(uri));
  const emptyTree = (uri: vscode.Uri) => uri.scheme === 'git' && ['sha1', 'sha256'].some(algorithm =>
    query(uri).ref === createHash(algorithm).update('tree 0\0').digest('hex'));
  const gitSide = async (uri: vscode.Uri): Promise<ReviewSnapshot | undefined> => {
    if (!folder) return undefined;
    if (uri.scheme === 'file' || uri.scheme === 'vscode-remote') return workingSnapshot(folder);
    if (uri.scheme === 'git') {
      const ref = query(uri).ref;
      if (emptyTree(uri)) {
        // Git uses the empty tree as a per-file sentinel for added/deleted files,
        // not as proof that the entire compared repository is empty.
        const other = uri === input.original ? input.modified : input.original;
        if (other.scheme !== 'file' && !(other.scheme === 'git' && ['HEAD', '~', '', 'index'].includes(String(query(other).ref)))) {
          throw new Error('This empty-file diff does not identify its repository revision. Open the PR or a current Git changes diff.');
        }
        if (uri === input.original) return other.scheme === 'git' ? gitSnapshot(folder, 'HEAD') : indexSnapshot(folder);
        const index = await indexSnapshot(folder);
        return index.files.has(relative(uri)) ? workingSnapshot(folder) : index;
      }
      // VS Code's '~' original means index when staged, otherwise HEAD. The
      // full index is the same original for both cases and includes deletions.
      if (ref === '' || ref === '~' || ref === 'index') return indexSnapshot(folder);
      if (typeof ref === 'string' && !ref.startsWith('~')) return gitSnapshot(folder, ref);
    }
    if (uri.scheme === 'review' && typeof query(uri).commit === 'string') return gitSnapshot(folder, query(uri).commit as string);
    return undefined;
  };
  const [original, modified] = await Promise.all([vscode.workspace.openTextDocument(input.original), vscode.workspace.openTextDocument(input.modified)]);
  const beforeFile = relative(input.original), afterFile = relative(input.modified);
  const [beforeGit, afterGit] = await Promise.all([gitSide(input.original), gitSide(input.modified)]);
  const freeze = (source: string, file: string, label: string): ReviewSnapshot => ({ label, files: new Map([[file, sha256Text(source)]]),
    truncated: true, read: async () => source });
  const before = beforeGit ?? freeze(original.getText(), path.basename(beforeFile), 'Editor before');
  const after = afterGit ?? freeze(modified.getText(), path.basename(afterFile), 'Editor after');
  const actualBefore = beforeGit ? beforeFile : path.basename(beforeFile), actualAfter = afterGit ? afterFile : path.basename(afterFile);
  // Provider URIs are not trusted revision metadata. Refuse a stale/mismatched
  // document rather than displaying a different change than the editor shows.
  for (const [snapshot, file, text] of [[before, actualBefore, original.getText()], [after, actualAfter, modified.getText()]] as const) {
    if ((snapshot.files.has(file) ? await snapshot.read(file) : '') !== text) throw new Error('The code diff changed while it was opening. Refresh the code diff and try again.');
  }
  const session: ReviewSession = { title: 'Code diff', folder, before, after, preferredFile: actualAfter,
    editorDiff: { original: input.original, modified: input.modified, beforeFile: actualBefore, afterFile: actualAfter },
    refresh: () => reviewEditorDiff(input, read, cache),
    locate: uri => {
      if (uri.toString() === input.original.toString()) return { file: actualBefore, side: 'before' };
      if (uri.toString() === input.modified.toString()) return { file: actualAfter, side: 'after' };
      const value = query(uri), file = relative(uri);
      if ((uri.scheme === 'file' || uri.scheme === 'vscode-remote') && afterGit && !after.commit && after.files.has(file)) return { file, side: 'after' };
      if (uri.scheme === 'git' || uri.scheme === 'review') {
        const ref = value.commit ?? value.ref;
        if (before.commit && ref === before.commit && before.files.has(file)) return { file, side: 'before' };
        if (after.commit && ref === after.commit && after.files.has(file)) return { file, side: 'after' };
      }
      return undefined;
    } };
  return preferCursor(session);
}
