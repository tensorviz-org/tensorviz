import { matchesHostFile } from './workspaceFiles';
import { lstat, readFile, realpath } from 'node:fs/promises';
import * as path from 'node:path';
import * as vscode from 'vscode';
import { sha256Text } from './workspaceSources';
import { GraphOpeningError } from '@tensorviz/graph-host/graphOpening';

export interface SetupFileSnapshot { text: string; exists: boolean; identity: string | null; version: number | null }
import type { PreparedGraphSetup } from '@tensorviz/graph-host/graphOpening';
export type { PreparedGraphSetup } from '@tensorviz/graph-host/graphOpening';

export class SetupFileError extends GraphOpeningError {}

/** Generated setup can only touch these host-owned destinations, never arbitrary view paths. */
async function safeFile(root: string, file: string): Promise<string> {
  if (file !== 'tensorviz.toml' && !/^\.tensorviz\/setup_[A-Za-z_][A-Za-z0-9_]*_[a-f0-9]{12}\.py$/.test(file)) {
    throw new SetupFileError('The setup destination is invalid.');
  }
  const canonicalRoot = await realpath(root);
  const absolute = path.join(root, file);
  const segments = file.split('/');
  for (let index = 1; index <= segments.length; index++) {
    const item = path.join(root, ...segments.slice(0, index));
    try {
      const stat = await lstat(item);
      if (stat.isSymbolicLink() || (index === segments.length ? !stat.isFile() || stat.nlink !== 1 : !stat.isDirectory())) {
        throw new SetupFileError('Setup files must be ordinary workspace files, without symbolic or hard links.');
      }
      const physical = await realpath(item);
      const relative = path.relative(canonicalRoot, physical);
      if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
        throw new SetupFileError('The setup destination is outside this workspace.');
      }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
      break;
    }
  }
  return absolute;
}

export async function readSetupFile(root: string, file = 'tensorviz.toml'): Promise<SetupFileSnapshot> {
  const absolute = await safeFile(root, file);
  const document = vscode.workspace.textDocuments.find(doc => matchesHostFile(doc.uri, absolute));
  if (document?.isDirty) throw new SetupFileError('Save or discard changes to the setup file in VS Code first.');
  try {
    const before = await lstat(absolute);
    if (before.size > 262144) throw new SetupFileError('The setup file exceeds the supported size.');
    const text = await readFile(absolute, 'utf8');
    if (Buffer.byteLength(text) > 262144) throw new SetupFileError('The setup file exceeds the supported size.');
    await safeFile(root, file);
    const after = await lstat(absolute);
    const identity = (stat: typeof before) => `${stat.dev}:${stat.ino}:${stat.size}:${stat.mtimeMs}`;
    if (identity(before) !== identity(after) || document && document.getText() !== text) {
      throw new SetupFileError('The setup file changed while it was being read. Retry with the current file.');
    }
    return { text, exists: true, identity: identity(after), version: document?.version ?? null };
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT' && !document) {
      return { text: '', exists: false, identity: null, version: null };
    }
    throw error;
  }
}

const equal = (left: SetupFileSnapshot, right: SetupFileSnapshot) => left.exists === right.exists
  && left.identity === right.identity && left.text === right.text
  && (left.version === null || left.version === right.version);

/** Create a reviewable helper without replacing an existing or edited one. */
export async function openSetupHelper(root: string, file: string, source: string,
  current: () => boolean, signal: AbortSignal): Promise<void> {
  const check = () => {
    if (signal.aborted || !current()) throw new SetupFileError('The model changed or opening was cancelled.');
    if (!vscode.workspace.isTrusted) throw new SetupFileError('Trust this workspace before creating setup.');
  };
  check();
  const before = await readSetupFile(root, file);
  const uri = vscode.Uri.file(await safeFile(root, file));
  if (!before.exists) {
    const edit = new vscode.WorkspaceEdit();
    edit.createFile(uri, { overwrite: false, ignoreIfExists: false });
    edit.insert(uri, new vscode.Position(0, 0), source);
    check();
    if (!equal(before, await readSetupFile(root, file)) || !await vscode.workspace.applyEdit(edit)) {
      throw new SetupFileError('The helper changed before creation. Reopen setup.');
    }
  }
  check();
  await vscode.window.showTextDocument(await vscode.workspace.openTextDocument(uri), { preview: false });
}

/** An explicit opening or form submission saves these checked bytes before capture. */
export async function saveSetupFiles(root: string, before: SetupFileSnapshot, prepared: PreparedGraphSetup,
  current: () => boolean, signal: AbortSignal): Promise<void> {
  const check = () => {
    if (signal.aborted || !current()) throw new SetupFileError('The model changed or opening was cancelled.');
    if (!vscode.workspace.isTrusted) throw new SetupFileError('Trust this workspace before saving execution setup.');
  };
  check();
  if (!equal(before, await readSetupFile(root))) throw new SetupFileError('tensorviz.toml changed while setup was open. Reopen setup to preserve those changes.');
  const files = [{ file: 'tensorviz.toml', text: prepared.config, before }];
  if (prepared.factory) {
    const previous = await readSetupFile(root, prepared.factory.file);
    if (previous.exists && previous.text !== prepared.factory.source) throw new SetupFileError('The generated factory was edited. Keep that file and select it as an existing factory.');
    if (!previous.exists) files.push({ file: prepared.factory.file, text: prepared.factory.source, before: previous });
  }
  const edit = new vscode.WorkspaceEdit();
  for (const file of files) {
    const absolute = await safeFile(root, file.file);
    const uri = vscode.Uri.file(absolute);
    if (file.before.exists) {
      const document = await vscode.workspace.openTextDocument(uri);
      if (document.isDirty || document.getText() !== file.before.text) throw new SetupFileError('The setup file changed. Retry with its current contents.');
      edit.replace(uri, new vscode.Range(document.positionAt(0), document.positionAt(file.before.text.length)), file.text);
    } else {
      edit.createFile(uri, { overwrite: false, ignoreIfExists: false });
      edit.insert(uri, new vscode.Position(0, 0), file.text);
    }
  }
  for (const file of files) {
    if (!equal(file.before, await readSetupFile(root, file.file))) throw new SetupFileError('A setup destination changed before saving. Retry with current files.');
  }
  check();
  if (!await vscode.workspace.applyEdit(edit)) throw new SetupFileError('VS Code could not apply the setup files.');
  for (const file of files) {
    check();
    await safeFile(root, file.file);
    const document = await vscode.workspace.openTextDocument(vscode.Uri.file(path.join(root, file.file)));
    if (document.getText() !== file.text || !await document.save()) throw new SetupFileError('Save the prepared setup files in VS Code to continue.');
    if (sha256Text((await readSetupFile(root, file.file)).text) !== sha256Text(file.text)) {
      throw new SetupFileError('The saved setup changed. Reopen it before capturing the model.');
    }
  }
  check();
}
