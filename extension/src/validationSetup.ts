/** Host-owned configuration buffers. No config values live in workspace storage. */
import * as vscode from 'vscode';
import * as path from 'node:path';
import { randomUUID } from 'node:crypto';
import type { ValidationRequirement } from '@tensorviz/contracts/protocol';
import type { ValidationTarget } from '@tensorviz/graph-host/validationController';
import { CONFIG_STARTER, locateConfigField } from './validationConfig';

export class ValidationSetup implements vscode.Disposable {
  private readonly diagnostics = vscode.languages.createDiagnosticCollection('tensorviz-config');
  private reviewing = false;
  private readonly changes = vscode.workspace.onDidChangeTextDocument(event => this.diagnostics.delete(event.document.uri));
  dispose(): void { this.changes.dispose(); this.diagnostics.dispose(); }
  clear(root: string): void { this.diagnostics.delete(vscode.Uri.file(path.join(root, 'tensorviz.toml'))); }

  /** Navigation only: a missing saved profile must never open a creation wizard. */
  async reveal(target: ValidationTarget, field: (string | number)[]): Promise<void> {
    if (!target.current()) return;
    const document = await vscode.workspace.openTextDocument(vscode.Uri.file(path.join(target.root, 'tensorviz.toml')));
    if (!target.current()) return;
    const location = locateConfigField(document.getText(), field);
    await vscode.window.showTextDocument(document, { preview: false,
      selection: new vscode.Range(document.positionAt(location.start), document.positionAt(location.end)) });
  }

  async open(target: ValidationTarget, field: (string | number)[] = [], starter = CONFIG_STARTER): Promise<void> {
    if (!target.current()) return;
    const uri = vscode.Uri.file(path.join(target.root, 'tensorviz.toml'));
    let document: vscode.TextDocument;
    try { document = await vscode.workspace.openTextDocument(uri); }
    catch (error) {
      // Only a genuinely absent file permits creation; permission/read errors do not.
      try { await vscode.workspace.fs.stat(uri); } catch (statError) {
        if (!(statError instanceof vscode.FileSystemError) || statError.code !== 'FileNotFound') throw error;
        await this.create(target, uri, starter);
        return;
      }
      throw error;
    }
    if (!target.current()) return;
    const location = locateConfigField(document.getText(), field);
    await vscode.window.showTextDocument(document, { preview: false,
      selection: new vscode.Range(document.positionAt(location.start), document.positionAt(location.end)) });
  }

  async show(target: ValidationTarget, requirements: ValidationRequirement[]): Promise<void> {
    if (!target.current() || !requirements.length) return;
    const uri = vscode.Uri.file(path.join(target.root, 'tensorviz.toml'));
    try {
      const document = await vscode.workspace.openTextDocument(uri);
      if (!target.current()) return;
      this.diagnostics.set(uri, requirements.map(requirement => {
        const location = locateConfigField(document.getText(), requirement.field);
        const diagnostic = new vscode.Diagnostic(
          new vscode.Range(document.positionAt(location.start), document.positionAt(location.end)),
          `${requirement.field.join('.')}: ${requirement.reason} ${requirement.remedy}`,
          vscode.DiagnosticSeverity.Error);
        diagnostic.source = 'tensorViz setup';
        diagnostic.code = requirement.code;
        return diagnostic;
      }));
    } catch { /* Missing config: the host picker can create an unsaved buffer. */ }
    const selected = await vscode.window.showQuickPick(requirements.map(requirement => ({
      label: requirement.field.join('.'), description: requirement.reason,
      detail: `${requirement.remedy} Open tensorviz.toml in VS Code.`, requirement,
    })), { title: 'Needs setup — choose a field to edit tensorviz.toml', ignoreFocusOut: true });
    if (selected && target.current()) await this.open(target, selected.requirement.field);
  }

  private async create(target: ValidationTarget, uri: vscode.Uri, starter: string): Promise<void> {
    if (this.reviewing || !target.current() || !vscode.workspace.isTrusted) return;
    this.reviewing = true;
    const scheme = `tensorviz-config-${randomUUID()}`;
    const before = vscode.Uri.from({ scheme, path: '/before/tensorviz.toml' });
    const after = vscode.Uri.from({ scheme, path: '/after/tensorviz.toml' });
    const provider = vscode.workspace.registerTextDocumentContentProvider(scheme, {
      provideTextDocumentContent: value => value.toString() === before.toString() ? '' : starter,
    });
    try {
      await vscode.commands.executeCommand('vscode.diff', before, after, 'Create tensorviz.toml — review starter', { preview: true });
      const choice = await vscode.window.showInformationMessage('Apply this starter to an unsaved tensorviz.toml buffer?', 'Apply', 'Discard');
      if (choice !== 'Apply' || !target.current() || !vscode.workspace.isTrusted) return;
      try {
        await vscode.workspace.fs.stat(uri);
        void vscode.window.showInformationMessage('tensorviz.toml was created during review. Open it again to inspect the current file.');
        return;
      } catch (error) {
        if (!(error instanceof vscode.FileSystemError) || error.code !== 'FileNotFound') throw error;
      }
      const document = await vscode.workspace.openTextDocument(uri.with({ scheme: 'untitled' }));
      if (!target.current() || document.getText() || !vscode.workspace.isTrusted) return;
      const edit = new vscode.WorkspaceEdit();
      edit.insert(document.uri, new vscode.Position(0, 0), starter);
      if (!await vscode.workspace.applyEdit(edit)) throw new Error('Configuration edit was not applied.');
      await vscode.window.showTextDocument(document, { preview: false });
    } finally {
      const tabs = vscode.window.tabGroups.all.flatMap(group => group.tabs).filter(tab =>
        tab.input instanceof vscode.TabInputTextDiff && tab.input.modified.scheme === scheme);
      try { if (tabs.length) await vscode.window.tabGroups.close(tabs, true); }
      finally { provider.dispose(); this.reviewing = false; }
    }
  }
}
