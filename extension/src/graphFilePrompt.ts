import * as path from 'node:path';
import * as vscode from 'vscode';
import { OPEN_AS_GRAPH } from './commandIds';
import { LensClassIndex } from './openAsGraphLenses';
import { WorkspaceClassIndex } from './workspaceClassIndex';
import { isHostFile } from './workspaceFiles';

/** Offer discovered models without executing them or starting another scan. */
export class GraphFilePrompt implements vscode.Disposable {
  private readonly offered = new Set<string>();
  private readonly subscriptions: vscode.Disposable[];
  private disposed = false;

  constructor(private readonly index: LensClassIndex) {
    this.subscriptions = [
      vscode.window.onDidChangeActiveTextEditor(() => this.check()),
      index.onDidChangeClasses(() => this.check()),
      vscode.workspace.onDidChangeConfiguration(event => {
        if (event.affectsConfiguration('tensorviz.showGraphPrompt')) this.check();
      }),
    ];
    this.check();
  }

  private enabled(): boolean {
    return vscode.workspace.getConfiguration('tensorviz').get('showGraphPrompt', true);
  }

  private check(): void {
    const document = vscode.window.activeTextEditor?.document;
    if (this.disposed || !this.enabled() || !vscode.workspace.isTrusted
      || !document || document.languageId !== 'python' || document.isDirty
      || !isHostFile(document.uri)) return;
    const uri = document.uri.toString();
    if (this.offered.has(uri)) return;
    const root = WorkspaceClassIndex.rootFor(document.uri);
    const relative = path.relative(root, document.uri.fsPath).split(path.sep).join('/');
    if (!this.index.classesFor(root)?.some(entry => entry.file === relative)) return;
    // Index changes and editor focus may arrive together. Reserve before awaiting
    // the notification so neither rescans nor tab switching repeat it this session.
    this.offered.add(uri);
    void this.offer(uri, path.basename(document.uri.fsPath)).catch(error => {
      void vscode.window.showErrorMessage(`tensorViz: ${String(error)}`);
    });
  }

  private async offer(uri: string, file: string): Promise<void> {
    const action = await vscode.window.showInformationMessage(
      `tensorViz can visualize the PyTorch models in ${file}.`,
      'Open as Graph', 'Disable graph prompts',
    );
    if (this.disposed) return;
    if (action === 'Disable graph prompts') {
      await vscode.workspace.getConfiguration('tensorviz').update(
        'showGraphPrompt', false, vscode.ConfigurationTarget.Global,
      );
    } else if (action === 'Open as Graph' && this.enabled() && vscode.workspace.isTrusted) {
      // Bind to the offered file even if focus changed while the popup was visible.
      // The existing command chooses among multiple models and handles setup.
      await vscode.commands.executeCommand(OPEN_AS_GRAPH, { uri });
    }
  }

  dispose(): void {
    this.disposed = true;
    for (const subscription of this.subscriptions) subscription.dispose();
  }
}
