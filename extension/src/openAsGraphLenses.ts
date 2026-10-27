import * as path from 'node:path';
import * as vscode from 'vscode';
import { isHostFile } from './workspaceFiles';

import { OPEN_AS_GRAPH } from './commandIds';
import { WorkspaceClass, WorkspaceClassIndex } from './workspaceClassIndex';

/** What the lens needs from the class index.
 *
 *  Structural, so the provider cannot be handed a second index: `WorkspaceClassIndex`
 *  is a window-wide singleton because the Models view and the lenses must never
 *  disagree about which classes exist, and because `workspaceClasses` throws while
 *  the backend is starting — only the shared index's keep-the-last-scan behaviour
 *  makes that harmless. */
export interface LensClassIndex {
  classesFor(root: string): WorkspaceClass[] | undefined;
  refresh(uri: vscode.Uri): Promise<void>;
  readonly onDidChangeClasses: vscode.Event<void>;
}

/** Whether the canvas on screen is ALREADY drawing this class. */
export type AlreadyDrawn = (root: string, relativeFile: string, cls: string) => boolean;

/**
 * "Open as Graph" above every `nn.Module` class — restored, with the condition that
 * made removing it right in the first place.
 *
 * The lens shipped in M3 and was deleted by the project canvas work, for a reason
 * that is still true: "on an experiment-based canvas the class is already drawn, so
 * a lens offering to *open* it offers something that has already happened." What was
 * lost with it is the single-file journey — open one model file and there is no
 * visible way into the graph at all, only a palette command nobody finds.
 *
 * So the offer is withheld exactly where it was meaningless, and made everywhere
 * else. `alreadyDrawn` is that test, and it is a map lookup: the project panel keeps
 * the span map of the document it published, keyed by the backend's node ids, and a
 * class node's id is `<relative file>::<ClassName>`. No line maths, no await — this
 * runs on every keystroke-idle.
 *
 * The original hard requirement is unchanged: FINDING the classes must not import
 * the file. The index comes from the backend's AST scan, a cold index renders
 * nothing and asks for a scan in the background, and a file that stops parsing keeps
 * the lenses the last good scan produced.
 */
export class OpenAsGraphLenses implements vscode.CodeLensProvider, vscode.Disposable {
  private readonly changed = new vscode.EventEmitter<void>();
  readonly onDidChangeCodeLenses = this.changed.event;
  private readonly subscriptions: vscode.Disposable[] = [];

  constructor(
    private readonly index: LensClassIndex,
    /** Defaults to "nothing is drawn", which is the honest answer for a window that
     *  has never opened a project canvas. */
    private readonly alreadyDrawn: AlreadyDrawn = () => false,
  ) {
    this.subscriptions.push(this.index.onDidChangeClasses(() => this.changed.fire()));
  }

  /**
   * Redraw because what the canvas draws has changed — a different experiment
   * focused, a frame collapsed, the canvas closed.
   *
   * `onDidChangeClasses` cannot carry this: it fires when the class INDEX changes,
   * and opening a canvas changes no class. Without this call the lens keeps showing
   * the answer from whenever the file was last saved.
   */
  refreshDrawn(): void {
    this.changed.fire();
  }

  provideCodeLenses(document: vscode.TextDocument): vscode.CodeLens[] {
    if (document.languageId !== 'python' || !isHostFile(document.uri)) {
      return [];
    }
    const root = WorkspaceClassIndex.rootFor(document.uri);
    const classes = this.index.classesFor(root);
    if (!classes) {
      // Cold index: ask, render nothing, and let onDidChangeCodeLenses bring us back.
      // Never await here — VS Code asks on every keystroke-idle.
      void this.index.refresh(document.uri);
      return [];
    }
    const relative = path
      .relative(root, document.uri.fsPath)
      .split(path.sep)
      .join('/');
    const lenses: vscode.CodeLens[] = [];
    for (const entry of classes) {
      if (entry.file !== relative) {
        continue;
      }
      if (this.alreadyDrawn(root, relative, entry.name)) {
        continue;
      }
      // The backend reports the `class` statement's own line; a decorated class
      // still anchors there, which is where a reader looks for the lens.
      const line = Math.max(0, entry.line - 1);
      lenses.push(
        new vscode.CodeLens(new vscode.Range(line, 0, line, 0), {
          title: '$(circuit-board) Open as Graph',
          tooltip: `Draw ${entry.name} on the tensorViz canvas`,
          command: OPEN_AS_GRAPH,
          // Naming the class is load-bearing: OPEN_AS_GRAPH with no `cls` opens the
          // whole PROJECT, so a per-class lens that omitted it would look
          // class-specific and behave project-wide.
          arguments: [{ uri: document.uri.toString(), cls: entry.name }],
        }),
      );
    }
    return lenses;
  }

  dispose(): void {
    for (const subscription of this.subscriptions) {
      subscription.dispose();
    }
    this.changed.dispose();
  }
}
