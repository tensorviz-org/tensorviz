import * as path from 'node:path';
import * as vscode from 'vscode';
import { isHostFile } from './workspaceFiles';

/** One `nn.Module` subclass, as `/api/ide/workspace/classes` reports it. */
export interface WorkspaceClass {
  name: string;
  /** Relative to the root that was scanned, forward slashes. */
  file: string;
  /** 1-based line of the `class` statement. */
  line: number;
  bases: string[];
  ctor_required: number;
}

/** How the provider reaches the backend. Injected so the unit tests can drive it
 *  without a running server, and so the provider survives a backend restart
 *  (the origin is read per call, never captured).
 *
 *  **Reject rather than resolving `[]` when you cannot scan.** An empty array is a
 *  real answer that this folder holds no `nn.Module` classes, and the Models view
 *  says exactly that. A source that returns it for "the backend is not up yet"
 *  makes the view state a falsehood about the user's own file, and hides the
 *  honest message that would have told them what to do. */
export interface ClassIndexSource {
  (root: string): Promise<WorkspaceClass[]>;
}

/**
 * Where a workspace's `nn.Module` classes are, cached per root.
 *
 * The lens that reads this — "Open as Graph" above every `nn.Module` class — lives
 * in `openAsGraphLenses.ts`. It was deleted outright when the project canvas made the
 * offer meaningless on an experiment-based canvas, where the class is already drawn
 * and a lens offering to *open* it offers something that has already happened
 * (step 2.5). It is back, withholding the offer for exactly that case and making it
 * everywhere else, because deleting it also deleted the single-file journey: open one
 * model file and there was no visible way into a graph at all.
 *
 * The index is deliberately separate from the lens and always was: the Models view
 * reads the same one, so the two can never disagree about which classes exist, and it
 * must still be built without importing a single file.
 *
 * The scan is per *root* (the workspace folder, or the file's own directory when
 * no folder is open) and cached, because a filesystem walk per keystroke would
 * be felt. `refresh()` is what open and save call.
 */
export class WorkspaceClassIndex implements vscode.Disposable {
  private readonly changed = new vscode.EventEmitter<void>();
  /** Fires when a scan changes what the index believes. The Models view redraws
   *  from it; it used to drive the lenses too. */
  readonly onDidChangeClasses = this.changed.event;


  /** root -> classes. Kept across a failed scan on purpose: a backend that is
   *  still starting must not make the Models view go empty. */
  private readonly index = new Map<string, WorkspaceClass[]>();
  private readonly inFlight = new Map<string, Promise<boolean>>();

  constructor(private readonly source: ClassIndexSource) {}

  /** The folder a file's classes are indexed under. */
  static rootFor(uri: vscode.Uri): string {
    return (
      vscode.workspace.getWorkspaceFolder(uri)?.uri.fsPath ?? path.dirname(uri.fsPath)
    );
  }

  /** Re-scan the root this file belongs to. */
  async refresh(uri: vscode.Uri): Promise<void> {
    if (!isHostFile(uri)) return;
    const root = WorkspaceClassIndex.rootFor(uri);
    await this.refreshRoot(root);
  }

  /** A model's project root may be nested inside its VS Code workspace folder. */
  async refreshRoot(root: string): Promise<boolean> {
    const running = this.inFlight.get(root);
    if (running) {
      return running;
    }
    const scan = this.source(root)
      .then((classes) => {
        this.index.set(root, classes);
        this.changed.fire();
        return true;
      })
      .catch(() => {
        // Deliberately silent, and deliberately keeping the old index: this runs
        // on every save, and a backend that has not started yet is the normal
        // case, not an error worth a notification.
        return false;
      })
      .finally(() => {
        this.inFlight.delete(root);
      });
    this.inFlight.set(root, scan);
    return scan;
  }

  /** What the index believes, for the tests. */
  classesFor(root: string): WorkspaceClass[] | undefined {
    return this.index.get(root);
  }

  dispose(): void {
    this.changed.dispose();
    this.index.clear();
    this.inFlight.clear();
  }
}
