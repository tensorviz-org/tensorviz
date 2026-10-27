import * as vscode from 'vscode';

import { ShapeHint } from './projection';

/**
 * Observed tensor output shapes immediately after the executed expression.
 * For `self.head(self.norm(x))`, each call gets its own hint, rather than two
 * ambiguous shapes at the end of the line or on the constructor declarations.
 *
 * Two rules keep it honest, and they are the whole design:
 *
 *  * only **observed** shapes are shown. An inlay hint is read as a fact about the
 *    program, and rendering an inferred one in the same grey as a measured one
 *    would be a lie in the same font. `LocalBrowserGraph.shapesFor` filters on
 *    `_shape_provenance === 'observed'` for exactly this reason.
 *  * the hints come from the projection the canvas is currently showing, so they
 *    disappear the moment the buffer moves on and reappear when the next save is
 *    traced. A stale shape presented as current is the one failure mode here.
 */
export class ShapeInlayHints implements vscode.InlayHintsProvider, vscode.Disposable {
  private readonly changed = new vscode.EventEmitter<void>();
  readonly onDidChangeInlayHints = this.changed.event;
  private readonly disposables: vscode.Disposable[] = [];

  constructor(private readonly source: (uri: vscode.Uri) => ShapeHint[]) {}

  /** Wire an event that means "the shapes changed" — the controller's. */
  refreshOn(event: vscode.Event<void>): void {
    this.disposables.push(event(() => this.changed.fire()));
  }

  provideInlayHints(document: vscode.TextDocument, range: vscode.Range): vscode.InlayHint[] {
    const hints: vscode.InlayHint[] = [];
    for (const shape of this.source(document.uri)) {
      const line = shape.line - 1;  // spans count from 1, Position from 0
      if (line < 0 || line < range.start.line || line > range.end.line || line >= document.lineCount) {
        continue;
      }
      const text = document.lineAt(line);
      if (!shape.shape || shape.column < 0 || shape.column > text.text.length) {
        continue;
      }
      const hint = new vscode.InlayHint(
        new vscode.Position(line, shape.column),
        ` → ${shape.shape}`,
        vscode.InlayHintKind.Type,
      );
      hint.paddingLeft = true;
      hint.tooltip = new vscode.MarkdownString(
        `**${shape.nodeId}** output \`${shape.shape}\`\n\n` +
          `Observed by running this model in your interpreter (\`${shape.provenance}\`).`,
      );
      hints.push(hint);
    }
    return hints;
  }

  dispose(): void {
    this.changed.dispose();
    this.disposables.forEach((item) => item.dispose());
  }
}
