import * as vscode from 'vscode';

/**
 * Make room for source beside a Canvas that fills the window.
 *
 * The Canvas lives in the secondary side bar. Maximized, it hides every editor, and
 * VS Code brings them back only when a reveal changes the visible editor or activates
 * its group, so a file already open behind the Canvas would stay hidden. Restoring
 * does nothing unless the side bar is maximized.
 */
export async function splitBesideCanvas(): Promise<void> {
  try {
    await vscode.commands.executeCommand('workbench.action.restoreAuxiliaryBar');
  } catch {
    // A VS Code without the command keeps its layout; the reveal still happens.
  }
}
