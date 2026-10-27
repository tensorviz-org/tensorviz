import * as vscode from 'vscode';
import { OPEN_AS_GRAPH, OPEN_PROJECT, OPEN_PROJECT_IN_BROWSER } from './commandIds';
import { LocalBrowserGraph } from './localBrowserGraph';
import { SidebarCanvas, SHOW_SIDEBAR_CANVAS } from './sidebarCanvas';

export const HIDE_CANVAS = 'tensorviz.hideCanvas';
export const OPEN_CANVAS_IN_BROWSER = 'tensorviz.openCanvasInBrowser';
export const TOGGLE_CANVAS = 'tensorviz.toggleCanvas';

/** The selected text tab remains available while a view or palette owns focus. */
function selectedTextUri(): string | undefined {
  const editor = vscode.window.activeTextEditor;
  if (editor) return editor.document.uri.toString();
  const input = vscode.window.tabGroups.activeTabGroup.activeTab?.input;
  return input instanceof vscode.TabInputText ? input.uri.toString() : undefined;
}

/** All user entry points share one target owner. Legacy graphSurface values no
 * longer route commands; the browser is an explicit action on that owner. */
export function registerCanvasCommands(
  canvas: Pick<SidebarCanvas, 'show' | 'reveal' | 'hide' | 'visible'>,
  graph: Pick<LocalBrowserGraph, 'open' | 'openProject' | 'openSelectedInBrowser'>,
  resolveModel?: (target: { uri?: string; cls?: string }) => Promise<{ uri: string; cls: string } | undefined>,
): vscode.Disposable[] {
  const openProject = async (argument?: vscode.Uri | { folder?: string }) => {
    const active = vscode.window.activeTextEditor?.document.uri;
    const folder = argument instanceof vscode.Uri ? argument.fsPath : argument?.folder;
    const target = folder ?? (active && vscode.workspace.getWorkspaceFolder(active))?.uri.fsPath
      ?? vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    if (await canvas.show()) await graph.openProject({ folder: target }, 'canvas');
  };
  return [
    vscode.commands.registerCommand('tensorviz.openCanvas', () => canvas.reveal()),
    vscode.commands.registerCommand(SHOW_SIDEBAR_CANVAS, async (argument?: { uri?: string; cls?: string }) => {
      if (argument?.uri && argument.cls) {
        if (await canvas.show()) await graph.open(argument, 'canvas');
      } else await canvas.reveal();
    }),
    vscode.commands.registerCommand(HIDE_CANVAS, () => canvas.hide()),
    vscode.commands.registerCommand(TOGGLE_CANVAS, () => canvas.visible ? canvas.hide() : canvas.reveal()),
    vscode.commands.registerCommand(OPEN_AS_GRAPH, async (argument?: vscode.Uri | { uri?: string; cls?: string }) => {
      // Snapshot the selected file before discovery or revealing Canvas moves focus.
      let target = argument instanceof vscode.Uri ? { uri: argument.toString() } : argument;
      const uri = target?.uri ?? selectedTextUri();
      let revealed = false;
      if (resolveModel) {
        // Reveal service progress before class discovery, while keeping the
        // clicked file independent of any editor focus moved by Canvas.
        if (!(revealed = await canvas.show())) return;
        target = await resolveModel({ ...target, uri });
        if (!target) return;
      }
      if (!target?.cls) return openProject();
      const selected = { ...target, uri: target.uri ?? uri };
      if (revealed || await canvas.show()) await graph.open(selected, 'canvas');
    }),
    vscode.commands.registerCommand(OPEN_PROJECT, openProject),
    vscode.commands.registerCommand(OPEN_CANVAS_IN_BROWSER, () => graph.openSelectedInBrowser()),
    // Compatibility alias: never select the workspace of an incidental editor.
    vscode.commands.registerCommand(OPEN_PROJECT_IN_BROWSER, () => graph.openSelectedInBrowser()),
  ];
}
