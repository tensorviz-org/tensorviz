import * as assert from 'node:assert/strict';
import * as vscode from 'vscode';

/** Exercise the real picker, inspection and reopen path with the test runtime. */
export async function selectManagedEnvironment(): Promise<void> {
  const surface = vscode.window as unknown as Record<string, unknown>;
  const original = surface.showQuickPick;
  surface.showQuickPick = async (items: ReadonlyArray<{ choice?: { kind: string } }>) => {
    const managed = items.find(item => item.choice?.kind === 'managed');
    assert.ok(managed, 'the managed environment must be an explicit choice');
    return managed;
  };
  try {
    assert.equal(await vscode.commands.executeCommand('tensorviz.selectEnvironment'), true);
  } finally { surface.showQuickPick = original; }
}
