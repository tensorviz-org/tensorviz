// Interactive installed-VSIX acceptance fixture. Use qa-host.mjs --driver with
// this file and inspect the real editor/Canvas; this driver never simulates clicks.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const vscode = require('vscode');
const source = `import torch
from torch import nn


class ScaledBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)
        self.scale = nn.Parameter(torch.ones(8))

    def forward(self, x):
        x = self.proj(x)
        return x * self.scale

    def reset_scale(self):
        with torch.no_grad():
            self.scale.fill_(1)


def unrelated_helper(x):
    return x + 1
`;

exports.run = async () => {
  const root = process.env.TV_QA_ROOT;
  assert.ok(root, 'Launch with qa-host.mjs in a disposable profile');
  await vscode.workspace.getConfiguration('extensions').update('autoUpdate', false, vscode.ConfigurationTarget.Global);
  await vscode.workspace.getConfiguration('extensions').update('autoCheckUpdates', false, vscode.ConfigurationTarget.Global);
  const file = path.join(root, 'fixture/focus.py');
  fs.writeFileSync(file, source);
  const extension = vscode.extensions.getExtension('tensorviz.tensorviz');
  assert.ok(extension);
  const api = (await extension.activate()).__test;
  const document = await vscode.workspace.openTextDocument(file);
  await vscode.window.showTextDocument(document);
  await vscode.commands.executeCommand('tensorviz.openAsGraph', { uri: document.uri.toString(), cls: 'ScaledBlock' });
  const write = (name, value) => fs.writeFileSync(path.join(root, 'control', name), JSON.stringify(value, null, 2));
  let busy = false;
  await new Promise((resolve, reject) => {
    const timer = setInterval(async () => {
      if (busy) return;
      busy = true;
      try {
        const state = api.sidebarState();
        write('focus-state.json', { extensionPath: extension.extensionPath, vscode: vscode.version,
          target: api.localBrowserTarget(), visible: state?.visible, ready: state?.ready,
          selection: state?.messages.filter(message => message.type === 'viewSelection').at(-1),
          options: vscode.workspace.getConfiguration('tensorviz').get('codeFocus'),
          dirty: document.isDirty, unchanged: document.getText() === source,
          opening: api.graphOpeningState()?.phase });
        const commandFile = path.join(root, 'control/command.json');
        if (!fs.existsSync(commandFile)) return;
        const command = JSON.parse(fs.readFileSync(commandFile, 'utf8'));
        fs.unlinkSync(commandFile);
        if (command.type === 'finish') {
          assert.equal(document.getText(), source, 'Focus controls must never edit source');
          assert.equal(document.isDirty, false);
          clearInterval(timer); resolve();
        } else if (command.type === 'hide' || command.type === 'show') {
          await vscode.commands.executeCommand(command.type === 'hide' ? 'tensorviz.hideCanvas' : 'tensorviz.showSidebarCanvas');
        } else if (command.type === 'browser') {
          await vscode.commands.executeCommand('tensorviz.openCanvasInBrowser');
          write('page.json', api.localBrowserSession());
        }
      } catch (error) { clearInterval(timer); reject(error); }
      finally { busy = false; }
    }, 200);
  });
};
