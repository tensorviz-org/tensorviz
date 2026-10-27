/** Extension-host regression gate. Run with a fresh copied fixture and a manifest
 * captured BEFORE VS Code starts; activation itself used to write workspace state.
 * TV_QA_ROOT contains fixture/, evidence/fixture-before.json and control/.
 * Use this file as @vscode/test-electron's extensionTestsPath with the installed
 * VSIX as extensionDevelopmentPath. Never point it at a user's working tree. */
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const vscode = require('vscode');

exports.run = async () => {
  const root = process.env.TV_QA_ROOT;
  assert.ok(root, 'TV_QA_ROOT must name the disposable installed QA fixture');
  const fixture = path.join(root, 'fixture');
  assert.equal(vscode.workspace.workspaceFolders?.[0]?.uri.fsPath, fixture);
  const before = JSON.parse(fs.readFileSync(path.join(root, 'evidence/fixture-before.json'), 'utf8'));
  const extension = vscode.extensions.getExtension('tensorviz.tensorviz');
  assert.ok(extension);
  const api = (await extension.activate()).__test;
  const document = await vscode.workspace.openTextDocument(path.join(fixture, 'models/net.py'));
  await vscode.window.showTextDocument(document);
  await vscode.commands.executeCommand('tensorviz.openAsGraph', { uri: document.uri.toString(), cls: 'Net' });
  await vscode.commands.executeCommand('tensorviz.openCanvasInBrowser');
  assert.equal(api.localBrowserTarget()?.file, 'models/net.py');
  const after = {};
  function walk(dir) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const file = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        assert.ok(Object.keys(before).some(name => name.startsWith(path.relative(fixture, file) + path.sep)), `Unexpected workspace directory: ${file}`);
        walk(file);
      } else {
        after[path.relative(fixture, file)] = createHash('sha256').update(fs.readFileSync(file)).digest('hex');
      }
    }
  }
  walk(fixture);
  assert.deepEqual(after, before, 'Activation and static graph opening must write nothing into the workspace');
  console.log('Installed workspace purity: passed');
};
