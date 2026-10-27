/** Installed VSIX smoke. qa-host owns the copied fixture and disposable profile. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vscode = require('vscode');
const { chromium } = require('../../canvas/node_modules/playwright');

async function until(label, read, timeout = 120000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    const result = await read();
    if (result) return result;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error(`Timed out: ${label}`);
}

exports.run = async () => {
  const root = process.env.TV_QA_ROOT;
  assert.ok(root && process.env.TV_QA_DEBUG_PORT, 'Use qa-host with a disposable root and debug port.');
  const fixture = path.join(root, 'fixture');
  assert.equal(vscode.workspace.workspaceFolders[0].uri.fsPath, fixture);
  const extension = vscode.extensions.getExtension('tensorviz.tensorviz');
  assert.ok(extension);
  assert.ok(extension.extensionPath.startsWith(path.join(root, 'extensions') + path.sep), 'Load the installed package, not the checkout.');
  assert.deepEqual(fs.readdirSync(path.join(extension.extensionPath, 'out')), ['extension.js']);
  // An explicit deterministic recipe keeps this smoke independent of setup inference.
  fs.writeFileSync(path.join(fixture, 'tensorviz.toml'), `version = 1
[profiles.smoke]
kind = "model"
model = "models.net:Net"
mode = "eval"
device = "cpu"
grad_enabled = false
init = { args = [], kwargs = {} }
inputs = { args = [{ kind = "tensor", shape = [2, 8], dtype = "float32", requires_grad = false, generator = { kind = "normal", mean = 0.0, std = 1.0, seed = 0 } }], kwargs = {} }
`);
  const api = (await extension.activate()).__test;
  api.seedManagedRuntimeForTests();
  // Attach before creating the native webview so its out-of-process frames are
  // observed from creation, as in the existing sidebar acceptance driver.
  const browser = await chromium.connectOverCDP(`http://127.0.0.1:${process.env.TV_QA_DEBUG_PORT}`);
  try {
    const document = await vscode.workspace.openTextDocument(path.join(fixture, 'models/net.py'));
    await vscode.window.showTextDocument(document);
    await vscode.commands.executeCommand('tensorviz.openAsGraph', { uri: document.uri.toString(), cls: 'Net' });
    // Opening is the request to run: no click is made, and a waiting review fails fast.
    const opening = await until('measured capture without a Run click', () => {
      const value = api.graphOpeningState();
      if (value?.phase === 'awaiting_execution') throw new Error('Opening a saved recipe waited for Run instead of capturing.');
      if (['failed', 'cancelled', 'needs_setup'].includes(value?.phase)) throw new Error(`Capture ${value.phase}: ${value.detail}`);
      return value?.phase === 'ready' && value.hasCapture && value;
    });
    assert.equal(opening.execution?.phase, 'succeeded'); assert.equal(opening.execution?.device, 'cpu', 'The saved device runs.');
    console.log('Installed smoke: measured capture ready. Checking rendered Canvas.');
    const frame = await until('native Canvas iframe', () => browser.contexts().flatMap(context => context.pages())
      .flatMap(page => page.frames()).find(item => /[?&]view=canvas/.test(item.url())));
    const page = frame.page();
    await frame.locator('[data-node-id]').first().waitFor({ state: 'visible', timeout: 60000 });
    await until('current Canvas projection receipt', () => api.messages().some(message =>
      message.type === 'projectionApplied' && message.sourceDigest === opening.target.sourceDigest
      && message.traced === true && message.greyed === false && message.nodeCount > 0));
    const count = await frame.locator('[data-node-id]').count();
    assert.ok(count > 0, 'A ready host must also render actual graph nodes.');
    await vscode.commands.executeCommand('notifications.clearAll');
    await page.screenshot({ path: path.join(root, 'evidence/canvas.png') });
    fs.writeFileSync(path.join(root, 'evidence/workflow.json'), JSON.stringify({
      status: 'passed', vscode: vscode.version, extension: extension.packageJSON.version,
      class: opening.target.cls, hasCapture: opening.hasCapture, renderedNodes: count, runOnOpen: true,
    }, null, 2));
    console.log('Installed package: saved-input capture and native Canvas render passed.');
  } catch (error) {
    // URL queries can contain launch credentials; retain only safe diagnostic fields.
    const pages = browser.contexts().flatMap(context => context.pages()).map(page => page.frames().map(frame => {
      const url = new URL(frame.url() || 'about:blank');
      return { protocol: url.protocol, path: url.pathname, view: url.searchParams.get('view') };
    }));
    fs.writeFileSync(path.join(root, 'evidence/render-failure.json'), JSON.stringify({ pages, projections: api.messages().filter(message => message.type === 'projectionApplied') }, null, 2));
    throw error;
  } finally {
    await browser.close();
  }
};
