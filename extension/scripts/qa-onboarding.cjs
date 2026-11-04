/** Cold installed-package acceptance. Real pinned download/dependencies and renderer;
 * notification responses and environment selection are controlled, not native UI proof. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
let vscode = require('vscode');
const { chromium } = require('../../canvas/node_modules/playwright');

const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(label, read, timeout = 180000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) { const value = await read(); if (value) return value; await pause(100); }
  throw new Error(`Timed out: ${label}`);
}

exports.run = async () => {
  const root = process.env.TV_QA_ROOT;
  assert.ok(root && process.env.TV_QA_DEBUG_PORT && process.env.TENSORVIZ_TEST_PYTHON);
  assert.equal(process.env.TENSORVIZ_TEST_PRESEED_MANAGED, '0');
  assert.ok(!process.env.BACKEND_PYTHON && !process.env.TENSORVIZ_TEST_RUNTIME_ARCHIVE && !process.env.TENSORVIZ_TEST_RUNTIME_MANIFEST);
  const extension = vscode.extensions.getExtension('tensorviz.tensorviz-vscode');
  assert.ok(extension.extensionPath.startsWith(path.join(root, 'extensions') + path.sep));
  // VS Code scopes its API objects by extension path. Intercept the installed
  // extension's API, not the external driver's unrelated API object.
  vscode = require('node:module').createRequire(path.join(extension.extensionPath, 'out/extension.js'))('vscode');
  const fixture = path.join(root, 'fixture');
  assert.deepEqual(fs.readdirSync(fixture), [], 'cold launcher must start without model-triggered setup');
  let consent = undefined, graphAction = undefined;
  const notifications = [];
  const surface = vscode.window;
  const original = { info: surface.showInformationMessage, picker: surface.showQuickPick, input: surface.showInputBox,
    error: surface.showErrorMessage, warn: surface.showWarningMessage };
  for (const kind of ['showErrorMessage', 'showWarningMessage']) surface[kind] = async (message, ...args) => {
    notifications.push({ kind, message, options: args.find(value => value && typeof value === 'object') });
    save(); return undefined;
  };
  surface.showInformationMessage = async (message, ...args) => {
    const buttons = args.filter(value => typeof value === 'string');
    const options = args.find(value => value && typeof value === 'object');
    notifications.push({ message, buttons, options });
    save();
    if (message.startsWith('Install tensorViz’s runtime')) {
      assert.deepEqual(buttons, ['Install']);
      assert.equal(options.modal, true);
      assert.match(options.detail, /dependency downloads/);
      return consent;
    }
    if (message.startsWith('tensorViz can visualize')) return graphAction;
    return undefined;
  };
  surface.showQuickPick = async (items, options) => {
    if (options?.title === 'Choose the Python environment for tensorViz') {
      return (await items).find(item => item.action === 'path');
    }
    return original.picker.call(surface, items, options);
  };
  surface.showInputBox = async options => options?.title === 'Python environment'
    ? process.env.TENSORVIZ_TEST_PYTHON : original.input.call(surface, options);
  let browser;
  const evidence = { status: 'running', nativeDialogsVerified: false, controlledResponses: true, checks: [] };
  const save = () => {
    evidence.notifications = notifications;
    fs.writeFileSync(path.join(root, 'evidence/onboarding.json'), JSON.stringify(evidence, null, 2));
  };
  try {
    const api = (await extension.activate()).__test;
    const runtime = path.join(api.globalStorageDir(), 'runtime');
    assert.ok(!fs.existsSync(path.join(runtime, '.tensorviz-provisioned.json')));
    await vscode.commands.executeCommand('tensorviz.openCanvas');
    assert.ok(!fs.existsSync(runtime), 'the starting screen must not provision a runtime');
    assert.equal(api.backendOrigin(), undefined);
    assert.equal(notifications.filter(item => item.message.startsWith('Install tensorViz’s runtime')).length, 0);
    evidence.checks.push('starting screen is visible without runtime setup'); save();
    await vscode.commands.executeCommand('tensorviz.checkEnvironment');
    assert.ok(!fs.existsSync(runtime), 'dismissed consent must install nothing');
    assert.equal(api.backendOrigin(), undefined);
    evidence.checks.push('dismissed cold consent installs nothing'); save();

    consent = 'Install';
    const started = Date.now();
    await vscode.commands.executeCommand('tensorviz.checkEnvironment');
    assert.ok(api.backendOrigin(), 'the actual packaged backend must start');
    evidence.serviceMs = Date.now() - started;
    evidence.setup = JSON.parse(fs.readFileSync(path.join(runtime, 'last-setup.json'), 'utf8'));
    assert.equal(evidence.setup.outcome, 'ready');
    assert.ok(evidence.setup.timingsMs.download >= 0);
    evidence.archiveBytes = fs.readdirSync(path.join(runtime, 'downloads')).filter(file => file.endsWith('.tar.gz'))
      .map(file => fs.statSync(path.join(runtime, 'downloads', file)).size);
    evidence.checks.push('real cold pinned runtime and dependencies reach backend ready'); save();

    fs.mkdirSync(path.join(fixture, 'models'), { recursive: true });
    fs.writeFileSync(path.join(fixture, 'models/__init__.py'), '');
    const model = 'import torch.nn as nn\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.layer = nn.Linear(8, 8)\n    def forward(self, x):\n        return self.layer(x)\n';
    const uri = vscode.Uri.file(path.join(fixture, 'models/net.py'));
    fs.writeFileSync(uri.fsPath, model);
    fs.writeFileSync(path.join(fixture, 'tensorviz.toml'), `version = 1
[profiles.qa]
kind = "model"
model = "models.net:Net"
mode = "eval"
device = "cpu"
grad_enabled = false
init = { args = [], kwargs = {} }
inputs = { args = [{ kind = "tensor", shape = [2, 8], dtype = "float32", requires_grad = false, generator = { kind = "normal", mean = 0.0, std = 1.0, seed = 0 } }], kwargs = {} }
`);
    browser = await chromium.connectOverCDP(`http://127.0.0.1:${process.env.TV_QA_DEBUG_PORT}`);
    graphAction = 'Open as Graph';
    const graphStarted = Date.now();
    await vscode.window.showTextDocument(await vscode.workspace.openTextDocument(uri));
    let selected = false;
    const opening = await until('popup-driven measured capture', async () => {
      const value = api.graphOpeningState();
      if (evidence.openingPhase !== value?.phase) {
        evidence.openingPhase = value?.phase;
        evidence.openingDetail = value?.detail;
        save();
      }
      if (value?.phase === 'failed') throw new Error(value.detail);
      if (value?.phase === 'needs_setup' && value.failure?.kind === 'environment' && !selected) {
        selected = true;
        await vscode.commands.executeCommand('tensorviz.selectEnvironment');
      }
      if (value?.phase === 'needs_setup' && value.failure?.kind !== 'environment') {
        throw new Error(`Saved-input fixture requires unexpected setup: ${value.detail}`);
      }
      return value?.phase === 'ready' && value.hasCapture && value;
    });
    assert.equal(opening.target.cls, 'Net');
    const frame = await until('visible Canvas', () => browser.contexts().flatMap(context => context.pages())
      .flatMap(page => page.frames()).find(item => /[?&]view=canvas/.test(item.url())));
    await frame.locator('[data-node-id]').first().waitFor({ state: 'visible', timeout: 60000 });
    await until('matching measured render receipt', () => api.messages().some(message => message.type === 'projectionApplied'
      && message.sourceDigest === opening.target.sourceDigest && message.traced === true && message.greyed === false && message.nodeCount > 0));
    evidence.graphMs = Date.now() - graphStarted;
    evidence.renderedNodes = await frame.locator('[data-node-id]').count();
    await frame.page().screenshot({ path: path.join(root, 'evidence/onboarding-canvas.png') });
    evidence.checks.push('model-file offer opens a measured graph with matching visible render receipt');

    graphAction = 'Disable graph prompts';
    const other = vscode.Uri.file(path.join(fixture, 'models/other.py'));
    fs.writeFileSync(other.fsPath, model);
    await vscode.window.showTextDocument(await vscode.workspace.openTextDocument(other));
    await until('disable preference persisted', () => vscode.workspace.getConfiguration('tensorviz').get('showGraphPrompt') === false);
    const offers = () => notifications.filter(item => item.message.startsWith('tensorViz can visualize')).length;
    const offered = offers();
    await vscode.window.showTextDocument(await vscode.workspace.openTextDocument(uri));
    await pause(500);
    assert.equal(offers(), offered);
    assert.equal(api.localBrowserTarget().file, 'models/net.py', 'disable must not open the other model');
    evidence.checks.push('disable action persists and suppresses subsequent offers');
    const consents = notifications.filter(item => item.message.startsWith('Install tensorViz’s runtime')).length;
    await vscode.commands.executeCommand('tensorviz.openCanvas');
    assert.equal(notifications.filter(item => item.message.startsWith('Install tensorViz’s runtime')).length, consents);
    evidence.checks.push('ready service reuse does not prompt or provision again');
    evidence.status = 'passed';
  } catch (error) {
    evidence.status = 'failed'; evidence.error = String(error); throw error;
  } finally {
    save(); surface.showInformationMessage = original.info;
    surface.showQuickPick = original.picker; surface.showInputBox = original.input;
    surface.showErrorMessage = original.error; surface.showWarningMessage = original.warn;
    if (browser) await browser.close();
  }
};
