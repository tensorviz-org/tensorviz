/** Installed VSIX acceptance. Use qa-host.mjs --vsix --vscode --debug-port in a disposable root. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vscode = require('vscode');
const { chromium } = require('../../canvas/node_modules/playwright');
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(label, read, timeout = 90000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { const value = await read(); if (value) return value; await pause(100); }
  throw new Error(`Timed out: ${label}`);
}
exports.run = async () => {
  const root = process.env.TV_QA_ROOT;
  assert.ok(root && process.env.TV_QA_DEBUG_PORT, 'Use the disposable QA host with a debug port');
  const mark = text => { console.log(`Native audit: ${text}`); fs.appendFileSync(path.join(root, 'evidence/native-progress.txt'), text + '\n'); };
  mark('activating extension');
  const api = (await vscode.extensions.getExtension('tensorviz.tensorviz-vscode').activate()).__test;
  mark('connecting to VS Code CDP');
  const browser = await chromium.connectOverCDP(`http://127.0.0.1:${process.env.TV_QA_DEBUG_PORT}`);
  mark('CDP connected');
  const results = [], errors = [];
  let page, app;
  const save = extra => fs.writeFileSync(path.join(root, 'evidence/native-audit.json'), JSON.stringify({
    scope: process.env.TV_QA_SETUP_ONLY ? 'Installed VSIX model-setup acceptance only; disposable native sidebar and 96-layer source fixture. Saving runs one CPU capture; no extra run or user workspace changes.'
      : 'Installed VSIX in disposable desktop VS Code; actual sidebar iframe and CPU-only 96-layer fixture. No user workspace or GPU execution.',
    vscode: vscode.version, platform: process.platform, arch: process.arch, passed: results, workbenchErrors: errors, ...extra }, null, 2) + '\n');
  try {
    page = await until('VS Code workbench', () => browser.contexts().flatMap(context => context.pages()).find(page => page.url().includes('workbench')));
    mark('workbench found');
    page.setDefaultTimeout(20000);
    mark('creating CPU fixture');
    const canvas = () => until('native iframe', () => page.frames().find(frame => /[?&]view=canvas/.test(frame.url())));
    const uri = vscode.Uri.file(path.join(root, 'fixture/models/manual.py'));
    fs.writeFileSync(uri.fsPath, `import torch.nn as nn

class ManualModel(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(features, features) for _ in range(96)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
`);
    const doc = await vscode.workspace.openTextDocument(uri); await vscode.window.showTextDocument(doc);
    const lens = await until('manual-model CodeLens', async () => (await vscode.commands.executeCommand('vscode.executeCodeLensProvider', uri))?.find(lens => lens.command?.arguments?.[0]?.cls === 'ManualModel'));
    await vscode.commands.executeCommand(lens.command.command, ...lens.command.arguments);
    mark('model selected');
    await vscode.commands.executeCommand('notifications.clearAll');
    app = await canvas();
    page.on('pageerror', error => errors.push({ message: error.message, stack: error.stack }));
    await app.evaluate(() => {
      window.nativeAuditErrors = [];
      window.addEventListener('error', event => window.nativeAuditErrors.push({ type: event.type, message: event.message, error: String(event.error), filename: event.filename, line: event.lineno, at: performance.now() }));
      window.addEventListener('unhandledrejection', event => window.nativeAuditErrors.push({ type: event.type, reason: String(event.reason), reasonType: typeof event.reason, stack: event.reason?.stack, at: performance.now() }));
    });
    const panel = app.getByRole('region', { name: 'Open captured graph' });
    await panel.getByRole('button', { name: 'Set up with agent', exact: true }).waitFor();
    assert.equal(await panel.getByLabel('features *', { exact: true }).isVisible(), false);
    assert.equal(await panel.getByRole('button', { name: 'Open graph', exact: true }).count(), 0);
    assert.equal(await panel.getByRole('button', { name: 'Set up with agent', exact: true }).evaluate(node => node === document.activeElement), true);
    assert.equal(await panel.getByRole('button', { name: 'Set up with agent', exact: true }).isVisible(), true);
    await page.screenshot({ path: path.join(root, 'evidence/native-setup-choice.png') });
    await panel.getByRole('button', { name: 'Set up manually', exact: true }).click();
    await panel.getByLabel('features *', { exact: true }).fill('8');
    await panel.getByLabel('x shape', { exact: true }).fill('2, 8');
    await panel.getByRole('button', { name: 'Setup methods', exact: true }).click();
    await panel.getByRole('button', { name: 'Set up manually', exact: true }).click();
    assert.equal(await panel.getByLabel('x shape', { exact: true }).inputValue(), '2, 8');
    await panel.getByTitle('Open declaration of features', { exact: true }).click();
    await until('constructor declaration revealed', () => vscode.window.activeTextEditor?.document.uri.fsPath === uri.fsPath
      && vscode.window.activeTextEditor.selection.active.line === 3);
    assert.equal(await panel.getByLabel('features *', { exact: true }).inputValue(), '8');
    await panel.getByTitle('Open declaration of x', { exact: true }).click();
    await until('forward declaration revealed', () => vscode.window.activeTextEditor?.selection.active.line === 7);
    await page.screenshot({ path: path.join(root, 'evidence/native-setup-source-fields.png') });
    await panel.locator('.graph-setup-body').evaluate(node => node.scrollTop = 0);
    await page.screenshot({ path: path.join(root, 'evidence/native-required-inputs.png') });
    await panel.getByText('Saved setup and source evidence', { exact: true }).click();
    await panel.getByRole('button', { name: 'Edit tensorviz.toml', exact: true }).waitFor();
    await panel.getByText('Saved setup and source evidence', { exact: true }).click();
    assert.match(await panel.locator('footer').innerText(), /then runs the model to capture the graph/);
    await panel.getByRole('button', { name: 'Open graph', exact: true }).click();
    await until('saved setup captured without a Run click', () => {
      const opening = api.graphOpeningState();
      if (opening?.phase === 'awaiting_execution') throw new Error('Saving setup waited for Run instead of capturing.');
      if (['failed', 'cancelled', 'needs_setup'].includes(opening?.phase)) throw new Error(`Capture ${opening.phase}: ${opening.detail}`);
      return opening?.phase === 'ready' && opening.hasCapture;
    }, 180000);
    assert.equal(api.graphOpeningState().execution.phase, 'succeeded'); assert.equal(api.graphOpeningState().execution.device, 'cpu');
    results.push('Explicit recommended agent/manual choice without competing form or submit; returning preserves edits; verified constructor/forward links reveal exact lines without losing edits; saving runs one CPU capture without a Run click'); save();
    if (process.env.TV_QA_SETUP_ONLY) {
      const canvasErrors = await app.evaluate(() => window.nativeAuditErrors);
      save({ status: canvasErrors.length ? 'failed' : 'passed', canvasErrors });
      assert.deepEqual(canvasErrors, []); return;
    }
    await app.getByRole('button', { name: 'Run model', exact: true }).click();
    await app.getByLabel('Device for this run', { exact: true }).selectOption('cpu');
    assert.equal(await app.getByLabel('Measure latency', { exact: true }).count(), 0);
    await app.locator('.execution-sheet').getByRole('button', { name: 'Run', exact: true }).click();
    await until('details closed after Run', async () => await app.getByRole('button', { name: 'Execution details', exact: true }).getAttribute('aria-expanded') === 'false');
    await page.screenshot({ path: path.join(root, 'evidence/native-run-closed.png') });
    await until('CPU capture settled', () => ['succeeded', 'failed'].includes(api.graphOpeningState()?.execution?.phase), 180000);
    assert.equal(api.graphOpeningState().execution.phase, 'succeeded', api.graphOpeningState().execution.detail);
    await vscode.commands.executeCommand('notifications.clearAll');
    assert.equal(await app.getByRole('button', { name: 'Latency details', exact: true }).count(), 0);
    results.push('Explicit CPU run closes the details and captures without a timing control'); save();
    const expand = app.getByRole('button', { name: 'Expand toolbar', exact: true }); if (await expand.isVisible()) await expand.click();
    const find = app.getByRole('button', { name: 'Find a layer', exact: true }); await find.click();
    const search = app.getByRole('dialog', { name: 'Find a layer', exact: true });
    const input = search.getByRole('textbox', { name: 'Quick jump to node' }); await input.fill('Linear');
    await search.getByText('1–20 of 96 results', { exact: true }).waitFor();
    await search.getByRole('button', { name: 'Next', exact: true }).click();
    await search.getByText('21–40 of 96 results', { exact: true }).waitFor();
    const paths = await search.locator('.local-canvas-search-path').allTextContents();
    assert.ok(paths.every(value => value.startsWith('layers.')), 'registered paths must be readable instead of opaque invocation IDs');
    await page.screenshot({ path: path.join(root, 'evidence/native-search.png') });
    await input.fill('layers.75');
    await search.getByText('layers.75', { exact: true }).waitFor();
    await input.press('ArrowDown'); await app.locator(':focus').press('Enter');
    await search.waitFor({ state: 'hidden' });
    await until('captured shapes rendered', () => app.locator('.tl-shape').count());
    await page.screenshot({ path: path.join(root, 'evidence/native-graph.png') });
    results.push('Native captured model search counts 96 repeated layers, pages beyond twenty, shows registered paths and jumps by keyboard');
    await app.getByRole('button', { name: 'Canvas guide and shortcuts', exact: true }).click();
    await app.getByRole('dialog', { name: 'Canvas guide', exact: true }).getByText(/Opening a model runs it once/).waitFor();
    await page.screenshot({ path: path.join(root, 'evidence/native-guide.png') });
    results.push('Native Canvas guide explains running on open and the Run model review');
    const canvasErrors = await app.evaluate(() => window.nativeAuditErrors);
    save({ status: canvasErrors.length ? 'failed' : 'passed', footer, benchmarkValues: values, canvasErrors });
    assert.deepEqual(canvasErrors, []);
  } catch (error) {
    if (page) await page.screenshot({ path: path.join(root, 'evidence/native-failure.png') });
    save({ status: 'failed', failure: String(error), opening: api.graphOpeningState(), canvasErrors: app ? await app.evaluate(() => window.nativeAuditErrors).catch(() => undefined) : undefined }); throw error;
  } finally { await browser.close(); }
};
