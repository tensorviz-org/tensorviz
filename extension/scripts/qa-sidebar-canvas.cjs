/** Real VS Code + production iframe acceptance. Launched only by qa-host.mjs in
 * a copied fixture and fresh profile. CDP inspects the actual webview frame;
 * host state alone is never counted as renderer delivery. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vscode = require('vscode');
const { chromium } = require('../../canvas/node_modules/playwright');
const root = process.env.TV_QA_ROOT;
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(label, read, timeout = 30000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { const value = await read(); if (value) return value; await pause(50); }
  throw new Error(`Timed out: ${label}`);
}
exports.run = async () => {
  assert.ok(root && process.env.TV_QA_DEBUG_PORT, 'Use qa-host.mjs --debug-port with a disposable --root');
  const results = [];
  const check = async (name, test) => {
    await test(); results.push(name); console.log(`PASS sidebar: ${name}`);
    fs.writeFileSync(path.join(root, 'evidence/sidebar-results.json'), JSON.stringify({ host: vscode.version, platform: process.platform, arch: process.arch, passed: results }, null, 2));
  };
  const extension = vscode.extensions.getExtension('tensorviz.tensorviz');
  const api = (await extension.activate()).__test;
  const browser = await chromium.connectOverCDP(`http://127.0.0.1:${process.env.TV_QA_DEBUG_PORT}`);
  let client;
  try {
    const page = await until('VS Code workbench', () => browser.contexts().flatMap(c => c.pages()).find(p => p.url().includes('workbench')));
    page.setDefaultTimeout(15000);
    // Complete zoom commands synchronously: hiding before the first animation
    // frame otherwise measures the pre-gesture camera, not checkpoint freshness.
    await page.emulateMedia({ reducedMotion: 'reduce' });
    page.on('pageerror', error => fs.appendFileSync(path.join(root, 'evidence/renderer-errors.txt'), `${error.stack || error}\n`));
    page.on('console', message => { if (message.type() === 'error') fs.appendFileSync(path.join(root, 'evidence/renderer-errors.txt'), `${message.text()}\n`); });
    const frame = () => page.frames().find(f => /[?&]view=canvas/.test(f.url()));
    const canvas = async () => until('native iframe', frame);
    const uri = vscode.Uri.file(path.join(root, 'fixture/models/net.py'));
    const doc = await vscode.workspace.openTextDocument(uri);
    const command = (id, ...args) => vscode.commands.executeCommand(id, ...args);
    const target = () => api.localBrowserTarget();
    let before;
    await check('empty reveal registers one visible secondary-sidebar Canvas', async () => {
      await command('tensorviz.openCanvas');
      await until('actual visibility', () => api.sidebarState()?.visible, 180000);
      await until('host starting-screen state', () => api.sidebarState()?.homeVisible);
      assert.equal(await page.locator('.auxiliarybar').getByText('tensorViz', { exact: true }).count() > 0, true);
      assert.equal(api.localBrowserSession(), undefined, 'show must not mint a browser session');
    });
    let lens;
    await check('real CodeLens selects an exact class and repeated selection reuses the renderer', async () => {
      await vscode.window.showTextDocument(doc);
      lens = await until('class CodeLens', async () => (await command('vscode.executeCodeLensProvider', uri))?.find(l => l.command?.arguments?.[0]?.cls === 'Net'), 60000);
      await command(lens.command.command, ...lens.command.arguments);
      await (await canvas()).locator('[data-node-id="up"]').waitFor();
      assert.equal(target().cls, 'Net');
      before = { ...target() };
      const session = api.sidebarState().sessionId;
      await command(lens.command.command, ...lens.command.arguments);
      assert.deepEqual(target(), before);
      assert.equal(api.sidebarState().sessionId, session);
      const otherUri = vscode.Uri.file(path.join(root, 'fixture/complex_models/network.py'));
      await vscode.window.showTextDocument(await vscode.workspace.openTextDocument(otherUri));
      const blockLens = await until('second exact class CodeLens', async () => (await command('vscode.executeCodeLensProvider', otherUri))?.find(l => l.command?.arguments?.[0]?.cls === 'Block'));
      await command(blockLens.command.command, ...blockLens.command.arguments);
      await (await canvas()).locator('[data-node-id="conv"]').waitFor();
      assert.equal(target().cls, 'Block');
      assert.equal(target().file, 'complex_models/network.py');
      await command(lens.command.command, ...lens.command.arguments);
      await (await canvas()).locator('[data-node-id="up"]').waitFor();
      before = { ...target() };
      assert.equal(api.sidebarState().sessionId, session, 'another exact class reuses the same renderer');
      assert.equal(page.frames().filter(f => /[?&]view=canvas/.test(f.url())).length, 1);
    });
    await check('editor switching keeps the selected model; source edits refresh that model', async () => {
      await vscode.window.showTextDocument(await vscode.workspace.openTextDocument(vscode.Uri.file(path.join(root, 'fixture/complex_models/network.py'))));
      await pause(400);
      assert.deepEqual(target(), before);
      const edit = new vscode.WorkspaceEdit();
      edit.insert(uri, new vscode.Position(0, 0), '# disposable sidebar refresh\n');
      assert.equal(await vscode.workspace.applyEdit(edit), true);
      await until('source projection refresh', () => target()?.documentVersion === doc.version && target()?.sourceDigest !== before.sourceDigest);
      await until('iframe refreshed', () => api.sidebarState().messages.some(m => m.type === 'projectionApplied' && m.sourceDigest === target().sourceDigest));
      await doc.save();
    });
    await check('narrow iframe controls and More menu stay inside the page', async () => {
      const app = await canvas();
      const width = await app.evaluate(() => innerWidth);
      assert.ok(width >= 200 && width <= 600, `expected native narrow sidebar, got ${width}`);
      const find = app.getByRole('button', { name: 'Find a layer', exact: true });
      await find.waitFor();
      assert.ok(await find.evaluate(el => { const r = el.getBoundingClientRect(); return r.width > 0 && r.left >= 0 && r.right <= innerWidth; }), 'layer finder must be usable');
      const overflow = await app.evaluate(() => document.documentElement.scrollWidth - innerWidth);
      assert.ok(overflow <= 1, `horizontal page overflow: ${overflow}`);
      for (const selector of ['[data-testid="local-browser-validate"]', '[data-testid="ui2-action-more"]']) {
        const control = app.locator(selector); await control.waitFor();
        const visible = await control.evaluate(el => { const r = el.getBoundingClientRect(); return r.left >= 0 && r.right <= innerWidth && r.bottom <= innerHeight; });
        assert.ok(visible, `${selector} is outside the iframe`);
      }
      await app.getByTestId('ui2-action-more').click();
      const menu = app.getByRole('menu'); await menu.waitFor();
      assert.ok(await menu.evaluate(el => { const r = el.getBoundingClientRect(); return r.left >= 0 && r.right <= innerWidth; }));
      await page.screenshot({ path: path.join(root, 'evidence/sidebar-narrow.png') });
      await app.locator('body').press('Escape');
    });
    await check('native MCP focus waits for the real iframe receipt and hidden views disappear', async () => {
      const connecting = command('tensorviz.connectMcp');
      // The current setup command asks which client to configure. Use the
      // connection guide so this disposable test never edits agent settings.
      await page.getByText('VS Code Copilot / another client', { exact: true }).click();
      await connecting;
      const connection = api.mcpConnection();
      const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
      const { StreamableHTTPClientTransport } = require('@modelcontextprotocol/sdk/client/streamableHttp.js');
      // The command opens credentials in an untitled editor; keep them out of screenshots.
      await vscode.window.showTextDocument(doc);
      client = new Client({ name: 'sidebar-electron', version: '1' });
      await client.connect(new StreamableHTTPClientTransport(new URL(connection.url), { requestInit: { headers: connection.headers } }));
      const viewId = api.sidebarState().viewId;
      // The save that ends the edit check starts a trace, and its capture can land
      // between discovery and focus. STALE_TARGET is the documented answer: rediscover, as an agent would.
      const stale = result => result.isError && result.structuredContent?.error?.code === 'STALE_TARGET';
      const reference = await until('focus on the current revision', async () => {
        const discovered = (await client.callTool({ name: 'list_models', arguments: {} })).structuredContent;
        const model = discovered.models[0];
        assert.ok(discovered.views.some(view => view.viewId === viewId));
        const reference = { modelId: model.modelId, revision: model.revision, viewId };
        const focused = await client.callTool({ name: 'focus_view', arguments: { ...reference, nodeIds: ['up'] } });
        if (stale(focused)) return undefined;
        assert.ok(!focused.isError, JSON.stringify(focused));
        assert.equal(focused.structuredContent.focused, true);
        const selection = await client.callTool({ name: 'get_selection', arguments: reference });
        if (stale(selection)) return undefined;
        assert.deepEqual(selection.structuredContent.nodeIds, ['up']);
        return reference;
      }, 60000);
      await command('tensorviz.hideCanvas');
      await until('hidden MCP presence', async () => !(await client.callTool({ name: 'list_models', arguments: {} })).structuredContent.views.some(view => view.viewId === viewId));
      const refused = await client.callTool({ name: 'focus_view', arguments: { ...reference, nodeIds: ['up'] } });
      assert.equal(refused.isError, true);
      await command('tensorviz.showSidebarCanvas');
      await until('native selection re-registered', async () => (await client.callTool({ name: 'list_models', arguments: {} })).structuredContent.views.some(view => view.viewId === viewId && view.current));
    });
    await check('explicit browser opening uses the selected model, without retargeting', async () => {
      const selected = { ...target() };
      await command('tensorviz.openCanvasInBrowser');
      assert.ok(api.localBrowserSession());
      assert.deepEqual(target(), selected);
      assert.ok(fs.readFileSync(path.join(root, 'control/opened-links.txt'), 'utf8').includes('session='));
    });
    await check('hide/reopen retains target and camera through the authenticated checkpoint', async () => {
      const app = await canvas();
      await app.evaluate(() => window.dispatchEvent(new CustomEvent('builder-command', { detail: { command: 'focus-node', payload: { nodeId: 'up' } } })));
      await until('presentation selected', () => api.sidebarState().presentation?.selected?.length);
      const targetBefore = { ...target() };
      await command('tensorviz.hideCanvas');
      await until('actual hidden', () => !api.sidebarState().visible);
      const presentation = api.sidebarState().presentation;
      assert.ok(presentation.camera.z > 0);
      assert.deepEqual(target(), targetBefore);
      await command('tensorviz.showSidebarCanvas');
      await until('actual visible', () => api.sidebarState().visible && api.sidebarState().ready);
      await (await canvas()).locator('[data-node-id="up"]').waitFor();
      assert.deepEqual(api.sidebarState().presentation.camera, presentation.camera);
      assert.deepEqual(api.sidebarState().presentation.selected, presentation.selected);
    });
    await check('validation completes while hidden and its result reaches the reopened iframe', async () => {
      const previousRuns = new Set(api.sidebarState().receipts.filter(r => r.type === 'validationState').map(r => r.runId));
      await (await canvas()).getByTestId('local-browser-validate').click();
      // Trace on save already captured the saved edit, so Run model opens the review. Choose Run as a user would.
      await (await canvas()).locator('#execution-sheet').getByRole('button', { name: 'Run', exact: true }).click();
      await (await canvas()).getByTestId('local-browser-validate-cancel').waitFor();
      const running = await until('current validation attempt', () => api.sidebarState().receipts.find(r =>
        r.type === 'validationState' && r.status === 'running' && !previousRuns.has(r.runId)
        && r.target.sourceDigest === target().sourceDigest), 180000);
      await command('tensorviz.hideCanvas');
      const outcome = await until('hidden validation result', () => !api.sidebarState().visible
        && api.sidebarState().receipts.find(r => r.type === 'validationState' && r.runId === running.runId && r.status !== 'running'), 180000);
      assert.equal(outcome.status, 'passed', outcome.detail);
      await command('tensorviz.showSidebarCanvas');
      await (await canvas()).getByRole('button', { name: 'Latest validation: Passed', exact: true }).waitFor();
      assert.equal(api.sidebarState().receipts.find(r => r.type === 'validationState' && r.runId === outcome.runId)?.status, 'passed');
    });
    await check('module expansion and camera survive immediate hide and real iframe recreation', async () => {
      const uri = vscode.Uri.file(path.join(root, 'fixture/complex_models/layered.py'));
      await command('tensorviz.openAsGraph', { uri: uri.toString(), cls: 'Layered' });
      const app = await canvas();
      await until('Layered capture completes', () => api.sidebarState().receipts.find(r =>
        r.type === 'validationState' && r.target.cls === 'Layered' && r.status === 'passed'), 180000);
      const block = app.getByRole('button', { name: 'Expand block', exact: true });
      const blockId = await block.evaluate(el => el.closest('[data-node-id]')?.getAttribute('data-node-id'));
      assert.ok(blockId, 'the capture must contain the imported block');
      // Expanded view opens block inline. Do not focus block itself: an exact
      // focus enters the parent's module-focus scope, where block is folded and
      // block.conv has no card. The capture root is always expanded, so wait on
      // block's own flag.
      await app.evaluate(() => window.dispatchEvent(new CustomEvent('builder-command', { detail: { command: 'view-expanded' } })));
      await until('expanded checkpoint', () => api.sidebarState().presentation?.nodes.find(n => n.id === blockId)?.expanded);
      const childId = await app.getByRole('button', { name: 'Expand block.conv', exact: true })
        .evaluate(el => el.closest('[data-node-id]')?.getAttribute('data-node-id'));
      assert.ok(childId, 'the fixture must contain an actual child to verify expanded selection');
      // Focusing the child opens block in module focus, so the checkpoint also
      // carries a focus scope, its camera and its selection.
      await app.evaluate(nodeId => window.dispatchEvent(new CustomEvent('builder-command', { detail: { command: 'focus-node', payload: { nodeId } } })), childId);
      await until('child selected in block focus', () => {
        const presentation = api.sidebarState().presentation;
        return presentation?.nodes.find(n => n.id === childId)?.selected && presentation.moduleFocus?.scopeId === blockId;
      });
      const previousZoom = api.sidebarState().presentation.camera.z;
      await app.evaluate(() => window.dispatchEvent(new CustomEvent('builder-command', { detail: { command: 'zoom-in' } })));
      // No 100 ms wait: the hide handshake must read the final camera now.
      await command('tensorviz.hideCanvas');
      const saved = api.sidebarState().presentation;
      assert.notEqual(saved.camera.z, previousZoom, 'hide must capture the latest zoom gesture, not the old checkpoint');
      assert.ok(saved.nodes.find(n => n.id === blockId)?.expanded, 'block stays expanded in the checkpoint');
      assert.equal(saved.moduleFocus?.scopeId, blockId);
      await command('tensorviz.showSidebarCanvas');
      const visible = await canvas();
      // Reload after evaluate returns: a reload inside it can detach the frame first.
      await visible.evaluate(() => { setTimeout(() => location.reload(), 0); });
      await until('recreated renderer', () => api.sidebarState().messages.filter(m => m.type === 'ready').length >= 2);
      // Inside block's own scope only its children have cards; block is in the breadcrumb.
      await (await canvas()).locator(`[data-node-id="${childId}"]`).waitFor();
      await until('restored expansion', () => api.sidebarState().presentation?.nodes.find(n => n.id === blockId)?.expanded);
      await pause(400); // Detect delayed initial-fit overriding restoration.
      await command('tensorviz.hideCanvas'); // Read back the recreated editor, not a cached host checkpoint.
      assert.deepEqual(api.sidebarState().presentation.camera, saved.camera);
      assert.deepEqual(api.sidebarState().presentation.selected, saved.selected);
      assert.deepEqual(api.sidebarState().presentation.nodes.filter(n => n.expanded).map(n => n.id), saved.nodes.filter(n => n.expanded).map(n => n.id));
      assert.equal(api.sidebarState().presentation.moduleFocus?.scopeId, saved.moduleFocus.scopeId);
      await command('tensorviz.showSidebarCanvas');
      await (await canvas()).locator(`[data-node-id="${childId}"]`).waitFor();
      await vscode.window.showTextDocument(await vscode.workspace.openTextDocument(uri));
      await command('notifications.clearAll');
      await command('tensorviz.showSidebarCanvas');
      await page.screenshot({ path: path.join(root, 'evidence/sidebar-restored.png') });
    });
  } catch (error) {
    fs.writeFileSync(path.join(root, 'evidence/sidebar-failure.txt'), String(error.stack || error));
    for (const page of browser.contexts().flatMap(c => c.pages())) {
      if (!page.url().includes('workbench')) continue;
      await page.screenshot({ path: path.join(root, 'evidence/sidebar-failure.png'), mask: [page.locator('.monaco-editor')] }).catch(() => {});
      for (const frame of page.frames()) if (frame.url().includes('view=canvas')) {
        fs.writeFileSync(path.join(root, 'evidence/sidebar-body.txt'), await frame.locator('body').innerText());
      }
    }
    fs.writeFileSync(path.join(root, 'evidence/sidebar-state.json'), JSON.stringify(api.sidebarState(), null, 2));
    throw error;
  } finally { await client?.close(); await browser.close(); }
};
