/** Installed-artifact starting screen acceptance, without provisioning or model execution. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vscode = require('vscode');
const { chromium } = require('../../canvas/node_modules/playwright');

async function until(read) {
  const deadline = Date.now() + 60000;
  while (Date.now() < deadline) {
    const value = await read();
    if (value) return value;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error('Starting screen did not appear in the native secondary sidebar.');
}

exports.run = async () => {
  const root = process.env.TV_QA_ROOT;
  assert.equal(process.env.TENSORVIZ_TEST_PRESEED_MANAGED, '0');
  const extension = vscode.extensions.getExtension('tensorviz.tensorviz');
  assert.ok(extension.extensionPath.startsWith(path.join(root, 'extensions') + path.sep));
  const manifest = JSON.parse(fs.readFileSync(path.join(extension.extensionPath, 'package.json'), 'utf8'));
  assert.equal(manifest.contributes.views['tensorviz-canvas'][0].when, undefined);
  assert.equal(manifest.contributes.viewsContainers.secondarySidebar[0].title, 'tensorViz');
  const browser = await chromium.connectOverCDP(`http://127.0.0.1:${process.env.TV_QA_DEBUG_PORT}`);
  try {
    const api = (await extension.activate()).__test;
    const runtime = path.join(api.globalStorageDir(), 'runtime');
    assert.equal(api.backendOrigin(), undefined);
    assert.ok(!fs.existsSync(runtime));
    await vscode.commands.executeCommand('workbench.action.closeAllEditors');
    // The generated native focus command is available before any graph request.
    await vscode.commands.executeCommand('tensorviz.sidebarCanvas.focus');
    await until(() => api.sidebarState()?.homeVisible);
    const page = await until(() => browser.contexts().flatMap(context => context.pages())
      .find(candidate => candidate.url().includes('workbench')));
    await page.locator('[id="workbench.parts.auxiliarybar"]').waitFor({ state: 'visible' });
    assert.ok(await page.locator('.auxiliarybar').getByText('tensorViz', { exact: true }).count());
    assert.equal(api.sidebarState().ready, false, 'the renderer/backend must remain unstarted');
    assert.equal(api.backendOrigin(), undefined);
    assert.ok(!fs.existsSync(runtime));
    await new Promise(resolve => setTimeout(resolve, 500));
    await page.screenshot({ path: path.join(root, 'evidence/canvas-home-first.png') });
    await vscode.commands.executeCommand('tensorviz.hideCanvas');
    await until(() => !api.sidebarState().visible);
    await vscode.commands.executeCommand('tensorviz.showSidebarCanvas');
    await until(() => api.sidebarState().homeVisible);
    await page.locator('[id="workbench.parts.auxiliarybar"]').waitFor({ state: 'visible' });
    await page.locator('[id="workbench.parts.auxiliarybar"]').getByText('tensorViz', { exact: true }).first()
      .waitFor({ state: 'visible' });
    assert.equal(api.backendOrigin(), undefined);
    assert.ok(!fs.existsSync(runtime));
    await new Promise(resolve => setTimeout(resolve, 500));
    await page.screenshot({ path: path.join(root, 'evidence/canvas-home.png') });
    const evidence = { status: 'passed', version: manifest.version, runtimePresent: false,
      nativeSidebar: true, rendererDomInspected: false, screenshotReviewRequired: true,
      checks: ['unconditional sidebar contribution', 'host starting-screen state and native sidebar before graph request',
        'no provisioning on first resolution', 'native hide and reopen preserve entry without backend startup'] };
    fs.writeFileSync(path.join(root, 'evidence/canvas-home.json'), JSON.stringify(evidence, null, 2));
    console.log('PASS installed sidebar starting screen without runtime setup');
  } finally { await browser.close(); }
};
