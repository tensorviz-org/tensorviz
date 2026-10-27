/** Staged production environment onboarding with a scripted host; no model execution. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { chromium } = require('../../canvas/node_modules/playwright');
const { isGraphOpeningState } = require('../../packages/contracts/dist/protocol');
const fixture = require('../../canvas/src/test/fixtures/canvas/graphReview.json');
const assets = path.resolve(__dirname, '../media/app');
const output = path.resolve(process.argv[2] || '.tensorviz-workflow/environment-onboarding');
const target = { workspaceFolder: '/qa', file: 'models/mamba.py', cls: 'Mamba', documentVersion: 1,
  sourceDigest: fixture.after.sourceDigest, backendRevision: fixture.after.revision };
const parent = `<!doctype html><style>body{margin:0}iframe{border:0;width:100%;height:100vh;display:block}</style>
<iframe src="/?view=canvas"></iframe><script>
const frame=document.querySelector('iframe');window.uiQa={messages:[],emit:m=>frame.contentWindow.postMessage(m,location.origin)};
addEventListener('message',e=>{if(e.source!==frame.contentWindow||e.origin!==location.origin)return;
uiQa.messages.push(e.data);if(e.data.type==='ready')uiQa.emit({type:'hostReady',protocolVersion:1,theme:'dark'});
if(e.data.type==='canvasConnect')uiQa.emit({type:'canvasConnected',sessionId:'environment-qa',viewId:'environment-view',theme:'dark'});});
</script>`;
async function run() {
  fs.mkdirSync(output, { recursive: true }); fs.rmSync(path.join(output, 'result.json'), { force: true });
  const missing = [], errors = [], external = [], screenshots = [];
  const server = http.createServer((req, res) => {
    const route = new URL(req.url, 'http://localhost').pathname;
    if (route === '/__qa_parent') { res.setHeader('Content-Type', 'text/html'); return res.end(parent); }
    const file = path.resolve(assets, '.' + (route === '/' ? '/index.html' : route));
    if (!file.startsWith(assets + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) { missing.push(route); res.statusCode = 404; return res.end(); }
    const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.woff2': 'font/woff2' };
    res.setHeader('Content-Type', mime[path.extname(file)] || 'application/octet-stream'); fs.createReadStream(file).pipe(res);
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const origin = `http://127.0.0.1:${server.address().port}`;
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 360, height: 800 } }); page.setDefaultTimeout(10000);
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/*', route => { if (route.request().url().startsWith(origin + '/')) return route.continue();
    external.push(route.request().url()); return route.abort(); });
  const frame = page.frameLocator('iframe');
  let sequence = 0;
  const host = message => page.evaluate(message => uiQa.emit(message), { type: 'canvasMessage', message: {
    ...message, protocolVersion: 1, workspaceSessionId: 'environment-qa', messageId: `m-${++sequence}`, hostSequence: sequence } });
  const actions = () => page.evaluate(() => uiQa.messages.filter(m => m.type === 'canvasMessage' && m.message.type === 'graphSetupRequest').map(m => m.message));
  const panel = frame.getByRole('region', { name: 'Python environment for model' });
  const candidate = { id: 'saved', interpreter: '/qa/runtime/venv/bin/python', prefix: '/qa/runtime/venv',
    source: 'tensorViz fallback', status: 'libraries_present', python: '3.12.2', torch: '2.7.1', selected: true };
  const setup = { id: 'mamba-opening', target, phase: 'needs_setup', hasCapture: false,
    detail: 'Required model libraries are not installed: triton. The saved environment has the capture libraries, but cannot import this model.',
    environment: { reason: 'missing_libraries', selectedId: 'saved', canCreate: true, requirements: ['triton>=3.5.0'],
      detail: 'Required model libraries are not installed: triton. The saved environment has the capture libraries, but cannot import this model.',
      candidates: [candidate, ...Array.from({ length: 12 }, (_, i) => ({ ...candidate, id: `other-${i}`, interpreter: `/qa/other-${i}/bin/python`, prefix: `/qa/other-${i}`, source: 'Known Python', selected: undefined }))] } };
  const open = state => { assert.equal(isGraphOpeningState(state), true, 'scripted host state obeys the production contract'); return host({ type: 'graphOpening', state }); };
  const shot = async name => { screenshots.push(name + '.png'); await page.screenshot({ path: path.join(output, name + '.png'), animations: 'disabled' }); };
  async function reachable(button) {
    assert.equal(await frame.locator('html').evaluate(node => node.scrollWidth > innerWidth), false, 'no document overflow');
    for (const selector of ['.graph-environment-panel', '.graph-environment-body'])
      assert.equal(await frame.locator(selector).evaluate(node => node.scrollWidth > node.clientWidth + 1), false, selector + ' fits');
    assert.equal(await button.evaluate(node => { const r = node.getBoundingClientRect();
      const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
      return r.top >= 0 && r.bottom <= innerHeight && r.left >= 0 && r.right <= innerWidth && node.contains(hit);
    }), true, 'primary action remains reachable');
    assert.equal(await panel.locator('.graph-environment-body').evaluate(node => node.clientHeight > 40), true, 'scrolling body retains usable space');
  }
  try {
    await page.goto(origin + '/__qa_parent'); await frame.getByRole('heading', { name: 'Explore your model' }).waitFor();
    await host({ type: 'activeTarget', target });
    await host({ type: 'projection', target, projection: { ...fixture.after, file: target.file, traced: false }, traced: false, greyed: false });
    await open(setup);
    const repair = panel.getByRole('button', { name: 'Add libraries here…', exact: true });
    assert.equal(await panel.getByText('Setup details', { exact: true }).locator('..').getAttribute('open'), null);
    assert.equal(await panel.getByText(setup.detail, { exact: true }).isVisible(), false);
    assert.equal(await panel.getByRole('button', { name: 'Create model environment…', exact: true }).isVisible(), false);
    assert.equal(await repair.evaluate(node => !!node.closest('footer')), true);
    for (const size of [{ width: 360, height: 800 }, { width: 200, height: 800 }, { width: 320, height: 320 }]) {
      await page.setViewportSize(size); await reachable(repair); await shot(`repair-${size.width}x${size.height}-dark`);
    }
    await page.setViewportSize({ width: 360, height: 800 });
    await panel.getByText('Change environment', { exact: true }).click();
    await panel.locator('.graph-environment-body').evaluate(node => { node.scrollTop = node.scrollHeight; });
    assert.equal(await panel.locator('.graph-environment-body').evaluate(node => node.scrollTop > 0), true);
    await reachable(repair); await shot('scrolled-360-dark'); await repair.click();
    await page.waitForFunction(() => uiQa.messages.some(m => m.type === 'canvasMessage' && m.message.action?.kind === 'install_dependencies'));
    const first = (await actions())[0]; assert.deepEqual(first.action, { kind: 'install_dependencies' });
    assert.equal(first.openingId, setup.id); assert.deepEqual(first.target, target);
    for (const detail of ['Resolving and downloading compatible model libraries…', 'Waiting for package approval in VS Code…', 'Downloading and installing the reviewed libraries…', 'Checking capture tools and required model libraries…']) {
      await open({ ...setup, phase: 'resolving', detail });
      await panel.locator('footer [role="status"]').filter({ hasText: detail }).waitFor();
      assert.equal(await panel.getByRole('progressbar').getAttribute('aria-valuenow'), null);
      assert.equal(await panel.getByRole('button', { name: 'Preparing environment…' }).isDisabled(), true);
      await page.setViewportSize({ width: 320, height: 320 });
      await reachable(panel.getByRole('button', { name: 'Preparing environment…' }));
      await page.setViewportSize({ width: 360, height: 800 });
    }
    await shot('checking-360-dark'); await panel.getByRole('button', { name: 'Stop', exact: true }).click();
    await page.waitForFunction(() => uiQa.messages.some(m => m.type === 'canvasMessage' && m.message.action?.kind === 'cancel'));
    await open({ ...setup, phase: 'cancelled', detail: 'Opening cancelled.' });
    await panel.getByRole('heading', { name: 'Environment setup stopped' }).waitFor();
    await panel.getByRole('button', { name: 'Retry library setup…' }).waitFor(); await shot('stopped-360-dark');
    const failed = { ...setup, phase: 'failed', failure: { kind: 'environment', code: 'PACKAGE_UNAVAILABLE' },
      detail: 'triton>=3.5.0: No distribution is available for this Python/platform. Choose a compatible environment, or inspect source without running the model.' };
    await open(failed); await panel.getByRole('heading', { name: 'Environment setup failed' }).waitFor();
    assert.equal(await panel.locator('footer [role="alert"]').getByText(failed.detail).isVisible(), true);
    assert.equal((await actions()).filter(m => m.action.kind === 'install_dependencies').length, 1, 'no automatic repair retry');
    for (const size of [{ width: 360, height: 800 }, { width: 200, height: 800 }, { width: 320, height: 320 }]) {
      await page.setViewportSize(size); await reachable(panel.getByRole('button', { name: 'Inspect source only', exact: true }));
      await shot(`failed-${size.width}x${size.height}-dark`);
    }
    await page.setViewportSize({ width: 360, height: 800 });
    await frame.getByRole('switch', { name: 'Dark canvas appearance' }).click(); await shot('failed-360-light');
    await panel.getByRole('button', { name: 'Inspect source only', exact: true }).click();
    await page.waitForFunction(() => uiQa.messages.some(m => m.type === 'canvasMessage' && m.message.action?.kind === 'source_only'));
    await panel.getByText('Change environment', { exact: true }).click();
    await panel.getByRole('radio').nth(1).check(); await panel.getByRole('button', { name: 'Use qa/other-0 instead' }).click();
    await page.waitForFunction(() => uiQa.messages.some(m => m.type === 'canvasMessage' && m.message.action?.kind === 'select_environment'));
    assert.deepEqual((await actions()).at(-1).action, { kind: 'select_environment', candidateId: 'other-0' });
    await panel.getByRole('button', { name: 'Browse…', exact: true }).click();
    await page.waitForFunction(() => uiQa.messages.some(m => m.type === 'canvasMessage' && m.message.action?.kind === 'environment'));
    assert.equal((await actions()).at(-1).openingId, setup.id);
    assert.equal((await actions()).filter(m => m.action.kind === 'install_dependencies').length, 1, 'Browse requests selection, not installation');
    const browsed = { ...setup, detail: 'This Python needs capture libraries.', environment: { ...setup.environment, selectedId: 'browsed',
      detail: 'This Python needs capture libraries.', requirements: ['torch', 'pydantic'], candidates: [
        { ...candidate, id: 'browsed', interpreter: '/qa/manual-env/bin/python', prefix: '/qa/manual-env', source: 'Saved for this folder',
          status: 'missing_libraries', torch: null, missing: ['torch', 'pydantic'] }, ...setup.environment.candidates] } };
    await open({ ...setup, phase: 'resolving', detail: 'Checking the selected Python environment…' });
    await open(browsed);
    await panel.locator('footer').getByText('qa/manual-env', { exact: true }).waitFor();
    assert.equal(await panel.getByRole('progressbar').count(), 0, 'selection does not start installation');
    await reachable(repair); await shot('browsed-selected-360-light'); await repair.click();
    await page.waitForFunction(() => uiQa.messages.filter(m => m.type === 'canvasMessage' && m.message.action?.kind === 'install_dependencies').length === 2);
    await open({ ...browsed, phase: 'resolving', detail: 'Downloading and installing the reviewed libraries…' });
    await panel.getByRole('progressbar').waitFor(); await reachable(panel.getByRole('button', { name: 'Preparing environment…' }));
    await shot('browsed-installing-360-light');
    await open({ ...setup, environment: { ...setup.environment, selectedId: 'other-0' }, phase: 'failed', failure: { kind: 'environment', code: 'DOWNLOAD_FAILED' }, detail: 'Library download could not connect.' });
    await panel.getByRole('button', { name: 'Retry library setup…' }).click();
    await page.waitForFunction(() => uiQa.messages.filter(m => m.type === 'canvasMessage' && m.message.action?.kind === 'install_dependencies').length === 3);
    const name = panel.getByRole('textbox', { name: 'Environment name' });
    if (!await name.isVisible()) await panel.getByText('Change environment', { exact: true }).click();
    await name.fill('mamba-next'); await panel.getByRole('button', { name: 'Create model environment…', exact: true }).click();
    await page.waitForFunction(() => uiQa.messages.some(m => m.type === 'canvasMessage' && m.message.action?.kind === 'create_environment'));
    assert.deepEqual((await actions()).at(-1).action, { kind: 'create_environment', name: 'mamba-next' });
    await open({ ...setup, phase: 'resolving', environmentName: 'mamba-next', detail: 'Checking required libraries…' });
    await panel.getByRole('heading', { name: 'Preparing mamba-next' }).waitFor();
    await open({ ...setup, environmentName: 'mamba-next', detail: 'Add an example call.', environment: { ...setup.environment,
      reason: 'ready', selectedId: 'new', detail: 'Ready', candidates: [{ ...candidate, id: 'new', prefix: '/qa/model-environments/model-123/mamba-next',
        interpreter: '/qa/model-environments/model-123/mamba-next/bin/python' }] },
      discovery: { model: 'models.mamba:Mamba', sourceDigest: target.sourceDigest, init: [{ name: 'width', kind: 'keyword', required: true }],
        inputs: [{ name: 'x', kind: 'keyword', required: true }], examples: [], factories: [], configs: [], parents: [], constraints: [], truncated: false } });
    await frame.getByText('Using mamba-next · Python 3.12.2 · Torch 2.7.1', { exact: true }).waitFor(); await frame.getByText('Required libraries available · configure the example call below.', { exact: true }).waitFor(); await shot('ready-model-setup-360-light');
    assert.deepEqual(errors, []); assert.deepEqual(external, []); assert.deepEqual(missing, []);
    const checks = ['compact details and secondary editable creation', 'saved-Python repair in fixed footer after scrolling 13 candidates',
      'real host stages and indeterminate progress', 'Stop and retained retry state', 'platform failure in fixed footer without automatic retry',
      'source inspection routed', 'different Python selectable after failure', 'Browse selects before Canvas installation and progress', 'explicit library retry', 'explicit named creation still supported',
      'verified library success advances to input form', '200px/360px and short-window primary reachability', 'dark/light appearance', 'no runtime/asset/external-request errors'];
    const result = { passed: true, scope: 'scripted host with staged production renderer; no package install, model execution or installed VS Code claim',
      rendererSha256: createHash('sha256').update(fs.readFileSync(path.join(assets, 'index.html'))).digest('hex'), checks, screenshots };
    fs.writeFileSync(path.join(output, 'result.json'), JSON.stringify(result, null, 2) + '\n'); console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    await shot('failed-driver');
    console.error(await panel.evaluate(node => Array.from(node.querySelectorAll('header,.graph-environment-body,footer,footer button')).map(n => ({ cls: n.className,
      height: n.getBoundingClientRect().height, display: getComputedStyle(n).display, padding: getComputedStyle(n).padding, margin: getComputedStyle(n).margin }))));
    fs.writeFileSync(path.join(output, 'failure.json'), JSON.stringify(await page.evaluate(() => ({ messages: uiQa.messages,
      text: document.querySelector('iframe').contentDocument.body.innerText })), null, 2) + '\n');
    throw error;
  } finally { await browser.close(); await new Promise(resolve => server.close(resolve)); }
}
run().catch(error => { console.error(error); process.exitCode = 1; });
