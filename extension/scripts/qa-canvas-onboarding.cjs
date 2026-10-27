/** Compiled native home + real staged renderer; scripted host, no model execution. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const { chromium } = require('../../canvas/node_modules/playwright');
require('../out/test/bootstrap');
const { canvasHomeHtml, canvasHomeParts } = require('../out/canvasHome');
const { shellHtml } = require('../out/shell');
const { emptyReviewProjection } = require('@tensorviz/graph-host/graphReviewCore');
const output = path.resolve(process.argv[2] || '.tensorviz-workflow/canvas-onboarding');
const assets = path.resolve(__dirname, '../media/app');
const home = { hasWorkspace: true, indexed: true, models: [],
  setup: { phase: 'checking', detail: 'Checking the tensorViz runtime. You can choose a model or PR while checks run.' } };
const review = { ...require('../../canvas/src/test/fixtures/canvas/graphReview.json'), id: 'onboarding-pr',
  title: 'Fixture pull request', beforeLabel: 'main', afterLabel: 'PR', changedFiles: ['model.py'],
  models: [], before: emptyReviewProjection('PR'), after: emptyReviewProjection('PR'),
  stale: false, notes: ['Dynamic source could not be resolved.'] };
const theme = `:root{--vscode-font-family:-apple-system,BlinkMacSystemFont,sans-serif;--vscode-font-size:13px;
  --vscode-sideBar-background:#181818;--vscode-editor-background:#1f1f1f;--vscode-foreground:#ccc;
  --vscode-descriptionForeground:#9d9d9d;--vscode-textLink-foreground:#3794ff;--vscode-focusBorder:#3794ff;
  --vscode-button-background:#0078d4;--vscode-button-foreground:white;--vscode-button-hoverBackground:#0078d4;
  --vscode-button-secondaryBackground:#313131;--vscode-button-secondaryForeground:#ccc;
  --vscode-widget-border:#454545;--vscode-list-hoverBackground:#2a2d2e}`;

async function run() {
  assert.ok(fs.existsSync(path.join(assets, 'index.html')), 'build:webview must stage the renderer first');
  fs.mkdirSync(output, { recursive: true });
  fs.rmSync(path.join(output, 'result.json'), { force: true });
  fs.rmSync(path.join(output, 'failed.png'), { force: true });
  let origin;
  const server = http.createServer((req, res) => {
    const route = new URL(req.url, 'http://localhost').pathname;
    if (route === '/__home' || route === '/__shell') {
      res.setHeader('Content-Type', 'text/html');
      return res.end(route === '/__home' ? canvasHomeHtml(home) : shellHtml({ appUrl: origin + '/?view=canvas',
        appOrigin: origin, port: server.address().port, workspaceDir: '/fixture' }, home));
    }
    const file = path.resolve(assets, '.' + (route === '/' ? '/index.html' : route));
    if (!file.startsWith(assets + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) { res.statusCode = 404; return res.end(); }
    res.setHeader('Content-Type', { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.woff2': 'font/woff2' }[path.extname(file)] || 'application/octet-stream');
    fs.createReadStream(file).pipe(res);
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  origin = `http://127.0.0.1:${server.address().port}`;
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 360, height: 760 } });
  page.setDefaultTimeout(8000);
  const errors = [], external = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.addInitScript(() => {
    window.hostPosts = []; let acquired = false;
    window.acquireVsCodeApi = () => {
      if (acquired) throw new Error('VS Code API acquired twice'); acquired = true;
      return { postMessage: message => window.hostPosts.push(message) };
    };
  });
  await page.route('**/*', route => {
    if (route.request().url().startsWith(origin + '/')) return route.continue();
    external.push(route.request().url()); return route.abort();
  });
  const emit = message => page.evaluate(message => window.postMessage(message, location.origin), message);
  const update = async state => emit({ type: 'canvasHomeState', html: canvasHomeParts(state).body });
  const shot = name => page.screenshot({ path: path.join(output, name + '.png'), animations: 'disabled' });
  try {
    await page.goto(origin + '/__home'); await page.addStyleTag({ content: theme });
    await page.getByRole('button', { name: 'Open PR', exact: true }).click();
    assert.deepEqual(await page.evaluate(() => window.hostPosts), [{ type: 'homeReady' }, { type: 'homeAction', action: 'pr' }]);
    await shot('checking');
    const missing = { ...home, setup: { phase: 'needs_setup', detail: 'Set up tensorViz’s own Python runtime once to open graphs.' } };
    await page.getByRole('button', { name: 'Open as Graph' }).focus();
    await update(missing); await page.getByText('Set up tensorViz’s own Python runtime once to open graphs.', { exact: true }).waitFor();
    assert.equal(await page.getByRole('button', { name: 'Choose Python' }).count(), 0);
    assert.equal(await page.locator(':focus').getAttribute('data-action'), 'open', 'readiness updates preserve action focus');
    await page.getByRole('button', { name: 'Set up tensorViz' }).click();
    await shot('missing');
    await update({ ...missing, error: 'tensorViz could not start. Your selected opening is retained.', retryAvailable: true });
    await page.getByRole('button', { name: 'Retry opening' }).click();
    await shot('recovery');
    await update({ ...missing, retryAvailable: true });
    await page.getByRole('button', { name: 'Continue opening' }).click();
    await update({ ...missing, error: 'tensorViz could not start. Your selected opening is retained.', retryAvailable: true });
    await page.setViewportSize({ width: 200, height: 760 });
    assert.equal(await page.locator('html').evaluate(node => node.scrollWidth > innerWidth), false);
    await shot('recovery-narrow');
    await page.setViewportSize({ width: 360, height: 760 });
    await update({ ...home, setup: { phase: 'untrusted', detail: 'Trust this workspace to check the tensorViz runtime.' } });
    await page.getByRole('button', { name: 'Manage Workspace Trust' }).waitFor();
    assert.equal(await page.getByRole('button', { name: 'Choose Python' }).count(), 0);
    await page.goto(origin + '/__shell'); await page.addStyleTag({ content: theme });
    await page.waitForFunction(() => window.hostPosts.some(message => message.type === 'canvasConnect'));
    await emit({ type: 'canvasConnected', sessionId: 'onboarding-qa', viewId: 'onboarding-qa-view' });
    const frame = page.frameLocator('#app');
    await frame.locator('h1').filter({ hasText: 'Explore your model' }).waitFor({ state: 'attached' });
    assert.equal(await page.locator('#app').isVisible(), false, 'connected React welcome stays hidden');
    assert.equal(await page.getByRole('heading', { name: 'Explore your model' }).count(), 1);
    await update({ ...home, setup: { phase: 'ready', detail: 'Choose a model or open a pull request. Model dependencies are checked after you select a model.' } });
    await page.getByText('tensorViz is ready', { exact: true }).waitFor(); await shot('connected-home');
    // The child cannot inject host-only HTML or show its own second starting screen.
    await frame.locator('body').evaluate(() => parent.postMessage({ type: 'canvasContent', visible: true }, location.origin));
    assert.equal(await page.locator('#app').isVisible(), false);
    await page.locator('body').evaluate(node => node.setAttribute('data-vscode-theme-kind', 'vscode-light'));
    await frame.locator('html[data-tv-theme="light"]').waitFor({ state: 'attached' });
    await emit({ type: 'canvasMessage', message: { type: 'graphReviewState', state: review,
      protocolVersion: 1, workspaceSessionId: 'onboarding-qa', messageId: 'review-1' } });
    await emit({ type: 'canvasContent', visible: true });
    await frame.getByRole('heading', { name: 'No model identified' }).waitFor();
    assert.equal(await page.locator('#canvas-home').isVisible(), false);
    assert.equal(await page.locator('body').getAttribute('data-motion'), 'paused'); await shot('review');
    await emit({ type: 'canvasMessage', message: { type: 'graphReviewState', state: null,
      protocolVersion: 1, workspaceSessionId: 'onboarding-qa', messageId: 'review-2' } });
    await emit({ type: 'canvasContent', visible: false });
    await page.getByRole('heading', { name: 'Explore your model' }).waitFor();
    assert.equal(await page.locator('#app').isVisible(), false);
    const target = { workspaceFolder: '/fixture', file: 'models/qwen3/model.py', cls: 'Qwen3Model',
      documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 0 };
    const opening = { id: 'missing-project-library', target, phase: 'needs_setup', hasCapture: false,
      detail: 'This project already declares torch_remat, but its dependency source is not supported by the guided installer.',
      environment: { reason: 'missing_libraries', selectedId: 'managed', requirements: [], missingModules: ['torch_remat'],
        detail: 'This project already declares torch_remat, but its dependency source is not supported by the guided installer.',
        candidates: [{ id: 'managed', interpreter: '/fixture/runtime/venv/bin/python', source: 'tensorViz fallback', managed: true,
          python: '3.12.14', torch: '2.7.1', status: 'libraries_present' }] } };
    await emit({ type: 'canvasMessage', message: { type: 'activeTarget', target,
      protocolVersion: 1, workspaceSessionId: 'onboarding-qa', messageId: 'missing-target', hostSequence: 1 } });
    await emit({ type: 'canvasMessage', message: { type: 'graphOpening', state: opening,
      protocolVersion: 1, workspaceSessionId: 'onboarding-qa', messageId: 'missing-opening', hostSequence: 2 } });
    await emit({ type: 'canvasContent', visible: true });
    const panel = frame.getByRole('region', { name: 'Python environment for model' });
    await panel.getByRole('heading', { name: 'Model libraries are missing' }).waitFor();
    assert.equal(await panel.getByRole('button', { name: 'Check again with this environment' }).count(), 0);
    assert.equal(await panel.getByText(/missing libraries above belong to this model/).isVisible(), true);
    for (const width of [360, 200]) {
      await page.setViewportSize({ width, height: 760 });
      assert.equal(await panel.evaluate(node => node.scrollWidth > node.clientWidth + 1), false, 'recovery fits sidebar');
      assert.equal(await panel.getByRole('button', { name: 'Use Agent', exact: true }).evaluate(node => {
        const bounds = node.getBoundingClientRect(); return bounds.top >= 0 && bounds.bottom <= innerHeight;
      }), true, 'help stays visible without scrolling');
      await shot('missing-project-library-' + width);
    }
    const requests = async () => page.evaluate(() => window.hostPosts.filter(message => message.type === 'canvasMessage'
      && message.message.type === 'graphSetupRequest').map(message => message.message));
    const waitAction = kind => page.waitForFunction(kind => window.hostPosts.some(message => message.type === 'canvasMessage'
      && message.message.type === 'graphSetupRequest' && message.message.action.kind === kind), kind);
    await panel.getByRole('button', { name: 'Use Agent', exact: true }).click();
    await waitAction('agent_help');
    assert.equal((await requests()).at(-1).action.kind, 'agent_help');
    // The scripted host acknowledges help; it does not run a coding agent.
    await emit({ type: 'canvasMessage', message: { type: 'graphOpening', state: opening,
      protocolVersion: 1, workspaceSessionId: 'onboarding-qa', messageId: 'help-ack', hostSequence: 3 } });
    await frame.getByRole('button', { name: 'Change inputs', exact: true }).click();
    await panel.getByRole('button', { name: 'Inspect source only' }).click();
    await waitAction('source_only');
    assert.equal((await requests()).at(-1).action.kind, 'source_only');
    assert.equal((await requests()).every(message => message.openingId === opening.id), true, 'actions keep the opening identity');
    assert.equal((await requests()).some(message => ['submit', 'install_dependencies', 'select_environment'].includes(message.action.kind)), false);
    await frame.getByRole('button', { name: 'Minimize agent chat', exact: true }).click();
    const createOpening = { ...opening, environment: { ...opening.environment, canCreate: true } };
    await emit({ type: 'canvasMessage', message: { type: 'graphOpening', state: createOpening,
      protocolVersion: 1, workspaceSessionId: 'onboarding-qa', messageId: 'create-offer', hostSequence: 4 } });
    const create = panel.getByRole('button', { name: 'Create model environment…', exact: true });
    for (const width of [360, 200]) {
      await page.setViewportSize({ width, height: 760 });
      await create.waitFor();
      assert.equal(await panel.evaluate(node => node.scrollWidth > node.clientWidth + 1), false, 'creation fits sidebar');
      assert.equal(await create.evaluate(node => { const bounds = node.getBoundingClientRect(); return bounds.top >= 0 && bounds.bottom <= innerHeight; }), true);
      await shot('create-model-environment-' + width);
    }
    await create.click(); await waitAction('create_environment');
    assert.equal((await requests()).at(-1).openingId, opening.id);
    await emit({ type: 'canvasMessage', message: { type: 'graphOpening', state: { ...createOpening, phase: 'resolving', detail: 'Downloading and installing the reviewed libraries…' },
      protocolVersion: 1, workspaceSessionId: 'onboarding-qa', messageId: 'create-progress', hostSequence: 5 } });
    await panel.getByRole('status').filter({ hasText: 'Downloading and installing' }).waitFor();
    assert.equal(await panel.getByRole('button', { name: 'Preparing environment…' }).isDisabled(), true);
    assert.equal(await panel.getByRole('button', { name: 'Cancel', exact: true }).isEnabled(), true);
    await shot('create-model-environment-progress');
    await panel.getByRole('button', { name: 'Cancel', exact: true }).click();
    await waitAction('cancel');
    assert.deepEqual(errors, []); assert.deepEqual(external, []);
    const result = { status: 'passed', scope: 'compiled native home and shell plus staged production React in Chromium; scripted host, not installed VS Code',
      checks: ['Open PR during runtime checks', 'home offers only owned runtime setup, no project Python picker', 'retained opening retry and continuation after setup',
        'readiness updates preserve keyboard focus',
        '200px recovery layout', 'workspace trust recovery', 'one welcome before and after service connection',
        'child cannot inject host home messages', 'live theme relay', 'PR handoff without project Python', 'return to same home',
        'hidden-home motion pause', 'unsupported model dependencies distinguish capture libraries',
        'unsupported dependencies promote help without repeating the same environment check', 'source-only action retains opening identity',
        '200px model recovery layout', 'native environment creation is reachable without an agent at 200px and 360px',
        'creation retains the opening identity', 'download progress prevents duplicates and keeps cancellation', 'no external assets or JavaScript errors'],
      screenshots: fs.readdirSync(output).filter(file => file.endsWith('.png')) };
    fs.writeFileSync(path.join(output, 'result.json'), JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    await shot('failed');
    console.error(JSON.stringify({ errors, external, hostPosts: await page.evaluate(() => window.hostPosts),
      frames: await Promise.all(page.frames().map(frame => frame.evaluate(() => ({ theme: document.documentElement.dataset.tvTheme,
        headings: [...document.querySelectorAll('h1')].map(node => node.textContent) })))) }, null, 2));
    throw error;
  } finally { await browser.close(); await new Promise(resolve => server.close(resolve)); }
}
run().catch(error => { console.error(error); process.exitCode = 1; });
