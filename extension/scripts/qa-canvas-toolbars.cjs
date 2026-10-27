/** Staged production renderer with scripted host/model data; no Python or real PR. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { chromium } = require('../../canvas/node_modules/playwright');
const { expect } = require('../../canvas/node_modules/@playwright/test');
const fixture = require('../../canvas/src/test/fixtures/canvas/graphReview.json');
const capture = require('../../canvas/src/test/fixtures/canvas/capture-document.json');
const assets = path.resolve(__dirname, '../media/app');
const output = path.resolve(process.argv[2] || '.tensorviz-workflow/canvas-toolbars');
const target = { workspaceFolder: '/qa', file: 'models/encoder.py', cls: 'MlpTiny', documentVersion: 1,
  sourceDigest: fixture.after.sourceDigest, backendRevision: fixture.after.revision };
const projection = { ...fixture.after, file: target.file, traced: true, document: capture };
const opening = { id: 'toolbar-opening', target, phase: 'ready', hasCapture: true, detail: 'Captured.',
  configFileExists: true, draft: { profileName: 'cpu_example', profile: {} } };
const review = { ...fixture, id: 'toolbar-review', title: 'Improve encoder projection', beforeLabel: 'main', afterLabel: 'PR #42',
  changedFiles: ['model.py'], stale: false, updated: false, notes: ['Dynamic source could not be resolved.'],
  models: [{ id: 'mlp', label: 'MlpTiny', file: 'model.py', status: 'modified' }], selectedModel: 'mlp' };
const parent = `<!doctype html><style>body{margin:0}iframe{border:0;width:100%;height:100vh;display:block}</style>
<iframe src="/?view=canvas"></iframe><script>
const frame=document.querySelector('iframe');window.uiQa={messages:[],emit:m=>frame.contentWindow.postMessage(m,location.origin)};
addEventListener('message',e=>{if(e.source!==frame.contentWindow||e.origin!==location.origin)return;
uiQa.messages.push(e.data);if(e.data.type==='ready')uiQa.emit({type:'hostReady',protocolVersion:1,theme:'dark'});
if(e.data.type==='canvasConnect')uiQa.emit({type:'canvasConnected',sessionId:'toolbar-qa',viewId:'toolbar-view',theme:'dark'});});
</script>`;
async function run() {
  fs.mkdirSync(output, { recursive: true });
  const missing = [];
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
  const page = await browser.newPage({ viewport: { width: 360, height: 800 } });
  const errors = [], external = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/*', route => { if (route.request().url().startsWith(origin + '/')) return route.continue();
    external.push(route.request().url()); return route.abort(); });
  const frame = page.frameLocator('iframe');
  let sequence = 0;
  const emit = message => page.evaluate(message => window.uiQa.emit(message), message);
  const host = message => emit({ type: 'canvasMessage', message: { ...message, protocolVersion: 1,
    workspaceSessionId: 'toolbar-qa', messageId: `m-${++sequence}`, hostSequence: sequence } });
  const shot = name => page.screenshot({ path: path.join(output, name + '.png'), animations: 'disabled' });
  async function reachable() {
    assert.equal(await frame.locator('html').evaluate(node => node.scrollWidth > innerWidth), false, 'no document overflow');
    assert.deepEqual(await frame.locator('.local-browser-header button, .graph-review-toolbar button, .graph-review-toolbar select').evaluateAll(nodes =>
      nodes.filter(node => node.getClientRects().length && (node.getBoundingClientRect().left < 0 || node.getBoundingClientRect().right > innerWidth + 1))
        .map(node => node.getAttribute('aria-label') || node.textContent)), [], 'toolbar controls fit the viewport');
    assert.deepEqual(await frame.locator('.local-browser-header button').evaluateAll(nodes =>
      nodes.filter(node => node.getClientRects().length && node.scrollWidth > node.clientWidth + 1)
        .map(node => node.getAttribute('aria-label') || node.textContent)), [], 'labels must not overflow compact icon buttons');
  }
  async function request(type, action) {
    await page.waitForFunction(({ type, action }) => uiQa.messages.some(m => m.type === 'canvasMessage' && m.message.type === type
      && (!action || m.message.action?.kind === action || m.message.action === action)), { type, action });
  }
  try {
    await page.goto(origin + '/__qa_parent');
    await frame.getByRole('heading', { name: 'Explore your model' }).waitFor();
    await host({ type: 'activeTarget', target });
    await host({ type: 'projection', target, projection, traced: true, greyed: false });
    await host({ type: 'syncState', target, state: 'runtime-traced' });
    await host({ type: 'graphOpening', state: opening });
    await frame.getByTestId('local-browser-strip').getByRole('button', { name: 'Change inputs' }).waitFor();
    await frame.getByRole('button', { name: 'Layers', exact: true }).waitFor();
    for (const width of [1280, 360, 320, 200]) {
      await page.setViewportSize({ width, height: 800 }); await reachable();
      await shot('toolbar-active-' + width + '-dark');
    }
    await page.setViewportSize({ width: 360, height: 800 });
    await frame.getByRole('switch', { name: 'Dark canvas appearance' }).click();
    await shot('toolbar-active-light');
    await frame.getByRole('button', { name: 'Expand toolbar' }).click();
    await frame.getByRole('button', { name: 'More tools' }).waitFor(); await reachable();
    await shot('toolbar-expanded-light');
    await frame.getByRole('button', { name: 'Model and file actions' }).click();
    await frame.getByRole('button', { name: 'Export graph' }).waitFor();
    await frame.getByRole('button', { name: 'Model and file actions' }).press('Escape');
    assert.equal(await frame.getByRole('button', { name: 'Model and file actions' }).evaluate(node => document.activeElement === node), true);
    await frame.getByTestId('local-browser-strip').getByRole('button', { name: 'Change inputs' }).click(); await request('graphSetupRequest', 'change_inputs');
    await frame.getByRole('button', { name: 'Collapse toolbar' }).click();
    const execution = { id: 'scroll-review', phase: 'review', kind: 'capture', device: 'cpu', configuredDevice: 'cpu',
      detail: 'Review', interpreter: '/qa/.venv/bin/python', inputs: JSON.stringify({ binding: { kwargs: { x: { shape: [1, 8], dtype: 'float32' } } } }),
      profileName: 'cpu_example', mode: 'eval', gradients: false,
      stages: [{ label: 'Capture', calls: 8 }], resources: { sampledAt: Date.now(), host: 'qa-worker', remote: false,
        cpu: 'QA processor', cores: 8, cpuPercent: 20, memoryFree: 8 * 1024 ** 3, memoryTotal: 16 * 1024 ** 3, devices: [], detail: 'Fixture' } };
    await host({ type: 'graphOpening', state: { ...opening, phase: 'awaiting_execution', execution } });
    const runModel = frame.getByRole('button', { name: 'Run model', exact: true });
    await runModel.click();
    assert.equal(await frame.getByRole('button', { name: 'Measure latency', exact: true }).count(), 0);
    await frame.getByText('Run settings', { exact: true }).click();
    for (const [width, height] of [[1280, 850], [320, 400]]) {
      await page.setViewportSize({ width, height });
      await expect(runModel.locator('.local-canvas-control-label')).toBeVisible();
      const region = frame.getByRole('region', { name: 'Run configuration' });
      await region.evaluate(node => { node.scrollTop = 0; });
      await region.hover(); await page.mouse.wheel(0, 300);
      await expect.poll(() => region.evaluate(node => node.scrollTop)).toBeGreaterThan(0);
      await region.focus(); await region.press('Home');
      await expect.poll(() => region.evaluate(node => node.scrollTop)).toBe(0);
      await region.press('PageDown');
      await expect.poll(() => region.evaluate(node => node.scrollTop)).toBeGreaterThan(0);
      await expect.poll(() => frame.locator('.execution-sheet footer').evaluate(node => {
        const bar = document.querySelector('.local-browser-footer').getBoundingClientRect();
        const actions = node.getBoundingClientRect();
        return actions.bottom <= bar.top && actions.top >= 0;
      })).toBe(true);
      const run = frame.locator('.execution-sheet').getByRole('button', { name: 'Run', exact: true });
      await expect.poll(() => run.evaluate(node => {
        const rect = node.getBoundingClientRect();
        return [0.25, 0.5, 0.75].every(fraction => node.contains(document.elementFromPoint(rect.left + rect.width * fraction, rect.top + rect.height / 2)));
      })).toBe(true);
      await run.click({ trial: true });
      await shot('execution-scroll-' + width + '-' + height);
    }
    await frame.getByRole('button', { name: 'Collapse execution details' }).click();
    await host({ type: 'graphOpening', state: opening });
    await page.setViewportSize({ width: 360, height: 800 });
    await host({ type: 'graphReviewState', state: review });
    await frame.getByRole('combobox', { name: 'Affected model' }).waitFor();
    assert.equal(await frame.getByRole('button', { name: 'Run model', exact: true }).count(), 0);
    for (const width of [1280, 360, 320, 200]) {
      await page.setViewportSize({ width, height: 800 }); await reachable();
      await shot('toolbar-review-' + width + '-light');
    }
    await page.setViewportSize({ width: 360, height: 800 });
    await frame.getByRole('button', { name: 'Before', exact: true }).click();
    assert.equal(await frame.getByRole('button', { name: 'Before', exact: true }).getAttribute('aria-pressed'), 'true');
    await frame.getByRole('button', { name: 'After', exact: true }).click();
    await frame.getByRole('button', { name: 'Diff', exact: true }).click();
    await frame.getByRole('button', { name: 'Next graph change' }).click(); await request('graphReviewRequest', 'reveal');
    await frame.getByRole('button', { name: 'Coverage limits (1)' }).click();
    await frame.getByRole('region', { name: 'Coverage limits for this model' }).press('Escape');
    assert.equal(await frame.getByRole('button', { name: 'Coverage limits (1)' }).evaluate(node => document.activeElement === node), true);
    await frame.getByRole('switch', { name: 'Dark canvas appearance' }).click(); await shot('toolbar-review-dark');
    await host({ type: 'graphReviewState', state: null });
    await host({ type: 'graphReviewState', state: { ...review, id: 'captured-toolbar-review', liveContext: target } });
    await frame.getByRole('group', { name: 'Graph revision' }).waitFor({ state: 'hidden' });
    await reachable(); await shot('toolbar-captured-review-dark');
    await frame.getByRole('button', { name: 'Open native code diff' }).click(); await request('graphReviewRequest', 'reveal');
    await frame.getByRole('button', { name: 'Refresh graph diff' }).click(); await request('graphReviewRequest', 'refresh');
    await frame.getByRole('button', { name: 'Close graph diff' }).click(); await request('graphReviewRequest', 'close');
    if (errors.length || external.length || missing.length) console.log(JSON.stringify({ errors: [...new Set(errors)], external: [...new Set(external)], missing: [...new Set(missing)] }));
    assert.deepEqual([...new Set(errors)], []); assert.deepEqual(external, []);
    const result = { status: 'passed', scope: 'staged production renderer in Chromium; scripted host and capture/source-review fixtures; no model execution or real GitHub PR',
      rendererIndexSha256: createHash('sha256').update(fs.readFileSync(path.join(assets, 'index.html'))).digest('hex'),
      checks: ['1280/360/320/200px control reachability', 'compact and expanded model toolbars', 'Paper/Ink themes', 'model menu Escape/focus',
        'existing input request', 'single labelled Run model action', 'wheel and keyboard execution scrolling above Canvas footer', 'Run action unobstructed by floating chat', 'source review modes and change navigation', 'coverage Escape/focus', 'captured review omits revision switches',
        'native diff/refresh/close requests', 'no external requests or JavaScript errors'],
      screenshots: fs.readdirSync(output).filter(file => file.endsWith('.png')) };
    fs.writeFileSync(path.join(output, 'result.json'), JSON.stringify(result, null, 2) + '\n'); console.log(JSON.stringify(result, null, 2));
  } finally { await browser.close(); await new Promise(resolve => server.close(resolve)); }
}
run().catch(error => { console.error(error); process.exitCode = 1; });
