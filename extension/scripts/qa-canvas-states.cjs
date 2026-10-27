/** Real packaged renderer in Chromium with a scripted parent host; no Python or model execution. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { chromium } = require('../../canvas/node_modules/playwright');
require('../out/test/bootstrap');
const { errorHtml } = require('../out/shell');
const { emptyReviewProjection } = require('@tensorviz/graph-host/graphReviewCore');

const output = path.resolve(process.argv[2] || '.tensorviz-workflow/canvas-states');
const assets = path.resolve(__dirname, '../media/app');
const review = { ...require('../../canvas/src/test/fixtures/canvas/graphReview.json'), id: 'qa-empty-review',
  title: 'Fixture pull request', beforeLabel: 'main', afterLabel: 'PR', changedFiles: ['model.py'],
  models: [], before: emptyReviewProjection('Pull request'), after: emptyReviewProjection('Pull request'),
  stale: false, notes: ['Dynamic source could not be resolved.'] };
const parent = `<!doctype html><html><head><style>body{margin:0}iframe{width:100%;height:100vh;display:block;border:0}</style></head><body>
<iframe id="canvas" src="/?view=canvas"></iframe><script>
const frame = document.querySelector('iframe');
window.uiQa = { messages: [], connect: () => frame.contentWindow.postMessage({type:'canvasConnected',sessionId:'ui-qa',viewId:'ui-qa-view',theme:'dark'},location.origin),
  emit: (message) => frame.contentWindow.postMessage(message,location.origin) };
addEventListener('message',event => {
  if(event.source !== frame.contentWindow || event.origin !== location.origin) return;
  window.uiQa.messages.push(event.data);
  if(event.data.type === 'ready') frame.contentWindow.postMessage({type:'hostReady',protocolVersion:1,theme:'dark'},location.origin);
});
</script></body></html>`;

async function run() {
  assert.ok(fs.existsSync(path.join(assets, 'index.html')), 'build:webview must stage the renderer first');
  fs.mkdirSync(output, { recursive: true });
  const server = http.createServer((req, res) => {
    const route = new URL(req.url, 'http://localhost').pathname;
    if (route === '/__qa_parent') { res.setHeader('Content-Type', 'text/html'); return res.end(parent); }
    if (route === '/__qa_failure' || route === '/__qa_progress') {
      res.setHeader('Content-Type', 'text/html');
      return res.end(route === '/__qa_progress'
        ? errorHtml('Checking Python environments…', 'Preparing the tensorViz service.\nCanvas will appear when the backend is ready.', '')
        : errorHtml('Canvas could not connect to the backend.', 'Check the tensorViz output and selected Python environment, then run tensorViz: Open as Graph to retry.', ''));
    }
    const file = path.resolve(assets, '.' + (route === '/' ? '/index.html' : route));
    if (!file.startsWith(assets + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) { res.statusCode = 404; return res.end(); }
    const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.woff2': 'font/woff2' };
    res.setHeader('Content-Type', mime[path.extname(file)] || 'application/octet-stream');
    fs.createReadStream(file).pipe(res);
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const origin = `http://127.0.0.1:${server.address().port}`;
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 360, height: 760 } });
  const errors = [], external = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/*', route => {
    if (route.request().url().startsWith(origin + '/')) return route.continue();
    external.push(route.request().url()); return route.abort();
  });
  const frame = page.frameLocator('#canvas');
  const shot = name => page.screenshot({ path: path.join(output, name + '.png'), animations: 'disabled' });
  const emit = message => page.evaluate(message => window.uiQa.emit(message), message);
  let sequence = 0;
  const reviewState = state => emit({ type: 'canvasMessage', message: { type: 'graphReviewState', state,
    protocolVersion: 1, workspaceSessionId: 'ui-qa', messageId: `m-${++sequence}` } });
  const requests = () => page.evaluate(() => window.uiQa.messages.flatMap(message => message.type === 'canvasMessage' ? [message.message] : []));
  const waitRequest = action => page.waitForFunction(action => window.uiQa.messages.some(message =>
    message.type === 'canvasMessage' && message.message.action === action), action);
  async function noOverflow() {
    assert.equal(await frame.locator('html').evaluate(node => node.scrollWidth > innerWidth), false);
  }
  try {
    await page.goto(origin + '/__qa_parent');
    await frame.getByRole('heading', { name: 'Connecting to Canvas' }).waitFor();
    await shot('canvas-connecting');
    await page.evaluate(() => window.uiQa.connect());
    await frame.getByRole('heading', { name: 'Explore your model' }).waitFor();
    await noOverflow();
    assert.equal(await frame.getByRole('button', { name: /(?:Pause|Play) animation/ }).count(), 0);
    await frame.getByRole('button', { name: 'Open as Graph' }).click();
    await page.waitForFunction(() => window.uiQa.messages.some(message => message.type === 'openAsGraph'));
    assert.equal(await page.evaluate(() => window.uiQa.messages.filter(message => message.type === 'openAsGraph').length), 1);
    await frame.getByRole('button', { name: 'Open PR' }).click();
    await waitRequest('openPR');
    assert.ok((await requests()).some(message => message.type === 'graphReviewRequest' && message.action === 'openPR'));
    await page.mouse.move(0, 0);
    await shot('canvas-ready-dark');
    await emit({ type: 'canvasTheme', theme: 'light' });
    await shot('canvas-ready-light');
    await page.setViewportSize({ width: 200, height: 760 });
    await noOverflow();
    await shot('canvas-ready-narrow');
    await page.setViewportSize({ width: 360, height: 760 });
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await frame.locator('.canvas-state[data-motion="paused"]').waitFor();
    assert.equal(await frame.locator('.canvas-state-signal').first().evaluate(node => getComputedStyle(node).animationName), 'none');
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await reviewState(review);
    await frame.getByRole('heading', { name: 'No model identified' }).waitFor();
    await frame.getByRole('button', { name: 'Coverage limits (1)' }).click();
    await frame.getByText('Dynamic source could not be resolved.', { exact: true }).waitFor();
    await noOverflow();
    await shot('canvas-review-empty-light');
    await emit({ type: 'canvasTheme', theme: 'dark' });
    await shot('canvas-review-empty-dark');
    await frame.getByRole('button', { name: 'View code changes' }).click();
    await waitRequest('reveal');
    assert.ok((await requests()).some(message => message.action === 'reveal' && message.reviewId === review.id && message.side === 'after'));
    await frame.getByRole('button', { name: 'Return to Canvas' }).click();
    await waitRequest('close');
    assert.ok((await requests()).some(message => message.action === 'close' && message.reviewId === review.id));
    await reviewState(null);
    await frame.getByRole('heading', { name: 'Explore your model' }).waitFor();
    // Exercise the actual native transport timeout, not an invented ended message.
    await page.goto(origin + '/__qa_parent');
    await frame.getByRole('heading', { name: 'Reconnect to your model' }).waitFor({ timeout: 25000 });
    await frame.getByText(/Show Canvas/).waitFor();
    await shot('canvas-session-ended');
    for (const state of ['progress', 'failure']) {
      await page.goto(origin + '/__qa_' + state);
      await page.addStyleTag({ content: `:root{--vscode-sideBar-background:#181818;--vscode-editor-foreground:#cccccc;
        --vscode-font-family:Arial,sans-serif;--vscode-descriptionForeground:#9d9d9d;--vscode-textLink-foreground:#3794ff;--vscode-widget-border:#454545}` });
      assert.equal(await page.locator('html').evaluate(node => node.scrollWidth > innerWidth), false);
      await shot('canvas-shell-' + state);
    }
    assert.deepEqual(errors, []);
    assert.deepEqual(external, [], 'empty and recovery screens must not need external assets');
    const result = { status: 'passed', scope: 'staged production React renderer in Chromium with scripted parent host; compiled shell HTML',
      rendererIndexSha256: createHash('sha256').update(fs.readFileSync(path.join(assets, 'index.html'))).digest('hex'),
      checks: ['connecting and connected empty Canvas', 'fixed Open as Graph message', 'Open PR request', 'no playback buttons', 'live reduced motion',
        'Paper/Ink themes', '200px layout', 'empty review retains coverage notes', 'source/close review requests', 'native transport timeout recovery',
        'shell progress/failure layout', 'no external requests or JavaScript errors'], screenshots: fs.readdirSync(output).filter(file => file.endsWith('.png')) };
    fs.writeFileSync(path.join(output, 'result.json'), JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify(result, null, 2));
  } finally { await browser.close(); await new Promise(resolve => server.close(resolve)); }
}
run().catch(error => { console.error(error); process.exitCode = 1; });
