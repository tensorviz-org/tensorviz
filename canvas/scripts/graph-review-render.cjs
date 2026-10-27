// Render the packaged native Canvas against the real Git/HTTP corpus. The shell
// is a test double; native VS Code acceptance remains a separate check.
const fs = require('node:fs');
const path = require('node:path');
const { gunzipSync } = require('node:zlib');
const assert = require('node:assert/strict');
const { chromium, expect } = require('@playwright/test');
const { LOCAL_BROWSER_PROTOCOL_VERSION } = require('@tensorviz/contracts/protocol');

const root = path.resolve(__dirname, '../..');
const assets = path.resolve(process.env.GRAPH_REVIEW_RENDER_ASSETS || path.join(root, 'extension/media/app'));
const corpus = JSON.parse(gunzipSync(fs.readFileSync(path.join(root, 'canvas/src/test/fixtures/canvas/graphReviewStress.json.gz'))));
const expected = new Map(JSON.parse(fs.readFileSync(process.env.GRAPH_REVIEW_STRESS_REPORT || '/tmp/graph-stress-final.json')).map(row => [row.id, row]));
const limit = Number(process.env.GRAPH_REVIEW_RENDER_LIMIT || 1000);
const output = process.env.GRAPH_REVIEW_RENDER_OUTPUT || '/tmp/graph-stress-render.json';
const origin = 'http://127.0.0.1:4179';
const session = 'stress-native-session';
const shell = `<!doctype html><html><body style="margin:0"><iframe title="Native Canvas" src="/?view=canvas" style="border:0;width:100vw;height:100vh"></iframe><script>
window.messages=[];
window.addEventListener('message',event=>{
  if(event.source!==document.querySelector('iframe').contentWindow)return;
  if(event.data.type==='canvasConnect'){event.source.postMessage({type:'canvasConnected',sessionId:'${session}',viewId:'stress-view'},location.origin);window.canvasReady=true;}
  if(event.data.type==='canvasMessage')window.messages.push(event.data.message);
});
window.sequence=0;
window.sendReview=state=>{
  const send=message=>document.querySelector('iframe').contentWindow.postMessage({type:'canvasMessage',message:{...message,protocolVersion:${LOCAL_BROWSER_PROTOCOL_VERSION},workspaceSessionId:'${session}',messageId:crypto.randomUUID()}},location.origin);
  send({type:'activeTarget',hostSequence:++window.sequence,target:{workspaceFolder:'/stress/'+state.id,file:'model.py',cls:state.after.cls,documentVersion:1,sourceDigest:state.after.sourceDigest,backendRevision:state.after.revision}});
  send({type:'graphReviewState',state});
};
</script></body></html>`;

async function reachable(locator) {
  await expect(locator).toBeVisible({ timeout: 10000 });
  assert.equal(await locator.evaluate(element => {
    const r = element.getBoundingClientRect();
    const hit = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
    return element.contains(hit);
  }), true, 'Control is visible but cannot receive a click');
}

(async () => {
  assert(limit > 0 && limit <= 1000);
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/*', async route => {
    const url = new URL(route.request().url());
    // Native Canvas uses the pinned tldraw asset CDN. Let its ordinary asset
    // requests run; aborting them manufactures image-decoding failures.
    if (url.origin !== origin) return route.continue();
    if (url.pathname === '/qa-shell') return route.fulfill({ contentType: 'text/html', body: shell });
    const file = path.resolve(assets, '.' + (url.pathname === '/' ? '/index.html' : url.pathname));
    if (!file.startsWith(assets + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) return route.fulfill({ status: 404, body: 'Not a packaged asset' });
    return route.fulfill({ path: file });
  });
  const results = [];
  try {
    for (const [index, item] of corpus.slice(0, limit).entries()) {
      const start = Date.now();
      const failures = [];
      try {
        await page.setViewportSize({ width: [430, 740, 1440][index % 3], height: 900 });
        if (index % 25 === 0) {
          await page.goto(origin + '/qa-shell');
          await page.waitForFunction(() => window.canvasReady === true, { timeout: 15000 });
        }
        const priorErrors = errors.length;
        await page.evaluate(state => { window.messages = []; window.sendReview(state); }, item.state);
        const frame = page.frameLocator('iframe');
        const toolbar = frame.getByRole('region', { name: 'Graph diff review' });
        await expect(toolbar.locator('.graph-review-revisions strong')).toHaveText(item.head, { timeout: 10000 });
        await expect(frame.getByLabel('Canvas status and navigation')).toContainText('Not executed');
        const wanted = expected.get(item.id);
        assert(wanted?.passed, 'Missing passing semantic review for this case');
        await expect(toolbar.locator('.graph-review-added')).toHaveText('+' + wanted.counts.added);
        await expect(toolbar.locator('.graph-review-removed')).toHaveText('−' + wanted.counts.removed);
        await expect(toolbar.locator('.graph-review-modified')).toHaveText('~' + wanted.counts.modified);
        // tldraw culls off-screen tiles in long graphs; the first DOM record
        // need not be visible. Require an actual visible graph tile instead.
        await expect(frame.locator('.tl-shapes [data-shape-id^="shape:node:"]:visible').first()).toBeVisible({ timeout: 10000 });
        for (const mode of ['Before', 'Diff', 'After']) {
          const button = toolbar.getByRole('button', { name: mode, exact: true });
          await reachable(button); await button.click();
          await expect(button).toHaveAttribute('aria-pressed', 'true');
        }
        if (item.kind === 'duplicate_call') {
          const semantic = item.state.after.document.semantic;
          const addition = semantic.nodes.find(n => n.params._raw_call === 'self.activation(x) + self.activation(x)');
          const incoming = semantic.edges.filter(e => e.target_node_id === addition.node_id);
          assert.equal(incoming.length, 2, 'The actual HTTP projection must retain both operands');
          for (const edge of incoming) {
            const shapeId = 'shape:edge:' + encodeURIComponent(JSON.stringify([
              'review:after:edge:' + edge.edge_id, 'review:after:' + edge.source_node_id, 'review:after:' + edge.target_node_id,
            ]));
            await expect(frame.locator(`[data-shape-id="${shapeId}"]`)).toHaveCount(1);
          }
        }
        const source = toolbar.getByRole('button', { name: 'Open native code diff' });
        await reachable(source); await source.click();
        await expect.poll(() => page.evaluate(() => window.messages.filter(m => m.type === 'graphReviewRequest' && m.action === 'reveal').length), { timeout: 10000 }).toBeGreaterThan(0);
        const next = toolbar.getByRole('button', { name: 'Next graph change' });
        if (wanted.changes.length) {
          await reachable(next); await next.click();
          await expect.poll(() => page.evaluate(() => window.messages.filter(m => m.type === 'graphReviewRequest' && m.nodeId).length), { timeout: 10000 }).toBeGreaterThan(0);
          const request = await page.evaluate(() => window.messages.filter(m => m.type === 'graphReviewRequest' && m.nodeId).at(-1));
          assert.equal(request.reviewId, item.id);
          assert(item.state[request.side].spanmap[request.nodeId], 'Navigation must name a source-mapped node in the chosen revision');
        } else await expect(next).toBeDisabled();
        assert.deepEqual(errors.slice(priorErrors), [], 'Uncaught error while rendering review');
      } catch (error) {
        failures.push(String(error));
        await page.screenshot({ path: output.replace(/\.json$/, `-${item.id}.png`) });
      }
      results.push({ id: item.id, passed: failures.length === 0, failures, elapsedMs: Date.now() - start });
      fs.writeFileSync(output, JSON.stringify({ shell: 'test double; production VSIX frontend', package: assets, cases: results }, null, 2));
      if ((index + 1) % 10 === 0 || failures.length) console.log(`Rendered ${index + 1}/${limit}; failures ${results.filter(r => !r.passed).length}`, failures.join('\n'));
      if (results.length === 3 && results.every(r => !r.passed)) throw new Error('First three cases failed; inspect the harness or package before continuing');
    }
  } finally { await browser.close(); }
  const failed = results.filter(r => !r.passed).length;
  console.log(`Completed ${results.length} packaged Canvas reviews; ${failed} failures`);
  process.exitCode = failed ? 1 : 0;
})().catch(error => { console.error(error); process.exitCode = 1; });
