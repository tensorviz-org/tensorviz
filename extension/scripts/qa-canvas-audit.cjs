/** Production renderer UI repairs with scripted host data; never executes Python. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { chromium } = require('../../canvas/node_modules/playwright');
const fixture = require('../../canvas/src/test/fixtures/canvas/graphReview.json');
const capture = require('../../canvas/src/test/fixtures/canvas/capture-document.json');
const assets = path.resolve(__dirname, '../media/app');
const output = path.resolve(process.argv[2] || '.tensorviz-workflow/canvas-audit');
const target = { workspaceFolder: '/qa', file: 'models/encoder.py', cls: 'MlpTiny', documentVersion: 1,
  sourceDigest: fixture.after.sourceDigest, backendRevision: fixture.after.revision };
const projection = { ...fixture.after, file: target.file, traced: true, document: capture };
const opening = { id: 'audit-ready', target, phase: 'ready', hasCapture: true, detail: 'Captured.',
  configFileExists: true, draft: { profileName: 'cpu_example', profile: {} } };
const setup = { id: 'audit-setup', target, phase: 'needs_setup', hasCapture: false, detail: 'Add the missing settings for one example call.',
  discovery: { model: 'models.encoder:MlpTiny', sourceDigest: target.sourceDigest,
    init: [{ name: 'features', kind: 'keyword', required: true }], inputs: [{ name: 'x', kind: 'keyword', required: true }],
    examples: [], factories: [], configs: [], parents: [], constraints: [], truncated: false } };
const parent = `<!doctype html><style>body{margin:0}iframe{border:0;width:100%;height:100vh;display:block}</style>
<iframe src="/?view=canvas"></iframe><script>
const f=document.querySelector('iframe');window.uiQa={messages:[],emit:m=>f.contentWindow.postMessage(m,location.origin)};
addEventListener('message',e=>{if(e.source!==f.contentWindow||e.origin!==location.origin)return;uiQa.messages.push(e.data);
if(e.data.type==='ready')uiQa.emit({type:'hostReady',protocolVersion:1,theme:'dark'});
if(e.data.type==='canvasConnect')uiQa.emit({type:'canvasConnected',sessionId:'audit',viewId:'audit',theme:'dark'});});</script>`;

async function run() {
  fs.mkdirSync(output, { recursive: true });
  const missing = [], errors = [], external = [], checks = [];
  const server = http.createServer((req, res) => {
    const route = new URL(req.url, 'http://localhost').pathname;
    if (route === '/__parent') { res.setHeader('Content-Type', 'text/html'); return res.end(parent); }
    const file = path.resolve(assets, '.' + (route === '/' ? '/index.html' : route));
    if (!file.startsWith(assets + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
      missing.push(route); res.statusCode = 404; return res.end();
    }
    res.setHeader('Content-Type', ({ '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.woff2': 'font/woff2' })[path.extname(file)] || 'application/octet-stream');
    fs.createReadStream(file).pipe(res);
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const origin = `http://127.0.0.1:${server.address().port}`;
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  page.setDefaultTimeout(12000);
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/*', route => { if (route.request().url().startsWith(origin + '/')) return route.continue(); external.push(route.request().url()); return route.abort(); });
  const frame = page.frameLocator('iframe');
  let sequence = 0;
  const host = message => page.evaluate(message => uiQa.emit(message), { type: 'canvasMessage', message: { ...message,
    protocolVersion: 1, workspaceSessionId: 'audit', messageId: `m-${++sequence}`, hostSequence: sequence } });
  const shot = name => page.screenshot({ path: path.join(output, name + '.png'), animations: 'disabled' });
  async function footerBounds() {
    assert.equal(await frame.locator('html').evaluate(node => node.scrollWidth > innerWidth), false, 'document fits');
    const failures = await frame.locator('.local-canvas-footer-tools button').evaluateAll(buttons => buttons.flatMap(button => {
      if (!button.getClientRects().length) return [];
      const rect = button.getBoundingClientRect();
      const label = button.getAttribute('aria-label') || button.textContent;
      const issues = [];
      if (rect.left < 0 || rect.right > innerWidth + 1 || rect.bottom > innerHeight + 1) issues.push(label + ': viewport');
      for (const span of button.querySelectorAll('span')) {
        const text = span.getBoundingClientRect();
        if (text.left < rect.left || text.right > rect.right + 1 || text.top < rect.top || text.bottom > rect.bottom + 1) issues.push(label + ': text');
      }
      if (!button.contains(document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2))) issues.push(label + ': hit target');
      return issues;
    }));
    assert.deepEqual(failures, [], 'footer values, controls and hit targets fit');
  }
  try {
    await page.goto(origin + '/__parent');
    await frame.getByRole('heading', { name: 'Explore your model' }).waitFor();
    await host({ type: 'activeTarget', target });
    await host({ type: 'projection', target, projection, traced: true, greyed: false });
    await host({ type: 'syncState', target, state: 'runtime-traced' });
    await host({ type: 'graphOpening', state: opening });
    await frame.getByRole('button', { name: 'Layers', exact: true }).waitFor();
    for (const theme of ['dark', 'light']) {
      if (theme === 'light') await frame.getByRole('switch', { name: 'Dark canvas appearance' }).click();
      for (const [width, height] of [[1440, 900], [420, 900], [320, 600], [320, 360]]) {
        await page.setViewportSize({ width, height });
        await footerBounds();
        checks.push(`footer ${theme} ${width}x${height}`);
        if (height > 360) await shot(`footer-${theme}-${width}`);
      }
    }
    assert.equal(await frame.getByRole('button', { name: 'Latency details', exact: true }).count(), 0);
    checks.push('timing details are absent from the footer');
    await host({ type: 'graphOpening', state: setup });
    await page.setViewportSize({ width: 420, height: 900 });
    const panel = frame.getByRole('region', { name: 'Open captured graph' });
    await panel.getByRole('heading', { name: 'Complete the example call' }).waitFor();
    assert.equal(await panel.locator('footer p').textContent(), 'Saves to tensorviz.toml, then continues to execution review.');
    for (const width of [1440, 420, 320]) {
      await page.setViewportSize({ width, height: 900 }); await shot(`setup-paper-${width}`);
    }
    await panel.getByRole('button', { name: 'Close inputs' }).click();
    await frame.getByRole('button', { name: 'Canvas guide and shortcuts' }).click();
    await frame.getByRole('dialog', { name: 'Canvas guide', exact: true }).getByText(/Run model opens execution review/).waitFor();
    await shot('guide-paper-320');
    checks.push('setup and guide describe save-to-review and explicit Run');
    await frame.getByRole('dialog', { name: 'Canvas guide', exact: true }).press('Escape');

    // 24 blocks, four identically named projections per block: all 96 must be reachable.
    const semantic = { graph_id: 'audit-repeated', name: 'RepeatedTransformer', revision: 0, inputs: [], nodes: [], edges: [] };
    const names = ['q_proj', 'k_proj', 'v_proj', 'out_proj'];
    for (let i = 0; i < 24; i++) {
      const base = `transformer.h.${i}`, children = names.map(name => `${base}.${name}`);
      semantic.nodes.push({ node_id: base, kind: 'module', op: 'Block', params: { name: base }, in_ports: ['x'], out_ports: ['y'],
        metadata: { parentId: null, isGroup: true, moduleInstanceId: base, moduleTemplateId: 'Block', moduleLayerCount: 4, moduleMembers: children, childNodeIds: children } });
      for (let j = 0; j < 4; j++) semantic.nodes.push({ node_id: children[j], kind: 'layer', op: 'Linear',
        params: { in_features: 768, out_features: 768, _source_file: 'model.py', _lineno: 10 + j }, in_ports: ['x'], out_ports: ['y'], metadata: { parentId: base } });
    }
    const denseTarget = { ...target, cls: 'RepeatedTransformer', file: 'models/transformer.py', backendRevision: target.backendRevision + 1 };
    const dense = { ...fixture.after, cls: denseTarget.cls, file: denseTarget.file, revision: denseTarget.backendRevision,
      graphId: semantic.graph_id, traced: false, viewRestored: true,
      document: { ...fixture.after.document, semantic: { ...fixture.after.document.semantic, ...semantic },
        view: { graph_id: semantic.graph_id, node_positions: Object.fromEntries(semantic.nodes.map((node, index) => [node.node_id,
          { x: Math.floor(index / 5) * 500 + (index % 5 ? 60 : 0), y: index % 5 * 190 }])),
        collapsed_modules: [], selection: [], layout: { strategy: 'imported' } } } };
    await page.setViewportSize({ width: 1440, height: 900 });
    await host({ type: 'activeTarget', target: denseTarget });
    await host({ type: 'projection', target: denseTarget, projection: dense, traced: false, greyed: false });
    await host({ type: 'syncState', target: denseTarget, state: 'source-only' });
    const find = frame.getByRole('button', { name: 'Find a layer', exact: true });
    await find.click();
    const search = frame.getByRole('dialog', { name: 'Find a layer', exact: true });
    const input = search.getByRole('textbox', { name: 'Quick jump to node' });
    await frame.locator('body').evaluate(() => { window.auditCommands = []; addEventListener('builder-command', event => window.auditCommands.push(event.detail)); });
    await input.fill('Linear');
    await search.getByText('1–20 of 96 results', { exact: true }).waitFor();
    assert.equal(await search.locator('.ui2-quickjump-list button').count(), 20);
    assert.equal(await search.locator('.local-canvas-search-path').first().innerText(), 'transformer.h.0.q_proj');
    const identities = new Set();
    for (let i = 0; i < 5; i++) {
      for (const identity of await search.locator('.local-canvas-search-path').allTextContents()) identities.add(identity);
      if (i < 4) await search.getByRole('button', { name: 'Next', exact: true }).click();
    }
    await search.getByText('81–96 of 96 results', { exact: true }).waitFor();
    assert.equal(identities.size, 96, 'every matching layer is reachable through pages');
    assert.ok(identities.has('transformer.h.17.q_proj'));
    assert.deepEqual(await frame.locator('body').evaluate(() => window.auditCommands), [], 'query and paging do not navigate the graph');
    for (const theme of ['light', 'dark']) {
      if (theme === 'dark') { await input.press('Escape'); await frame.getByRole('switch', { name: 'Dark canvas appearance' }).click(); await find.click(); await input.fill('Linear'); }
      for (const width of [1440, 420, 320]) {
        await page.setViewportSize({ width, height: 900 });
        assert.equal(await search.evaluate(node => node.scrollWidth > node.clientWidth + 1), false, 'search fits');
        assert.equal(await search.locator('.ui2-quickjump-list').evaluate(node => node.scrollHeight > node.clientHeight), true, 'only results scroll');
        await shot(`search-${theme}-${width}`);
      }
    }
    await input.fill('transformer.h.17.q_proj'); await input.press('ArrowDown');
    const exact = search.getByRole('button', { name: 'Linear — transformer.h.17.q_proj', exact: true });
    assert.equal(await exact.evaluate(node => node === document.activeElement), true);
    await exact.press('Enter');
    await search.waitFor({ state: 'hidden' });
    assert.equal(await find.evaluate(node => node === document.activeElement), true);
    assert.deepEqual((await frame.locator('body').evaluate(() => window.auditCommands)).filter(command => command.command === 'focus-node'),
      [{ command: 'focus-node', payload: { nodeId: 'transformer.h.17.q_proj' } }]);
    await find.click();
    await input.fill('type=Linear');
    await search.getByText('1–20 of 96 results · 96 matching nodes', { exact: true }).waitFor();
    await input.fill('attr:in_features=768');
    await search.getByText('1–20 of 96 results · 96 matching nodes', { exact: true }).waitFor();
    await input.fill('type=['); await search.getByRole('alert').waitFor();
    await input.fill('missing-layer'); await search.getByText('No matching nodes.', { exact: true }).waitFor();
    await input.press('Escape');
    // Retain the existing visible-ancestor behavior for typed matches inside closed modules.
    dense.document.view.collapsed_modules = semantic.nodes.filter(node => node.kind === 'module').map(node => node.node_id);
    const collapsedTarget = { ...denseTarget, cls: 'CollapsedTransformer', file: 'models/collapsed.py', backendRevision: denseTarget.backendRevision + 1 };
    const collapsed = structuredClone(dense);
    collapsed.graphId = collapsed.document.semantic.graph_id = collapsed.document.view.graph_id = 'audit-collapsed';
    Object.assign(collapsed, { cls: collapsedTarget.cls, file: collapsedTarget.file, revision: collapsedTarget.backendRevision });
    await host({ type: 'activeTarget', target: collapsedTarget });
    await host({ type: 'projection', target: collapsedTarget, projection: collapsed, traced: false, greyed: false });
    await find.click(); await input.fill('type=Linear');
    await search.getByText('1–20 of 24 results · 96 matching nodes', { exact: true }).waitFor();
    await search.getByRole('button', { name: /— transformer.h.0 — 4 matches/ }).waitFor();
    await shot('search-collapsed-dark-320');
    checks.push('96-hit search count, module paths, paging beyond 20, exact ID, arrow/Enter/Escape and no navigation while querying');
    checks.push('typed type/attribute queries, invalid/empty results and 96 descendants grouped into 24 closed modules');
    assert.deepEqual(errors, []); assert.deepEqual(external, []); assert.deepEqual(missing, []);
    fs.writeFileSync(path.join(output, 'result.json'), JSON.stringify({ status: 'passed',
      scope: 'Staged production renderer in Chromium; scripted model/host fixtures; no Python execution or native VS Code acceptance.',
      rendererIndexSha256: createHash('sha256').update(fs.readFileSync(path.join(assets, 'index.html'))).digest('hex'),
      checks, errors, external, missing }, null, 2) + '\n');
    console.log(`Canvas audit checks passed: ${checks.length}; evidence: ${output}`);
  } catch (error) { await shot('failure'); throw error; }
  finally { await browser.close(); await new Promise(resolve => server.close(resolve)); }
}
run().catch(error => { console.error(error); process.exitCode = 1; });
