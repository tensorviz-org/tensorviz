/** Staged production setup forms with scripted host data; no Python execution. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { chromium } = require('../../canvas/node_modules/playwright');
const fixture = require('../../canvas/src/test/fixtures/canvas/graphReview.json');
const capture = require('../../canvas/src/test/fixtures/canvas/capture-document.json');
const assets = path.resolve(__dirname, '../media/app');
const output = path.resolve(process.argv[2] || '.tensorviz-workflow/canvas-setup');
const target = { workspaceFolder: '/qa', file: 'models/encoder.py', cls: 'MlpTiny', documentVersion: 1,
  sourceDigest: fixture.after.sourceDigest, backendRevision: fixture.after.revision };
const projection = { ...fixture.after, file: target.file, traced: false };
const parent = `<!doctype html><style>body{margin:0}iframe{border:0;width:100%;height:100vh;display:block}</style>
<iframe src="/?view=canvas"></iframe><script>
const frame=document.querySelector('iframe');window.uiQa={messages:[],emit:m=>frame.contentWindow.postMessage(m,location.origin)};
addEventListener('message',e=>{if(e.source!==frame.contentWindow||e.origin!==location.origin)return;
uiQa.messages.push(e.data);if(e.data.type==='ready')uiQa.emit({type:'hostReady',protocolVersion:1,theme:'dark'});
if(e.data.type==='canvasConnect')uiQa.emit({type:'canvasConnected',sessionId:'toolbar-qa',viewId:'toolbar-view',theme:'dark'});});
</script>`;
async function run() {
  fs.mkdirSync(output, { recursive: true });
  fs.rmSync(path.join(output, 'result.json'), { force: true });
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
  page.setDefaultTimeout(10000);
  const errors = [], external = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/*', route => { if (route.request().url().startsWith(origin + '/')) return route.continue();
    external.push(route.request().url()); return route.abort(); });
  const frame = page.frameLocator('iframe');
  let sequence = 0;
  const emit = message => page.evaluate(message => window.uiQa.emit(message), message);
  const host = message => emit({ type: 'canvasMessage', message: { ...message, protocolVersion: 1,
    workspaceSessionId: 'toolbar-qa', messageId: `m-${++sequence}`, hostSequence: sequence } });
  const screenshots = [];
  const shot = async name => { screenshots.push(name + '.png'); await page.screenshot({ path: path.join(output, name + '.png'), animations: 'disabled' }); };
  const tensor = shape => ({ kind: 'tensor', shape, dtype: 'float32', requires_grad: false,
    generator: { kind: 'normal', mean: 0, std: 1, seed: 7 } });
  const setup = { target, phase: 'needs_setup', hasCapture: false, detail: 'Add the missing settings for one example call.',
    discovery: { model: 'models.encoder:MlpTiny', sourceDigest: target.sourceDigest,
      init: [{ name: 'features', kind: 'keyword', required: true, source: { file: target.file, line: 8 } }, { name: 'bias', kind: 'keyword', required: false, defaultValue: true, defaultText: 'True' }],
      inputs: [{ name: 'x', kind: 'keyword', required: true, source: { file: target.file, line: 12 } }, { name: 'mask', kind: 'keyword_only', required: true }],
      examples: [], factories: [], configs: [], parents: [], constraints: [], truncated: false } };
  let attempt = 0;
  const open = state => host({ type: 'graphOpening', state: JSON.parse(JSON.stringify({ ...state, id: 'setup-' + ++attempt })) });
  const actions = () => page.evaluate(() => uiQa.messages.filter(m => m.type === 'canvasMessage' && m.message.type === 'graphSetupRequest').map(m => m.message.action));
  const waitAction = kind => page.waitForFunction(kind => uiQa.messages.some(m => m.type === 'canvasMessage'
    && m.message.type === 'graphSetupRequest' && m.message.action.kind === kind), kind, { timeout: 10000 });
  const panel = frame.getByRole('region', { name: 'Open captured graph' });
  async function reachable() {
    assert.equal(await frame.locator('html').evaluate(node => node.scrollWidth > innerWidth), false, 'no document overflow');
    for (const selector of ['.graph-model-setup-panel', '.graph-setup-body']) {
      assert.equal(await frame.locator(selector).evaluate(node => node.scrollWidth > node.clientWidth + 1), false, selector + ' fits');
    }
    const primary = await panel.getByRole('button', { name: 'Open graph', exact: true }).count() ? panel.getByRole('button', { name: 'Open graph', exact: true }) : panel.getByRole('button', { name: 'Set up with agent', exact: true });
    assert.equal(await primary.evaluate(node => {
      const r = node.getBoundingClientRect(); return r.top >= 0 && r.bottom <= innerHeight && r.left >= 0 && r.right <= innerWidth;
    }), true, 'current primary action stays visible without scrolling');
    assert.equal(await panel.locator('.graph-setup-body').evaluate(node => node.clientHeight > 40), true, 'fields retain scroll space');
    assert.equal(await panel.evaluate(node => getComputedStyle(node).backgroundColor.startsWith('rgba(')), false, 'form surface must hide graph controls beneath it');
  }
  try {
    await page.goto(origin + '/__qa_parent');
    await frame.getByRole('heading', { name: 'Explore your model' }).waitFor();
    await host({ type: 'activeTarget', target });
    await host({ type: 'projection', target, projection, traced: false, greyed: false });
    await open(setup);
    await panel.getByRole('heading', { name: 'Set up this model' }).waitFor();
    for (const width of [1440, 420, 360, 320, 200]) {
      await page.setViewportSize({ width, height: 800 }); await reachable(); await shot('setup-required-' + width + '-dark');
    }
    assert.equal(await panel.getByLabel('features *', { exact: true }).isVisible(), false, 'manual fields do not compete with agent route');
    assert.equal(await panel.getByRole('button', { name: 'Open graph', exact: true }).count(), 0, 'no submit before choosing manual');
    assert.equal(await panel.getByText('Recommended', { exact: true }).evaluate(node => node.closest('button').getAttribute('aria-label')), 'Set up with agent');
    await page.setViewportSize({ width: 420, height: 800 });
    for (const button of ['.graph-setup-agent', '.graph-setup-manual-choice']) {
      assert.equal(await panel.locator(button).evaluate(node => {
        const icon = node.querySelector('svg').getBoundingClientRect(), text = (node.querySelector('.graph-setup-method-label') ?? node.querySelector('span')).getBoundingClientRect();
        return Math.abs(icon.y + icon.height / 2 - text.y - text.height / 2) <= 1;
      }), true, button + ' icon/text centers align');
    }
    await page.setViewportSize({ width: 420, height: 800 }); await shot('setup-choice-ink');
    await frame.getByRole('switch', { name: 'Dark canvas appearance' }).click(); await shot('setup-choice-paper');
    await page.setViewportSize({ width: 320, height: 360 }); await reachable(); await shot('setup-choice-short-paper');
    await page.setViewportSize({ width: 360, height: 800 });
    await panel.getByRole('button', { name: 'Set up manually', exact: true }).click(); await reachable(); await shot('setup-manual-paper');
    for (const width of [1440, 420, 320]) {
      await page.setViewportSize({ width, height: 800 }); await reachable(); await shot('setup-manual-' + width + '-paper');
    }
    await frame.getByRole('switch', { name: 'Dark canvas appearance' }).click();
    for (const width of [1440, 420, 320]) {
      await page.setViewportSize({ width, height: 800 }); await reachable(); await shot('setup-manual-' + width + '-ink');
    }
    await page.setViewportSize({ width: 360, height: 800 });
    const submit = panel.getByRole('button', { name: 'Open graph', exact: true });
    await submit.click();
    assert.equal((await actions()).filter(a => a.kind === 'submit').length, 0, 'empty required fields cannot execute');
    await panel.getByLabel('features *', { exact: true }).fill('64');
    await panel.getByLabel('x shape', { exact: true }).fill('not a shape');
    const mask = panel.getByLabel('mask shape', { exact: true });
    await mask.locator('..').locator('..').locator('summary').click();
    await panel.getByLabel('mask dtype').selectOption('bool');
    await mask.fill('2, 64');
    await submit.click();
    assert.equal((await actions()).filter(a => a.kind === 'submit').length, 0, 'invalid tensor dimensions cannot execute');
    await panel.getByLabel('x shape', { exact: true }).fill('2, 64');
    await panel.getByLabel('Model creation', { exact: true }).selectOption('python');
    await panel.getByLabel('Model builder file', { exact: true }).fill('model_setup.py');
    await panel.getByLabel('Model builder function', { exact: true }).fill('make_model');
    await panel.getByLabel('Model creation', { exact: true }).selectOption('settings');
    assert.equal(await panel.getByLabel('features *', { exact: true }).inputValue(), '64');
    await panel.getByLabel('Input source', { exact: true }).selectOption('python');
    await panel.getByLabel('Input function', { exact: true }).fill('example_inputs');
    await panel.getByLabel('Input source', { exact: true }).selectOption('entered');
    assert.equal(await panel.getByLabel('x shape', { exact: true }).inputValue(), '2, 64');
    await panel.getByLabel('x shape', { exact: true }).fill('2,');
    await panel.getByRole('button', { name: 'Setup methods', exact: true }).click();
    await panel.getByRole('button', { name: 'Set up manually', exact: true }).click();
    assert.equal(await panel.getByLabel('x shape', { exact: true }).inputValue(), '2,', 'incomplete shape text survives switching routes');
    await panel.getByLabel('x shape', { exact: true }).fill('2, 64');
    await panel.getByRole('button', { name: 'Setup methods', exact: true }).click();
    await panel.getByRole('button', { name: 'Set up with agent', exact: true }).click(); await waitAction('agent_help');
    const agentDraft = (await actions()).find(a => a.kind === 'agent_help').draft;
    assert.equal(agentDraft.profile.init.kwargs.features, 64);
    assert.deepEqual(agentDraft.profile.inputs.kwargs.x.shape, [2, 64]);
    // Recommended help opens chat and dismisses setup; reopening preserves the in-progress draft.
    await frame.getByRole('button', { name: 'Change inputs', exact: true }).click();
    await panel.getByRole('button', { name: 'Set up manually', exact: true }).click();
    if (!await panel.getByRole('button', { name: 'Edit tensorviz.toml' }).isVisible()) await panel.getByText('Saved setup and source evidence', { exact: true }).click();
    await panel.getByRole('button', { name: 'Edit tensorviz.toml' }).click(); await waitAction('edit_config');
    assert.equal(await panel.getByLabel('features *', { exact: true }).inputValue(), '64');
    await frame.getByRole('switch', { name: 'Dark canvas appearance' }).click();
    await panel.locator('.graph-setup-body').evaluate(node => node.scrollTop = 0); await shot('setup-required-light');
    await panel.getByText('Advanced setup', { exact: true }).click();
    await panel.getByText('Explicit bindings and limits', { exact: true }).click();
    await panel.getByLabel('Advanced profile').scrollIntoViewIfNeeded(); await reachable(); await shot('setup-advanced-light');
    await submit.click(); await waitAction('submit');
    const submitted = (await actions()).find(a => a.kind === 'submit').draft;
    assert.deepEqual(submitted.profile.init, { args: [], kwargs: { features: 64 } });
    assert.deepEqual(submitted.profile.inputs.kwargs.x.shape, [2, 64]);
    assert.deepEqual(submitted.profile.inputs.kwargs.mask.generator, { kind: 'full', value: false });
    assert.equal(submitted.profile.inputs.kwargs.mask.dtype, 'bool');
    await host({ type: 'projection', target, projection: null, traced: false, greyed: false });
    await open({ ...setup, draft: submitted, phase: 'capturing', detail: 'Running one forward call in the selected environment.' });
    await panel.getByRole('heading', { name: 'Capturing one forward call' }).waitFor();
    await page.emulateMedia({ reducedMotion: 'reduce' });
    assert.equal(await panel.locator('.graph-opening-indicator').evaluate(node => getComputedStyle(node).animationName), 'none');
    await shot('setup-capturing-light');
    await panel.getByRole('button', { name: 'Cancel', exact: true }).click(); await waitAction('cancel');
    await open({ ...setup, draft: submitted, phase: 'failed', failure: { kind: 'setup' }, requirements: ['inputs.x: match the model width.'], detail: 'The example needs a corrected input.' });
    await panel.getByRole('heading', { name: 'Capture needs attention' }).waitFor(); await panel.getByRole('button', { name: 'Set up manually', exact: true }).click(); await reachable(); await shot('setup-failed-light');
    await open({ ...setup, draft: { ...submitted, profile: { ...submitted.profile,
      init: undefined, factory: { file: 'model_setup.py', callable: 'make_model', kwargs: {} },
      inputs: { factory: { file: 'example_inputs.py', callable: 'graph_inputs', kwargs: {} } } } } });
    await panel.getByRole('button', { name: 'Set up manually', exact: true }).click();
    await panel.getByLabel('Model builder file', { exact: true }).waitFor(); await reachable(); await shot('setup-factories-light');
    const config = { ...setup, discovery: { ...setup.discovery, init: [{ name: 'config', kind: 'keyword', required: true }],
      inputs: [{ name: 'x', kind: 'keyword', required: true }], configs: [{ cls: 'EncoderConfig', file: target.file, parameter: 'config', construction: 'call',
        fields: [{ name: 'channels', kind: 'keyword', required: true }, { name: 'image_size', kind: 'keyword', required: false, defaultValue: 224, defaultText: '224' }] }] },
      evidence: [{ field: 'inputs.x', kind: 'example', detail: 'One synthetic image for this example call.', file: target.file, line: 12 }],
      draft: { profile: { kind: 'model', model: setup.discovery.model, mode: 'eval', device: 'cpu', grad_enabled: false,
        init: { args: [], kwargs: {} }, inputs: { args: [], kwargs: { x: tensor([1, 3, 224, 224]) } } }, configObject: { index: 0, parameter: 'config', values: {} } } };
    await open(config); await panel.getByRole('button', { name: 'Set up manually', exact: true }).click(); await panel.getByLabel('channels *', { exact: true }).fill('3'); await reachable(); await shot('setup-config-light');
    if (!await panel.getByRole('button', { name: 'Edit tensorviz.toml' }).isVisible()) await panel.getByText('Saved setup and source evidence', { exact: true }).click();
    await panel.getByText('Example setup · sources and assumptions', { exact: true }).click();
    await panel.getByText('Example choice', { exact: true }).waitFor();
    await host({ type: 'projection', target, projection: { ...projection, traced: true, document: capture }, traced: true, greyed: false });
    const nested = { ...setup, hasCapture: true, discovery: { ...setup.discovery, init: [], inputs: null },
      draft: { profileName: 'conditioning_example', profile: { kind: 'model', model: setup.discovery.model, mode: 'eval', device: 'cpu', grad_enabled: false,
        init: { args: [], kwargs: {} }, inputs: { args: [], kwargs: { conditioning: { kind: 'dict', items: {
          image: tensor([1, 3, 224, 224]), context: { kind: 'tuple', items: [tensor([1, 77, 768]), { kind: 'float', value: 1 }] },
        } } } } } } };
    await open(nested); await panel.getByRole('button', { name: 'Set up manually', exact: true }).click();
    await panel.getByLabel('conditioning.image shape', { exact: true }).waitFor();
    await panel.getByLabel('conditioning.image shape', { exact: true }).fill('1, 3, 256, 256');
    await panel.getByLabel('conditioning.context[1] value', { exact: true }).fill('1.25');
    await panel.getByLabel('conditioning.context[1] value', { exact: true }).scrollIntoViewIfNeeded();
    await reachable(); await shot('setup-nested-light');
    await page.setViewportSize({ width: 200, height: 800 }); await reachable(); await shot('setup-nested-200-light');
    const previous = (await actions()).filter(a => a.kind === 'submit').length;
    await panel.getByRole('button', { name: 'Open graph', exact: true }).click();
    await page.waitForFunction(count => uiQa.messages.filter(m => m.type === 'canvasMessage' && m.message.type === 'graphSetupRequest' && m.message.action.kind === 'submit').length > count, previous, { timeout: 10000 });
    const edited = (await actions()).filter(a => a.kind === 'submit').at(-1).draft.profile.inputs.kwargs.conditioning;
    assert.equal(edited.kind, 'dict'); assert.equal(edited.items.context.kind, 'tuple');
    assert.deepEqual(edited.items.image.shape, [1, 3, 256, 256]); assert.equal(edited.items.image.generator.seed, 7);
    assert.deepEqual(edited.items.context.items[1], { kind: 'float', value: 1.25 });
    await page.setViewportSize({ width: 320, height: 360 }); await open(setup); await panel.getByRole('button', { name: 'Set up manually', exact: true }).click(); await reachable();
    await panel.getByLabel('features *', { exact: true }).fill('32'); await reachable(); await shot('setup-short-light');
    await panel.getByRole('button', { name: 'Open graph', exact: true }).press('Escape');
    await panel.waitFor({ state: 'hidden' });
    assert.equal(await frame.getByRole('button', { name: 'Change inputs', exact: true }).evaluate(node => document.activeElement === node), true);
    assert.deepEqual(errors, []); assert.deepEqual(external, []); assert.deepEqual(missing, []);
    const result = { status: 'passed', scope: 'staged production renderer in Chromium; scripted setup/capture fixtures; no Python execution or real VS Code dialog',
      rendererIndexSha256: createHash('sha256').update(fs.readFileSync(path.join(assets, 'index.html'))).digest('hex'),
      checks: ['explicit agent/manual alternatives; no initial form/submit', 'recommendation belongs to agent action; icon/text centers align', 'incomplete text survives route switch', '1440/420/360/320/200px layouts and fixed capture action', '320x360 short window', 'opaque Paper/Ink surfaces',
        'required and invalid inputs cannot submit', 'constructor/tensor draft payload', 'factory and Python-input toggles preserve bindings',
        'manual/agent requests preserve draft', 'advanced fields scroll while submit stays visible', 'nested dict/tuple/float contracts and generator seed',
        'configuration fields and source/example evidence', 'capture cancel and reduced motion', 'setup failure correction', 'Escape restores header focus', 'no external requests or JavaScript errors'], screenshots };
    fs.writeFileSync(path.join(output, 'result.json'), JSON.stringify(result, null, 2) + '\n'); console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    await page.screenshot({ path: path.join(output, 'failure.png') });
    console.error(JSON.stringify({ actions: await actions(), errors, external, missing })); throw error;
  } finally { await browser.close(); await new Promise(resolve => server.close(resolve)); }
}
run().catch(error => { console.error(error); process.exitCode = 1; });
