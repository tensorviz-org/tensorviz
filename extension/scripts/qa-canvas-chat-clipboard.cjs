/** Native shell + staged React across origins; simulated host clipboard, no agent or OS clipboard use. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const { chromium } = require('../../canvas/node_modules/playwright');
const { expect } = require('../../canvas/node_modules/@playwright/test');
require('../out/test/bootstrap');
const { shellHtml } = require('../out/shell');
const fixture = require('../../canvas/src/test/fixtures/canvas/graphReview.json');
const assets = path.resolve(__dirname, '../media/app');
const output = path.resolve(process.argv[2] || '.tensorviz-workflow/canvas-chat-clipboard');
const editModifier = process.platform === 'darwin' ? 'Meta' : 'Control';
const listen = server => new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const close = server => new Promise(resolve => server.close(resolve));

async function run() {
  fs.mkdirSync(output, { recursive: true });
  fs.rmSync(path.join(output, 'result.json'), { force: true });
  const app = http.createServer((req, res) => {
    const route = new URL(req.url, 'http://localhost').pathname;
    const file = path.resolve(assets, '.' + (route === '/' ? '/index.html' : route));
    if (!file.startsWith(assets + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) { res.statusCode = 404; return res.end(); }
    res.setHeader('Content-Type', { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.woff2': 'font/woff2' }[path.extname(file)] || 'application/octet-stream');
    fs.createReadStream(file).pipe(res);
  });
  await listen(app);
  const appOrigin = `http://127.0.0.1:${app.address().port}`;
  const shell = http.createServer((_req, res) => {
    res.setHeader('Content-Type', 'text/html');
    res.end(shellHtml({ appUrl: appOrigin + '/?view=canvas', appOrigin, port: app.address().port, workspaceDir: '/fixture' }));
  });
  await listen(shell);
  const shellOrigin = `http://127.0.0.1:${shell.address().port}`;
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 380, height: 760 } });
  page.setDefaultTimeout(8000);
  const errors = [], external = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/*', route => {
    if ([appOrigin, shellOrigin].some(origin => route.request().url().startsWith(origin + '/'))) return route.continue();
    external.push(route.request().url()); return route.abort();
  });
  await page.addInitScript(() => {
    if (window !== top) return;
    window.clipboardQa = { requests: [], writes: [], text: '', held: false, fail: false, pending: null };
    window.emit = data => dispatchEvent(new MessageEvent('message', { data }));
    window.answerPaste = () => {
      const request = clipboardQa.pending;
      clipboardQa.pending = null;
      emit({ type: 'clipboardText', requestId: request.requestId, ...(clipboardQa.fail ? { error: true } : { text: clipboardQa.text.slice(0, 16000) }) });
    };
    window.acquireVsCodeApi = () => ({ postMessage(message) {
      clipboardQa.requests.push(message);
      if (message.type === 'canvasConnect') emit({ type: 'canvasConnected', sessionId: 'paste-session', viewId: 'paste-view' });
      if (message.type === 'readClipboard') { clipboardQa.pending = message; if (!clipboardQa.held) answerPaste(); }
      if (message.type === 'clipboard' && message.requestId) {
        if (!clipboardQa.fail) clipboardQa.writes.push(message.text);
        emit({ type: 'clipboardWritten', requestId: message.requestId, ...(clipboardQa.fail ? { error: true } : {}) });
      }
    } });
  });
  const frame = page.frameLocator('#app');
  const field = frame.getByRole('textbox', { name: 'Message to coding agent' });
  let sequence = 0;
  const host = message => page.evaluate(message => emit({ type: 'canvasMessage', message }), {
    ...message, protocolVersion: 1, workspaceSessionId: 'paste-session', messageId: `paste-${++sequence}`,
    ...(message.type === 'agentState' ? {} : { hostSequence: sequence }),
  });
  const reads = () => page.evaluate(() => clipboardQa.requests.filter(message => message.type === 'readClipboard').length);
  const seed = text => page.evaluate(text => { clipboardQa.text = text; }, text);
  try {
    await page.goto(shellOrigin);
    await frame.getByRole('heading', { name: 'Explore your model' }).waitFor();
    const target = { workspaceFolder: '/fixture', file: 'model.py', cls: fixture.after.cls, documentVersion: 1,
      sourceDigest: fixture.after.sourceDigest, backendRevision: fixture.after.revision };
    await host({ type: 'activeTarget', target });
    await host({ type: 'projection', target, projection: fixture.after, traced: false, greyed: false });
    await frame.getByRole('button', { name: 'Open agent chat', exact: true }).click();
    await field.fill('before OLD after');
    assert.equal(await reads(), 0, 'opening/focusing/typing never reads clipboard');
    for (const key of ['Meta+v', 'Control+v', 'Meta+Shift+v']) {
      await field.fill('before OLD after');
      await field.evaluate(node => node.setSelectionRange(7, 10));
      await seed('two\r\nlines');
      await field.press(key);
      await expect(field).toHaveValue('before two\nlines after');
      await field.evaluate(node => new Promise(resolve => requestAnimationFrame(() => resolve(node.selectionStart)))).then(caret => assert.equal(caret, 16));
      await field.press(editModifier + '+z');
      await expect(field).toHaveValue('before OLD after');
      await field.press(editModifier + '+Shift+z');
      await expect(field).toHaveValue('before two\nlines after');
    }
    await field.fill('keep');
    await page.evaluate(() => { clipboardQa.held = true; clipboardQa.text = 'late text'; });
    await field.press('Meta+v');
    await page.waitForFunction(() => clipboardQa.pending !== null);
    await field.press('End'); await field.press('!');
    await page.evaluate(() => answerPaste());
    await field.evaluate(node => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve(node.value)))));
    assert.equal(await field.inputValue(), 'keep!', 'late reply preserves continued typing');
    await page.evaluate(() => { clipboardQa.held = false; clipboardQa.fail = true; });
    await field.press('Meta+v');
    await frame.getByRole('alert').filter({ hasText: 'Could not paste. Try again.' }).waitFor();
    assert.equal(await field.inputValue(), 'keep!');
    await page.evaluate(() => { clipboardQa.fail = false; });
    const response = 'The model needs its own environment.\nCreate it, install the required libraries, then continue.';
    await host({ type: 'agentState', viewId: 'paste-view', state: { sessionId: 'chat-session', revision: 1, provider: 'codex', status: 'idle',
      messages: [{ id: 'reply', role: 'assistant', text: response }], activity: 'Ready' } });
    const reply = frame.getByText(response, { exact: true });
    await reply.waitFor();
    assert.equal(await page.evaluate(() => clipboardQa.writes.length), 0, 'receiving agent text never copies it');
    for (const shortcut of ['Meta+c', 'Control+c']) {
      const before = await page.evaluate(() => clipboardQa.writes.length);
      await reply.evaluate(node => {
        const range = document.createRange(); range.setStart(node.firstChild, 0); range.setEnd(node.firstChild, 15);
        const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
      });
      await page.keyboard.press(shortcut);
      await page.waitForFunction(before => clipboardQa.writes.length === before + 1 && clipboardQa.writes.at(-1) === 'The model needs', before);
    }
    await frame.getByRole('button', { name: 'Copy agent response' }).click();
    await page.waitForFunction(response => clipboardQa.writes.at(-1) === response, response);
    await page.evaluate(() => { clipboardQa.fail = true; });
    await frame.getByRole('button', { name: 'Copy agent response' }).click();
    await frame.getByRole('alert').filter({ hasText: 'Could not copy. Try again.' }).waitFor();
    assert.equal(await page.evaluate(() => clipboardQa.requests.filter(message => message.type === 'canvasMessage' && message.message?.action === 'send').length), 0, 'paste never submits');
    assert.deepEqual(errors, []); assert.deepEqual(external, []);
    const result = { status: 'passed', scope: 'compiled native shell + staged production React across HTTP origins in Chromium; simulated host clipboard, not installed VS Code',
      checks: ['no clipboard reads on open/focus/typing', 'Cmd+V, Ctrl+V and plain-text paste', 'selection replacement and caret', 'native Undo/Redo', 'multiline CRLF normalization', 'late reply preserves continued typing', 'host failure preserves draft with recovery', 'Cmd+C and Ctrl+C copy selected agent text', 'Copy agent response preserves the full reply', 'copy failure offers retry', 'no automatic clipboard writes', 'clipboard actions never send', 'no external requests or JavaScript errors'] };
    fs.writeFileSync(path.join(output, 'result.json'), JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify(result, null, 2));
  } finally { await browser.close(); await close(shell); await close(app); }
}
run().catch(error => { console.error(error); process.exitCode = 1; });
