/** Render the real local home HTML in Chromium; this is not an installed VS Code check. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('../../canvas/node_modules/playwright');
const { canvasHomeHtml } = require('../out/canvasHome');

const output = path.resolve(process.argv[2] || '.tensorviz-workflow/canvas-home-ui');
const themes = {
  dark: { background: '#181818', editor: '#1f1f1f', foreground: '#cccccc', description: '#9d9d9d', accent: '#3794ff',
    button: '#0078d4', buttonText: '#ffffff', secondary: '#313131', border: '#454545', hover: '#2a2d2e' },
  light: { background: '#f8f8f8', editor: '#ffffff', foreground: '#3b3b3b', description: '#616161', accent: '#005fb8',
    button: '#005fb8', buttonText: '#ffffff', secondary: '#e5e5e5', border: '#d4d4d4', hover: '#e8e8e8' },
  contrast: { background: '#000000', editor: '#000000', foreground: '#ffffff', description: '#ffffff', accent: '#ffff00',
    button: '#000000', buttonText: '#ffffff', secondary: '#000000', border: '#ffffff', hover: '#141414' },
};

async function run() {
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const errors = [];
  const requests = [];
  const context = await browser.newContext();
  await context.route('**/*', route => { requests.push(route.request().url()); return route.abort(); });
  await context.addInitScript(() => {
    window.homeMessages = [];
    window.acquireVsCodeApi = () => ({ postMessage: message => {
      if (message.type !== 'homeReady') window.homeMessages.push(message);
    } });
  });
  const page = await context.newPage();
  page.on('pageerror', error => errors.push(error.message));
  const state = { hasWorkspace: true, indexed: false, models: [] };
  async function render(theme, width, home = state) {
    await page.setViewportSize({ width, height: 760 });
    await page.goto('about:blank');
    await page.setContent(canvasHomeHtml(home));
    await page.addStyleTag({ content: `:root {
      --vscode-font-family: -apple-system, BlinkMacSystemFont, sans-serif; --vscode-font-size: 13px;
      --vscode-sideBar-background: ${theme.background}; --vscode-editor-background: ${theme.editor};
      --vscode-foreground: ${theme.foreground}; --vscode-descriptionForeground: ${theme.description};
      --vscode-textLink-foreground: ${theme.accent}; --vscode-focusBorder: ${theme.accent};
      --vscode-button-background: ${theme.button}; --vscode-button-foreground: ${theme.buttonText};
      --vscode-button-hoverBackground: ${theme.button}; --vscode-button-border: ${theme.border};
      --vscode-button-secondaryBackground: ${theme.secondary}; --vscode-button-secondaryForeground: ${theme.foreground};
      --vscode-button-secondaryHoverBackground: ${theme.hover}; --vscode-widget-border: ${theme.border};
      --vscode-list-hoverBackground: ${theme.hover}; }` });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, 'no horizontal overflow');
  }
  try {
    await render(themes.dark, 360);
    assert.equal(await page.locator('body').getAttribute('data-motion'), 'playing');
    const first = await page.locator('.signal').first().evaluate(node => getComputedStyle(node).strokeDashoffset);
    await page.waitForTimeout(200);
    assert.notEqual(await page.locator('.signal').first().evaluate(node => getComputedStyle(node).strokeDashoffset), first,
      'signals must actually animate in the empty screen');
    assert.equal(await page.getByRole('button', { name: /(?:Pause|Play) animation/ }).count(), 0);
    assert.deepEqual(await page.evaluate(() => window.homeMessages), [], 'animation cannot request setup');
    await page.evaluate(() => {
      Object.defineProperty(document, 'hidden', { configurable: true, value: true });
      document.dispatchEvent(new Event('visibilitychange'));
    });
    assert.equal(await page.locator('body').getAttribute('data-motion'), 'paused');
    await page.evaluate(() => {
      delete document.hidden;
      document.dispatchEvent(new Event('visibilitychange'));
    });
    assert.equal(await page.locator('body').getAttribute('data-motion'), 'playing');
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.waitForFunction(() => document.body.dataset.motion === 'paused');
    assert.equal(await page.locator('.signal').first().evaluate(node => getComputedStyle(node).animationName), 'none');
    await page.screenshot({ path: path.join(output, 'home-reduced-motion.png') });
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await page.keyboard.press('Tab');
    await page.getByRole('button', { name: 'Open as Graph' }).focus();
    assert.equal(await page.locator(':focus').evaluate(node => getComputedStyle(node).outlineWidth), '2px');
    await page.keyboard.press('Enter');
    assert.deepEqual(await page.evaluate(() => window.homeMessages), [{ type: 'homeAction', action: 'open' }]);
    await page.locator(':focus').blur();
    await page.screenshot({ path: path.join(output, 'home-dark.png'), animations: 'disabled' });

    await render(themes.light, 360, { ...state, hasWorkspace: false });
    await page.getByRole('button', { name: 'Open Folder', exact: true }).click();
    assert.deepEqual(await page.evaluate(() => window.homeMessages), [{ type: 'homeAction', action: 'folder' }]);
    await page.locator(':focus').blur();
    await page.screenshot({ path: path.join(output, 'home-light.png'), animations: 'disabled' });
    await render(themes.dark, 200);
    await page.screenshot({ path: path.join(output, 'home-narrow.png'), fullPage: true, animations: 'disabled' });
    await render(themes.contrast, 320);
    await page.locator('body').evaluate(node => node.classList.add('vscode-high-contrast'));
    await page.screenshot({ path: path.join(output, 'home-contrast.png'), animations: 'disabled' });

    await render(themes.dark, 260, { ...state, indexed: true, models: [
      { name: 'WhisperEncoder', file: 'whisper/model.py', uri: 'file:///not-exposed.py' },
      { name: '<script>window.injected=true</script>', file: 'models/' + 'long_name_'.repeat(8) + '.py', uri: 'file:///other.py' },
    ] });
    assert.equal(await page.locator('.network').count(), 0, 'model selection should not run a decorative animation');
    assert.equal(await page.evaluate(() => window.injected), undefined, 'model labels stay text');
    await page.getByRole('button', { name: /WhisperEncoder/ }).click();
    assert.deepEqual(await page.evaluate(() => window.homeMessages), [{ type: 'homeModel', index: 0 }]);
    assert.ok(!(await page.content()).includes('file:///not-exposed.py'), 'host-owned target URIs stay outside the HTML');
    await render(themes.dark, 260, { ...state, indexed: true, models: [
      { name: 'WhisperEncoder', file: 'whisper/model.py', uri: 'file:///encoder.py' },
      { name: 'WhisperDecoder', file: 'whisper/model.py', uri: 'file:///decoder.py' },
    ] });
    await page.mouse.move(0, 0);
    await page.screenshot({ path: path.join(output, 'home-models.png'), fullPage: true });
    await render(themes.light, 420, { ...state, indexed: true });
    assert.ok(await page.getByText('No PyTorch models were found', { exact: false }).isVisible());
    const recovery = { ...state, setup: { phase: 'needs_setup', detail: 'The tensorViz runtime needs repair. Set up tensorViz again or inspect the log.' } };
    for (const [name, theme] of Object.entries(themes)) {
      await render(theme, 320, recovery);
      if (name === 'contrast') await page.locator('body').evaluate(node => node.classList.add('vscode-high-contrast'));
      await page.getByRole('button', { name: 'Set up tensorViz' }).waitFor();
      await page.screenshot({ path: path.join(output, 'setup-' + name + '.png'), fullPage: true, animations: 'disabled' });
    }
    await page.setViewportSize({ width: 320, height: 360 });
    await page.getByRole('button', { name: 'Set up tensorViz' }).scrollIntoViewIfNeeded();
    assert.equal(await page.getByRole('button', { name: 'Set up tensorViz' }).isVisible(), true, 'short views can scroll to recovery');
    await page.screenshot({ path: path.join(output, 'setup-short.png'), animations: 'disabled' });
    await render(themes.dark, 320, { ...state, setup: { phase: 'ready',
      detail: 'Choose a model or open a pull request. Model dependencies are checked after you select a model.' } });
    await page.getByText('tensorViz is ready', { exact: true }).waitFor();
    assert.equal(await page.getByRole('button', { name: 'Choose Python' }).count(), 0);
    assert.equal(await page.getByText(/Incompatible:|Missing:|Selected in VS Code/).count(), 0);
    await page.screenshot({ path: path.join(output, 'runtime-ready.png'), fullPage: true, animations: 'disabled' });
    assert.deepEqual(requests, [], 'the local starting screen needs no network assets');
    assert.deepEqual(errors, [], 'the nonce script executes without JavaScript errors');
    const result = { status: 'passed', scope: 'real compiled home HTML in standalone Chromium with mocked VS Code message API',
      checks: ['animated signals without playback buttons or setup messages', 'hidden-document pause', 'live reduced-motion preference',
        'keyboard action and visible focus', 'Open Folder action', 'host-owned model index message', 'escaped labels',
        '200px layout', 'light/dark/high-contrast renders', 'themed runtime recovery and short-view scrolling',
        'ready home has no project Python warnings or picker', 'no network requests or JavaScript errors'],
      screenshots: fs.readdirSync(output).filter(file => file.endsWith('.png')) };
    fs.writeFileSync(path.join(output, 'result.json'), JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify(result, null, 2));
  } finally { await browser.close(); }
}

run().catch(error => { console.error(error); process.exitCode = 1; });
