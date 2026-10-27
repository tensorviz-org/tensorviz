import { test, expect, Page } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { selectedNodeEdit } from './local-browser-node-controls';

// The real disposable installed extension supplies its authenticated page URL.
const pageFile = process.env.TENSORVIZ_QA_PAGE_FILE;
test.skip(!pageFile, 'Requires the disposable installed-VSIX host and TENSORVIZ_QA_PAGE_FILE');
const open = async (page: Page) => {
  await page.goto(JSON.parse(fs.readFileSync(pageFile!, 'utf8')).pageUrl);
  await page.getByRole('button', { name: 'Expand toolbar' }).click();
};
async function select(page: Page) {
  await page.getByRole('button', { name: 'More tools', exact: true }).press('ControlOrMeta+p');
  const jump = page.getByRole('textbox', { name: 'Quick jump to node' });
  await jump.fill('branch_fc');
  await jump.press('Tab');
  await page.keyboard.press('Enter');
  await expect(page.locator('[data-shape-id="shape:node:branch_fc"] .layer-node')).toHaveClass(/\bselected\b/);
}

test('selection works without a chatbot, Inspector or reserved canvas space', async ({ page }) => {
  await open(page);
  await select(page);
  await page.keyboard.press('i');
  await expect(page.locator('.ui2-inspector-overlay, .ui2-inspector-tab')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Ask about this model', exact: true })).toHaveCount(0);
  await expect(page.locator('.ui2-builder-stage')).toHaveCSS('--ui2-canvas-inset-left', '0px');
  await expect(page.getByRole('button', { name: 'Reveal in editor', exact: true })).toBeEnabled();

  // What replaced the sidebar must not reach this page either: the parameter
  // editor edits source, and the browser page does not write source. Clicking the
  // card is the gesture that opens it everywhere it IS allowed.
  const card = await page.locator('[data-shape-id="shape:node:enc2.fc"]').boundingBox();
  if (!card) throw new Error('the enc2.fc card has no bounding box');
  await page.mouse.click(card.x + card.width / 2, card.y + 14);
  await expect(page.getByTestId('node-parameters')).toHaveCount(0);
});

for (const [width, height] of [[698, 994], [1024, 768], [1440, 900]]) {
  test(`installed controls and source form fit ${width}x${height}`, async ({ page }) => {
    await page.setViewportSize({ width, height });
    await open(page);
    await select(page);
    for (const name of ['Find a layer', 'Fit graph to view', 'More tools']) {
      const box = await page.getByRole('button', { name, exact: true }).boundingBox();
      expect(box).not.toBeNull();
      expect(box!.x).toBeGreaterThanOrEqual(0);
      expect(box!.x + box!.width).toBeLessThanOrEqual(width);
      expect(box!.y + box!.height).toBeLessThanOrEqual(height);
    }
    await expect(page.locator('.local-browser-header [data-testid="local-browser-validate"]')).toBeInViewport();
    await page.getByRole('button', { name: 'More tools', exact: true }).click();
    await expect(page.getByRole('menuitem', { name: 'Auto layout', exact: true })).toBeInViewport();
    await page.getByRole('button', { name: 'More tools', exact: true }).press('Escape');
    const canvas = page.locator('.tldraw-canvas-wrapper');
    const before = await canvas.boundingBox();
    await (await selectedNodeEdit(page)).click();
    await expect(page.getByRole('region', { name: 'Edit node source', exact: true })).toBeInViewport();
    expect(await canvas.boundingBox()).toEqual(before);
    await page.getByRole('button', { name: 'Close source editor' }).click();
    expect(await page.evaluate(() => ({ width: document.documentElement.scrollWidth, height: document.documentElement.scrollHeight }))).toEqual({ width, height });
  });
}

test('source form Escape preserves selection, focus and read-only architecture', async ({ page }) => {
  const failures: string[] = [];
  page.on('pageerror', error => failures.push(String(error)));
  page.on('response', response => { if (response.status() === 401) failures.push('HTTP 401'); });
  page.on('request', request => { if (/\/api\/(projects|ai)(?:[/?]|$)/.test(request.url())) failures.push('Legacy API request'); });
  await open(page);
  await select(page);
  const edit = await selectedNodeEdit(page);
  await edit.click();
  await page.getByRole('textbox', { name: 'Parameter value' }).press('Escape');
  await expect(page.getByRole('region', { name: 'Edit node source', exact: true })).toHaveCount(0);
  await expect(edit).toBeFocused();
  await expect(page.locator('[data-shape-id="shape:node:branch_fc"] .layer-node')).toHaveClass(/\bselected\b/);
  await page.getByRole('button', { name: 'Move focus to canvas', exact: true }).press('Enter');
  await expect.poll(() => page.evaluate(() => document.activeElement?.getAttribute('role'))).toBe('application');
  const count = await page.getByTitle('Model summary', { exact: true }).innerText();
  for (const key of ['Delete', 'Backspace', 'ControlOrMeta+d']) await page.keyboard.press(key);
  await expect(page.getByTitle('Model summary', { exact: true })).toHaveText(count);
  await expect(page.locator('[title="Drag to connect"], [title="Drop a connection here"]')).toHaveCount(0);
  expect(failures).toEqual([]);
});

test('a reconnected installed canvas can still navigate to the exact source file', async ({ page }) => {
  let closeView = () => {};
  let connections = 0;
  await page.routeWebSocket('**/local/ws*', socket => {
    connections++;
    socket.connectToServer();
    closeView = () => socket.close({ code: 1012, reason: 'QA network interruption' });
  });
  await open(page);
  await select(page);
  const before = connections;
  closeView();
  await expect.poll(() => connections).toBeGreaterThan(before);
  await select(page);
  await page.getByRole('button', { name: 'Reveal in editor', exact: true }).click();
  await expect.poll(() => JSON.parse(fs.readFileSync(path.join(path.dirname(pageFile!), 'state.json'), 'utf8')).editor?.file).toMatch(/complex_models\/network\.py$/);
  await expect(page.getByTitle('Model summary', { exact: true })).toContainText('102');
});
