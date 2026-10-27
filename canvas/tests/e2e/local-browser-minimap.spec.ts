import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';

const pageFile = process.env.TENSORVIZ_QA_PAGE_FILE;
test.skip(!pageFile, 'Requires the installed host on the complex-model fixture');

for (const [width, height] of [[698, 994], [1024, 768], [1440, 900]]) {
  test(`minimap fits the full-width canvas at ${width}x${height}`, async ({ page }) => {
    await page.setViewportSize({ width, height });
    await page.goto(JSON.parse(fs.readFileSync(pageFile!, 'utf8')).pageUrl);
    const overview = page.getByRole('region', { name: 'Graph overview', exact: true });
    await expect(overview).toBeVisible();
    const initial = await overview.boundingBox();
    expect(initial!.x + initial!.width).toBeLessThanOrEqual(width);
    expect(initial!.y + initial!.height).toBeLessThanOrEqual(height);
    await expect(page.locator('.ui2-inspector-overlay, .ui2-inspector-tab')).toHaveCount(0);
    expect(initial!.x).toBeGreaterThanOrEqual(0);
    expect(await page.evaluate(() => ({ width: document.documentElement.scrollWidth, height: document.documentElement.scrollHeight }))).toEqual({ width, height });
    const evidence = path.join(path.dirname(pageFile!), '../evidence');
    await page.screenshot({ path: path.join(evidence, `minimap-${width}x${height}.png`) });
  });
}

test('minimap mouse and keyboard navigation change the viewport without changing graph selection', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(JSON.parse(fs.readFileSync(pageFile!, 'utf8')).pageUrl);
  await page.getByRole('button', { name: 'Expand toolbar' }).click();
  await page.getByRole('button', { name: 'More tools', exact: true }).press('ControlOrMeta+p');
  await page.getByRole('textbox', { name: 'Quick jump to node' }).fill('branch.fc');
  await page.getByRole('button', { name: 'branch.fc — Linear', exact: true }).click();
  const nav = page.getByRole('group', { name: 'Minimap navigation', exact: true });
  const card = page.locator('[data-shape-id="shape:node:branch.fc"]');
  const before = await card.boundingBox();
  await nav.press('+');
  await expect.poll(async () => (await card.boundingBox())?.width ?? 0).toBeGreaterThan(before!.width);
  const zoomed = await card.boundingBox();
  await nav.press('ArrowRight');
  await expect.poll(async () => (await card.boundingBox())?.x ?? 0).toBeLessThan(zoomed!.x);
  const canvas = page.getByTestId('minimap.canvas');
  const minimapImage = () => canvas.evaluate(element => (element as HTMLCanvasElement).toDataURL());
  const beforeMouse = await minimapImage();
  const box = await canvas.boundingBox();
  await canvas.dblclick({ position: { x: box!.width * .85, y: box!.height * .5 } });
  await expect.poll(minimapImage).not.toBe(beforeMouse);
  await nav.press('Home');
  await expect(page.locator('[data-shape-id="shape:node:branch.fc"] .layer-node')).toHaveClass(/\bselected\b/);
  await nav.press('Escape');
  await expect(page.getByRole('button', { name: 'Show minimap', exact: true })).toBeFocused();
  await expect(nav).toHaveCount(0);
  await page.getByRole('button', { name: 'Show minimap', exact: true }).press('Enter');
  await expect(nav).toBeVisible();
  await expect(page.getByTitle('Model summary', { exact: true })).toContainText('102');
});

test('released minimap drags stop panning and leave node interaction usable', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(JSON.parse(fs.readFileSync(pageFile!, 'utf8')).pageUrl);
  const nav = page.getByRole('group', { name: 'Minimap navigation', exact: true });
  const minimap = page.getByTestId('minimap.canvas');
  const card = page.locator('[data-shape-id="shape:node:branch_fc"]');
  await expect(card).toBeVisible();
  await nav.press('Home');
  const box = (await minimap.boundingBox())!;
  const start = { x: box.x + box.width * .45, y: box.y + box.height * .5 };
  const end = { x: start.x + 20, y: start.y };
  const cardPosition = () => card.evaluate(element => {
    const rect = element.getBoundingClientRect();
    return { x: rect.x, y: rect.y };
  });
  await page.mouse.move(start.x, start.y);
  const before = await cardPosition();
  await page.mouse.down();
  await page.mouse.move(end.x, end.y, { steps: 8 });
  await page.mouse.up();
  await expect.poll(cardPosition).not.toEqual(before);
  const released = await cardPosition();
  // Re-entering the minimap with no button pressed must never continue the drag.
  await page.mouse.move(box.x - 30, box.y + box.height + 30);
  await page.mouse.move(start.x - 15, start.y + 15, { steps: 8 });
  await expect.poll(cardPosition, { timeout: 2000 }).toEqual(released);
  await nav.press('Home');
  const selectedBox = (await card.boundingBox())!;
  await page.mouse.click(selectedBox.x + selectedBox.width / 2, selectedBox.y + selectedBox.height / 2);
  await expect(page.locator('[data-shape-id="shape:node:branch_fc"] .layer-node')).toHaveClass(/\bselected\b/);
  await nav.press('Escape');
  await expect(page.getByRole('button', { name: 'Show minimap', exact: true })).toBeFocused();
  await page.getByRole('button', { name: 'Show minimap', exact: true }).press('Enter');
  await expect(nav).toBeVisible();
  await expect(page.getByTitle('Model summary', { exact: true })).toContainText('102');
});

test('canvas uses visible system cursors over nodes and during hand panning', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(JSON.parse(fs.readFileSync(pageFile!, 'utf8')).pageUrl);
  const card = page.locator('[data-shape-id="shape:node:branch_fc"]');
  await expect(card).toBeVisible();
  await page.getByRole('button', { name: 'Fit graph to view', exact: true }).click();
  // tldraw hit-tests the canvas; node HTML deliberately ignores pointer events.
  const cardBox = (await card.boundingBox())!;
  await page.mouse.move(cardBox.x + cardBox.width / 2, cardBox.y + cardBox.height / 2);
  await expect(card).toHaveCSS('cursor', 'default', { timeout: 3000 });
  const canvas = page.locator('.tl-canvas');
  const box = (await canvas.boundingBox())!;
  const point = { x: box.x + box.width * .4, y: box.y + box.height * .8 };
  await page.mouse.click(point.x, point.y);
  await page.keyboard.down('Space');
  try {
    await expect(canvas).toHaveCSS('cursor', 'grab');
    await page.mouse.down();
    try {
      await page.mouse.move(point.x + 60, point.y + 20, { steps: 8 });
      await expect(canvas).toHaveCSS('cursor', 'grabbing');
    } finally {
      await page.mouse.up();
    }
    await expect(canvas).toHaveCSS('cursor', 'grab');
  } finally {
    await page.keyboard.up('Space');
  }
  await page.mouse.move(point.x + 65, point.y + 25);
  await expect(canvas).toHaveCSS('cursor', 'default');
  await page.getByRole('button', { name: 'Fit graph to view', exact: true }).click();
  const selectedBox = (await card.boundingBox())!;
  await page.mouse.click(selectedBox.x + selectedBox.width / 2, selectedBox.y + selectedBox.height / 2);
  await expect(card).toHaveCSS('cursor', 'default', { timeout: 3000 });
  await expect(page.locator('[data-shape-id="shape:node:branch_fc"] .layer-node')).toHaveClass(/\bselected\b/);
});
