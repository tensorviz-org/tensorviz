import { test, expect, Page } from '@playwright/test';
import fs from 'node:fs';

const pageFile = process.env.TENSORVIZ_QA_PAGE_FILE;
test.skip(!pageFile, 'Requires a disposable extension host and TENSORVIZ_QA_PAGE_FILE');
const open = (page: Page) => page.goto(JSON.parse(fs.readFileSync(pageFile!, 'utf8')).pageUrl);
const more = (page: Page) => page.getByRole('button', { name: 'More tools', exact: true });

for (const width of [300, 380, 600, 601, 1280]) {
  test(`overview and connection focus stay separate at ${width}px`, async ({ page }) => {
    test.setTimeout(45_000);
    await page.setViewportSize({ width, height: 820 });
    await open(page);
    await page.getByRole('button', { name: 'Expand toolbar' }).click();
    await page.getByRole('button', { name: 'Find a layer', exact: true }).click();
    await page.getByRole('textbox', { name: 'Quick jump to node' }).fill('head');
    await page.getByRole('button', { name: /(?:^head —| — head · Linear$)/ }).click();
    const focus = page.getByRole('region', { name: 'Connection focus', exact: true });
    const overview = page.getByRole('region', { name: 'Graph overview', exact: true });
    await expect(focus).toBeVisible();
    for (const expanded of [false, true, false]) {
      if (expanded !== (await overview.getAttribute('data-expanded') === 'true')) {
        await overview.getByRole('button').click();
      }
      const a = (await focus.boundingBox())!, b = (await overview.boundingBox())!;
      expect(a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height,
        `selection ${JSON.stringify(a)} intersects overview ${JSON.stringify(b)}`).toBe(false);
      const count = focus.locator('.graph-focus-caption span');
      await expect(count).toContainText('connections');
      for (const control of [count, overview.getByRole('button'), focus.getByRole('button', { name: 'Trace upstream inputs' })]) {
        expect(await control.evaluate(element => {
          const r = element.getBoundingClientRect();
          return r.left >= 0 && r.right <= innerWidth && element.contains(document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2));
        })).toBe(true);
      }
      await focus.getByRole('button', { name: 'Trace upstream inputs' }).click();
      await expect(focus.getByRole('button', { name: 'Trace upstream inputs' })).toHaveAttribute('aria-pressed', 'true');
      await focus.getByRole('button', { name: 'Neighbors', exact: true }).click();
      if (expanded && width === 300) await page.screenshot({ path: test.info().outputPath('overview-expanded.png') });
    }
    const canvas = (await page.locator('.tldraw-canvas-wrapper').boundingBox())!;
    await page.mouse.click(canvas.x + canvas.width - 4, canvas.y + canvas.height / 2);
    await expect(focus).toHaveCount(0);
    await expect(overview).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(width);
  });
}

for (const width of [1280, 380, 320]) {
  test(`canvas chrome stays reachable at ${width}px without covering the graph by default`, async ({ page }) => {
    await page.setViewportSize({ width, height: 820 });
    await open(page);
    await expect(page.locator('.tl-shapes [data-shape-id^="shape:node:"]').first()).toBeVisible();
    await expect(page.getByText('Connection guide', { exact: true })).toHaveCount(0);
    await expect(page.getByRole('searchbox', { name: 'Search the graph' })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Show minimap' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Find a layer' })).toHaveCount(0);
    for (const name of ['Layers', 'All operations', 'Change inputs', 'Refresh graph', 'Expand toolbar']) {
      await expect(page.getByRole('button', { name, exact: true })).toBeInViewport();
    }
    await page.getByRole('button', { name: 'Expand toolbar' }).click();
    for (const name of ['Find a layer', 'Fit graph to view', 'More tools', 'Canvas guide and shortcuts']) {
      await expect(page.getByRole('button', { name, exact: true })).toBeInViewport();
    }
    await more(page).click();
    const menu = page.getByRole('menu', { name: 'Canvas' });
    for (const group of ['View', 'Tools', 'Navigate', 'Export']) await expect(menu.getByRole('group', { name: group, exact: true })).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'Download SVG' })).toBeInViewport();
    await menu.press('?');
    const guide = page.getByRole('dialog', { name: 'Canvas guide' });
    await expect(guide).toBeFocused();
    await expect(menu).toHaveCount(0);
    await guide.press('Escape');
    await more(page).click();
    await menu.press('Escape');
    await expect(more(page)).toBeFocused();
    await page.getByRole('button', { name: 'Canvas guide and shortcuts' }).click();
    await expect(guide).toContainText('Residual / skip');
    await guide.press('Escape');
    await page.getByRole('button', { name: 'Find a layer' }).click();
    const search = page.getByRole('textbox', { name: 'Quick jump to node' });
    await search.fill('type=Linear');
    await expect(page.locator('.local-canvas-search .ui2-quickjump-item').first()).toBeVisible();
    await search.press('Escape');
    await expect(page.getByRole('button', { name: 'Find a layer' })).toBeFocused();
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(width);
  });
}

test('menus keep renderer, zoom, overlay and fullscreen actions connected to the live canvas', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(String(error)));
  await open(page);
  await page.getByRole('button', { name: 'Expand toolbar' }).click();
  await expect(more(page)).toBeVisible();
  await more(page).click();
  await page.getByRole('menuitem', { name: 'Display settings' }).click();
  await page.getByRole('button', { name: 'Parameters', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Parameters', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Back to canvas actions' }).click();
  await page.getByRole('menuitemcheckbox', { name: 'SVG overview' }).click();
  await expect(page.getByTestId('graph-svg-layer')).toBeVisible();
  const transform = () => page.getByTestId('graph-svg-content').getAttribute('style');
  await page.getByRole('button', { name: 'Fit graph to view' }).click();
  const before = await transform();
  await page.getByRole('button', { name: 'Zoom options' }).click();
  await page.getByRole('button', { name: /Readable detail/ }).click();
  await expect.poll(transform).not.toBe(before);
  await more(page).click();
  await page.getByRole('menuitem', { name: 'Fullscreen canvas' }).click();
  await expect(page.locator('.ui2-builder-stage-fullscreen')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Canvas guide and shortcuts' })).toBeInViewport();
  await more(page).click();
  await page.getByRole('menuitem', { name: 'Exit fullscreen' }).click();
  await expect(page.locator('.ui2-builder-stage-fullscreen')).toHaveCount(0);
  expect(errors).toEqual([]);
});
