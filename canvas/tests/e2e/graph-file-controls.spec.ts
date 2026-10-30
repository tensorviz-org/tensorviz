import { expect, Locator, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { createGraphFile } from '../../src/services/graphFile';
import { nestedGraph } from '../../src/test/fixtures/graphFileFixture';

// Exercise the actual production viewer and CSS, without a backend or GUI host.
// Build first: npm run build:viewer
// Run: E2E_DISABLE_WEBSERVER=1 npx playwright test tests/e2e/graph-file-controls.spec.ts
const dist = path.resolve(__dirname, '../../viewer-dist');
const origin = 'http://127.0.0.1:4175';
test.setTimeout(30_000);
test.use({ actionTimeout: 10_000 });
const graph = nestedGraph();
for (const node of graph.nodes) {
  const shape = node.id === 'output' ? [1, 160] : [1, 320];
  Object.assign(node.data, {
    parameterCount: node.type === 'moduleNode' ? 20 : node.id === 'fc' ? 10 : 0,
    outputShape: shape, shapeEvidence: 'measured',
    outputTensor: { shape, dtype: 'float32', logicalBytes: shape[1] * 4 },
  });
}
const file = createGraphFile(graph, { name: 'Measured nested model', status: 'traced' });

async function expectReachable(control: Locator) {
  await control.scrollIntoViewIfNeeded();
  await expect(control).toBeVisible();
  // Visibility alone passes for the old overlapping panels. Check the actual
  // pointer target, including after results change the panel heights.
  expect(await control.evaluate(element => {
    const rect = element.getBoundingClientRect();
    const hit = document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2);
    return element.contains(hit) ? 'self' : `${hit?.tagName}: ${hit?.getAttribute('aria-label') || hit?.textContent}`;
  })).toBe('self');
}

test.beforeEach(async ({ page }) => {
  await page.route('**/*', async route => {
    const url = new URL(route.request().url());
    if (url.origin !== origin) return route.abort();
    const asset = path.resolve(dist, `.${url.pathname === '/' ? '/index.html' : url.pathname}`);
    if (!asset.startsWith(`${dist}${path.sep}`)) return route.abort();
    await route.fulfill({ path: asset });
  });
  await page.goto(origin);
  await page.getByLabel('Graph file', { exact: true }).setInputFiles({
    name: 'measured.tensorviz.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(file)),
  });
  await expect(page.locator('[data-node-id="block"]')).toBeVisible();
});

for (const viewport of [{ width: 1440, height: 900 }, { width: 1024, height: 768 }, { width: 740, height: 640 }, { width: 430, height: 680 }]) {
  test(`navigation and searches remain reachable at ${viewport.width}px`, async ({ page }) => {
    await page.setViewportSize(viewport);
    await expect(page.getByRole('button', { name: 'Output tensor bytes', exact: true })).toHaveCount(0);
    const search = page.getByRole('searchbox', { name: 'Search the graph' });
    await expectReachable(search);
    await search.fill('type=linear');
    const groupedHit = page.locator('.graph-query-results button');
    await expect(groupedHit).toContainText('block');
    await expectReachable(groupedHit);
    await page.getByRole('textbox', { name: 'Search layers' }).fill('module');
    await expect(page.getByLabel('Search results').getByRole('button')).toHaveCount(2);
    if (viewport.width === 430) {
      const stack = page.locator('.graph-focus-overlay');
      await page.getByRole('button', { name: 'Fit', exact: true }).hover();
      await page.mouse.wheel(0, 600);
      await expect.poll(() => stack.evaluate(element => element.scrollTop)).toBeGreaterThan(0);
    }
    await expectReachable(search);
    await expectReachable(page.getByText('Connection guide', { exact: true }));
    for (const name of ['Fit', 'Zoom in', 'Zoom out', 'Blocks', 'Expand all']) {
      await expectReachable(page.getByRole('button', { name, exact: true }));
    }
    await page.getByRole('textbox', { name: 'Search layers' }).press('Escape');
    await expect(page.getByLabel('Search results')).toHaveCount(0);
    await expect(page.getByRole('textbox', { name: 'Search layers' })).toBeFocused({ timeout: 3000 });
  });
}

test('search keeps its two navigation semantics and saved views/discussions keep source read-only', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.getByRole('searchbox', { name: 'Search the graph' }).fill('type=linear');
  await page.locator('.graph-query-results button').click();
  await expect(page.locator('[data-node-id="block"]')).toHaveClass(/collapsed/);
  await page.getByRole('button', { name: 'Clear search', exact: true }).click();
  await page.getByRole('textbox', { name: 'Search layers' }).fill('fc');
  await page.getByRole('textbox', { name: 'Search layers' }).press('Enter');
  await expect(page.locator('[data-node-id="block"]')).not.toHaveClass(/collapsed/);
  await expect(page.getByRole('complementary', { name: 'Layer details' })).toContainText('fc');

  await page.getByRole('button', { name: 'Saved views', exact: true }).click();
  await page.getByRole('button', { name: 'Share this view', exact: true }).click();
  await expect(page.getByLabel('Current saved view')).toContainText('This view opens first');
  await expectReachable(page.getByRole('searchbox', { name: 'Search the graph' }));
  await page.getByRole('button', { name: 'Blocks', exact: true }).click();
  await expect(page.locator('[data-node-id="block"]')).toHaveClass(/collapsed/);
  await page.getByRole('button', { name: 'Show saved view', exact: true }).click();
  await expect(page.locator('[data-node-id="block"]')).not.toHaveClass(/collapsed/);
  await page.getByRole('button', { name: 'Close saved views' }).click();
  await page.getByRole('button', { name: 'Discuss layer' }).click();
  await expect(page.getByRole('complementary', { name: 'Layer discussions' })).toContainText('Selected: fc');
  await page.getByRole('button', { name: 'Start discussion' }).click();
  await page.getByRole('textbox', { name: 'Message', exact: true }).fill('Measured layer retained.');
  await page.getByRole('button', { name: 'Add message' }).click();
  await expectReachable(page.getByRole('searchbox', { name: 'Search the graph' }));

  const downloaded = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download graph file', exact: true }).click();
  const download = await downloaded;
  const saved = JSON.parse(await readFile((await download.path())!, 'utf8'));
  expect(saved.document).toEqual(file.document);
  expect(saved.collaboration.savedViews[0].view.expandedModuleIds).toEqual(expect.arrayContaining(['block', 'inner']));
  expect(saved.collaboration.savedViews[0].view.selectedNodeIds).toEqual(['fc']);
  expect(saved.collaboration.threads).toHaveLength(1);
});
