import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { createGraphFile } from '../../src/services/graphFile';
import type { ModelGraph } from '../../src/types';

// The production viewer must also rename graphs that predate captureRef metadata.
// Build first: npm run build:viewer
const dist = path.resolve(__dirname, '../../viewer-dist');
const origin = 'http://127.0.0.1:4175';
const operations = [
  ['aten._native_batch_norm_legit_no_training.default', 'Batch norm'],
  ['aten.max_pool2d_with_indices.default', 'Max pool 2D'],
  ['torch.ops.aten.upsample_bilinear2d.default', 'Bilinear resize 2D'],
  ['aten::embedding', 'Embedding lookup'],
  ['aten.hardtanh_.default', 'Hard tanh (in place)'],
  ['aten._scaled_dot_product_cudnn_attention_backward.default', 'Scaled attention backward'],
];
const graph: ModelGraph = {
  nodes: operations.map(([target], index) => ({
    id: `fx_${index}`, type: 'layerNode', position: { x: 450 + (index % 2) * 320, y: Math.floor(index / 2) * 220 },
    data: {
      layerType: { id: target, name: target, displayName: target, category: 'misc', icon: 'Box', description: '', parameters: [] },
      parameters: {}, outputShape: [1, 8, 16, 16],
    },
  })), edges: [],
};
const file = createGraphFile(graph, { name: 'Imported ATen operations', status: 'traced' });

test('legacy graph tiles, both searches and downloaded graph keep consistent names and exact targets', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.route('**/*', async route => {
    const url = new URL(route.request().url());
    if (url.origin !== origin) return route.abort();
    const asset = path.resolve(dist, `.${url.pathname === '/' ? '/index.html' : url.pathname}`);
    if (!asset.startsWith(`${dist}${path.sep}`)) return route.abort();
    await route.fulfill({ path: asset });
  });
  await page.goto(origin);
  await page.getByLabel('Graph file', { exact: true }).setInputFiles({
    name: 'legacy.tensorviz.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(file)),
  });
  for (const [index, [target, title]] of operations.entries()) {
    const tile = page.locator(`[data-node-id="fx_${index}"]`);
    await expect(tile.locator('.ui2-node-title')).toHaveText(title);
    await expect(tile).not.toContainText(target);
  }
  const search = page.getByRole('textbox', { name: 'Search layers' });
  for (const query of ['Max pool 2D', 'aten.max_pool2d']) {
    await search.fill(query);
    await expect(page.getByLabel('Search results').getByRole('button')).toContainText('Max pool 2D');
  }
  await search.press('Enter');
  await expect(page.getByRole('complementary', { name: 'Layer details' }).getByRole('heading')).toHaveText('Max pool 2D');
  await page.getByRole('searchbox', { name: 'Search the graph' }).fill('type=Max pool');
  await expect(page.locator('.graph-query-results button strong')).toHaveText('Max pool 2D');

  const downloading = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download graph file', exact: true }).click();
  const download = await downloading;
  const saved = JSON.parse(await readFile((await download.path())!, 'utf8'));
  expect(saved.document).toEqual(file.document);
});
