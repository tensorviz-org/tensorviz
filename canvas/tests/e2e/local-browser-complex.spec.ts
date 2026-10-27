import { test, expect, Page } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';

const pageFile = process.env.TENSORVIZ_QA_PAGE_FILE;
test.skip(!pageFile, 'Requires the installed host on the complex-model fixture');
const control = () => path.dirname(pageFile!);
const openPage = async (page: Page) => {
  await page.goto(JSON.parse(fs.readFileSync(pageFile!, 'utf8')).pageUrl);
  await page.getByRole('button', { name: 'Expand toolbar' }).click();
  await expect(page.getByRole('banner')).toContainText('ComplexNet · complex_models/network.py');
  await expect(page.getByTitle('Model summary', { exact: true })).toContainText('102');
};
const jump = async (page: Page, id: string, type: string) => {
  await page.getByRole('button', { name: 'More tools', exact: true }).press('ControlOrMeta+p');
  await page.getByRole('textbox', { name: 'Quick jump to node' }).fill(id);
  await page.getByRole('button', { name: `${id} — ${type}`, exact: true }).click();
};

test('complex graph keeps dotted and flat identities, cards, source navigation separate', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await openPage(page);
  await jump(page, 'branch.fc', 'Linear');
  for (const [id, file] of [['branch.fc', 'heads.py'], ['branch_fc', 'network.py']]) {
    await jump(page, id, 'Linear');
    await expect(page.locator(`[data-shape-id="shape:node:${id}"] .layer-node`)).toHaveClass(/\bselected\b/);
    await expect(page.locator('.ui2-node-subtitle').filter({ hasText: new RegExp(`^#${id.replace('.', '\\.')}$`) })).toHaveCount(1);
    await page.getByRole('button', { name: 'Reveal in editor', exact: true }).click();
    await expect.poll(() => JSON.parse(fs.readFileSync(path.join(control(), 'state.json'), 'utf8')).editor?.file).toMatch(new RegExp(`complex_models/${file.replace('.', '\\.')}$`));
  }
  // Both cards exist simultaneously after branch expansion; neither overwrites the other.
  await expect(page.locator('.ui2-node-subtitle').filter({ hasText: /^#branch\.fc$/ })).toHaveCount(1);
  await expect(page.locator('.ui2-node-subtitle').filter({ hasText: /^#branch_fc$/ })).toHaveCount(1);
  expect(errors).toEqual([]);
});

test('four nested levels expand, collapse and reselect without losing repeated instances', async ({ page }) => {
  await openPage(page);
  for (const trunk of ['trunk1', 'trunk2', 'trunk1']) {
    const id = `${trunk}.stage2.block2.second.conv`;
    await jump(page, id, 'Conv2D');
    await expect(page.locator(`[data-shape-id="shape:node:${id}"] .layer-node`)).toHaveClass(/\bselected\b/);
    await page.getByRole('button', { name: 'Reveal in editor', exact: true }).click();
    await expect.poll(() => JSON.parse(fs.readFileSync(path.join(control(), 'state.json'), 'utf8')).editor?.file).toMatch(/complex_models\/blocks\.py$/);
    await page.getByRole('button', { name: 'Fit graph to view', exact: true }).click();
    await page.getByRole('button', { name: `Collapse ${trunk}`, exact: true }).click();
    await expect(page.getByRole('button', { name: `Expand ${trunk}`, exact: true })).toHaveCount(1);
    await expect(page.locator('.ui2-node-subtitle').filter({ hasText: `#${id}` })).toHaveCount(0);
  }
  await page.getByRole('button', { name: 'Fit graph to view', exact: true }).click();
  await expect(page.getByTitle('Model summary', { exact: true })).toContainText('102');
});

test('quick jump keeps an exact parent reachable among more than twenty descendants', async ({ page }) => {
  await openPage(page);
  await jump(page, 'trunk1.stage2.block2.second.conv', 'Conv2D');
  await page.getByRole('button', { name: 'More tools', exact: true }).press('ControlOrMeta+p');
  const input = page.getByRole('textbox', { name: 'Quick jump to node' });
  await input.fill('trunk1');
  await expect(page.locator('.ui2-quickjump-item').first()).toHaveAccessibleName('trunk1 — Module');
  await input.press('Enter');
  await expect(page.locator('[data-shape-id="shape:node:trunk1"] .module-node')).toHaveClass(/\bselected\b/);
});

test('dirty dependency edits refresh repeated modules and syntax errors preserve the last valid graph', async ({ page }) => {
  await openPage(page);
  const edit = async (file: string, before: string, after: string) => {
    const id = `${Date.now()}-${Math.random()}`;
    fs.writeFileSync(path.join(control(), 'command.json'), JSON.stringify({ type: 'edit', id, file, before, after }));
    await expect.poll(() => {
      const result = path.join(control(), 'edit-result.json');
      return fs.existsSync(result) ? JSON.parse(fs.readFileSync(result, 'utf8')).id : null;
    }).toBe(id);
  };
  const blocks = 'complex_models/blocks.py';
  let blocksEdited = false;
  let rootEdited = false;
  try {
    await edit(blocks, 'nn.ReLU()', 'nn.GELU()');
    blocksEdited = true;
    await jump(page, 'trunk2.stage2.block2.second.act', 'GELU');
    await expect(page.locator('[data-shape-id="shape:node:trunk2.stage2.block2.second.act"] .layer-node')).toHaveClass(/\bselected\b/);
    await edit('complex_models/network.py', 'def forward(self, x):\n', 'def forward(self, x)\n');
    rootEdited = true;
    await expect(page.getByRole('banner')).toContainText(/stale/i);
    await expect(page.getByTitle('Model summary', { exact: true })).toContainText('102');
  } finally {
    // Restore buffers without saving to the fixture, even after an assertion fails.
    if (rootEdited) await edit('complex_models/network.py', 'def forward(self, x)\n', 'def forward(self, x):\n');
    if (blocksEdited) await edit(blocks, 'nn.GELU()', 'nn.ReLU()');
  }
  await jump(page, 'trunk2.stage2.block2.second.act', 'ReLU');
  await expect(page.getByRole('banner')).not.toContainText(/stale/i);
});
