import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import { selectedNodeEdit } from './local-browser-node-controls';

const pageFile = process.env.TENSORVIZ_QA_PAGE_FILE;
test.skip(!pageFile, 'Requires the installed host on the complex-model fixture');

test('selection exposes source editing on the detailed node without a floating chip', async ({ page }) => {
  await page.goto(JSON.parse(fs.readFileSync(pageFile!, 'utf8')).pageUrl);
  await page.getByRole('button', { name: 'Expand toolbar' }).click();
  await page.getByRole('button', { name: 'Find a layer', exact: true }).click();
  await page.getByRole('textbox', { name: 'Quick jump to node' }).fill('branch_fc');
  await page.getByRole('button', { name: /^branch_fc —/ }).click();
  await expect(page.locator('.selection-chips')).toHaveCount(0);
  await (await selectedNodeEdit(page)).click();
  await expect(page.getByRole('region', { name: 'Edit node source' })).toBeVisible();
  await expect(page.locator('dialog.source-edit-dialog')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Review in VS Code' })).toBeVisible();
});
