import { test, expect } from '@playwright/test';
import fs from 'node:fs';

const pageFile = process.env.TENSORVIZ_QA_PAGE_FILE;
test.skip(!pageFile, 'Requires the installed host on the complex-model fixture');

test('selection adds no floating chip and no source editing to the node', async ({ page }) => {
  await page.goto(JSON.parse(fs.readFileSync(pageFile!, 'utf8')).pageUrl);
  await page.getByRole('button', { name: 'Expand toolbar' }).click();
  await page.getByRole('button', { name: 'Find a layer', exact: true }).click();
  await page.getByRole('textbox', { name: 'Quick jump to node' }).fill('branch_fc');
  await page.getByRole('button', { name: /^branch_fc —/ }).click();
  await expect(page.locator('.selection-chips')).toHaveCount(0);
  await page.getByRole('button', { name: 'Zoom options', exact: true }).click();
  await page.getByRole('button', { name: /Readable detail/ }).click();
  await expect(page.locator('.selected')).not.toHaveCount(0);
  await expect(page.locator('.selected').getByRole('button', { name: /^Edit / })).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Edit node source' })).toHaveCount(0);
});
