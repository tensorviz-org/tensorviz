import { expect, Page } from '@playwright/test';

/** Preserve the selection while zooming far enough to expose its node controls. */
export async function selectedNodeEdit(page: Page) {
  await page.getByRole('button', { name: 'Zoom options', exact: true }).click();
  await page.getByRole('button', { name: /Readable detail/ }).click();
  const edit = page.locator('.selected .node-inline-edit-button').first();
  for (let step = 0; step < 4 && await edit.count() === 0; step++) {
    await page.getByRole('button', { name: 'Zoom in', exact: true }).click();
    await expect(page.locator('.selected')).not.toHaveCount(0);
  }
  await expect(edit).toBeVisible();
  return edit;
}
