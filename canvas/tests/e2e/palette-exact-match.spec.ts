import { expect, test } from '@playwright/test';

/**
 * The palette must offer an exact match first.
 *
 * A post-audit report said typing `ReLU` and pressing Enter inserted **PReLU** —
 * a fuzzy match beating the exact one. It does not reproduce: the layer filter
 * preserves declaration order and `relu` is declared before `prelu`, so ReLU is
 * row 0 and Enter takes row 0.
 *
 * The test stays anyway, because "the exact match ranks first" is a property
 * nobody was checking, and the filter is one `sort` away from losing it. This is
 * the cheapest possible guard on a class of bug that is very hard to notice: you
 * get a layer that is nearly the one you asked for.
 */
test('an exact palette match ranks first and is what Enter inserts', async ({ page }) => {
  await page.goto('/builder?ui=v2');
  await expect(page.locator('.ui2-builder-toolbar')).toBeVisible({ timeout: 60_000 });

  await page.keyboard.press('Shift+A');
  const search = page.getByLabel('Search layers, blocks and templates');
  await expect(search).toBeVisible();
  await search.fill('ReLU');

  const first = await page
    .locator('.ui2-palette-drawer button')
    .first()
    .innerText();
  // ReLU, not ReLU6, LeakyReLU, PReLU or RReLU — all of which also match "relu".
  expect(first.split('\n')[0].trim()).toBe('ReLU');

  await search.press('Enter');
  await expect.poll(() => page.locator('.ui2-node-card').count(), { timeout: 20_000 }).toBe(1);
  const inserted = await page.locator('.ui2-node-card').first().innerText();
  expect(inserted).toContain('ReLU');
  expect(inserted).not.toContain('PReLU');
});
