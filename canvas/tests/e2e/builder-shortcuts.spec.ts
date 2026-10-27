import { expect, test } from '@playwright/test';

// P2d: builder keyboard shortcuts (quick-jump + hide-all chrome).
test.use({ viewport: { width: 1440, height: 900 } });

const MOD = process.platform === 'darwin' ? 'Meta' : 'Control';

test('quick-jump palette opens on Cmd/Ctrl+P and closes on Escape', async ({ page }) => {
  await page.goto('/builder?ui=v2');
  await expect(page.locator('.ui2-builder-toolbar')).toBeVisible();

  await page.keyboard.press(`${MOD}+p`);
  await expect(page.getByPlaceholder('Jump to node…')).toBeVisible();

  await page.keyboard.press('Escape');
  await expect(page.getByPlaceholder('Jump to node…')).toHaveCount(0);
});

test('Cmd/Ctrl+\\ hides all floating chrome, and restores it', async ({ page }) => {
  await page.goto('/builder?ui=v2');
  const toolbar = page.locator('.ui2-builder-toolbar');
  await expect(toolbar).toBeVisible();

  await page.keyboard.press(`${MOD}+\\`);
  await expect(toolbar).toBeHidden();

  await page.keyboard.press(`${MOD}+\\`);
  await expect(toolbar).toBeVisible();
});
