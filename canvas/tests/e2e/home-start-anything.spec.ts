import { expect, test } from '@playwright/test';

test('unmatched search stays on the home page without starting an AI request', async ({ page }) => {
  await page.goto('/');
  const box = page.getByLabel('Find a model');
  await box.fill('no-model-with-this-unique-name-7f341');
  await expect(page.getByText('No matching model', { exact: true })).toBeVisible();
  await box.press('Enter');
  await expect(page).toHaveURL(/\/$/);
  await expect(box).toHaveValue('no-model-with-this-unique-name-7f341');
  await expect(page.getByTestId('ai-task-input')).toHaveCount(0);
});

test('an existing model opens from search', async ({ page }) => {
  await page.goto('/');
  // Whatever this machine's first saved model is called — the spec must not
  // depend on a fixture name that may not exist here.
  const menuLabel = await page.getByLabel(/^More actions for /).first().getAttribute('aria-label');
  test.skip(!menuLabel, 'no saved models on this backend');
  const name = (menuLabel as string).replace('More actions for ', '');

  const box = page.getByLabel('Find a model');
  await box.fill(name);
  await expect(page.getByText('↵ to open')).toBeVisible();
  await box.press('Enter');
  await expect(page).toHaveURL(/\/builder\/[0-9a-f]{8,}$/);
});

test('the template band counts the starters the gallery actually serves', async ({ page }) => {
  await page.goto('/');

  await expect(page.getByText('Start from a template')).toBeVisible();
  const note = page.getByText(/\d+ in the gallery/);
  await expect(note).toBeVisible();
  const text = (await note.textContent()) || '';
  // A real, non-zero count of what the gallery serves.
  expect(text).toMatch(/^[1-9]\d* in the gallery/);
  await expect(page.getByText(/starter blocks?$/)).toHaveCount(0);
  if (!/saved/.test(text)) {
    // Nothing saved on this machine: the band still says what there is to start from.
    await expect(page.getByText(/starter architectures are in the gallery/)).toBeVisible();
    await expect(page.getByText('No saved templates yet — the gallery has starter architectures to begin from.')).toHaveCount(0);
  }
});
