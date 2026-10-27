import { expect, test, Page } from '@playwright/test';

/**
 * Two tabs on one graph: the second tab's edit may be refused, but it may never
 * be refused silently.
 *
 * There is no live sync between tabs, so a second tab holding an older revision
 * is expected. The server protects the data — a stale commit gets a 409 naming
 * the expected and current revision — and the pill says so once. What went wrong
 * is what happened to that message on the NEXT edit: the pre-debounce journal
 * effect set the pill to "Saving…" for every change, so the conflict wording was
 * replaced by a cyan "Saving…" that never resolved. The queue had frozen its
 * conflicted head, so that edit was never sent at all; the pill claimed a save
 * was in flight for an edit that had already been dropped.
 *
 * Every assertion below is on what the stale tab TELLS the user after its edit,
 * not on whether the commit succeeded.
 */

test.use({ viewport: { width: 1440, height: 900 } });

const CARD = '.ui2-node-card';

async function addLayer(page: Page, query: string): Promise<void> {
  await page.bringToFront();
  const before = await page.locator(CARD).count();
  await page.keyboard.press('Shift+A');
  const search = page.getByLabel('Search layers, blocks and templates');
  await expect(search).toBeVisible();
  await search.fill(query);
  await search.press('Enter');
  await expect.poll(() => page.locator(CARD).count(), { timeout: 20_000 }).toBe(before + 1);
}

async function savePill(page: Page): Promise<string> {
  const pill = page.locator('.ui2-saved-pill').first();
  return (await pill.count()) === 0 ? '(no pill)' : (await pill.innerText()).trim();
}

test('a stale tab keeps saying its edit was refused, and never claims to be saving it', async ({
  browser,
}) => {
  const context = await browser.newContext();
  const first = await context.newPage();
  await first.goto('/builder?ui=v2');
  await expect(first.locator('.ui2-builder-toolbar')).toBeVisible({ timeout: 60_000 });
  await addLayer(first, 'input');
  await addLayer(first, 'linear');
  await first.waitForURL(/\/builder\/[^/]+$/, { timeout: 60_000 });
  const graphId = new URL(first.url()).pathname.split('/').pop();
  await expect.poll(() => savePill(first), { timeout: 30_000 }).toContain('Saved');

  // The second tab opens the same graph at the same revision.
  const second = await context.newPage();
  await second.goto(`/builder/${graphId}?ui=v2`);
  await expect(second.locator('.ui2-builder-toolbar')).toBeVisible({ timeout: 60_000 });
  await expect.poll(() => second.locator(CARD).count(), { timeout: 30_000 }).toBe(2);

  // The first tab commits a revision the second tab knows nothing about.
  await addLayer(first, 'relu');
  await expect.poll(() => savePill(first), { timeout: 30_000 }).toContain('Saved · just now');

  // The second tab's own edit is now stale, and the server refuses it.
  await addLayer(second, 'dropout');
  await expect
    .poll(() => savePill(second), { timeout: 30_000 })
    .toContain('reload to reconcile');

  // A second edit in the stale tab. It cannot be sent either — the queue keeps
  // its conflicted head frozen until the graph is reloaded — so the refusal has
  // to still be on screen. Sampled over several seconds because the regression
  // is a delayed overwrite: the wording came back as "Saving…" ~300ms later and
  // stayed there.
  await addLayer(second, 'sigmoid');
  const readings: string[] = [];
  for (let i = 0; i < 12; i++) {
    readings.push(await savePill(second));
    await second.waitForTimeout(500);
  }
  expect(readings.filter((pill) => !pill.includes('reload to reconcile'))).toEqual([]);

  // And the edit itself is still on the canvas: refused, not reverted.
  expect(await second.locator(CARD).count()).toBe(4);
});
