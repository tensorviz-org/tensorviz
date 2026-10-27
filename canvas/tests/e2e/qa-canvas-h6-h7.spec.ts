import { expect, test, Page, APIRequestContext } from '@playwright/test';

// Regression guards for two QA-found canvas bugs:
//  H6 — programmatic React→tldraw syncs polluted the undo stack, so pressing undo
//       right after a template load erased canvas shapes (looked like data loss).
//  (H7 context-region scoping is unit-covered in qa_context_region.test.ts.)

const API = process.env.REACT_APP_API_URL || 'http://localhost:8765';
const SHAPE = '.ui2-node-card, .layer-node, .module-node';
const UNDO = process.platform === 'darwin' ? 'Meta+z' : 'Control+z';

test.use({ viewport: { width: 1440, height: 900 } });

async function openBuilder(page: Page, request: APIRequestContext) {
  const healthy = await request.get(`${API}/health`).then((r) => r.ok()).catch(() => false);
  test.skip(!healthy, `backend not reachable at ${API}`);
  await page.goto('/?ui=v2');
  await page.getByText('Choose from Template', { exact: false }).first().click();
  const useBtn = page.getByRole('button', { name: /Use Template/i });
  await expect(useBtn.first()).toBeVisible();
  await useBtn.first().click();
  await page.waitForURL(/\/builder\/[^/]+/, { timeout: 30_000 });
  await expect(page.locator(SHAPE).first()).toBeVisible({ timeout: 20_000 });
}

test('H6: undo right after a template load does not erase the canvas', async ({ page, request }) => {
  await openBuilder(page, request);
  const shape = page.locator(SHAPE);
  const before = await shape.count();
  expect(before).toBeGreaterThan(0);

  // No user edit has happened yet, so the affordance itself must be off. This is
  // the half the old it.skip pin in qa_canvas_correctness.test.ts described and
  // could not test: a programmatic sync that records history lights the Undo
  // button before the user has done anything at all.
  await expect(page.getByTestId('ui2-action-undo')).toBeDisabled();

  // Focus the canvas and press undo a few times anyway.
  await page.mouse.click(720, 460);
  for (let i = 0; i < 3; i++) {
    await page.keyboard.press(UNDO);
    await page.waitForTimeout(150);
  }

  // Before the fix, undo reverted the programmatic sync and shapes vanished.
  expect(await shape.count()).toBe(before);
  await expect(page.getByTestId('ui2-action-undo')).toBeDisabled();
});
