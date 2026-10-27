import { test, expect } from '@playwright/test';

/**
 * A blank graph used to be a black rectangle with nothing to act on.
 *
 * "Blank graph" is the first and most prominent of the six start options on Home,
 * so it is the one a newcomer picks — and it landed them on an empty canvas whose
 * bottom bar is icon-only, whose Validate button is disabled and silent, and
 * whose layer palette is behind an unadvertised Shift+A. Measured on the live
 * app: 6m50s to escape that screen, against ~11s to a validated model via a
 * template.
 *
 * These tests pin the way in, not the wording: that something explains the state,
 * that the control offered actually opens the palette, and that it gets out of
 * the way the moment there is a graph to look at.
 */
test.describe('an empty canvas offers a way in', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/builder');
    await page.waitForSelector('[data-testid="ui2-action-more"]', { timeout: 30000 });
    await page.waitForTimeout(2500);
  });

  test('explains the empty state and offers a control that opens the palette', async ({ page }) => {
    const empty = page.locator('[data-testid="builder-empty-state"]');
    await expect(empty).toBeVisible();

    // The palette is the app's strongest affordance and the thing a newcomer
    // cannot find, so the empty state has to lead there.
    await page.click('[data-testid="builder-empty-add-layer"]');
    await expect(page.locator('.ui2-palette-drawer')).toHaveCount(1);
  });

  test('gets out of the way as soon as the canvas has a layer', async ({ page }) => {
    await expect(page.locator('[data-testid="builder-empty-state"]')).toBeVisible();

    await page.click('[data-testid="builder-empty-add-layer"]');
    await page.waitForTimeout(800);
    await page.keyboard.type('linear');
    await page.waitForTimeout(1000);
    await page.keyboard.press('Enter');

    // Not merely hidden behind the canvas — gone, so it can never cover a graph.
    await expect(page.locator('[data-testid="builder-empty-state"]')).toHaveCount(0);
  });

  test('never intercepts a click meant for the canvas', async ({ page }) => {
    // The overlay covers the whole canvas frame, so pointer-events must stay off
    // on the container or it would swallow every drag onto an empty canvas.
    const container = page.locator('[data-testid="builder-empty-state"]');
    await expect(container).toHaveCSS('pointer-events', 'none');
  });
});
