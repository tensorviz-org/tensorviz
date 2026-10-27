import { test, expect } from '@playwright/test';

/**
 * Regression pin for the ⋯ More menu painting outside the toolbar's clip.
 *
 * `.ui2-builder-toolbar` sets `overflow-x: auto` so its ~995px of controls stay
 * reachable on a narrow window; CSS computes the other axis to
 * `overflow-y: hidden`, and the menu opens upward at `bottom: 100%` — entirely
 * outside the bar's box, so it was clipped out of existence. Roughly ten
 * controls live only in that menu and none of them could be used.
 *
 * The reason no test caught it is the reason these assertions look the way they
 * do: the menu stayed in the DOM with a correct getBoundingClientRect, so
 * `toBeVisible()` passed for every item. Only hit-testing tells the truth, so
 * this file asks the browser what is actually on top at each item's centre and
 * then really clicks one.
 */
test.describe('the ⋯ More menu is reachable', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/builder');
    await page.waitForSelector('[data-testid="ui2-action-more"]', { timeout: 30000 });
    await page.waitForTimeout(2000);
  });

  test('every item in the menu is the topmost element at its own centre', async ({ page }) => {
    await page.click('[data-testid="ui2-action-more"]');
    const menu = page.locator('[role="menu"].ui2-panel');
    await expect(menu).toBeVisible();

    const report = await page.evaluate(() => {
      const items = Array.from(document.querySelectorAll('[role="menuitem"]')) as HTMLElement[];
      return items.map((item) => {
        const box = item.getBoundingClientRect();
        const top = document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2);
        return {
          label: (item.textContent || '').trim(),
          covered: !(top === item || item.contains(top)),
        };
      });
    });

    expect(report.length).toBeGreaterThanOrEqual(6);
    expect(report.filter((r) => r.covered)).toEqual([]);
  });

  test('clicking a menu item runs it rather than landing on the canvas', async ({ page }) => {
    await page.click('[data-testid="ui2-action-more"]');
    const item = page.locator('[role="menuitem"]:not([disabled])').first();
    await expect(item).toBeVisible();

    // A real click with no force: if the canvas were on top this times out,
    // which is exactly what it did before the menu was portalled out.
    await item.click({ timeout: 5000 });

    // Running an item closes the menu — proof the click reached the button and
    // not the drawing surface underneath it.
    await expect(page.locator('[role="menu"].ui2-panel')).toHaveCount(0);
  });

  test('the menu stays anchored to its trigger after a resize', async ({ page }) => {
    await page.click('[data-testid="ui2-action-more"]');
    await page.setViewportSize({ width: 1100, height: 800 });
    await page.waitForTimeout(300);

    const drift = await page.evaluate(() => {
      const trigger = document.querySelector('[data-testid="ui2-action-more"]')!.getBoundingClientRect();
      const menu = document.querySelector('[role="menu"].ui2-panel')!.getBoundingClientRect();
      return { dx: Math.abs(menu.left - trigger.left), gap: trigger.top - menu.bottom };
    });

    expect(drift.dx).toBeLessThan(4);
    expect(drift.gap).toBeGreaterThanOrEqual(0);
    expect(drift.gap).toBeLessThan(20);
  });
});
