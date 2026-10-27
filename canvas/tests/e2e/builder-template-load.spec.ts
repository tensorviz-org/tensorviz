import { expect, test } from '@playwright/test';

// Regression guard for the "choose a template → empty canvas" bug: template
// nodes loaded into React state but never rendered as tldraw shapes because the
// editor mounted after the graph was set (a mount-order race in useTldrawBridge).
// This asserts the graph actually renders on the canvas — and survives a hard
// reload, which is the timing case that deterministically exposed the race.

const API = process.env.REACT_APP_API_URL || 'http://localhost:8765';

test.use({ viewport: { width: 1440, height: 900 } });

test('choosing a template renders its nodes on the canvas (and survives reload)', async ({ page, request }) => {
  // Backend-dependent: instantiate + saveGraph need the API. Skip cleanly if down.
  const healthy = await request.get(`${API}/health`).then((r) => r.ok()).catch(() => false);
  test.skip(!healthy, `backend not reachable at ${API}`);

  await page.goto('/?ui=v2');
  await page.getByText('Choose from Template', { exact: false }).first().click();

  const useButtons = page.getByRole('button', { name: /Use Template/i });
  await expect(useButtons.first()).toBeVisible();
  await useButtons.first().click();

  // Lands on the builder for the newly created graph.
  await page.waitForURL(/\/builder\/[^/]+/, { timeout: 30_000 });

  const shape = page.locator('.ui2-node-card, .layer-node, .module-node');
  await expect(shape.first()).toBeVisible({ timeout: 20_000 });
  const afterSelect = await shape.count();
  expect(afterSelect).toBeGreaterThan(0);

  // Hard reload at /builder/:id — the deterministic race case. Nodes must persist.
  await page.reload();
  await expect(shape.first()).toBeVisible({ timeout: 20_000 });
  const afterReload = await shape.count();
  // eslint-disable-next-line no-console
  console.log(`[template-load] shapes after select=${afterSelect} after reload=${afterReload}`);
  expect(afterReload).toBeGreaterThan(0);
});
