import { expect, test } from '@playwright/test';

/**
 * Regression guard: the read-only page must still MOVE nodes.
 *
 * The external page fed its SOURCE-read-only policy into tldraw's presentation
 * lock (`editor.updateInstanceState({ isReadonly: readOnly })`). That lock is
 * all-or-nothing — tldraw's SelectTool refuses to enter `translating` while it is
 * set (`PointingShape.startTranslating`) — so a card highlighted on pointer-down
 * and then never moved, on a page whose whole job is reading a graph and whose own
 * contract promises layout/fit.
 *
 * WHAT THIS TEST HAD TO LEARN, because two earlier versions passed against the very
 * build they were written to catch:
 *   1. Never assert on a bounding box. A refused drag falls through to canvas
 *      panning, which moves everything on screen; screen movement is not shape
 *      movement.
 *   2. Never compare shapes without filtering to NODES. The first elements in
 *      `[data-shape-id]` DOM order are EDGES, whose geometry is recomputed for
 *      reasons unrelated to the drag.
 * The assertion is the node's own `style.transform`, which tldraw writes in page
 * space — panning moves the container, not this — so it changes if and only if the
 * shape really moved. Verified to fail against the locked build.
 */

const API = process.env.REACT_APP_API_URL || 'http://localhost:8765';

test.use({ viewport: { width: 1440, height: 900 } });

test('a read-only local-browser canvas can still move a node', async ({ page, request }) => {
  const healthy = await request.get(`${API}/health`).then((r) => r.ok()).catch(() => false);
  test.skip(!healthy, `backend not reachable at ${API}`);

  await page.addInitScript(() => {
    (window as unknown as { __TENSORVIZ_LOCAL_BROWSER__?: boolean }).__TENSORVIZ_LOCAL_BROWSER__ = true;
  });

  await page.goto('/?ui=v2');
  await page.getByText('Choose from Template', { exact: false }).first().click();
  const useButtons = page.getByRole('button', { name: /Use Template/i });
  await expect(useButtons.first()).toBeVisible();
  await useButtons.first().click();
  await page.waitForURL(/\/builder\/[^/]+/, { timeout: 30_000 });

  // The mode really is the read-only one, or this proves nothing about the bug.
  expect(
    await page.evaluate(
      () => (window as unknown as { __TENSORVIZ_LOCAL_BROWSER__?: boolean }).__TENSORVIZ_LOCAL_BROWSER__,
    ),
  ).toBe(true);

  const node = page.locator('.tl-shapes [data-shape-id^="shape:node:"]').first();
  await expect(node).toBeVisible({ timeout: 20_000 });

  const shapeId = await node.getAttribute('data-shape-id');
  const transform = () =>
    page.evaluate(
      (id) => (document.querySelector(`[data-shape-id="${id}"]`) as HTMLElement | null)?.style.transform ?? '',
      shapeId,
    );

  const before = await transform();
  expect(before).not.toBe('');

  const box = (await node.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  // Past tldraw's drag threshold, in steps so it reads as a drag and not a jump.
  await page.mouse.move(box.x + box.width / 2 + 160, box.y + box.height / 2 + 110, { steps: 12 });
  await page.mouse.up();

  await expect
    .poll(transform, {
      timeout: 5_000,
      message: `the node's transform never changed from ${before} — the canvas is locked and the drag only panned`,
    })
    .not.toBe(before);
});
