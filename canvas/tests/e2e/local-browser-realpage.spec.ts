import { expect, test } from '@playwright/test';
import fs from 'node:fs';

/**
 * The real page, served by a real extension host — not the standalone builder with
 * `__TENSORVIZ_LOCAL_BROWSER__` forced on.
 *
 * That distinction is the point. Every other local-browser assertion in this repo runs
 * against jsdom with the canvas mocked away, or against the dev server with the mode
 * flag set by hand. Neither exercises the broker, the host transport, the projection
 * that arrives over the wire, or the capabilities the broker actually grants a view —
 * so a defect in any of them is invisible until someone opens the product.
 *
 * Fixture-agnostic on purpose: it locates whatever nodes the graph has rather than
 * naming any, so it runs against any workspace `extension/scripts/qa-host.mjs` boots.
 * Point TENSORVIZ_QA_PAGE_FILE at that host's control/page.json.
 */

const pageFile = process.env.TENSORVIZ_QA_PAGE_FILE;
test.skip(!pageFile, 'Requires a running disposable host (extension/scripts/qa-host.mjs)');

const pageUrl = () => JSON.parse(fs.readFileSync(pageFile!, 'utf8')).pageUrl as string;

test.use({ viewport: { width: 1440, height: 900 } });

test('the graph renders, and a node can be moved', async ({ page }) => {
  await page.goto(pageUrl());

  const nodes = page.locator('.tl-shapes [data-shape-id^="shape:node:"]');
  await expect(nodes.first()).toBeVisible({ timeout: 30_000 });
  expect(await nodes.count()).toBeGreaterThan(1);

  // The node's own transform, in page space. A bounding box would also move when a
  // refused drag falls through to canvas panning, which is how a locked canvas passed
  // a drag test twice before.
  const shapeId = await nodes.first().getAttribute('data-shape-id');
  const transform = () =>
    page.evaluate(
      (id) => (document.querySelector(`[data-shape-id="${id}"]`) as HTMLElement | null)?.style.transform ?? '',
      shapeId,
    );

  const before = await transform();
  expect(before).not.toBe('');

  const box = (await nodes.first().boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 150, box.y + box.height / 2 + 100, { steps: 12 });
  await page.mouse.up();

  await expect
    .poll(transform, {
      timeout: 5_000,
      message: `the node's transform never changed from ${before} — the real page cannot move a node`,
    })
    .not.toBe(before);
});

test('the header capture action reports a verdict', async ({ page }) => {
  await page.goto(pageUrl());
  const validate = page.getByTestId('local-browser-validate');
  await expect(validate).toBeVisible({ timeout: 30_000 });
  await expect(validate).toHaveClass(/local-canvas-control/);
  await expect(page.locator('.local-browser-header [data-testid="local-browser-validate"]')).toHaveCount(1);
  await expect(page.locator('.ui2-builder-toolbar')).toHaveCount(0);

  await validate.click();
  // Either it runs and settles, or it is refused with a reason — both are verdicts.
  // What must NOT happen is a button that reports nothing at all, which is the bug
  // the toast exists to prevent.
  await expect(page.locator('.ui2-toast')).toBeVisible({ timeout: 240_000 });
  const toast = (await page.locator('.ui2-toast').first().innerText()).trim();
  expect(toast.length).toBeGreaterThan(0);
  test.info().annotations.push({ type: 'verdict', description: toast });
});

test('a parameter node carries no "unresolved external class" warning', async ({ page }) => {
  await page.goto(pageUrl());
  await expect(page.locator('.tl-shapes [data-shape-id^="shape:node:"]').first()).toBeVisible({ timeout: 30_000 });

  // nn.Parameter is a canonical framework construct, not a third-party class that
  // failed to resolve. Every parameter card used to carry a warning saying its own
  // class was unavailable.
  const body = await page.locator('body').innerText();
  expect(body).not.toContain('its classes stay opaque');
});
