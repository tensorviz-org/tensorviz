import { expect, test } from '@playwright/test';
import fs from 'node:fs';

/**
 * A card must not be drawn on top of the block beside it.
 *
 * The layout reserved `MODULE_COLLAPSED_H` for a collapsed module, but a collapsed
 * module card is auto-height and draws taller than the box dagre spaced it by. On
 * the QA fixture that put the `x.reshape(...)` op card 47 canvas units INSIDE the
 * `stack` repeat frame, over its "Expand Block" affordance — so the one thing the
 * frame exists to offer could not be clicked.
 *
 * `repeatBlocks.test.ts` pins the geometry in Jest. This pins what the user sees:
 * the two shapes, drawn by the real canvas in a real browser, do not intersect.
 * Asserted on the DRAWN rectangles rather than the shape boxes, because the bug was
 * precisely that those two disagreed.
 */

const pageFile = process.env.TENSORVIZ_QA_PAGE_FILE;
test.skip(!pageFile, 'Requires a running disposable host (extension/scripts/qa-host.mjs)');

const pageUrl = () => JSON.parse(fs.readFileSync(pageFile!, 'utf8')).pageUrl as string;

test.use({ viewport: { width: 1440, height: 900 } });

test('a neighbouring card is not drawn inside the repeat block', async ({ page }) => {
  await page.goto(pageUrl());
  await expect(page.locator('.tl-shapes [data-shape-id^="shape:node:"]').first()).toBeVisible({
    timeout: 30_000,
  });

  // `_repeat_count` is a traced fact, so the frame only exists after a run.
  await page.getByRole('button', { name: 'Validate', exact: true }).click();
  await expect(page.getByText(/Traced . shapes measured/)).toBeVisible({ timeout: 240_000 });

  const frame = page.locator('[data-shape-id^="shape:node:repeat%3A"]').first();
  await expect(frame).toBeVisible({ timeout: 15_000 });

  // The neighbour dagre placed below it. Matched by its text rather than its id,
  // which encodes a generated `python@Network#N` name.
  const neighbour = page.locator('.tl-shapes [data-shape-id]').filter({ hasText: 'x.reshape' }).first();
  await expect(neighbour).toBeVisible();

  const [a, b] = [await frame.boundingBox(), await neighbour.boundingBox()];
  expect(a, 'the repeat frame is not drawn').not.toBeNull();
  expect(b, 'the neighbouring card is not drawn').not.toBeNull();

  const overlap =
    a!.x < b!.x + b!.width &&
    b!.x < a!.x + a!.width &&
    a!.y < b!.y + b!.height &&
    b!.y < a!.y + a!.height;

  expect(
    overlap,
    `the block ${JSON.stringify(a)} and its neighbour ${JSON.stringify(b)} intersect`,
  ).toBe(false);

  // And the affordance the overlap used to cover is reachable. `toBeVisible` is not
  // enough — an element under another one is still "visible" — so this asks the
  // canvas what it would actually hit at that point.
  // Deliberately no "is the CTA clickable" hit test here. `elementFromPoint` lands
  // on tldraw's own layer container rather than a shape element, so it answers null
  // whether or not anything covers the button — an assertion that cannot tell the
  // two apart is worse than none. The geometry above is what the bug was.
});
