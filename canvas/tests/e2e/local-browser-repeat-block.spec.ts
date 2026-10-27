import { expect, test } from '@playwright/test';
import fs from 'node:fs';

/**
 * A folded `nn.ModuleList` is a block you can open, not a bare layer card.
 *
 * The tracer does not emit one node per member: `introspect.build_tree` groups
 * structurally identical siblings and recurses into the first only, so four
 * `nn.Linear`s arrive as ONE node carrying `_repeat_count: 4`. That fold is
 * deliberate — a Stable Diffusion UNet drawn one card per sibling is unreadable —
 * but what reached the canvas was a bare `Linear`, indistinguishable from a lone
 * layer, and the other three read as missing.
 *
 * The frame holds ONE canonical member, not four. Drawing four in a chain would
 * invent three edges nobody measured: the tracer folds their dataflow onto the
 * canonical node, so nothing in the graph says they run in series.
 */

const pageFile = process.env.TENSORVIZ_QA_PAGE_FILE;
test.skip(!pageFile, 'Requires a running disposable host (extension/scripts/qa-host.mjs)');

const pageUrl = () => JSON.parse(fs.readFileSync(pageFile!, 'utf8')).pageUrl as string;

test.use({ viewport: { width: 1440, height: 900 } });

test('a folded ModuleList is drawn as an expandable block', async ({ page }) => {
  await page.goto(pageUrl());
  await expect(page.locator('.tl-shapes [data-shape-id^="shape:node:"]').first()).toBeVisible({
    timeout: 30_000,
  });

  // The fold is a TRACED fact: `_repeat_count` comes from introspect.build_tree
  // grouping identical siblings, and the static tier never sets it. Without a run
  // there is nothing to wrap, and the page says so — "Not run — shapes read from
  // the source".
  await page.getByRole('button', { name: 'Validate', exact: true }).click();
  await expect(page.getByText(/Traced . shapes measured/)).toBeVisible({ timeout: 240_000 });

  // The frame is invented by the canvas, so its id is the one the projection never
  // sends — that is exactly what makes it a frame rather than a node.
  // The colon is percent-encoded in a tldraw shape id, so match the encoded form —
  // `shape:node:repeat%3Astack.0`.
  const frame = page.locator('[data-shape-id^="shape:node:repeat%3A"]');
  await expect(frame).toHaveCount(1);

  // It says how many of what, which the bare card could not.
  await expect(frame.first()).toContainText(/×\s*4/);

  // And it opens. Collapsed, the member is not drawn; expanded, it is.
  const memberSelector = '[data-shape-id="shape:node:stack.0"]';
  const expand = frame.first().getByRole('button', { name: /^Expand / });
  await expect(expand).toBeVisible();
  await expand.click();

  await expect
    .poll(() => page.locator(memberSelector).count(), {
      timeout: 10_000,
      message: 'the block did not open — its canonical member never appeared',
    })
    .toBeGreaterThan(0);
});
