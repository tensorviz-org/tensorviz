import { expect, test } from '@playwright/test';

// Phase 2 gate: on a 1440x900 viewport with default chrome, the builder canvas
// (the tldraw surface) must occupy >= 88% of the viewport area. During the
// layout-inversion PRs the threshold is soft; it is raised to 0.88 in the
// inspector-overlay PR once the canvas is truly full-bleed.
const GATE_MIN_RATIO = 0.88; // Phase 2 gate: the canvas must own >= 88% of the viewport

test.use({ viewport: { width: 1440, height: 900 } });

test('builder canvas occupies enough of the viewport at 1440x900', async ({ page }) => {
  await page.goto('/builder?ui=v2');
  const canvas = page.getByTestId('builder-canvas');
  await expect(canvas).toBeVisible();

  const box = await canvas.boundingBox();
  if (!box) throw new Error('builder-canvas has no bounding box');

  const ratio = (box.width * box.height) / (1440 * 900);
  // eslint-disable-next-line no-console
  console.log(`[canvas-gate] area=${(ratio * 100).toFixed(1)}%  box=${Math.round(box.width)}x${Math.round(box.height)}`);

  await page.screenshot({ path: 'test-results/builder-canvas-gate.png' });

  expect(ratio).toBeGreaterThanOrEqual(GATE_MIN_RATIO);
});
