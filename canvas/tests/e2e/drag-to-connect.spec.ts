import { expect, test, Page } from '@playwright/test';

/**
 * Drag-to-connect and click-an-edge-to-delete, measured.
 *
 * The repo had zero e2e coverage of edges before this, so every assertion here
 * is a DELTA on a counted selector — never `count > 0`, which would pass on a
 * canvas that ignored the gesture entirely.
 *
 * `.model-edge-path` matches two paths per edge (the stroke and its glow), so
 * the count excludes the glow to get exactly one node per edge.
 */

test.use({ viewport: { width: 1440, height: 900 } });

const CARD = '.ui2-node-card';
const EDGE = '.model-edge-path:not(.model-edge-path-glow)';

const edgeCount = (page: Page) => page.locator(EDGE).count();
const cardCount = (page: Page) => page.locator(CARD).count();

/**
 * Two separate things make a port unpressable, and BOTH have to clear before any
 * measurement here is worth acting on:
 *
 *  - `.tl-hit-test-blocker` is `pointer-events: all` until the camera reports
 *    idle, so a press lands on the blocker and reaches tldraw as a plain canvas
 *    click — the gesture never starts and the test fails looking like a logic bug;
 *  - a camera still animating moves the card out from under a point measured a
 *    moment earlier, with the same symptom.
 *
 * So this demands the blocker hidden AND every shape's screen position unchanged
 * across consecutive samples. Observed directly: without the second condition the
 * click-to-connect case failed roughly one run in three.
 */
async function waitForCanvasSettled(page: Page): Promise<void> {
  let previous = '';
  let stable = 0;
  await expect
    .poll(
      async () => {
        const blocker = page.locator('.tl-hit-test-blocker');
        const hidden = (await blocker.count()) === 0 || (await blocker.first().isHidden());
        const geometry = await page
          .locator('.tl-shape[data-shape-type]')
          .evaluateAll((els) =>
            els
              .map((el) => {
                const rect = el.getBoundingClientRect();
                return `${Math.round(rect.x)},${Math.round(rect.y)}`;
              })
              .join('|')
          );
        stable = hidden && geometry !== '' && geometry === previous ? stable + 1 : 0;
        previous = geometry;
        return stable >= 2;
      },
      { timeout: 30_000, interval: 250 }
    )
    .toBe(true);
}

async function addLayer(page: Page, query: string): Promise<void> {
  const before = await cardCount(page);
  await page.keyboard.press('Shift+A');
  const search = page.getByLabel('Search layers, blocks and templates');
  await expect(search).toBeVisible();
  await search.fill(query);
  await search.press('Enter');
  await expect.poll(() => cardCount(page), { timeout: 20_000 }).toBe(before + 1);
}

/** Input -> Linear -> ReLU, wired by the palette's own auto-wire: 3 cards, 2 edges. */
async function buildChain(page: Page): Promise<void> {
  await page.goto('/builder?ui=v2');
  await expect(page.locator('.ui2-builder-toolbar')).toBeVisible({ timeout: 60_000 });

  await addLayer(page, 'input');
  await addLayer(page, 'linear');
  await addLayer(page, 'relu');

  await expect.poll(() => edgeCount(page), { timeout: 20_000 }).toBe(2);

  // Fit first: the palette appends to the RIGHT of the chain, so the last card
  // can land outside the viewport, and tldraw culls off-screen shapes — their
  // ports then exist in the DOM with no bounding box at all.
  await page.getByTestId('ui2-action-fit-view').click();
  // Then wait for shape inference, which now runs on its own after every edit
  // rather than only on Refresh. It GROWS each card by its In/Out rows, moving
  // every port and every wire — so a geometry-stability check alone can pass in
  // the gap before it lands and hand back coordinates that are about to be wrong.
  // The rows appearing is the positive signal that it has happened.
  await expect.poll(() => page.locator('.ui2-node-shape-row').count(), { timeout: 30_000 })
    .toBeGreaterThan(0);
  await waitForCanvasSettled(page);
}

interface Point {
  x: number;
  y: number;
}

/** Shape ids of one shape type, left to right — the order addNode lays them out in. */
async function shapeIdsLeftToRight(page: Page, shapeType: string): Promise<string[]> {
  const found = await page.locator(`.tl-shape[data-shape-type="${shapeType}"]`).evaluateAll((els) =>
    els.map((el) => ({
      id: el.getAttribute('data-shape-id') || '',
      x: el.getBoundingClientRect().x,
    }))
  );
  return found.sort((a, b) => a.x - b.x).map((entry) => entry.id);
}

async function portCentre(page: Page, shapeId: string, side: 'source' | 'target'): Promise<Point> {
  // A card now draws a dot on each side its direction allows (target: left+top,
  // source: right+bottom), so there is more than one per direction. Any of them
  // starts the same gesture; take the first.
  const port = page.locator(`[data-shape-id="${shapeId}"] [data-testid="node-port-${side}"]`).first();
  await expect(port).toBeVisible();
  const box = await port.boundingBox();
  // A culled shape keeps its DOM but reports no box, which would otherwise make
  // the drag click empty canvas and the assertion fail for the wrong reason.
  if (!box) throw new Error(`${side} port on ${shapeId} is culled — fit the view first`);
  return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
}

async function cardCentre(page: Page, shapeId: string): Promise<Point> {
  const box = await page.locator(`[data-shape-id="${shapeId}"]`).boundingBox();
  if (!box) throw new Error(`${shapeId} has no bounding box`);
  return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
}

async function dragConnect(page: Page, from: Point, to: Point): Promise<void> {
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  // Multiple steps: the gesture only becomes a drag past 4px of travel, and one
  // jump would land as a click-to-connect instead.
  await page.mouse.move(to.x, to.y, { steps: 14 });
  await page.mouse.up();
}

test('a port renders on every side that can legally carry an edge', async ({ page }) => {
  await buildChain(page);

  // Input has no target port and nothing feeds it; the two layers have both.
  // Direction is enforced by which ports exist, so this IS the direction rule:
  // each source direction draws two dots (right + bottom), each target two
  // (left + top). Input = 2 sources, the two layers = 4 sources + 4 targets.
  expect(await page.locator('[data-testid="node-port-source"]').count()).toBe(6);
  expect(await page.locator('[data-testid="node-port-target"]').count()).toBe(4);
});

test('dragging a source port onto a valid target adds exactly one edge', async ({ page }) => {
  await buildChain(page);

  const [inputId] = await shapeIdsLeftToRight(page, 'input-node');
  const layerIds = await shapeIdsLeftToRight(page, 'layer');
  const reluId = layerIds[layerIds.length - 1];

  const before = await edgeCount(page);
  await dragConnect(
    page,
    await portCentre(page, inputId, 'source'),
    await portCentre(page, reluId, 'target')
  );

  await expect.poll(() => edgeCount(page), { timeout: 10_000 }).toBe(before + 1);
});

test('clicking a port then clicking the target connects them, with no drag', async ({ page }) => {
  // React Flow ships connectOnClick = true, so a press that never travels arms
  // the second click rather than cancelling.
  await buildChain(page);

  const [inputId] = await shapeIdsLeftToRight(page, 'input-node');
  const layerIds = await shapeIdsLeftToRight(page, 'layer');
  const reluId = layerIds[layerIds.length - 1];

  const from = await portCentre(page, inputId, 'source');
  const to = await portCentre(page, reluId, 'target');

  const before = await edgeCount(page);
  await page.mouse.click(from.x, from.y);
  await page.mouse.click(to.x, to.y);

  await expect.poll(() => edgeCount(page), { timeout: 10_000 }).toBe(before + 1);
});

test('dropping on empty canvas cancels silently and adds no edge', async ({ page }) => {
  await buildChain(page);

  const layerIds = await shapeIdsLeftToRight(page, 'layer');
  const reluId = layerIds[layerIds.length - 1];
  const from = await portCentre(page, reluId, 'source');

  const before = await edgeCount(page);
  const cardsBefore = await cardCount(page);
  // Well below the chain, clear of every card.
  await dragConnect(page, from, { x: from.x + 40, y: from.y + 300 });

  await page.waitForTimeout(500);
  expect(await edgeCount(page)).toBe(before);
  // And it must NOT open the palette, which would be its own kind of surprise.
  expect(await cardCount(page)).toBe(cardsBefore);
  await expect(page.getByLabel('Search layers, blocks and templates')).toHaveCount(0);
});

test('dropping on an Input node adds no edge — nothing may feed an Input', async ({ page }) => {
  await buildChain(page);

  const [inputId] = await shapeIdsLeftToRight(page, 'input-node');
  const layerIds = await shapeIdsLeftToRight(page, 'layer');
  const reluId = layerIds[layerIds.length - 1];

  const before = await edgeCount(page);
  await dragConnect(
    page,
    await portCentre(page, reluId, 'source'),
    await cardCentre(page, inputId)
  );

  await page.waitForTimeout(500);
  expect(await edgeCount(page)).toBe(before);
});

test('clicking an edge and pressing Delete removes that edge and no node', async ({ page }) => {
  await buildChain(page);

  const [inputId] = await shapeIdsLeftToRight(page, 'input-node');
  const [linearId] = await shapeIdsLeftToRight(page, 'layer');

  const start = await portCentre(page, inputId, 'source');
  const end = await portCentre(page, linearId, 'target');
  // The cubic's t=0.5 point is exactly the chord midpoint for every control-point
  // choice edgeCurve makes, so this lands ON the stroke rather than near it.
  const onCurve = { x: (start.x + end.x) / 2, y: (start.y + end.y) / 2 };

  const edgesBefore = await edgeCount(page);
  const cardsBefore = await cardCount(page);

  await page.mouse.click(onCurve.x, onCurve.y);
  await page.keyboard.press('Delete');

  await expect.poll(() => edgeCount(page), { timeout: 10_000 }).toBe(edgesBefore - 1);
  expect(await cardCount(page)).toBe(cardsBefore);
});

/**
 * Insert-on-the-wire. Both assertions below are about the POINTER, not the
 * element: `locator.click()` drives the mouse to whatever box the element
 * reports, so it passes happily on a `+` drawn somewhere the user's cursor will
 * never be. That is exactly how a projection bug shipped — the affordance was
 * placed with pageToScreen inside a container inset 40px below the viewport, so
 * it drew 40px off its wire, and travelling to it left the edge's hit radius and
 * unmounted it. Hence: hover the wire, then press WITHOUT MOVING.
 */
async function hoverEdgeMidpoint(
  page: Page,
  sourceId: string,
  targetId: string
): Promise<Point> {
  let midpoint: Point = { x: 0, y: 0 };
  // Re-measure on every attempt rather than once up front. Anything that moves a
  // card — inference finishing, a save settling — moves the wire out from under a
  // point measured a moment earlier, and a single mouse.move would then never
  // fire another pointer_move, so the + could never appear no matter how long the
  // assertion waited. The 1px jiggle is what makes the retry a real move.
  await expect
    .poll(
      async () => {
        const from = await portCentre(page, sourceId, 'source');
        const to = await portCentre(page, targetId, 'target');
        // The cubic's t=0.5 point is the chord midpoint for every control-point
        // choice edgeCurve makes, so this is on the stroke, not near it.
        midpoint = { x: (from.x + to.x) / 2, y: (from.y + to.y) / 2 };
        await page.mouse.move(midpoint.x + 1, midpoint.y + 1);
        await page.mouse.move(midpoint.x, midpoint.y);
        return page.getByTestId('edge-insert-plus').count();
      },
      { timeout: 30_000, interval: 300 }
    )
    .toBe(1);
  return midpoint;
}

test('the + on a hovered wire is drawn under the pointer, and presses without moving', async ({
  page,
}) => {
  await buildChain(page);

  const [inputId] = await shapeIdsLeftToRight(page, 'input-node');
  const [linearId] = await shapeIdsLeftToRight(page, 'layer');
  const midpoint = await hoverEdgeMidpoint(page, inputId, linearId);

  const box = await page.getByTestId('edge-insert-plus').boundingBox();
  if (!box) throw new Error('the + reports no bounding box');
  // Within a pixel: the button is 22px across, and being merely "close" is what
  // the 40px regression looked like from a distance.
  expect(Math.abs(box.x + box.width / 2 - midpoint.x)).toBeLessThanOrEqual(1);
  expect(Math.abs(box.y + box.height / 2 - midpoint.y)).toBeLessThanOrEqual(1);

  // No mouse.move — the cursor is already on it, which is the point.
  await page.mouse.down();
  await page.mouse.up();
  await expect(page.getByTestId('edge-insert-picker')).toBeVisible();
});

test('choosing a layer in the wire picker splices it into that one connection', async ({ page }) => {
  await buildChain(page);

  const [inputId] = await shapeIdsLeftToRight(page, 'input-node');
  const [linearId] = await shapeIdsLeftToRight(page, 'layer');
  await hoverEdgeMidpoint(page, inputId, linearId);

  const edgesBefore = await edgeCount(page);
  const cardsBefore = await cardCount(page);

  await page.mouse.down();
  await page.mouse.up();
  await page.locator('.model-edge-insert-input').fill('sli');
  await page.keyboard.press('Enter');

  // One edge becomes two, so both counts rise by exactly one: spliced in, not
  // dropped loose on the canvas next to an untouched connection.
  await expect.poll(() => cardCount(page), { timeout: 10_000 }).toBe(cardsBefore + 1);
  expect(await edgeCount(page)).toBe(edgesBefore + 1);
  await expect(page.getByTestId('edge-insert-picker')).toHaveCount(0);

  // And it lands already applied, on the LAST axis, selected whole. `x[...` is
  // how the card renders a negative dim — see opSummary — and start:end spanning
  // the axis is what makes the default safe to apply without being asked, because
  // the shape it hands on is the shape it was given.
  await expect
    .poll(async () => (await page.locator(CARD).allInnerTexts()).join('|'), { timeout: 15_000 })
    .toMatch(/x\[\.\.\., 0:(\d+)\]/);
});
