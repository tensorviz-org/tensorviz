import { expect, test, Page } from '@playwright/test';

/**
 * CV-11, closed by removal.
 *
 * The Inspector was `position:absolute; width:320px` inside the builder stage
 * while the tldraw surface was `inset:0` of that same stage, so the canvas ran
 * UNDERNEATH it: `document.elementsFromPoint` in that strip returned the panel
 * and no `.tl-shape`, and the click never reached tldraw. The card in the strip
 * could not be selected, and its drag-to-connect ports sat even deeper into it.
 * The fix at the time was to reserve the 320px so `.tl-container` genuinely
 * narrowed; this file measured that reserve, and its last test was the
 * counterweight that stopped the gutter becoming permanent.
 *
 * The panel is gone. Parameters are edited on the node, in a popover anchored to
 * the card, which is why the reserve went with it. That makes the dead-strip
 * class structurally impossible — nothing is docked and nothing is opaque for
 * longer than the gesture that opened it — so what is left to prove is exactly
 * that: the canvas is full-bleed whether or not the editor is open, and whatever
 * the editor does cover is one keystroke or one click away.
 *
 * The file is kept rather than deleted because the guarantee is the same one;
 * only the mechanism that provides it changed.
 */

test.use({ viewport: { width: 1440, height: 900 } });

const CARD = '.ui2-node-card';
const EDITOR = '[data-testid="node-parameters"]';

interface Card {
  id: string;
  x: number;
  width: number;
}

async function addLayer(page: Page, query: string): Promise<void> {
  const before = await page.locator(CARD).count();
  await page.keyboard.press('Shift+A');
  const search = page.getByLabel('Search layers, blocks and templates');
  await expect(search).toBeVisible();
  await search.fill(query);
  await search.press('Enter');
  await expect.poll(() => page.locator(CARD).count(), { timeout: 20_000 }).toBe(before + 1);
}

/**
 * Both things that make a point unclickable have to clear first: the
 * `.tl-hit-test-blocker` is `pointer-events:all` until the camera reports idle,
 * and a camera still animating moves the card out from under a coordinate
 * measured a moment earlier. Shape inference also grows every card by its In/Out
 * rows once it lands, which moves them again.
 */
async function waitForCanvasSettled(page: Page): Promise<void> {
  let previous = '';
  let stable = 0;
  await expect
    .poll(
      async () => {
        const blocker = page.locator('.tl-hit-test-blocker');
        const hidden = (await blocker.count()) === 0 || (await blocker.first().isHidden());
        const geometry = await page.locator('.tl-shape[data-shape-type]').evaluateAll((els) =>
          els
            .map((el) => {
              const rect = el.getBoundingClientRect();
              return `${Math.round(rect.x)},${Math.round(rect.y)},${Math.round(rect.height)}`;
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

/** The node cards, left to right — the order the palette lays them out in. */
async function cardsLeftToRight(page: Page): Promise<Card[]> {
  const found = await page
    .locator('.tl-shape[data-shape-type="layer"], .tl-shape[data-shape-type="input-node"]')
    .evaluateAll((els) =>
      els.map((el) => {
        const rect = el.getBoundingClientRect();
        return { id: el.getAttribute('data-shape-id') || '', x: rect.x, width: rect.width };
      })
    );
  return found.sort((a, b) => a.x - b.x);
}

/** Input -> Linear -> ReLU. */
async function buildChain(page: Page): Promise<Card[]> {
  await page.goto('/builder?ui=v2');
  await expect(page.locator('.ui2-builder-toolbar')).toBeVisible({ timeout: 60_000 });
  await addLayer(page, 'input');
  await addLayer(page, 'linear');
  await addLayer(page, 'relu');
  await page.getByTestId('ui2-action-fit-view').click();
  // Inference runs on its own now and grows each card, moving every coordinate.
  await expect.poll(() => page.locator('.ui2-node-shape-row').count(), { timeout: 30_000 })
    .toBeGreaterThan(0);
  await waitForCanvasSettled(page);
  return cardsLeftToRight(page);
}

/** Click a card's header, which is the gesture that opens the editor. */
async function clickCard(page: Page, card: Card): Promise<void> {
  const box = await page.locator(`[data-shape-id="${card.id}"]`).boundingBox();
  if (!box) throw new Error(`${card.id} has no bounding box — it is culled`);
  await page.mouse.click(box.x + box.width / 2, box.y + 14);
}

/** What the browser says is on top at a point, outermost class first. */
async function stackAt(page: Page, x: number, y: number): Promise<string[]> {
  return page.evaluate(
    ([px, py]) =>
      document
        .elementsFromPoint(px as number, py as number)
        .slice(0, 6)
        .map((el) => (el.className || el.tagName).toString()),
    [x, y]
  );
}

test('the canvas is full-bleed with the parameter editor open', async ({ page }) => {
  // The property the reserve used to buy conditionally, now unconditional: with
  // nothing docked, `.tl-container` is the full viewport and no coordinate on it
  // is dead. Unfixed — a panel restored, or the 352px reserve left behind in the
  // stylesheet — this measures short.
  const cards = await buildChain(page);
  await clickCard(page, cards[1]);
  await expect(page.locator(EDITOR)).toBeVisible();

  const frame = await page.getByTestId('builder-canvas').boundingBox();
  const container = await page.locator('.tl-container').boundingBox();
  if (!frame || !container) throw new Error('missing bounding box');
  expect(Math.round(frame.width)).toBe(1440);
  expect(Math.round(container.x)).toBe(0);
});

test('the card being edited keeps the ports you drag connections from', async ({ page }) => {
  // The generalisation that made CV-11 worse than a selection bug: the panel
  // covered the drag-to-connect ports of the card in its strip. The editor opens
  // clear of the card it belongs to, so that card's own ports stay reachable —
  // which is the case a popover could plausibly get wrong by overlapping its
  // anchor.
  const cards = await buildChain(page);
  await clickCard(page, cards[1]);
  await expect(page.locator(EDITOR)).toBeVisible();

  const port = page.locator(`[data-shape-id="${cards[1].id}"] [data-testid="node-port-source"]`).first();
  await expect(port).toBeVisible();
  const box = await port.boundingBox();
  if (!box) throw new Error('the source port reports no bounding box');

  const stack = await stackAt(page, box.x + box.width / 2, box.y + box.height / 2);
  expect(stack.join(' | ')).not.toContain('model-node-params');
});

test('clicking the canvas puts back whatever the editor was covering', async ({ page }) => {
  // The counterweight, and what makes a popover admissible where a panel was not:
  // it is opaque only while you are using it. A click on empty canvas clears the
  // selection it belongs to, and it goes with it.
  const cards = await buildChain(page);
  await clickCard(page, cards[1]);
  await expect(page.locator(EDITOR)).toBeVisible();

  await page.mouse.click(40, 860);
  await expect(page.locator(EDITOR)).toHaveCount(0);
});

test('opening the editor never hides a toolbar control', async ({ page }) => {
  // The bar holds ~995px of controls and scrolls on overflow, so anything that
  // narrows it clips its right end out of sight — and took Validate and Export
  // Code with it when the reserve was first added. Nothing narrows it now; this
  // is what would catch a reserve coming back for the popover.
  await page.setViewportSize({ width: 1280, height: 900 });
  const cards = await buildChain(page);
  await clickCard(page, cards[1]);
  await expect(page.locator(EDITOR)).toBeVisible();

  const bar = page.locator('.ui2-builder-toolbar');
  const overflow = await bar.evaluate((el) => el.scrollWidth - el.clientWidth);
  expect(overflow).toBe(0);

  for (const name of ['Validate', 'Export Code']) {
    const button = page.getByRole('button', { name, exact: true });
    await expect(button).toBeVisible();
    const box = await button.boundingBox();
    const barBox = await bar.boundingBox();
    if (!box || !barBox) throw new Error(`${name} or the toolbar has no bounding box`);
    expect(box.x).toBeGreaterThanOrEqual(barBox.x);
    expect(box.x + box.width).toBeLessThanOrEqual(barBox.x + barBox.width + 1);
  }
});
