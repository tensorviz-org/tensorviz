import { expect, test, Page } from '@playwright/test';

/**
 * The parameter editor: what it opens on, and that an edit keeps itself.
 *
 * It was the Inspector's Properties tab, docked in a 320px panel. That panel is
 * gone — it reserved canvas width whether or not anything was selected, and it
 * opened on the pointer-down that begins a drag — and the fields it held now open
 * as a popover anchored to the card you clicked. The write path underneath is the
 * same debounced commit, which is why every assertion below survived the move.
 *
 * There is no Apply button any more, which puts the burden of correctness on the
 * debounced write. Two ways that went wrong while it was being built, both of
 * them silent, and both pinned below:
 *
 *  - the draft's values are React state and lag a render behind the selection, so
 *    comparing them against a synchronously-updated baseline made autosave decide
 *    the new node's parameters differed from the previous node's values — and
 *    write the difference. Building a three-layer chain erased the parameters of
 *    every layer but the last, and the panel came up EMPTY.
 *  - a write still inside the debounce window was cancelled, not flushed, when the
 *    selection moved, so the last thing you typed vanished if you clicked away
 *    within half a second.
 *
 * Every assertion here is on a value read back after a round trip, never on the
 * field having been filled.
 */

test.use({ viewport: { width: 1440, height: 900 } });

const CARD = '.ui2-node-card';

async function addLayer(page: Page, query: string): Promise<void> {
  const before = await page.locator(CARD).count();
  await page.keyboard.press('Shift+A');
  const search = page.getByLabel('Search layers, blocks and templates');
  await expect(search).toBeVisible();
  await search.fill(query);
  await search.press('Enter');
  await expect.poll(() => page.locator(CARD).count(), { timeout: 20_000 }).toBe(before + 1);
}

/** Ids of the layer cards, left to right — the order addNode lays them out in. */
async function layerIds(page: Page): Promise<string[]> {
  const found = await page.locator('.tl-shape[data-shape-type="layer"]').evaluateAll((els) =>
    els.map((el) => ({ id: el.getAttribute('data-shape-id') || '', x: el.getBoundingClientRect().x }))
  );
  return found.sort((a, b) => a.x - b.x).map((entry) => entry.id);
}

/**
 * Both things that make a card unclickable have to clear first: the
 * `.tl-hit-test-blocker` is `pointer-events: all` until the camera reports idle,
 * so a press lands on it and reaches tldraw as a plain canvas click — the node is
 * never selected and the panel never opens, which reads exactly like a logic bug.
 * A camera still animating has the same symptom.
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

/** Click a card's header. Not its middle: that is over the parameter rows. */
async function selectCard(page: Page, shapeId: string): Promise<void> {
  const box = await page.locator(`[data-shape-id="${shapeId}"]`).boundingBox();
  if (!box) throw new Error(`${shapeId} has no bounding box — fit the view first`);
  await page.mouse.click(box.x + box.width / 2, box.y + 14);
  await expect(page.getByTestId('node-parameters')).toBeVisible();
}

async function buildChain(page: Page): Promise<string[]> {
  await page.goto('/builder?ui=v2');
  await expect(page.locator('.ui2-builder-toolbar')).toBeVisible({ timeout: 60_000 });
  await addLayer(page, 'input');
  await addLayer(page, 'linear');
  await addLayer(page, 'relu');
  await page.getByTestId('ui2-action-fit-view').click();
  // Inference runs on its own now and grows every card, which moves them. Wait for
  // the rows to exist before measuring anything.
  await expect.poll(() => page.locator('.ui2-node-shape-row').count(), { timeout: 30_000 })
    .toBeGreaterThan(0);
  await waitForCanvasSettled(page);
  return layerIds(page);
}

const field = (page: Page, name: string) => page.locator(`[aria-label="parameter-${name}"]`);

test('a click opens the editor on that card, with the layer’s real parameters in it', async ({ page }) => {
  const ids = await buildChain(page);
  await selectCard(page, ids[0]);

  // Linear's own three, not an empty editor: the erasure bug showed up exactly
  // here, as a Reset chip with nothing above it.
  await expect(field(page, 'in_features')).toHaveValue('512');
  await expect(field(page, 'out_features')).toHaveValue('256');
  await expect(field(page, 'bias')).toHaveCount(1);

  // It is beside the card it edits, not docked at the viewport edge — which is
  // the whole reason the panel was removed.
  const editor = await page.getByTestId('node-parameters').boundingBox();
  const card = await page.locator(`[data-shape-id="${ids[0]}"]`).boundingBox();
  if (!editor || !card) throw new Error('the editor or its card has no bounding box');
  expect(Math.abs(editor.x - (card.x + card.width))).toBeLessThan(60);
  expect(Math.abs(editor.y - card.y)).toBeLessThan(40);
});

test('the editor goes away with Escape, so it never sits over the canvas for good', async ({ page }) => {
  // A popover that could not be dismissed would be the docked panel again, just
  // in a different place: whatever it covers has to be one keystroke away.
  const ids = await buildChain(page);
  await selectCard(page, ids[0]);
  await page.keyboard.press('Escape');
  await expect(page.getByTestId('node-parameters')).toHaveCount(0);
});

test('an edit saves itself — there is no Apply to press', async ({ page }) => {
  const ids = await buildChain(page);
  await selectCard(page, ids[0]);

  await field(page, 'out_features').fill('321');
  await expect(page.locator('.ui2-chip', { hasText: 'Apply' })).toHaveCount(0);

  // Read it back off the CARD, which renders the committed graph rather than the
  // draft — so this fails if the write never happened.
  await expect
    .poll(async () => (await page.locator(CARD).allInnerTexts()).join('|'), { timeout: 15_000 })
    .toContain('321');
});

test('an edit still inside the debounce survives the selection moving away', async ({ page }) => {
  const ids = await buildChain(page);
  await selectCard(page, ids[0]);

  await field(page, 'out_features').fill('321');
  // Immediately — well inside the 450ms window. The write must be flushed, not
  // dropped, and it must land on the Linear rather than on whatever is selected
  // by the time it goes out.
  //
  // The INPUT node, deliberately, and not the ReLU to its right: the editor opens
  // to the RIGHT of the card it belongs to, so a click aimed at the next card in
  // the chain can land on the editor and the selection never moves — which would
  // make this test pass without ever testing anything. The Input is to the left
  // and always clear of it.
  const inputId = await page
    .locator('.tl-shape[data-shape-type="input-node"]')
    .first()
    .getAttribute('data-shape-id');
  await selectCard(page, inputId as string);
  await expect(field(page, 'name')).toHaveCount(1);

  await selectCard(page, ids[0]);
  await expect(field(page, 'out_features')).toHaveValue('321');
  await expect(field(page, 'in_features')).toHaveValue('512');
});

test('the commit carries real types, and leaves the keys you did not touch alone', async ({
  page,
}) => {
  const ids = await buildChain(page);
  await selectCard(page, ids[0]);
  await expect(field(page, 'bias')).toHaveValue('true');

  // Listen to what actually crosses the boundary, rather than inferring it from
  // the card: the corruption was invisible on screen, because both a real boolean
  // and the string 'True' render as "True".
  await page.evaluate(() => {
    (window as any).__commits = [];
    window.addEventListener('builder-command', (event: any) => {
      if (event.detail?.command === 'set-parameters') {
        (window as any).__commits.push(event.detail.payload);
      }
    });
  });

  await field(page, 'out_features').fill('321');
  await expect
    .poll(() => page.evaluate(() => (window as any).__commits.length), { timeout: 15_000 })
    .toBeGreaterThan(0);

  const committed = await page.evaluate(() => (window as any).__commits.at(-1).parameters);
  // The field that moved: a number, not the string the input handed over.
  expect(committed.out_features).toBe(321);
  // The field that did NOT move: still a real boolean. This is the whole point —
  // the commit writes the entire map, and `bias` used to come back as the string
  // 'True'. A stringified `bias: 'False'` is truthy in Python, so it silently
  // switches the bias on in the exported model while validation stays green.
  expect(committed.bias).toBe(true);
  expect(typeof committed.bias).toBe('boolean');
  expect(committed.in_features).toBe(512);
});
