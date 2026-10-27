import { expect, test, Locator } from '@playwright/test';

// S-m2: "Grouping layers into a module is irreversible — there is no Ungroup."
//
// Pressing G collapsed a selection into a module and nothing anywhere undid it.
// The model-history stack lives in a ref and dies on reload, so a group made
// yesterday was permanent, and the user guide said as much.
//
// The interesting half is the edges. Grouping REWRITES every edge that crosses
// the selection boundary onto the wrapper and keeps no record of the child it
// came from, so the failure mode to guard against is not "the block is still
// there" — it is an ungroup that leaves the graph quietly severed. This asserts
// the layer count comes back AND the graph is still connected afterwards.

const API = process.env.REACT_APP_API_URL || 'http://localhost:8765';
const SHAPE = '.ui2-node-card, .layer-node, .module-node';
const MODULE = '.module-node, .module-container';

test.use({ viewport: { width: 1440, height: 900 } });

// Raw pointer input is necessary inside tldraw, but it does not get Playwright's
// normal stability checks. Selection opens the parameter editor and moves the canvas.
async function settledBox(locator: Locator) {
  await expect(locator).toBeVisible();
  await expect(locator.page().locator('.tl-hit-test-blocker')).toBeHidden();
  return locator.evaluate(element => new Promise<{ x: number; y: number; width: number; height: number }>(resolve => {
    let previous = '', stable = 0;
    const frame = () => {
      const rect = element.getBoundingClientRect();
      const next = [rect.x, rect.y, rect.width, rect.height].map(Math.round).join(',');
      stable = next === previous ? stable + 1 : 0;
      previous = next;
      if (stable >= 12) resolve({ x: rect.x, y: rect.y, width: rect.width, height: rect.height });
      else requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }));
}

test('a module made with G can be ungrouped again, edges intact', async ({ page, request }) => {
  const healthy = await request.get(`${API}/health`).then((r) => r.ok()).catch(() => false);
  test.skip(!healthy, `backend not reachable at ${API}`);

  // Grouping asks for the class name through window.prompt.
  page.on('dialog', (dialog) => dialog.accept('AuditFixBlock'));

  await page.goto('/?ui=v2');
  await page.getByText('Choose from Template', { exact: false }).first().click();
  const useButtons = page.getByRole('button', { name: /Use Template/i });
  await expect(useButtons.first()).toBeVisible();
  await useButtons.first().click();
  await page.waitForURL(/\/builder\/[^/]+/, { timeout: 30_000 });

  const shape = page.locator(SHAPE);
  await expect(shape.first()).toBeVisible({ timeout: 20_000 });
  const before = await shape.count();
  expect(before).toBeGreaterThan(2);
  const edgeCount = () => page.locator('.model-edge, [data-shape-type="model-edge"]').count();
  const edgesBefore = await edgeCount();

  // Select two adjacent layers and group them. tldraw owns pointer input and its
  // background swallows Playwright's element click, so drive the real mouse at the
  // cards' headers (parameter rows handle their own clicks); shift-click is how
  // the builder multi-selects, and G is the
  // shortcut the guide documents.
  const clickCard = async (index: number, shiftHeld = false) => {
    const box = await settledBox(shape.nth(index));
    if (shiftHeld) await page.keyboard.down('Shift');
    await page.mouse.click(box.x + box.width / 2, box.y + 14);
    if (shiftHeld) await page.keyboard.up('Shift');
  };
  await clickCard(1);
  await expect(page.locator('.ui2-node-card.selected')).toHaveCount(1);
  await clickCard(2, true);
  await expect(page.locator('.ui2-node-card.selected')).toHaveCount(2);
  await page.keyboard.press('g');

  const moduleBlock = page.locator(MODULE);
  await expect(moduleBlock.first()).toBeVisible({ timeout: 15_000 });
  const grouped = await shape.count();
  // eslint-disable-next-line no-console
  console.log(`[ungroup] shapes before=${before} after group=${grouped} edges before=${edgesBefore}`);
  expect(grouped).toBeLessThan(before);

  // Now the thing that did not exist: ungroup, from the block's own controls.
  //
  // NOT from the More menu, even though the item is there too: that menu panel is
  // painted behind the tldraw canvas (tl-background__wrapper carries z-index 100,
  // the floating toolbar's stacking context only 30), so every item in it is
  // invisible and unclickable in a real browser. That is a pre-existing chrome
  // bug, unrelated to ungrouping — but it is why the affordance a user can
  // actually reach is on the module card.
  const ungroupButton = page.getByTitle('Ungroup — break this module back into its layers');
  const buttonBox = await settledBox(ungroupButton.first());
  // The card is inside a tldraw shape, so the button reacts to pointerdown and
  // the real mouse has to press it (see canvasButtonHandlers in ModuleShapeUtil).
  await page.mouse.click(buttonBox.x + buttonBox.width / 2, buttonBox.y + buttonBox.height / 2);

  // The wrapper is gone and every layer is back on the parent canvas.
  await expect(moduleBlock).toHaveCount(0, { timeout: 15_000 });
  await expect.poll(() => shape.count(), { timeout: 15_000 }).toBe(before);
  // And the boundary edges landed on real children rather than vanishing with
  // the wrapper — an ungroup that loses edges is worse than no ungroup at all.
  await expect.poll(edgeCount, { timeout: 15_000 }).toBe(edgesBefore);
  // eslint-disable-next-line no-console
  console.log(`[ungroup] after ungroup shapes=${await shape.count()} edges=${await edgeCount()}`);
});
