import { expect, test } from '@playwright/test';

/**
 * S-M4: a validated model must stop saying "Never run".
 *
 * Every card on Home carried a "Never run" badge and could never carry anything
 * else. The builder recorded verification in browser localStorage; Home reads
 * status out of the saved document's metadata, which nothing ever wrote; and the
 * function bridging the two had zero callers. So the status system — the thing
 * that is supposed to tell you which of your models work — was decorative, and
 * the getting-started guide promised a flip that could not happen.
 *
 * Two assertions, because they fail for different reasons: that the outcome
 * reaches the SAVED DOCUMENT (the write side, read back over the API so it cannot
 * pass on a stale render), and that Home then shows something other than
 * "Never run" (the read side).
 *
 * A template is the starting point on purpose: it lands a graph that already
 * validates, with an id, without this spec having to drive parameter fields
 * through a settling canvas.
 */
const API = process.env.REACT_APP_API_URL || 'http://localhost:8765';

test.use({ viewport: { width: 1440, height: 900 } });

test('a validated model records its result in the saved document, and Home reads it', async ({ page, request }) => {
  const healthy = await request.get(`${API}/health`).then((r) => r.ok()).catch(() => false);
  test.skip(!healthy, `backend not reachable at ${API}`);

  await page.goto('/?ui=v2');
  await page.getByText('Choose from Template', { exact: false }).first().click();
  // Use Template is disabled until the gallery has a card selected, and which
  // card that is depends on how fast the category request lands.
  const useButton = page.locator('button:not([disabled])', { hasText: 'Use Template' }).first();
  await expect(useButton).toBeVisible({ timeout: 60_000 });
  await useButton.click();
  await page.waitForURL(/\/builder\/[^/]+/, { timeout: 30_000 });
  const graphId = page.url().match(/\/builder\/([^/?#]+)/)![1];
  await expect(page.locator('.ui2-node-card').first()).toBeVisible({ timeout: 20_000 });

  // Before: nothing has been validated, so there is no record.
  const recordFor = async () => {
    const projects = await request.get(`${API}/api/projects`);
    if (!projects.ok()) return null;
    const body = await projects.json();
    for (const project of body.projects || body) {
      const graphs = await request.get(`${API}/api/projects/${project.id}/graphs`);
      if (!graphs.ok()) continue;
      const list = (await graphs.json()).graphs || [];
      const found = list.find((g: { id: string }) => g.id === graphId);
      if (found) return found.document?.semantic?.metadata?.verification ?? null;
    }
    return null;
  };
  expect(await recordFor()).toBeNull();

  await page.getByTestId('ui2-action-validate').click();
  await expect
    .poll(async () => (await page.locator('body').innerText()).toLowerCase(), { timeout: 120_000 })
    .toMatch(/verified|error/);

  // The write side.
  const record = await expect
    .poll(recordFor, { timeout: 60_000, intervals: [1000, 2000, 3000] })
    .not.toBeNull()
    .then(() => recordFor());
  // eslint-disable-next-line no-console
  console.log('[status] saved verification record:', JSON.stringify(record));
  expect(record).toMatchObject({ ok: expect.any(Boolean), revision: expect.any(Number) });

  // The read side: Home no longer says "Never run" for this model. Located by
  // name rather than by markup, so a redesign of the card does not silently turn
  // this into a test of nothing.
  const graphName = await page.getByTestId('ui2-model-name').innerText().catch(async () => {
    const projects = await request.get(`${API}/api/projects`);
    const body = await projects.json();
    for (const project of body.projects || body) {
      const graphs = await request.get(`${API}/api/projects/${project.id}/graphs`);
      if (!graphs.ok()) continue;
      const found = ((await graphs.json()).graphs || []).find((g: { id: string }) => g.id === graphId);
      if (found) return found.name as string;
    }
    return '';
  });
  expect(graphName).not.toBe('');

  await page.goto('/?ui=v2');
  // Cards are buttons whose whole text is the card, status included. Matched on
  // "carries this model's name AND a status" so the sidebar's pinned shortcut —
  // same name, no status — cannot be mistaken for the card. Most-recently-edited
  // is first, so the first such card is this model's.
  const statusCards = async () =>
    (await page.locator('button').allInnerTexts())
      .map((text) => text.replace(/\s+/g, ' ').trim())
      .filter((text) => text.includes(graphName) && /Never run|Verified|error/i.test(text));

  await expect.poll(async () => (await statusCards()).length, { timeout: 60_000 })
    .toBeGreaterThan(0);
  const cardText = (await statusCards())[0];
  // eslint-disable-next-line no-console
  console.log('[status] home card:', JSON.stringify(cardText.slice(0, 200)));
  expect(cardText).not.toContain('Never run');
  expect(cardText).toContain('Verified');

  // Clean up after ourselves. A spec that leaves a graph behind on every run is
  // how the home page came to hold 214 of them.
  const projects = await request.get(`${API}/api/projects`);
  if (projects.ok()) {
    const body = await projects.json();
    for (const project of body.projects || body) {
      const deleted = await request.delete(`${API}/api/projects/${project.id}/graphs/${graphId}`);
      if (deleted.ok()) break;
    }
  }
});
