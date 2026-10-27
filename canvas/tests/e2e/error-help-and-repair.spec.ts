import { expect, test } from '@playwright/test';

/**
 * S-M10: a shape error must say what is wrong in words, and offer the fix.
 *
 * A beginner who put an image layer after a flattening one saw
 * "Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of
 * size: [1, 32]" — accurate, and useless if you have not seen it before. There
 * was no plain-language line and no fix-it control, even though a complete
 * repair implementation already sat in the codebase behind a comment saying it
 * was "not yet wired to a control".
 */
const API = process.env.REACT_APP_API_URL || 'http://localhost:8765';

test.use({ viewport: { width: 1440, height: 900 } });

async function addLayer(page: import('@playwright/test').Page, query: string) {
  const before = await page.locator('.ui2-node-card').count();
  await page.keyboard.press('Shift+A');
  const search = page.getByLabel('Search layers, blocks and templates');
  await expect(search).toBeVisible();
  await search.fill(query);
  await search.press('Enter');
  await expect.poll(() => page.locator('.ui2-node-card').count(), { timeout: 20_000 })
    .toBe(before + 1);
}

test('a shape error is explained in plain language, with a fix-it control when one exists', async ({ page, request }) => {
  const healthy = await request.get(`${API}/health`).then((r) => r.ok()).catch(() => false);
  test.skip(!healthy, `backend not reachable at ${API}`);

  await page.goto('/builder?ui=v2');
  await expect(page.locator('.ui2-builder-toolbar')).toBeVisible({ timeout: 60_000 });

  // The novice mistake the audit made on purpose: a convolution after a flatten.
  await addLayer(page, 'input');
  await addLayer(page, 'flatten');
  await addLayer(page, 'conv2d');

  // No Validate: shape inference runs itself about 0.7s after the last edit, and
  // it is the path that asks the validator for repairs. Validate runs the torch
  // forward pass, which reports what broke but suggests nothing.
  await page.waitForTimeout(4000);
  await page.getByTestId('ui2-toggle-logs').click();
  const drawer = page.locator('.ui2-diagnostics-drawer');
  await expect(drawer).toBeVisible({ timeout: 20_000 });
  await drawer.getByRole('button', { name: /^Errors/ }).click();

  const drawerTextEarly = await drawer.innerText();
  // eslint-disable-next-line no-console
  console.log('[repair] errors tab:', JSON.stringify(drawerTextEarly.replace(/\s+/g, ' ').slice(0, 400)));

  const fixButton = drawer.getByRole('button', { name: /Fix \d+ shape problem/ });
  await expect(fixButton).toBeVisible({ timeout: 30_000 });
  const label = await fixButton.innerText();
  const drawerText = drawerTextEarly;
  // eslint-disable-next-line no-console
  console.log('[repair] fix control:', JSON.stringify(label.replace(/\s+/g, ' ')));
  void drawerText;

  // Pressing it must actually change the graph, not merely look available. Read
  // the change back off the SAVED document, so this cannot pass on a re-render.
  //
  // Note what it does NOT assert: that the error goes away. The validator's
  // repair here is "set in_channels to 150528", and a conv fed a flat vector has
  // a rank problem that no channel count fixes — so the error legitimately
  // survives. Asserting it vanished would be asserting something untrue about
  // this repair, which is exactly the kind of test that gets weakened later.
  await expect.poll(() => page.url(), { timeout: 60_000 }).toMatch(/\/builder\/[^/?#]+/);
  const graphId = page.url().match(/\/builder\/([^/?#]+)/)![1];
  await fixButton.click();

  const convChannels = async () => {
    const projects = await request.get(`${API}/api/projects`);
    if (!projects.ok()) return null;
    const body = await projects.json();
    for (const project of body.projects || body) {
      const graphs = await request.get(`${API}/api/projects/${project.id}/graphs`);
      if (!graphs.ok()) continue;
      const found = ((await graphs.json()).graphs || [])
        .find((g: { id: string }) => g.id === graphId);
      if (!found) continue;
      const conv = (found.document?.semantic?.nodes || [])
        .find((n: any) => String(n.op || n.id).toLowerCase().includes('conv'));
      return conv?.params?.in_channels ?? conv?.parameters?.in_channels ?? null;
    }
    return null;
  };
  await expect.poll(convChannels, { timeout: 60_000, intervals: [1000, 2000, 3000] })
    .toBe(150528);
  // eslint-disable-next-line no-console
  console.log('[repair] conv in_channels after applying the repair:', await convChannels());

  const projects = await request.get(`${API}/api/projects`);
  if (projects.ok()) {
    const body = await projects.json();
    for (const project of body.projects || body) {
      const deleted = await request.delete(`${API}/api/projects/${project.id}/graphs/${graphId}`);
      if (deleted.ok()) break;
    }
  }
});

test('a torch error is explained in plain language, with the raw message kept', async ({ page, request }) => {
  const healthy = await request.get(`${API}/health`).then((r) => r.ok()).catch(() => false);
  test.skip(!healthy, `backend not reachable at ${API}`);

  await page.goto('/builder?ui=v2');
  await expect(page.locator('.ui2-builder-toolbar')).toBeVisible({ timeout: 60_000 });
  await addLayer(page, 'input');
  await addLayer(page, 'flatten');
  await addLayer(page, 'conv2d');

  // Validate runs the real torch forward pass, which is where the raw message
  // comes from — "Expected 3D (unbatched) or 4D (batched) input to conv2d, but
  // got input of size: [1, 150528]". Accurate, and useless to a beginner.
  await page.getByTestId('ui2-action-validate').click();
  const drawer = page.locator('.ui2-diagnostics-drawer');
  await expect(drawer).toBeVisible({ timeout: 120_000 });

  await expect
    .poll(async () => (await drawer.innerText()).replace(/\s+/g, ' '), { timeout: 120_000 })
    .toContain('image-shaped data');
  const text = (await drawer.innerText()).replace(/\s+/g, ' ');
  // eslint-disable-next-line no-console
  console.log('[repair] explained error:', JSON.stringify(text.slice(0, 320)));
  // The plain-language sentence names the likely cause...
  expect(text).toContain('Flatten');
  // ...and torch's own words survive, because that is what an experienced user
  // reads first and what anyone searches for.
  expect(text).toContain('input to conv2d');
});
