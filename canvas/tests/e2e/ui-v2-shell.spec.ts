import { expect, Page, test } from '@playwright/test';

/**
 * The v2 shell: the nav that gets you between routes, the builder's on-surface
 * command bar, and the guidance a disabled control owes the user.
 *
 * This was ui-v2-visual.spec.ts, skipped since 2026-07-12 and described in the
 * register as needing "a human visual re-baseline". It never had a baseline to
 * re-take — there is no screenshot, no toMatchSnapshot, nothing visual in it. Every
 * assertion was a DOM query, so the file is checkable by machine and was renamed
 * to stop implying otherwise.
 *
 * Four of its six tests asserted chrome that #20–#23 deliberately removed (a
 * "Recent Projects" heading, header-new-project-button, a right-panel show/hide
 * toggle, disabled Help and Account buttons) and are gone rather than rewritten:
 * asserting removed chrome is how the frontend smoke gate came to assert nothing.
 * The parameter editor is now covered by node-parameters-reach-the-canvas.spec.ts. What
 * survives is what is still true and still uncovered — no other spec navigates the
 * rail or switches to Layer Code.
 */

test.use({ viewport: { width: 1440, height: 900 } });

test('the workspace rail reaches each of its destinations', async ({ page }) => {
  const destinations = [
    { label: 'Models', url: /\/projects$/, anchor: (p: Page) => p.getByRole('tablist', { name: 'Filter models by status' }) },
    { label: 'Sandboxes', url: /\/sandboxes$/, anchor: (p: Page) => p.getByRole('heading', { name: 'Sandboxes' }) },
    { label: 'Builder', url: /\/builder$/, anchor: (p: Page) => p.locator('.ui2-builder-toolbar') },
  ];

  for (const { label, url, anchor } of destinations) {
    await page.goto('/?ui=v2');
    await page
      .getByRole('navigation', { name: 'Primary' })
      .getByRole('button', { name: label, exact: true })
      .click();
    await expect(page).toHaveURL(url);

    // A URL alone proves navigate() ran, not that anything is behind it: the
    // catch-all route renders "Page not found" for a path with no page. So each
    // destination is identified by something only that page renders.
    await expect(page.getByText('Page not found')).toHaveCount(0);
    await expect(anchor(page), `${label} did not render`).toBeVisible({ timeout: 60_000 });
  }
});

test('the builder keeps its command bar on the surface and switches to Layer Code', async ({ page }) => {
  await page.goto('/builder?ui=v2');
  await expect(page.locator('.ui2-builder-toolbar')).toBeVisible({ timeout: 60_000 });

  // The controls the toolbar keeps in reach rather than behind More. Everything
  // else is checked by more-menu-reachable.spec.ts, which hit-tests the menu.
  const onSurface = [
    'ui2-view-graph',
    'ui2-view-code',
    'ui2-action-layout',
    'ui2-action-fit-view',
    'ui2-action-undo',
    'ui2-action-redo',
    'ui2-action-toggle-connection',
    'ui2-action-delete-selected',
    'ui2-action-more',
    'ui2-toggle-logs',
    'ui2-toggle-history',
    'ui2-action-export-code',
    'ui2-action-validate',
  ];
  for (const testid of onSurface) {
    await expect(page.getByTestId(testid), `${testid} is not on the toolbar`).toBeVisible();
  }

  const code = page.getByRole('heading', { name: 'Selected Layer Code' });
  await expect(code).toHaveCount(0);
  await page.getByTestId('ui2-view-code').click();
  await expect(code).toBeVisible();
  await page.getByTestId('ui2-view-graph').click();
  await expect(code).toHaveCount(0);
  await expect(page.getByTestId('builder-canvas')).toBeVisible();
});

test('a control that cannot act says what would let it act', async ({ page }) => {
  await page.goto('/builder?ui=v2');
  await expect(page.locator('.ui2-builder-toolbar')).toBeVisible({ timeout: 60_000 });

  // Nothing is selected on a fresh builder, so each of these is off — and a
  // control that is off without saying why is the complaint this test inherited.
  const connect = page.getByTestId('ui2-action-toggle-connection');
  await expect(connect).toBeDisabled();
  await expect(connect).toHaveAttribute('title', /Select exactly two layers/i);

  const remove = page.getByTestId('ui2-action-delete-selected');
  await expect(remove).toBeDisabled();
  await expect(remove).toHaveAttribute('title', /Select a layer first/i);

  await page.getByTestId('ui2-action-more').click();
  const group = page.getByRole('menuitem', { name: /Group as module/i });
  await expect(group).toBeDisabled();
  await expect(group).toHaveAttribute('title', /Select two or more layers to group/i);

  const ungroup = page.getByTestId('ui2-action-ungroup');
  await expect(ungroup).toBeDisabled();
  await expect(ungroup).toHaveAttribute('title', /Select a module block/i);
});
