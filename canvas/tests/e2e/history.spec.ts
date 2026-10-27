import { expect, test } from '@playwright/test';

/**
 * The You/AI version timeline, rewritten against the panel that actually ships.
 *
 * The old file drove /agent-demo's "Session History" and "Paper Session History"
 * panels through helpers.ts. /agent-demo is a redirect to /builder now, and every
 * selector those helpers used (session-history-toggle, paper-panel-toggle,
 * paper-history-list) is gone from src/ — so the file could only have been revived
 * by inventing a page. What replaced those panels is HistoryPanelV2, one timeline
 * of who changed the saved graph, and it had no browser coverage at all.
 *
 * It needed no new testids to get some: the toolbar button carries
 * data-testid="ui2-toggle-history" and the panel is a role="dialog" named
 * "Version history".
 *
 * The second revision is committed over the API with author=agent rather than by
 * driving a real AI turn. Telling You from AI is the entire point of this panel and
 * only the agent author produces the AI badge, so the alternative was to duplicate
 * qa-ai-oncanvas.spec.ts's whole turn just to reach a badge.
 */
const API = process.env.REACT_APP_API_URL || 'http://localhost:8765';

test.use({ viewport: { width: 1440, height: 900 } });

test('an unsaved graph is told to save, not shown an empty timeline', async ({ page }) => {
  await page.goto('/builder?ui=v2');
  await expect(page.locator('.ui2-builder-toolbar')).toBeVisible({ timeout: 60_000 });

  await page.getByTestId('ui2-toggle-history').click();
  const panel = page.getByRole('dialog', { name: 'Version history' });
  await expect(panel).toBeVisible();
  await expect(panel).toContainText('Save this graph to start tracking its history.');
  // Not "No versions yet." — there is no graph to have versions of yet, and the
  // two states are different instructions to the user.
  await expect(panel.getByText('No versions yet.')).toHaveCount(0);

  await panel.getByRole('button', { name: 'Close' }).click();
  await expect(panel).toHaveCount(0);
});

test('the timeline says who changed the graph, newest first, restorable except the newest', async ({ page, request }) => {
  const healthy = await request.get(`${API}/health`).then((r) => r.ok()).catch(() => false);
  test.skip(!healthy, `backend not reachable at ${API}`);

  // A template is the starting point for the same reason verified-status-reaches-home
  // uses one: it lands a graph that is already saved, with an id, without driving
  // the canvas.
  await page.goto('/?ui=v2');
  await page.getByText('Choose from Template', { exact: false }).first().click();
  const useButton = page.locator('button:not([disabled])', { hasText: 'Use Template' }).first();
  await expect(useButton).toBeVisible({ timeout: 60_000 });
  await useButton.click();
  await page.waitForURL(/\/builder\/[^/]+/, { timeout: 30_000 });
  const graphId = page.url().match(/\/builder\/([^/?#]+)/)![1];
  await expect(page.locator('.ui2-node-card').first()).toBeVisible({ timeout: 20_000 });

  const projects = (await (await request.get(`${API}/api/projects`)).json()).projects || [];
  let projectId = '';
  let graph: { current_revision: number; document: unknown } | null = null;
  for (const project of projects) {
    const response = await request.get(`${API}/api/projects/${project.id}/graphs/${graphId}`);
    if (!response.ok()) continue;
    projectId = project.id;
    graph = (await response.json()).graph;
    break;
  }
  expect(graph, `template graph ${graphId} not found under any project`).not.toBeNull();

  // Revision 2, attributed to the AI. The template landing was revision 1 and the
  // user made it, so the panel now has one of each author to tell apart.
  const commit = await request.patch(`${API}/api/projects/${projectId}/graphs/${graphId}`, {
    data: {
      document: graph!.document,
      expected_revision: graph!.current_revision,
      author: 'agent',
      op_summary: 'assistant edit',
    },
  });
  expect(commit.ok(), `committing an agent revision failed: ${commit.status()}`).toBeTruthy();

  await page.getByTestId('ui2-toggle-history').click();
  const panel = page.getByRole('dialog', { name: 'Version history' });
  await expect(panel).toBeVisible();
  await expect(panel.getByText('No versions yet.')).toHaveCount(0);
  await expect(panel.getByText('Save this graph to start tracking its history.')).toHaveCount(0);

  const rows = panel.locator('.ui2-panel.p-3');
  await expect(rows).toHaveCount(2);
  // eslint-disable-next-line no-console
  console.log(`[history] panel rows: ${JSON.stringify(await rows.allInnerTexts())}`);

  // Newest first, and the AI edit is attributed to the AI.
  await expect(rows.nth(0)).toContainText('rev 2');
  await expect(rows.nth(0)).toContainText('assistant edit');
  await expect(rows.nth(0).getByText('AI', { exact: true })).toBeVisible();
  // You cannot restore the version you are already on, so the newest entry has no
  // Restore control.
  await expect(rows.nth(0).getByRole('button', { name: 'Restore' })).toHaveCount(0);

  // The template landing was the user's own doing, so it must not say AI.
  await expect(rows.nth(1)).toContainText('rev 1');
  await expect(rows.nth(1).getByText('You', { exact: true })).toBeVisible();
  await expect(rows.nth(1).getByText('AI', { exact: true })).toHaveCount(0);
  await expect(rows.nth(1).getByRole('button', { name: 'Restore' })).toHaveCount(1);

  // Clean up after ourselves, the way verified-status-reaches-home does: a spec
  // that leaves a graph behind on every run is how Home came to hold 214 of them.
  await request.delete(`${API}/api/projects/${projectId}/graphs/${graphId}`);
});
