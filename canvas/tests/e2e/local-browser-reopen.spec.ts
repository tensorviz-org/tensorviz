import { expect, test } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';

/**
 * Close the tab, run "Open as Graph" again, get a tab back.
 *
 * The host used to decide "a page is open" from its OWN socket, which closing a
 * browser tab does not touch — so the second invocation published a retarget into
 * an empty roster and opened nothing. Nothing in the repo could catch that: the
 * host, the broker and the page all behaved correctly in isolation, and only their
 * composition was wrong.
 *
 * Needs the disposable host from `extension/scripts/qa-host.mjs`, which sets
 * TENSORVIZ_LOCAL_BROWSER_LINK_FILE so every page the extension would hand to a
 * browser is appended to a file instead. That file is the observable: without it
 * "opened a tab" and "silently did nothing" look identical from outside.
 */

const pageFile = process.env.TENSORVIZ_QA_PAGE_FILE;
test.skip(!pageFile, 'Requires a running disposable host (extension/scripts/qa-host.mjs)');

const controlDir = () => path.dirname(pageFile!);
const linkFile = () => path.join(controlDir(), 'opened-links.txt');

const opens = (): string[] =>
  fs.existsSync(linkFile())
    ? fs.readFileSync(linkFile(), 'utf8').split('\n').filter(Boolean)
    : [];

/** Ask the extension host to run a command. The driver polls at 500ms. */
const command = (body: Record<string, unknown>) =>
  fs.writeFileSync(path.join(controlDir(), 'command.json'), JSON.stringify(body), 'utf8');

test('Open as Graph reopens the page after every view has closed', async ({ browser }) => {
  const url = JSON.parse(fs.readFileSync(pageFile!, 'utf8')).pageUrl as string;
  const before = opens().length;

  // A real view, attached and then gone — the broker learns the roster emptied only
  // when the socket actually closes, so the context has to close, not just navigate.
  const context = await browser.newContext();
  const page = await context.newPage();
  await page.goto(url);
  await expect(page.locator('.tl-shapes [data-shape-id^="shape:node:"]').first()).toBeVisible({
    timeout: 30_000,
  });
  await context.close();

  // Name the file: openAsGraph resolves from the ACTIVE editor, and this test has
  // just been driving a browser.
  command({ type: 'open', file: 'complex_models/network.py' });

  await expect
    .poll(() => opens().length, {
      timeout: 30_000,
      message:
        'the extension opened nothing after every view closed — it retargeted into an empty roster',
    })
    .toBeGreaterThan(before);

  // Reopened, not minted: a fresh session would orphan this one until its TTL.
  const reopened = opens().at(-1)!;
  const session = JSON.parse(fs.readFileSync(pageFile!, 'utf8')).workspaceSessionId as string;
  expect(reopened).toContain(session);
});
