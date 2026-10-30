import { test, expect } from '@playwright/test';
import { ChildProcess, spawn } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import * as net from 'node:net';
import * as path from 'node:path';
import { mintLocalBrowserSession, createLocalBrowserHostTransport,
  type LocalBrowserHostTransport } from '@tensorviz/graph-host/localBrowserHostTransport';

/**
 * The packaged local browser Canvas needs a host-owned model session. Exercise
 * the real broker/host transport and shipped bytes with a scripted projection;
 * this is renderer/session acceptance, not installed extension capture evidence.
 * A sessionless page must show reconnection, never a pretend live canvas.
 *
 * Prerequisites, both outside git: an interpreter with torch + fastapi + uvicorn
 * (TENSORVIZ_TEST_PYTHON, or BACKEND_PYTHON) and the staged app
 * (`npm run build:webview` in extension/). Without them the spec skips with the
 * reason named, rather than failing a gate on a machine that never built them.
 */

const REPO = path.resolve(__dirname, '..', '..', '..');
const STATIC_DIR = path.join(REPO, 'extension', 'media', 'app');
const PYTHON = process.env.TENSORVIZ_TEST_PYTHON || process.env.BACKEND_PYTHON || '';
const TOKEN = 'e2e-stage-a-token';

/** The licence timer fires at 5 s; 8.5 s is comfortably past it and still short
 *  enough that a hang reads as a hang. */
const LICENCE_GRACE_MS = 8500;

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address() as net.AddressInfo;
      server.close(() => resolve(address.port));
    });
  });
}

async function waitForHealth(origin: string, budgetMs: number): Promise<void> {
  const deadline = Date.now() + budgetMs;
  let why = 'never attempted';
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${origin}/health`);
      if (response.ok) {
        const body = (await response.json()) as { capabilities?: { static?: boolean } };
        if (body.capabilities?.static) return;
        why = 'capabilities.static was false';
      } else {
        why = `status ${response.status}`;
      }
    } catch (error) {
      why = error instanceof Error ? error.message : String(error);
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`the backend never served the app at ${origin}: ${why}`);
}

test.describe('the IDE build served over http', () => {
  let backend: ChildProcess | undefined;
  let host: LocalBrowserHostTransport | undefined;
  let origin = '';
  let pageUrl = '';

  test.skip(!PYTHON, 'TENSORVIZ_TEST_PYTHON / BACKEND_PYTHON is unset');
  test.skip(
    !existsSync(path.join(STATIC_DIR, 'index.html')),
    'extension/media/app is not built — run `npm run build:webview` in extension/',
  );

  test.beforeAll(async () => {
    const port = await freePort();
    origin = `http://127.0.0.1:${port}`;
    backend = spawn(
      PYTHON,
      ['-m', 'uvicorn', 'server_v2:app', '--host', '127.0.0.1', '--port', String(port)],
      {
        cwd: path.join(REPO, 'backend'),
        env: {
          ...process.env,
          AUTH_DISABLED: 'true',
          TENSORVIZ_STATIC_DIR: STATIC_DIR,
          TENSORVIZ_LOCAL_TOKEN: TOKEN,
          KMP_DUPLICATE_LIB_OK: 'TRUE',
          PYTHONUNBUFFERED: '1',
        },
        stdio: ['ignore', 'pipe', 'pipe'],
      },
    );
    backend.stderr?.on('data', (buffer: Buffer) => process.stderr.write(`[ide-backend] ${buffer}`));
    await waitForHealth(origin, 90_000);
    const session = await mintLocalBrowserSession({ backendBaseUrl: origin, launchOrigin: origin, launchToken: TOKEN });
    pageUrl = session.pageUrl;
    host = createLocalBrowserHostTransport({ backendBaseUrl: origin, session });
    const fixture = JSON.parse(readFileSync(path.join(REPO, 'canvas/src/test/fixtures/canvas/graphReview.json'), 'utf8')).after;
    const projection = { ...fixture, traced: true,
      document: JSON.parse(readFileSync(path.join(REPO, 'canvas/src/test/fixtures/canvas/capture-document.json'), 'utf8')) };
    const target = { workspaceFolder: '/static-serve-fixture', file: projection.file, cls: projection.cls,
      documentVersion: 1, sourceDigest: projection.sourceDigest, backendRevision: projection.revision };
    const snapshot = () => host!.publish({ type: 'snapshot', state: { activeTarget: target, projection,
      traced: true, greyed: false, syncState: 'runtime-traced', connectionState: 'connected' } });
    host.onMessage(message => { if (message.type === 'snapshotRequest') snapshot(); });
    await host.connect();
    snapshot();
  });

  test.afterAll(() => {
    host?.dispose();
    backend?.kill('SIGTERM');
  });

  test('tldraw is still mounted well past the licence grace period', async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on('pageerror', (error) => consoleErrors.push(error.message));

    await page.goto(`${origin}/?token=${TOKEN}${new URL(pageUrl).hash}`, { waitUntil: 'domcontentloaded' });

    // The token is a credential; the app lifts it out of the URL on startup so it
    // does not sit in a history entry or a screenshot (task.md §6.5).
    await expect.poll(() => new URL(page.url()).searchParams.get('token')).toBeNull();
    await expect.poll(() => new URL(page.url()).hash).toBe('');
    await expect(page.locator('.tl-container')).toBeVisible();
    await expect(page.locator('.ui2-node-card').first()).toBeVisible();

    await page.waitForTimeout(LICENCE_GRACE_MS);

    const containers = await page.evaluate(
      () => document.querySelectorAll('.tl-container').length,
    );
    expect(
      containers,
      'the authenticated Canvas must remain mounted after its initial render',
    ).toBeGreaterThanOrEqual(1);
    expect(consoleErrors, 'the app threw while rendering the canvas').toEqual([]);
    expect(host!.protocolViolations).toBe(0);
  });

  test('a page without a model session asks to reconnect without mounting a graph', async ({ page }) => {
    await page.goto(origin);
    await expect(page.getByRole('heading', { name: 'Reconnect to your model', exact: true })).toBeVisible();
    await expect(page.locator('.tl-container')).toHaveCount(0);
  });

  test('the API refuses a request that does not carry the launch token', async ({ request }) => {
    // The same origin serves both, which is what makes the canvas same-origin with
    // its API and is why nothing here needed a CORS change (task.md D12).
    expect((await request.get(`${origin}/api/layer-types`)).status()).toBe(401);
    expect(
      (
        await request.get(`${origin}/api/layer-types`, {
          headers: { 'X-tensorViz-Token': TOKEN },
        })
      ).status(),
    ).toBe(200);
  });
});
