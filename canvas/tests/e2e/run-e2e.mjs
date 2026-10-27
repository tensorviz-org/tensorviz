#!/usr/bin/env node

/**
 * Run Playwright over an explicit spec list and fail if NOTHING actually ran.
 *
 * This replaces run-ui2-smoke.mjs, which had three separate ways of reporting
 * success without testing anything:
 *
 *  1. It always ran ui-v2-visual.spec.ts + history.spec.ts, both entirely
 *     `test.describe.skip`. Playwright collects 8, skips 8, and exits 0.
 *  2. Its frontend readiness probe fetched /agent-demo?ui=v2 — a route that is
 *     now only a redirect — and a CRA dev server answers 200 for *any* path, so
 *     the probe could not fail. It "passed" for a page that no longer exists.
 *  3. Nothing checked the backend the specs need, so on a machine without one
 *     every backend-dependent spec self-skipped and the run still went green.
 *
 * So: the spec list is an argument (no hidden default set of dead files), the
 * readiness probes assert something that can actually be false, and the exit
 * code is derived from the number of tests that RAN, not just from Playwright's.
 *
 * Usage:  node tests/e2e/run-e2e.mjs <spec> [<spec> ...]
 * Env:    E2E_FRONTEND_URL (default http://localhost:3000)
 *         E2E_BACKEND_URL  (default http://localhost:8765)
 *         E2E_REQUIRE_BACKEND_HEALTH=0  skip the backend probe (specs may then
 *           self-skip — the zero-executed-tests guard still fails the run)
 *         E2E_DISABLE_WEBSERVER=1  a frontend is managed externally
 */

import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const specs = process.argv.slice(2);
if (specs.length === 0) {
  console.error('[e2e] No spec files given. Usage: node tests/e2e/run-e2e.mjs <spec> [<spec> ...]');
  process.exit(2);
}

const frontendUrl = (process.env.E2E_FRONTEND_URL || 'http://localhost:3000').replace(/\/$/, '');
const backendUrl = (process.env.E2E_BACKEND_URL || 'http://localhost:8765').replace(/\/$/, '');
const probeTimeoutMs = Number(process.env.E2E_FRONTEND_PROBE_TIMEOUT_MS || 8000);
const requireBackend = process.env.E2E_REQUIRE_BACKEND_HEALTH !== '0';

async function getText(url, timeoutMs) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { signal: controller.signal });
    return { ok: response.ok, status: response.status, body: await response.text() };
  } catch (error) {
    return { ok: false, status: 0, body: '', error: error instanceof Error ? error.message : String(error) };
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * Is the app itself being served? A CRA dev server returns index.html with 200
 * for every path, so "200" proves nothing. The app shell does: the mount point
 * and a bundle to fill it.
 */
async function isFrontendServingTheApp(url, timeoutMs) {
  const result = await getText(url, timeoutMs);
  if (!result.ok) return { serving: false, why: result.error || `status ${result.status}` };
  const hasMount = /id="root"/.test(result.body);
  const hasBundle = /static\/js\//.test(result.body) || /<script[^>]+src=/.test(result.body);
  if (!hasMount || !hasBundle) {
    return {
      serving: false,
      why: `200 but not the app shell (root mount: ${hasMount}, script bundle: ${hasBundle})`,
    };
  }
  return { serving: true };
}

/**
 * The backend probe hits the endpoint the builder cannot render without —
 * /api/layer-types populates the palette — rather than /health, so a backend
 * that is up but not serving the model plane fails here instead of at test time.
 */
async function isBackendServingTheApi(url, timeoutMs) {
  const result = await getText(`${url}/api/layer-types`, timeoutMs);
  if (!result.ok) return { serving: false, why: result.error || `status ${result.status}` };
  try {
    const payload = JSON.parse(result.body);
    const count = Array.isArray(payload)
      ? payload.length
      : Object.values(payload).reduce((n, v) => n + (Array.isArray(v) ? v.length : 0), 0);
    if (count === 0) return { serving: false, why: '/api/layer-types returned no layer types' };
    return { serving: true, count };
  } catch {
    return { serving: false, why: '/api/layer-types did not return JSON' };
  }
}

if (requireBackend) {
  const backend = await isBackendServingTheApi(backendUrl, probeTimeoutMs);
  if (!backend.serving) {
    console.error(
      `[e2e] Backend not usable at ${backendUrl}: ${backend.why}\n` +
        '[e2e] Backend-dependent specs would self-skip and the run would look green. Refusing.\n' +
        '[e2e] Start it, or set E2E_REQUIRE_BACKEND_HEALTH=0 to run only the specs that need no backend.'
    );
    process.exit(1);
  }
  console.log(`[e2e] Backend serving ${backend.count} layer types at ${backendUrl}`);
}

const env = { ...process.env };
if (env.E2E_DISABLE_WEBSERVER === '1') {
  console.log('[e2e] E2E_DISABLE_WEBSERVER=1: using an externally managed frontend.');
} else {
  const frontend = await isFrontendServingTheApp(frontendUrl, probeTimeoutMs);
  if (frontend.serving) {
    env.E2E_DISABLE_WEBSERVER = '1';
    console.log(`[e2e] Reusing the frontend already serving at ${frontendUrl}.`);
  } else {
    console.log(`[e2e] No app at ${frontendUrl} (${frontend.why}); Playwright will start one.`);
    if (!env.E2E_REUSE_EXISTING_SERVER && process.env.CI) env.E2E_REUSE_EXISTING_SERVER = '0';
  }
}

const reportDir = mkdtempSync(join(tmpdir(), 'e2e-report-'));
const reportPath = join(reportDir, 'results.json');
env.PLAYWRIGHT_JSON_OUTPUT_NAME = reportPath;

const npxCmd = process.platform === 'win32' ? 'npx.cmd' : 'npx';
const child = spawn(npxCmd, ['playwright', 'test', '--reporter=list,json', ...specs], {
  stdio: 'inherit',
  env,
});

child.on('error', (error) => {
  console.error(`[e2e] Failed to launch Playwright: ${error.message}`);
  rmSync(reportDir, { recursive: true, force: true });
  process.exit(1);
});

child.on('exit', (code) => {
  let stats = null;
  try {
    stats = JSON.parse(readFileSync(reportPath, 'utf8')).stats;
  } catch (error) {
    console.error(
      `[e2e] Could not read the Playwright JSON report at ${reportPath}: ` +
        `${error instanceof Error ? error.message : String(error)}`
    );
  }
  rmSync(reportDir, { recursive: true, force: true });

  if (!stats) {
    console.error('[e2e] No test report to check — treating the run as failed.');
    process.exit(1);
  }

  const executed = (stats.expected || 0) + (stats.unexpected || 0) + (stats.flaky || 0);
  console.log(
    `[e2e] executed=${executed} (passed ${stats.expected || 0}, failed ${stats.unexpected || 0}, ` +
      `flaky ${stats.flaky || 0}), skipped=${stats.skipped || 0}`
  );

  if (executed === 0) {
    console.error(
      '[e2e] ZERO tests executed. A run that tests nothing is a failure, whatever ' +
        'Playwright’s exit code says — that is exactly how five npm scripts reported ' +
        'green for a month. Check the spec list and whether the specs self-skipped.'
    );
    process.exit(1);
  }

  if (process.env.E2E_FAIL_ON_SKIP === '1' && stats.skipped > 0) {
    console.error('[e2e] Required feature checks were skipped.');
    process.exit(1);
  }
  process.exit(code ?? 1);
});
