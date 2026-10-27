import { defineConfig } from '@playwright/test';

const baseURL = process.env.E2E_FRONTEND_URL || 'http://localhost:3000';
const backendURL = process.env.E2E_BACKEND_URL || 'http://127.0.0.1:8765';
const isLocalBaseUrl = /localhost|127\.0\.0\.1/.test(baseURL);
const webServerDisabled = process.env.E2E_DISABLE_WEBSERVER === '1';
const reuseExistingServer =
  process.env.E2E_REUSE_EXISTING_SERVER != null
    ? process.env.E2E_REUSE_EXISTING_SERVER !== '0'
    : !process.env.CI;

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: false,
  workers: 1,
  // A failed regression must stay visible instead of being hidden by a retry.
  retries: 0,
  // Was 12 minutes. Nothing here legitimately takes that long — the substantive
  // specs finish in under ten seconds each — so the only thing a 12-minute budget
  // bought was a hung test costing 12 minutes of a serial run before anyone found
  // out.
  timeout: 3 * 60 * 1000,
  expect: {
    timeout: 60 * 1000,
  },
  reporter: [['list'], ['html', { open: 'never' }]],
  webServer: isLocalBaseUrl && !webServerDisabled
    ? {
        command: 'npm start',
        url: baseURL,
        reuseExistingServer,
        timeout: 120 * 1000,
        // The test UI's environment is pinned here, not in the operator's
        // shell or .env.local: when Playwright launches the dev server, the
        // build boots with Firebase auth disabled and the CRA proxy aiming
        // /api, /health and /local at the local backend, so the plain gate
        // command works without exports. webServer.env merges over
        // process.env (tests/e2e/run-e2e.mjs picks up E2E_BACKEND_URL the
        // same way, so one override retargets probes and proxy together).
        // Externally managed frontends keep their own env, as always.
        env: {
          REACT_APP_AUTH_DISABLED: 'true',
          REACT_APP_DEV_API_PROXY_TARGET: backendURL,
        },
      }
    : undefined,
  use: {
    baseURL,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    actionTimeout: 60 * 1000,
    navigationTimeout: 90 * 1000,
  },
});
