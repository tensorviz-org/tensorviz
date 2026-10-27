#!/usr/bin/env node

// 8765, not 8000: the backend binds 8765 (README explains why — editor plugins have
// been seen squatting on 8000). Defaulting to 8000 made this precheck fail against a
// perfectly healthy backend, which is the whole reason `npm run test:ui2:gate` did not
// run out of the box.
const backendUrl = process.env.E2E_BACKEND_URL || 'http://localhost:8765';
const requireHealth = process.env.E2E_REQUIRE_BACKEND_HEALTH !== '0';
const timeoutMs = Number(process.env.E2E_BACKEND_HEALTH_TIMEOUT_MS || 7000);

if (!requireHealth) {
  console.log(
    `[ui2-precheck] Skipping backend health check (E2E_REQUIRE_BACKEND_HEALTH=0). backend=${backendUrl}`
  );
  process.exit(0);
}

const controller = new AbortController();
const timeout = setTimeout(() => controller.abort(), timeoutMs);

try {
  const url = `${backendUrl.replace(/\/$/, '')}/health`;
  const response = await fetch(url, {
    method: 'GET',
    signal: controller.signal,
    headers: { Accept: 'application/json' },
  });

  if (!response.ok) {
    console.error(`[ui2-precheck] Backend health failed: status=${response.status} url=${url}`);
    process.exit(1);
  }

  console.log(`[ui2-precheck] Backend healthy: status=${response.status} url=${url}`);
  process.exit(0);
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`[ui2-precheck] Backend health request failed: ${message}`);
  process.exit(1);
} finally {
  clearTimeout(timeout);
}
