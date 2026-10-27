#!/usr/bin/env node
/**
 * Copy `backend/` into the extension so the .vsix carries its own backend.
 *
 * Installed extensions resolve this staged copy without requiring a repository checkout.
 *
 * What is left out and why:
 *   tests/       ~ test code and fixtures that can shadow backend imports.
 *   evals/       ~ development evaluation datasets.
 *   scripts/     ~ developer entry points, some of which import excluded evals.
 *   reports/     ~ local developer audit and validation output.
 *   .runtime/    ~ a provisioned virtualenv from someone's machine. Absolute paths inside.
 *   __pycache__/ ~ bytecode for whichever interpreter happened to build the package.
 *   .env*        ~ local credentials and machine-specific configuration.
 *
 * `backend-identity.json` is written last: a digest of the staged files, which the
 * execution cache uses so a result is never reused across backend builds.
 *
 * Print the staged size and require the built page, runtime pins and requirements.
 * `package` does not rebuild media/app; run `build:webview` after frontend changes.
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { stageBackend, writeBackendIdentity } from '../../tooling/runtime/stage-backend.mjs';
import { stageRuntimeAssets } from '../../tooling/runtime/assets.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const extensionDir = resolve(here, '..');
const repoRoot = resolve(extensionDir, '..');
const dest = join(extensionDir, 'backend');

stageBackend(repoRoot, dest, 'extension');
// Content identity of the staged backend, read by the execution cache key.
writeBackendIdentity(dest);
stageRuntimeAssets(repoRoot, extensionDir);

function measure(dir) {
  let bytes = 0;
  let files = 0;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      const inner = measure(full);
      bytes += inner.bytes;
      files += inner.files;
    } else if (entry.isFile()) {
      bytes += statSync(full).size;
      files += 1;
    }
  }
  return { bytes, files };
}

const { bytes, files } = measure(dest);
console.log(
  `stage-backend: ${files} files, ${(bytes / 1024 / 1024).toFixed(1)} MiB -> ${relative(extensionDir, dest)}`,
);

if (!existsSync(join(extensionDir, 'media', 'app', 'index.html'))) {
  console.error(
    `stage-backend: media/app/index.html is missing — the .vsix would ship no canvas.\n` +
      `Run \`npm run build:webview\` first (and always after changing canvas/src).`,
  );
  process.exit(1);
}

// The optional managed-runtime fallback needs pinned downloads in the installed VSIX.
const manifestPath = join(extensionDir, 'runtime-manifest.json');
try {
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  const platforms = manifest.platforms ?? {};
  for (const required of ['darwin-arm64', 'linux-x64', 'win32-x64']) {
    const pin = platforms[required];
    if (!pin || typeof pin.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(pin.sha256)) {
      throw new Error(`platforms.${required} is missing or its sha256 is not a pinned hash`);
    }
  }
} catch (err) {
  console.error(`stage-backend: runtime-manifest.json is absent or invalid: ${err}`);
  process.exit(1);
}

// Managed-runtime provisioning reads these requirements from the staged backend.
for (const requirements of ['requirements.txt', 'requirements-cpu.txt']) {
  if (!existsSync(join(dest, requirements))) {
    console.error(
      `stage-backend: backend/${requirements} is missing — the managed runtime cannot provision.`,
    );
    process.exit(1);
  }
}
