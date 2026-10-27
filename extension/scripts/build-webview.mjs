#!/usr/bin/env node
/**
 * Build the React app for the IDE and stage it inside the extension.
 *
 * Stage A (task.md D2): the backend serves this build over http:// and the
 * webview shell iframes it, because tldraw 5 unmounts its editor on a
 * `vscode-webview:` origin without a licence key. So the .vsix has to carry the
 * built app, and this is what puts it there.
 *
 *   REACT_APP_IDE=true            -> src/index.tsx loads src/ide/main instead of App
 *   REACT_APP_AUTH_DISABLED=true  -> firebase.ts:16-18 short-circuits; no popup
 *                                    sign-in, which cannot run in a webview anyway
 *   PUBLIC_URL=.                  -> asset URLs relative to the document
 *   CI=true                       -> CRA's interactive prompts are off — and
 *                                    lint warnings become fatal, which is the
 *                                    point: a releasable webview builds clean.
 *
 * The output is extension/media/app/, which is gitignored and included in the
 * .vsix (see .vscodeignore).
 */
import { execFileSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, rmSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const extensionDir = resolve(here, '..');
const appDir = resolve(extensionDir, '..', 'canvas');
const buildDir = join(appDir, 'build');
const target = join(extensionDir, 'media', 'app');

if (!existsSync(join(appDir, 'package.json'))) {
  console.error(`build-webview: no frontend package at ${appDir}`);
  process.exit(1);
}
if (!existsSync(join(appDir, 'node_modules'))) {
  console.error(
    `build-webview: ${join(appDir, 'node_modules')} is missing.\n` +
      `Run \`npm install\` in ${appDir} first — this script builds the app, it does not install it.`,
  );
  process.exit(1);
}

const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
console.log(`build-webview: building ${appDir} for the IDE…`);
execFileSync(npm, ['run', 'build'], {
  cwd: appDir,
  stdio: 'inherit',
  env: {
    ...process.env,
    CI: 'true',
    PUBLIC_URL: '.',
    REACT_APP_IDE: 'true',
    REACT_APP_AUTH_DISABLED: 'true',
  },
});

if (!existsSync(join(buildDir, 'index.html'))) {
  console.error(`build-webview: ${buildDir}/index.html was not produced`);
  process.exit(1);
}

rmSync(target, { recursive: true, force: true });
mkdirSync(dirname(target), { recursive: true });
cpSync(buildDir, target, { recursive: true });
console.log(`build-webview: staged ${target}`);

// The graph-file viewer is an independent static entry, bundled for local use
// and also available as canvas/viewer-dist for static hosting.
execFileSync(npm, ['run', 'build:viewer'], { cwd: appDir, stdio: 'inherit', env: process.env });
cpSync(join(appDir, 'viewer-dist'), join(target, 'viewer'), { recursive: true });
console.log('build-webview: staged the standalone graph-file viewer');
