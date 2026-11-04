/**
 * Build a local branch VSIX: ensure dependencies -> build:webview -> compile -> package.
 * `package` stages the backend and verifies the archive, but uses the existing page build.
 * This order matches release-vsix.yml; see docs/local-setup.md and docs/releasing.md.
 *
 * Missing node_modules are installed with lockfile-driven npm ci using the configured
 * registry. Existing dependencies are reused; this script does not update lockfiles.
 *
 * Usage:
 *   npm run vsix --prefix extension
 *   npm run vsix --prefix extension -- --skip-tests
 * --skip-tests skips this script's explicit compile gate, not package's prepublish compile.
 * This script does not run test suites.
 */
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const extensionDir = resolve(here, '..');
const repoRoot = resolve(extensionDir, '..');
const frontendDir = join(repoRoot, 'canvas');

// win32 needs the .cmd shim; build-webview.mjs does the same.
const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const skipCompile = process.argv.includes('--skip-tests');

const die = (message) => {
  console.error(`\nbuild-vsix: ${message}`);
  process.exit(1);
};
const step = (message) => console.log(`\n==> ${message}`);
const run = (cwd, args, env) =>
  execFileSync(npm, args, { cwd, stdio: 'inherit', env: env ? { ...process.env, ...env } : process.env });

if (!existsSync(join(extensionDir, 'package.json'))) die(`no extension package at ${extensionDir}`);
if (!existsSync(join(frontendDir, 'package.json'))) {
  die(
    `no frontend package at ${frontendDir}.\n` +
      'The webview canvas is built from canvas/; without it the .vsix opens an empty panel.',
  );
}

// Identify the source revision and local changes included in the build.
const git = (args, fallback) => {
  try {
    return execFileSync('git', ['-C', repoRoot, ...args], { encoding: 'utf8' }).trim();
  } catch {
    return fallback;
  }
};
step(`Building from ${git(['rev-parse', '--abbrev-ref', 'HEAD'], '(no git)')}`);
console.log(`    HEAD: ${git(['log', '--oneline', '-1'], '(unknown)')}`);
const dirty = git(['status', '--porcelain'], '');
if (dirty) console.log(`    note: ${dirty.split('\n').length} uncommitted change(s) are included in this build`);

// Install missing dependencies from the lockfile.
for (const [label, dir] of [
  ['shared packages', repoRoot],
  ['frontend (canvas)', frontendDir],
  ['extension', extensionDir],
]) {
  if (existsSync(join(dir, 'node_modules'))) {
    console.log(`    ${label} deps: present`);
    continue;
  }
  if (!existsSync(join(dir, 'package-lock.json'))) {
    die(`${dir} has no package-lock.json, so \`npm ci\` cannot run there. Install its deps manually.`);
  }
  step(`Installing ${label} deps (npm ci)`);
  run(dir, ['ci']);
}

// Refresh the page before packaging its staged assets.
step('Building the webview canvas (build:webview)');
run(extensionDir, ['run', 'build:webview']);

if (skipCompile) {
  console.log('\n    skipping the compile gate (--skip-tests)');
} else {
  step('Type + compile gate (compile)');
  run(extensionDir, ['run', 'compile']);
}

// stage:backend -> vsce package -> verify-package. verify-package is the one
// that fails if the backend never got staged or a runtime artifact leaked in.
step('Packaging (package)');
run(extensionDir, ['run', 'package']);

const { name, version } = JSON.parse(readFileSync(join(extensionDir, 'package.json'), 'utf8'));
const vsix = join(extensionDir, `${name}-${version}.vsix`);
if (!existsSync(vsix)) die(`package finished but ${vsix} is not there`);

step('Done.');
console.log(`    VSIX: ${vsix}`);
console.log(`    Size: ${(statSync(vsix).size / 1024 / 1024).toFixed(1)} MB`);
console.log('\nInstall it with:\n');
console.log(`    code --install-extension "${vsix}" --force\n`);
