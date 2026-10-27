import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { defineConfig } from '@vscode/test-cli';

// Interpreter-dependent suites require TENSORVIZ_TEST_PYTHON. Select a backend-free
// run explicitly with TENSORVIZ_TEST_NO_BACKEND=1.
const backendFree = process.env.TENSORVIZ_TEST_NO_BACKEND === '1';

// Explicit list of suites requiring Torch, FastAPI and Uvicorn.
const NEEDS_BACKEND = [
  'coldDiscovery',
  'openCanvas',
  'graphReview',
  'localBrowserGraph',
  'runtimeFlow',
  'zeroCeremony',
];

const deselected = backendFree ? NEEDS_BACKEND : [];
// Select one suite locally: TENSORVIZ_TEST_ONLY=localBrowserGraph npm run test:electron
const onlySuite = process.env.TENSORVIZ_TEST_ONLY;

// Optional compatibility run against a built fork. Obtain its executable path with
// ide/scripts/smoke.sh --print-binary <bundle>; test-cli uses useInstallation.fromPath.
const forkBinary = process.env.TENSORVIZ_IDE_BINARY;
const editorBinary = forkBinary || process.env.TENSORVIZ_TEST_EDITOR_BINARY;

// Preseed from TENSORVIZ_TEST_PYTHON to avoid first-use prompts. runtimeFlow disables
// this for its consent/provisioning cases; do not treat preseeded runs as first-install proof.
process.env.TENSORVIZ_TEST_PRESEED_MANAGED = process.env.TENSORVIZ_TEST_PRESEED_MANAGED ?? '1';

// Use an existing installation or download stable VS Code, with identical launch settings.
const shared = {
  ...(editorBinary ? { useInstallation: { fromPath: editorBinary } } : { version: 'stable' }),
  launchArgs: ['--disable-extensions'],
  mocha: {
    ui: 'tdd',
    timeout: 60000,
  },
};

// test-electron otherwise reuses .vscode-test/user-data, including runtime stamps,
// environment choices and settings from previous builds. Each host owns a fresh
// profile; leave the downloaded Code and companion-extension caches reusable.
function isolatedLaunchArgs(label) {
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), `tensorviz-electron-${label}-`));
  process.once('exit', () => fs.rmSync(profile, { recursive: true, force: true }));
  return [...shared.launchArgs, `--user-data-dir=${profile}`];
}

// zeroCeremony needs the fixture folder open to test workspace-wide model discovery.
const ZERO_CEREMONY_FIXTURE =
  process.env.TENSORVIZ_ZERO_CEREMONY_FIXTURE ??
  fileURLToPath(new URL('../backend/tests/fixtures/ide_repo', import.meta.url));

// Electron suites use a real host; unit suites run separately with the vscode stub.
export default defineConfig([
  {
    ...shared,
    label: 'suites',
    launchArgs: isolatedLaunchArgs('suites'),
    // First-use provisioning must precede any suite that starts a backend from
    // the preseeded runtime. test-cli deduplicates the overlap with the glob.
    files: ['out/test/electron/runtimeFlow.test.js', 'out/test/electron/**/!(coldDiscovery).test.js'],
    mocha: {
      ...shared.mocha,
      // Anchor the suite name: a test title can mention another suite. Exclude
      // zeroCeremony here because its separate configuration supplies the fixture.
      ...(onlySuite
        ? { grep: `^${onlySuite} ` }
        : { grep: `^(${[...new Set([...deselected, 'zeroCeremony'])].join('|')}) `, invert: true }),
    },
  },
  {
    ...shared,
    label: 'coldDiscovery',
    launchArgs: isolatedLaunchArgs('coldDiscovery'),
    files: 'out/test/electron/coldDiscovery.test.js',
    mocha: {
      ...shared.mocha,
      // A dedicated process is essential: another suite's warm backend masks this bug.
      ...(backendFree || (onlySuite && onlySuite !== 'coldDiscovery') ? { grep: '^notSelected ' } : {}),
    },
  },
  {
    ...shared,
    label: 'zeroCeremony',
    launchArgs: isolatedLaunchArgs('zeroCeremony'),
    files: 'out/test/electron/zeroCeremony.test.js',
    workspaceFolder: ZERO_CEREMONY_FIXTURE,
    mocha: {
      ...shared.mocha,
      // Selecting another suite also deselects this separate configuration.
      ...(onlySuite && onlySuite !== 'zeroCeremony' ? { grep: `^${onlySuite} ` } : {}),
    },
  },
]);
