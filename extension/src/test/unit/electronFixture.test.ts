import * as assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';

const EXTENSION = path.resolve(__dirname, '..', '..', '..');
const SELECTED = '(activation|openCanvas|openAsGraph|localBrowserGraph|runtimeFlow|traceOnSave|applyEdit|workspacePalette)';

interface Configuration {
  label: string;
  files: string | string[];
  launchArgs: string[];
  mocha: { grep: string; invert?: boolean };
}

/** Load the actual runner configuration in its own process, as test-cli does.
 * A second load models a new invocation while the first profile is still dirty. */
function profiles(exitCode = 0): { first: Configuration[]; directories: string[] } {
  const child = spawnSync(process.execPath, ['--input-type=module', '-e', `
    import * as assert from 'node:assert/strict';
    import * as fs from 'node:fs';
    import * as path from 'node:path';
    const directories = [];
    async function load(run) {
      const configs = (await import('./.vscode-test.mjs?' + run)).default;
      for (const config of configs) {
        const arg = config.launchArgs.find(arg => arg.startsWith('--user-data-dir='));
        assert.ok(arg, config.label + ' must own its test profile');
        const directory = arg.slice('--user-data-dir='.length);
        assert.ok(path.isAbsolute(directory));
        assert.deepEqual(fs.readdirSync(directory), [], 'no settings, consent or runtime from another host');
        // Both a prior run and another configuration must be unable to reuse this stamp.
        fs.writeFileSync(path.join(directory, 'stale-runtime-stamp'), 'old requirements');
        directories.push(directory);
      }
      return configs;
    }
    const first = await load('first');
    await load('second');
    console.log(JSON.stringify({ first, directories }));
    process.exitCode = ${exitCode};
  `], {
    cwd: EXTENSION,
    env: { ...process.env, TENSORVIZ_TEST_ONLY: SELECTED },
    encoding: 'utf8',
    timeout: 10_000,
  });
  assert.equal(child.status, exitCode, child.stderr || String(child.error));
  return JSON.parse(child.stdout);
}

suite('Electron fixture isolation', () => {
  test('separate hosts and repeated runs cannot inherit stale runtime stamps; owned profiles are cleaned up', () => {
    // Cleanup must also preserve a failing runner's status, not turn it green.
    for (const exitCode of [0, 1]) {
      const { first, directories } = profiles(exitCode);
      assert.deepEqual(first.map(config => config.label), ['suites', 'coldDiscovery', 'zeroCeremony']);
      assert.equal(directories.length, 6, 'all three configurations on both invocations');
      assert.equal(new Set(directories).size, 6);
      for (const directory of directories) {
        assert.equal(fs.existsSync(directory), false, 'runner exit must clean only its owned profile');
      }
    }
  });

  test('first-use runs before backend owners and the combined selection still includes every requested suite', () => {
    const { first } = profiles();
    const suites = first.find(config => config.label === 'suites')!;
    assert.ok(Array.isArray(suites.files));
    assert.equal(suites.files[0], 'out/test/electron/runtimeFlow.test.js');
    assert.ok(suites.files.includes('out/test/electron/**/!(coldDiscovery).test.js'), 'keep other coverage without warming the cold-discovery host');
    const coldDiscovery = first.find(config => config.label === 'coldDiscovery')!;
    assert.equal(coldDiscovery.files, 'out/test/electron/coldDiscovery.test.js', 'cold discovery keeps its own host');
    const selected = new RegExp(suites.mocha.grep);
    assert.ok(!suites.mocha.invert);
    for (const name of SELECTED.slice(1, -1).split('|')) {
      assert.ok(selected.test(`${name} behavior`), `${name} must not be silently excluded`);
    }
    assert.ok(!selected.test('zeroCeremony behavior'), 'fixture workspace has its own host');
  });
});
