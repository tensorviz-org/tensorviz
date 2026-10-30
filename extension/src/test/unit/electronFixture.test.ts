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
  workspaceFolder?: string;
  mocha: { grep: string; invert?: boolean };
}

/** Load the actual runner configuration in its own process, as test-cli does.
 * A second load models a new invocation while the first profile is still dirty. */
function profiles(exitCode = 0): { first: Configuration[]; directories: string[]; workspaces: string[] } {
  const child = spawnSync(process.execPath, ['--input-type=module', '-e', `
    import * as assert from 'node:assert/strict';
    import * as fs from 'node:fs';
    import * as path from 'node:path';
    const directories = [];
    const workspaces = [];
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
        if (config.workspaceFolder) {
          const workspace = config.workspaceFolder.endsWith('.code-workspace')
            ? path.dirname(config.workspaceFolder) : config.workspaceFolder;
          if (config.label === 'suites') {
            const folders = JSON.parse(fs.readFileSync(config.workspaceFolder, 'utf8')).folders;
            assert.deepEqual(folders, ['empty', 'canvas', 'browser/models', 'browser/tensor-expression-project', 'browser/measured-project'].map(path => ({ path })));
            assert.equal(process.env.TENSORVIZ_TEST_WORKSPACE, workspace);
            for (const folder of folders) assert.deepEqual(fs.readdirSync(path.join(workspace, folder.path)), []);
          }
          assert.ok(path.isAbsolute(workspace));
          assert.ok(!fs.existsSync(path.join(workspace, 'test-created-profile')), 'prior writes must not leak between invocations');
          fs.writeFileSync(path.join(workspace, 'test-created-profile'), 'owned fixture output');
          workspaces.push(workspace);
        }
      }
      return configs;
    }
    const first = await load('first');
    await load('second');
    console.log(JSON.stringify({ first, directories, workspaces }));
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
      const { first, directories, workspaces } = profiles(exitCode);
      assert.deepEqual(first.map(config => config.label), ['runtimeFlow', 'suites', 'coldDiscovery', 'zeroCeremony']);
      assert.equal(directories.length, 8, 'all four configurations on both invocations');
      assert.equal(new Set(directories).size, 8);
      assert.equal(workspaces.length, 6);
      assert.equal(new Set(workspaces).size, 6);
      for (const directory of [...directories, ...workspaces]) {
        assert.equal(fs.existsSync(directory), false, 'runner exit must clean only its owned profile');
      }
    }
  });

  test('first-use runs before backend owners and the combined selection still includes every requested suite', () => {
    const { first } = profiles();
    const suites = first.find(config => config.label === 'suites')!;
    const installer = first.find(config => config.label === 'runtimeFlow')!;
    assert.equal(installer.files, 'out/test/electron/runtimeFlow.test.js');
    assert.notEqual(installer.workspaceFolder, suites.workspaceFolder, 'installer-owned state cannot reach the general suites');
    assert.equal(suites.files, 'out/test/electron/**/!(runtimeFlow|coldDiscovery|zeroCeremony).test.js', 'dedicated hosts must not execute twice');
    const coldDiscovery = first.find(config => config.label === 'coldDiscovery')!;
    assert.equal(coldDiscovery.files, 'out/test/electron/coldDiscovery.test.js', 'cold discovery keeps its own host');
    const selected = new RegExp(suites.mocha.grep);
    assert.ok(!suites.mocha.invert);
    for (const name of SELECTED.slice(1, -1).split('|')) {
      if (name === 'runtimeFlow') assert.ok(!installer.mocha.grep, 'the selected installer must run in its own host');
      else assert.ok(selected.test(`${name} behavior`), `${name} must not be silently excluded`);
    }
    assert.ok(!selected.test('zeroCeremony behavior'), 'fixture workspace has its own host');
  });
});
