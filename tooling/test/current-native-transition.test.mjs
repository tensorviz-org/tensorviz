import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const workflow = readFileSync(new URL('../../.github/workflows/release-plugin.yml', import.meta.url), 'utf8');
test('0.1.48 requires pinned 0.1.47 upgrade and rollback without a version bypass', () => {
  const migration = workflow.split('\n  current_migration:\n')[1]?.split('\n  publish:\n')[0];
  assert.ok(migration);
  assert.match(migration, /needs: \[configuration, assemble\]/);
  assert.match(migration, /runner: ubuntu-22\.04/);
  assert.match(migration, /runner: macos-15/);
  assert.match(migration, /ref: 9c20428f515dda38b25d1b9436b7681969e41472/);
  assert.match(migration, /TENSORVIZ_MIGRATION_TARGET_VERSION: '0\.1\.48'/);
  assert.match(migration, /node --test plugin\/test\/migration\/native-version-transition\.cjs/);
  assert.doesNotMatch(migration, /continue-on-error|PLUGIN_PUBLISH_TOKEN|outputs\.version ==/);
  const publish = workflow.split('\n  publish:\n')[1];
  assert.match(publish, /needs: \[configuration, assemble, current_migration\]/);
  assert.match(publish, /if: always\(\) && needs\.assemble\.result == 'success' && needs\.current_migration\.result == 'success' && needs\.configuration\.outputs\.publish == 'true'/);
  assert.match(publish, /pattern: plugin-migration-current-\*/);
  assert.doesNotMatch(publish, /outputs\.version !=|continue-on-error/);
});
test('all six actual native archives and accepted Windows bootstrap provenance are mandatory', () => {
  const build = workflow.split('\n  build:\n')[1].split('\n  assemble:\n')[0];
  const assemble = workflow.split('\n  assemble:\n')[1].split('\n  current_migration:\n')[0];
  for (const target of ['darwin-arm64','darwin-x64','linux-x64','linux-arm64','win32-x64','win32-arm64']) {
    assert.ok(build.includes(`platform: ${target}`));
    assert.ok(assemble.includes(`$TENSORVIZ_RELEASE_VERSION-${target}.tar.gz`));
  }
  assert.match(build, /windows-production-mcp\.mjs/);
  assert.match(build, /unix-production-mcp\.mjs/);
  assert.match(build, /test\/native\/no-torch\.cjs/);
  assert.ok(build.indexOf('windows-production-mcp.mjs') < build.indexOf('attest-windows'));
  assert.match(build, /tensorviz-bootstrap-\*\.exe\.json/);
  assert.match(build, /tensorviz-bootstrap-\*\.NOTICES\.md/);
  assert.doesNotMatch(build, /continue-on-error|PLUGIN_PUBLISH_TOKEN/);
  assert.match(workflow, /version !== '0\.1\.47'/);
});

