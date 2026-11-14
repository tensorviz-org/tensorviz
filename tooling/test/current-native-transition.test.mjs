import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

test('0.1.44 publication requires native upgrade and rollback on compatible old-runtime hosts', () => {
  const workflow = readFileSync(new URL('../../.github/workflows/release-plugin.yml', import.meta.url), 'utf8');
  const migration = workflow.split('\n  current_migration:\n')[1]?.split('\n  publish:\n')[0];
  assert.ok(migration);
  assert.match(migration, /needs: \[configuration, assemble\]/);
  assert.match(migration, /runner: ubuntu-22\.04/);
  assert.match(migration, /runner: macos-15/);
  assert.match(migration, /ref: de268bb05a44d21fba51dc53b2d711ccee8a2738/);
  assert.match(migration, /TENSORVIZ_MIGRATION_TARGET_VERSION: '0\.1\.44'/);
  assert.match(migration, /node --test plugin\/test\/migration\/native-version-transition\.cjs/);
  assert.doesNotMatch(migration, /continue-on-error|PLUGIN_PUBLISH_TOKEN/);
  const publish = workflow.split('\n  publish:\n')[1];
  assert.match(publish, /needs: \[configuration, assemble, current_migration\]/);
  assert.match(publish, /needs\.current_migration\.result == 'success'/);
  assert.match(publish, /needs\.configuration\.outputs\.publish == 'true'/);
});
