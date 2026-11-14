import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
const read = name => readFileSync(new URL(`../../.github/workflows/${name}.yml`, import.meta.url), 'utf8');
test('native release uses the same isolated client and CPU torch versions as acceptance checks', () => {
 const release = read('release-plugin'), checks = read('release-checks');
 for (const packageName of ['@openai/codex', '@anthropic-ai/claude-code']) {
  const pattern = new RegExp(packageName + '@([^\\s]+)');
  const isolated = text => text.split('\n').filter(line => line.includes('npm install --prefix plugin/out/audit-clients'));
  assert.ok(isolated(release).length >= 2, 'capture and migration clients are isolated');
  for (const line of isolated(release)) assert.equal(line.match(pattern)?.[1], isolated(checks)[0].match(pattern)?.[1]);
 }
 assert.equal(release.match(/torch==([^\s]+) --index-url/)?.[1], checks.match(/torch==([^\s]+) --index-url/)?.[1]);
 assert.match(release, /npm install --prefix plugin\/out\/audit-clients --no-audit --no-fund/);
 assert.match(release, /plugin\/out\/audit-clients\/node_modules\/\.bin/);
 assert.doesNotMatch(release, /npm install --global/);
});
