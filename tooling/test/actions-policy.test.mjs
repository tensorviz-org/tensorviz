import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';

const directory = new URL('../../.github/workflows/', import.meta.url);

test('PRs, pushes, tags and schedules cannot start runners, including optional diagnostics', () => {
  const files = readdirSync(directory).filter(file => /\.ya?ml$/.test(file));
  assert.ok(files.length > 0);
  for (const file of files) {
    const source = readFileSync(new URL(file, directory), 'utf8');
    // Require a block mapping so shorthand or aliased event declarations fail closed.
    assert.equal((source.match(/^['"]?on['"]?:/gm) ?? []).length, 1, file);
    const block = source.match(/^on:\s*\n((?:[ \t][^\n]*\n|\n|#[^\n]*\n)*)/m)?.[1];
    assert.ok(block, `${file}: use an explicit event mapping`);
    const events = block.split('\n').filter(line => /^ {2}\S/.test(line) && !/^ {2}#/.test(line));
    const allowed = file === 'release-openvsx.yml'
      ? ['  workflow_run:', '  workflow_dispatch:']
      : ['  workflow_dispatch:'];
    assert.deepEqual(events.sort(), allowed.sort(), `${file}: automatic CI is disabled`);
  }
});

test('only a completed main extension release can chain Open VSX publishing', () => {
  const source = readFileSync(new URL('release-openvsx.yml', directory), 'utf8');
  assert.match(source, /workflow_run:\n    workflows: \[Release VSIX\]\n    types: \[completed\]\n    branches: \[main\]/);
  assert.match(source, /github\.event\.workflow_run\.conclusion == 'success'/);
  assert.match(source, /github\.event\.workflow_run\.head_repository\.full_name == github\.repository/);
});
