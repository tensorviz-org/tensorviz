import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const read = file => readFileSync(new URL(`../../${file}`, import.meta.url), 'utf8');

test('current standalone install surfaces include native platforms and capture limits', () => {
  for (const file of ['README.md', 'docs/install.md', 'docs/agent-plugin.md',
    'docs/quickstart.md', 'docs/environment-recovery.md',
    'docs/reference/agent-plugin.md', 'docs/.vitepress/theme/InstallOptions.vue']) {
    const source = read(file).replaceAll('&gt;', '>');
    for (const requirement of [/(?:Apple Silicon macOS|macOS ARM64\/x64)/, /Linux x64/, /glibc >=2\.35/,
      /CPython 3\.12/, /CPU capture/]) {
      assert.match(source, requirement, `${file}: ${requirement}`);
    }
    assert.doesNotMatch(source, /Darwin-arm64-only|remain Mac-only until/);
  }
});

test('install choices distinguish standalone exclusions from editor platforms', () => {
  for (const file of ['docs/install.md', 'docs/.vitepress/theme/InstallOptions.vue']) {
    const source = read(file);
    for (const platform of [/macOS ARM64\/x64/, /Linux x64/, /Linux ARM64/, /Windows x64\/ARM64/]) assert.match(source, platform);
    assert.match(source, /Windows support is through native Codex entries/);
    assert.match(source, /Windows Claude Code\/Cursor and musl are outside the published acceptance scope/);
    assert.match(source, /macOS, Windows, (?:and|or) Linux/);
    assert.doesNotMatch(source, /releases\/download\/v[\d.]+|open-vsx\.org\/api\/tensorviz/);
  }
});


test('all current installation commands and links use the canonical public repository', () => {
  for (const file of ['README.md', 'docs/install.md', 'docs/agent-plugin.md', 'docs/.vitepress/theme/InstallOptions.vue']) {
    const source = read(file);
    assert.doesNotMatch(source, /(?:codex|claude) plugin marketplace add tensorviz-org\/tensorviz-plugins/);
    assert.match(source, /codex plugin marketplace add tensorviz-org\/tensorviz(?:\n|\\n)/);
    assert.doesNotMatch(source, /github\.com\/tensorviz-org\/tensorviz-plugins(?:#|\/blob\/main|[)"'])/);
  }
});
