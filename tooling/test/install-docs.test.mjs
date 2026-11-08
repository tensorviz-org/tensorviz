import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const read = file => readFileSync(new URL(`../../${file}`, import.meta.url), 'utf8');

test('current standalone install surfaces include both published platforms and capture limits', () => {
  for (const file of ['README.md', 'docs/install.md', 'docs/agent-plugin.md',
    'docs/quickstart.md', 'docs/environment-recovery.md',
    'docs/reference/agent-plugin.md', 'docs/.vitepress/theme/InstallOptions.vue']) {
    const source = read(file).replaceAll('&gt;', '>');
    for (const requirement of [/Apple Silicon macOS/, /Linux x64/, /glibc >=2\.35/,
      /CPython 3\.12/, /CPU capture/]) {
      assert.match(source, requirement, `${file}: ${requirement}`);
    }
    assert.doesNotMatch(source, /Darwin-arm64-only|remain Mac-only until/);
  }
});

test('install choices distinguish standalone exclusions from editor platforms', () => {
  for (const file of ['docs/install.md', 'docs/.vitepress/theme/InstallOptions.vue']) {
    const source = read(file);
    assert.match(source, /Windows, Intel Mac, Linux ARM64, and musl are unsupported by the published standalone plugin/);
    assert.match(source, /macOS, Windows, (?:and|or) Linux/);
    assert.doesNotMatch(source, /releases\/download\/v[\d.]+|open-vsx\.org\/api\/tensorviz/);
  }
});
