const { test } = require('node:test');
const assert = require('node:assert/strict');

test('native target filenames match each OS and CPython ABI', async () => {
  const { nativePlatform, nativePlatforms } = await import('../../scripts/native-platforms.mjs');
  assert.equal(Object.keys(nativePlatforms).length, 6);
  for (const [key, target] of Object.entries(nativePlatforms)) {
    const selected = nativePlatform(target.os, target.arch);
    assert.equal(selected.key, key);
    assert.equal(selected.host, target.os === 'win32' ? 'tensorviz-host.exe' : 'tensorviz-host');
    assert.equal(selected.backend, `_tensorviz_backend.${target.abi}.${target.os === 'win32' ? 'pyd' : 'so'}`);
  }
  assert.throws(() => nativePlatform('linux', 'ia32'), /Unsupported/);
  assert.throws(() => nativePlatform('freebsd', 'x64'), /Unsupported/);
});
