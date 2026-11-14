import * as assert from 'node:assert/strict';
import * as path from 'node:path';

import {
  loadRuntimeManifest,
  runtimePlatformKey,
  selectRuntimeAsset,
} from '@tensorviz/graph-host/runtimeManifest';

/** The extension directory in the checkout (the compiled test lives in out/test/unit). */
const EXTENSION = path.join(__dirname, '..', '..', '..');

suite('runtimeManifest', () => {
  test('loads, validates, and selects exactly one pinned asset per supported platform', () => {
    const manifest = loadRuntimeManifest(EXTENSION);

    assert.equal(runtimePlatformKey('darwin', 'arm64'), 'darwin-arm64');
    assert.equal(runtimePlatformKey('darwin', 'x64'), 'darwin-x64');
    assert.equal(runtimePlatformKey('linux', 'x64'), 'linux-x64');
    assert.equal(runtimePlatformKey('win32', 'x64'), 'win32-x64');
    assert.equal(runtimePlatformKey('linux', 'arm64'), 'linux-arm64');
    assert.equal(runtimePlatformKey('win32', 'arm64'), 'win32-arm64');

    for (const key of ['darwin-arm64', 'linux-x64', 'win32-x64', 'linux-arm64', 'win32-arm64', 'darwin-x64']) {
      const selection = selectRuntimeAsset(manifest, key.split('-')[0] as NodeJS.Platform, key.split('-')[1]);
      assert.equal(selection.ok, true, key);
      if (!selection.ok) return;
      assert.match(selection.asset, /^cpython-3\.12\.14\+20260901-.*-install_only_stripped\.tar\.gz$/);
      assert.match(selection.sha256, /^[0-9a-f]{64}$/);
      assert.ok(selection.url.startsWith('https://'));
      assert.ok(!selection.url.includes('+'), 'the + in the version must be %2B on the wire');
      assert.ok(selection.url.includes('%2B'));
    }

    // The fork's layout rule, restated as data: python.exe is at the root on Windows and
    // under bin/ everywhere else.
    assert.equal(manifest.platforms['win32-x64'].interpreter, 'python.exe');
    assert.equal(manifest.platforms['darwin-arm64'].interpreter, 'bin/python3');
    assert.equal(manifest.platforms['linux-x64'].interpreter, 'bin/python3');
  });

  test('refuses unsupported platforms with a clear, honest message', () => {
    const manifest = loadRuntimeManifest(EXTENSION);
    assert.equal(runtimePlatformKey('freebsd', 'x64'), undefined);

    const selection = selectRuntimeAsset(manifest, 'freebsd', 'x64');
    assert.equal(selection.ok, false);
    if (selection.ok) return;
    assert.match(selection.message, /not available for freebsd-x64/);
    assert.match(selection.message, /darwin-arm64, linux-x64, win32-x64/);
  });

  test('the asset URL is byte-identical to the pinned standalone release', () => {
    // fetch-runtime.sh: url="$RUNTIME_BASE/$RUNTIME_RELEASE/<asset with + -> %2B>".
    const manifest = loadRuntimeManifest(EXTENSION);
    const selection = selectRuntimeAsset(manifest, 'darwin', 'arm64');
    assert.equal(selection.ok, true);
    if (!selection.ok) return;
    assert.equal(
      selection.url,
      'https://github.com/astral-sh/python-build-standalone/releases/download/20260901/' +
        'cpython-3.12.14%2B20260901-aarch64-apple-darwin-install_only_stripped.tar.gz',
    );
  });
});
