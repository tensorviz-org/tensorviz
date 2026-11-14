const {test} = require('node:test');
const assert = require('node:assert/strict');
test('Linux ABI ceiling remains strict for x64 and separately declares ARM64', async () => {
  const {verifyLinuxGlibc} = await import('../../scripts/package-native.mjs');
  const output = version => (command, args, options) => {
    assert.equal(command, 'readelf');
    assert.deepEqual(args, ['--version-info', 'fixture']);
    assert.equal(options.maxBuffer, 8 * 1024 * 1024);
    return 'GLIBC_2.17 GLIBC_' + version;
  };
  assert.doesNotThrow(() => verifyLinuxGlibc('fixture', output('2.31'), 'x64'));
  assert.throws(() => verifyLinuxGlibc('fixture', output('2.32'), 'x64'), /newer than 2.31/);
  assert.doesNotThrow(() => verifyLinuxGlibc('fixture', output('2.35'), 'arm64'));
  assert.throws(() => verifyLinuxGlibc('fixture', output('2.36'), 'arm64'), /newer than 2.35/);
  for (const arch of ['x64', 'arm64']) assert.throws(() => verifyLinuxGlibc('fixture', output('3.0'), arch), /newer/);
  assert.throws(() => verifyLinuxGlibc('fixture', output('2.17'), 'ia32'), /Unsupported/);
});
