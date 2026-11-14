/** Build targets. A catalog may advertise a target only after its native CI passes. */
export const nativePlatforms = {
  'darwin-arm64': { os: 'darwin', arch: 'arm64', machine: 'arm64', abi: 'cpython-312-darwin', pkg: 'node22.23.4-macos-arm64' },
  'darwin-x64': { os: 'darwin', arch: 'x64', machine: 'x86_64', abi: 'cpython-312-darwin', pkg: 'node22.23.4-macos-x64' },
  'linux-arm64': { os: 'linux', arch: 'arm64', machine: 'aarch64', abi: 'cpython-312-aarch64-linux-gnu', pkg: 'node22.23.4-linux-arm64' },
  'win32-x64': { os: 'win32', arch: 'x64', machine: 'AMD64', abi: 'cp312-win_amd64', pkg: 'node22.23.4-win-x64' },
  'win32-arm64': { os: 'win32', arch: 'arm64', machine: 'ARM64', abi: 'cp312-win_arm64', pkg: 'node22.23.4-win-arm64' },
  'linux-x64': { os: 'linux', arch: 'x64', machine: 'x86_64', abi: 'cpython-312-x86_64-linux-gnu', pkg: 'node22.23.4-linux-x64' },
};
export function nativePlatform(platform = process.platform, arch = process.arch) {
  const key = `${platform}-${arch}`;
  if (!Object.hasOwn(nativePlatforms, key)) throw new Error(`Unsupported compiled runtime target: ${key}`);
  const target = nativePlatforms[key];
  return { key, ...target, host: target.os === 'win32' ? 'tensorviz-host.exe' : 'tensorviz-host',
    backend: `_tensorviz_backend.${target.abi}.${target.os === 'win32' ? 'pyd' : 'so'}` };
}
export function runtimeEntries(lock) {
  return lock?.formatVersion === 2 ? Object.values(lock.runtimes ?? {}) : [lock];
}
export const runtimeArchive = lock => `tensorviz-runtime-${lock.version}-${lock.platform}.tar.gz`;
