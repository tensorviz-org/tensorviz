/** Only supported compiled targets may enter runtime locks. */
export const nativePlatforms = {
  'darwin-arm64': { os: 'darwin', arch: 'arm64', machine: 'arm64', abi: 'cpython-312-darwin', pkg: 'node22-macos-arm64' },
  'linux-x64': { os: 'linux', arch: 'x64', machine: 'x86_64', abi: 'cpython-312-x86_64-linux-gnu', pkg: 'node22-linux-x64' },
};
export function nativePlatform(platform = process.platform, arch = process.arch) {
  const key = `${platform}-${arch}`;
  if (!Object.hasOwn(nativePlatforms, key)) throw new Error(`Unsupported compiled runtime target: ${key}`);
  return { key, ...nativePlatforms[key] };
}
export function runtimeEntries(lock) {
  return lock?.formatVersion === 2 ? Object.values(lock.runtimes ?? {}) : [lock];
}
export const runtimeArchive = lock => `tensorviz-runtime-${lock.version}-${lock.platform}.tar.gz`;
