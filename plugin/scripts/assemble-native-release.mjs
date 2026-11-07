import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { exportPublic, validateRuntimeLock } from './export-public.mjs';
import { createReleaseCandidate } from './release-candidate.mjs';

/** Assemble verified native jobs before generating the one shared catalog/ZIP pair. */
export async function assembleNativeRelease(directory, archives) {
  const runtimes = {};
  let version;
  for (const archive of archives) {
    const lock = validateRuntimeLock(JSON.parse(await fs.readFile(`${archive}.json`, 'utf8')));
    if (lock.formatVersion !== 1 || runtimes[lock.platform] || version && lock.version !== version) throw new Error('Native build versions/platforms differ.');
    version = lock.version;
    runtimes[lock.platform] = lock;
  }
  const runtimeLock = validateRuntimeLock({ formatVersion: 2, version,
    runtimes: Object.fromEntries(Object.entries(runtimes).sort(([a], [b]) => a.localeCompare(b))) });
  await exportPublic(directory, { runtimeLock });
  return createReleaseCandidate(directory, archives);
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [directory, ...archives] = process.argv.slice(2);
  if (!directory || !archives.length) throw new Error('Usage: node assemble-native-release.mjs OUTPUT_INTEGRATION ARCHIVES...');
  console.log(await assembleNativeRelease(directory, archives));
}
