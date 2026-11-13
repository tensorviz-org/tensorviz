import { rm, statfs } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// These outputs have no consumers after packaging and native acceptance.
// Keep test-venv, native-runtime, audit-clients, integration and dist intact.
export const disposableBuildPaths = Object.freeze([
  'plugin/out/native-backend', 'plugin/out/native-host',
  'plugin/out/compiler-venv', 'canvas/node_modules',
]);
export async function cleanupBuildScratch(root) {
  const space = async () => { const s = await statfs(root); return s.bavail * s.bsize; };
  const before = await space();
  for (const relative of disposableBuildPaths) {
    await rm(path.join(root, relative), { recursive: true, force: true });
  }
  return { beforeBytes: before, afterBytes: await space(), removed: disposableBuildPaths };
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  console.log(JSON.stringify(await cleanupBuildScratch(fileURLToPath(new URL('../..', import.meta.url)))));
}
