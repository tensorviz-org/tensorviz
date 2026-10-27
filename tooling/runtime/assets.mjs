import { copyFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';

/** Stable installed paths, shared source ownership. */
export function stageRuntimeAssets(repository, destination) {
  mkdirSync(path.join(destination, 'licenses'), { recursive: true });
  copyFileSync(path.join(repository, 'runtime/python/manifest.json'), path.join(destination, 'runtime-manifest.json'));
  copyFileSync(path.join(repository, 'runtime/licenses/tldraw-5.0.1.md'), path.join(destination, 'licenses/tldraw-5.0.1.md'));
}
