import { execFileSync } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { stageBackend } from '../../tooling/runtime/stage-backend.mjs';

const root = fileURLToPath(new URL('../..', import.meta.url));
const python = process.env.TENSORVIZ_COMPILER_PYTHON;
if (!python || !path.isAbsolute(python)) throw new Error('Set TENSORVIZ_COMPILER_PYTHON to a dedicated Python environment with plugin/compiler-requirements.txt installed.');
const build = path.join(root, 'plugin/out/native-backend');
mkdirSync(build, { recursive: true });
stageBackend(root, path.join(build, 'source'), 'standalone');
execFileSync(python, [path.join(root, 'plugin/scripts/compile-backend.py'), path.join(build, 'source'),
  build, path.join(root, 'plugin/out/native-runtime/backend')], { stdio: 'inherit' });
