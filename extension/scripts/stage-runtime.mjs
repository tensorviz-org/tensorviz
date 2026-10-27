import { fileURLToPath } from 'node:url';
import { stageRuntimeAssets } from '../../tooling/runtime/assets.mjs';

stageRuntimeAssets(fileURLToPath(new URL('../..', import.meta.url)), fileURLToPath(new URL('..', import.meta.url)));
