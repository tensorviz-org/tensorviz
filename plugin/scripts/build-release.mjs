import { execFileSync } from 'node:child_process';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { verifyRendererLicense } from './renderer-license.mjs';
import { distributionRepository } from './release-version.mjs';
import { nativePlatform, runtimeArchive } from './native-platforms.mjs';
import { packageNative } from './package-native.mjs';

const root = fileURLToPath(new URL('../..', import.meta.url));

export function releaseBuildConfig(environment, version, platform = nativePlatform().key) {
  if (!/^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(version)) throw new Error('Invalid checked-in plugin version.');
  if (environment.TENSORVIZ_RELEASE_VERSION && environment.TENSORVIZ_RELEASE_VERSION !== version) throw new Error('Release version must match the checked-in plugin version.');
  if (!/^[a-f0-9]{40}$/.test(environment.TENSORVIZ_REVIEWED_SOURCE_SHA ?? '')) throw new Error('An exact reviewed source SHA is required.');
  if (!path.isAbsolute(environment.TENSORVIZ_COMPILER_PYTHON ?? '')) throw new Error('Set TENSORVIZ_COMPILER_PYTHON to a dedicated compiler environment.');
  if (environment.TLDRAW_DISTRIBUTION_CONFIRMED !== 'true') throw new Error('Renderer distribution coverage needs owner confirmation.');
  const key = (environment.REACT_APP_TLDRAW_LICENSE_KEY || environment.TLDRAW_LICENSE_KEY || '').replace(/[\u200B-\u200D\uFEFF\r\n]/g, '');
  if (!key) throw new Error('A production renderer license is required. The key is never printed.');
  return { key, url: `https://github.com/${distributionRepository}/releases/download/v${version}/${runtimeArchive({ version, platform })}`,
    env: { ...environment, TENSORVIZ_RELEASE_VERSION: version, TLDRAW_LICENSE_KEY: key, REACT_APP_TLDRAW_LICENSE_KEY: key } };
}

/** Matrix job: build one native archive. Assemble all platforms in a separate job. */
export async function buildRelease({ command = execFileSync } = {}) {
  const { version } = JSON.parse(await fs.readFile(path.join(root, 'plugin/package.json'), 'utf8'));
  const config = releaseBuildConfig(process.env, version);
  const run = (tool, args, options = {}) => command(tool, args, { cwd: root, env: config.env, stdio: 'inherit', ...options });
  if (run('git', ['rev-parse', 'HEAD'], { encoding: 'utf8', stdio: 'pipe' }).trim() !== process.env.TENSORVIZ_REVIEWED_SOURCE_SHA) throw new Error('Checkout does not match the reviewed source SHA.');
  if (run('git', ['status', '--porcelain', '--untracked-files=no'], { encoding: 'utf8', stdio: 'pipe' }).trim()) throw new Error('Native release requires a clean reviewed source working tree.');
  await verifyRendererLicense(config.key);
  for (const name of ['native-runtime', 'native-host', 'native-backend']) await fs.rm(path.join(root, 'plugin/out', name), { recursive: true, force: true });
  run('npm', ['run', 'build:shared']);
  run(process.execPath, ['plugin/scripts/build.mjs']);
  run(process.execPath, ['canvas/scripts/build-plugin.cjs']);
  run(process.execPath, ['canvas/scripts/build-plugin.cjs', '--panel']);
  // Verify the production bundle contains the signed key without logging it.
  const includesKey = async directory => {
    for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
      const file = path.join(directory, entry.name);
      if (entry.isDirectory() && await includesKey(file)) return true;
      if (entry.isFile() && entry.name.endsWith('.js') && (await fs.readFile(file, 'utf8')).includes(config.key)) return true;
    }
    return false;
  };
  if (!await includesKey(path.join(root, 'plugins/tensorviz/browser'))) throw new Error('Production browser does not embed the renderer license.');
  run(process.execPath, ['plugin/scripts/build-native-backend.mjs']);
  run(process.execPath, ['plugin/scripts/build-native-host.mjs']);
  // Build scripts must not stamp or rewrite reviewed source manifests.
  if (run('git', ['status', '--porcelain', '--untracked-files=no'], { encoding: 'utf8', stdio: 'pipe' }).trim()) throw new Error('Native build changed tracked source; review generated changes before release.');
  return packageNative(config.url);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 3 || process.argv[2] !== '--native-archive') throw new Error('Usage: node plugin/scripts/build-release.mjs --native-archive');
  await buildRelease();
}
