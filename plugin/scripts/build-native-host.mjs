import { execFileSync } from 'node:child_process';
import { cp, mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { nativePlatform } from './native-platforms.mjs';
import { pluginVersion } from './release-version.mjs';

const root = fileURLToPath(new URL('../..', import.meta.url));
const target = nativePlatform();
const output = path.join(root, 'plugin/out/native-runtime');
const build = path.join(root, 'plugin/out/native-host');
const version = pluginVersion();
await mkdir(build, { recursive: true });
await mkdir(output, { recursive: true });
const host = await readFile(path.join(root, 'plugins/tensorviz/runtime/host.cjs'), 'utf8');
await writeFile(path.join(build, 'host.cjs'), `if (process.argv[2] === '--verify-native-package') {
  if (require('node:fs').readFileSync(__filename, 'utf8') !== 'source-code-not-available') throw new Error('Host source is present in the executable');
  console.log(JSON.stringify({ sourceIncluded: false, node: process.versions.node, platform: process.platform, architecture: process.arch }));
} else {\n${host}\n}\n`);
// A synthetic build manifest controls pkg's source-disclosure policy. It does not
// change the repository's license or the licenses of bundled dependencies.
await writeFile(path.join(build, 'package.json'), JSON.stringify({ name: 'tensorviz-native-host',
  version, private: true, license: 'UNLICENSED', bin: 'host.cjs',
  pkg: { scripts: ['host.cjs'], targets: [target.pkg], bytecode: true,
    public: false, publicPackages: [], fallbackToSource: false, sea: false,
    ignore: ['**/node_modules/**'] } }, null, 2));
const log = execFileSync(path.join(root, 'plugin/node_modules/.bin/pkg'),
  [path.join(build, 'package.json'), '--debug', '--output', path.join(output, 'tensorviz-host')],
  { encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 });
await writeFile(path.join(build, 'build.log'), log);
if (/Content of[^\n]*host\.cjs|Failed to make bytecode|fallback.to.source/i.test(log)) throw new Error('The host was not packaged exclusively as bytecode.');
const checked = JSON.parse(execFileSync(path.join(output, 'tensorviz-host'), ['--verify-native-package'], { encoding: 'utf8' }));
if (checked.sourceIncluded !== false || checked.platform !== target.os || checked.architecture !== target.arch) throw new Error('Native host verification failed.');
await writeFile(path.join(build, 'verification.json'), JSON.stringify(checked, null, 2) + '\n');
for (const name of ['browser', 'runtime-manifest.json', 'plugin.json', 'licenses']) {
  await cp(path.join(root, 'plugins/tensorviz', name), path.join(output, name), { recursive: true });
}
const manifest = JSON.parse(await readFile(path.join(output, 'plugin.json'), 'utf8'));
await writeFile(path.join(output, 'plugin.json'), JSON.stringify({ ...manifest, version }, null, 2) + '\n');
console.log(`Built native host ${version}; audit log: ${path.join(build, 'build.log')}`);
