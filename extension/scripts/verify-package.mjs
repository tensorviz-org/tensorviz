#!/usr/bin/env node
/**
 * Post-package gate: inspect the .vsix vsce just wrote and refuse a contract violation.
 *
 * `npm run stage:backend` guards what goes INTO the package; this scripts guards what came
 * OUT — presence of the runtime manifest and the requirements the managed runtime provisions
 * from, and absence of everything the managed runtime must never ship (the interpreter
 * archive, an extracted interpreter, a venv, download caches). A .vsix that offends either
 * side looks identical in the file list a user skims; make it the packager's failure, not the
 * first user's.
 *
 * Runs `unzip -Z1` so the gate lists the same entries a package inspection would,
 * and a packaging regression fails here rather than in a manual read.
 */
import { execFileSync } from 'node:child_process';
import { readdirSync, statSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { verifyRendererLicense } from '../../plugin/scripts/renderer-license.mjs';

const extensionDir = resolve(dirname(fileURLToPath(import.meta.url)), '..');

const RELEASE_KEY_MESSAGE = 'verify-package: --release requires REACT_APP_TLDRAW_LICENSE_KEY. Refusing a production package whose Canvas would show the tldraw watermark. The key is not printed.';

function releaseRendererKey() {
  return process.env.REACT_APP_TLDRAW_LICENSE_KEY?.replace(/[\u200B-\u200D\uFEFF\r\n]/g, '');
}

// `npm run package:release` runs this before `vsce package`. The same message is
// the post-package --release failure, so a missing key never starts packaging.
if (process.argv.includes('--require-release-key') || process.argv.includes('--release')) {
  if (!releaseRendererKey()) {
    console.error(RELEASE_KEY_MESSAGE);
    process.exit(1);
  }
}
if (process.argv.includes('--require-release-key')) {
  process.exit(0);
}

// The newest .vsix in the extension directory — vsce just wrote it.
const candidates = readdirSync(extensionDir)
  .filter((name) => name.endsWith('.vsix'))
  .map((name) => ({ name, mtimeMs: statSync(join(extensionDir, name)).mtimeMs }))
  .sort((a, b) => b.mtimeMs - a.mtimeMs);
if (candidates.length === 0) {
  console.error('verify-package: no .vsix found — run `npm run package` first.');
  process.exit(1);
}
const vsix = join(extensionDir, candidates[0].name);

const entries = execFileSync('unzip', ['-Z1', vsix], { encoding: 'utf8' })
  .split('\n')
  .map((line) => line.trim())
  .filter(Boolean);
if (entries.length === 0) {
  console.error(`verify-package: ${vsix} listed no entries — it is not a readable zip.`);
  process.exit(1);
}

const REQUIRED = [
  'extension/LICENSE.txt', // vsce normalizes the source LICENSE filename
  'extension/THIRD_PARTY_NOTICES.md',
  'extension/DEPENDENCY_NOTICES.txt',
  'extension/licenses/tldraw-5.0.1.md',
  // The pinned runtime contract installed installs download from.
  'extension/runtime-manifest.json',
  'extension/backend/extension_environment.py',
  // What that runtime provisions against.
  'extension/backend/requirements.txt',
  'extension/backend/requirements-cpu.txt',
  // The rest of the package inspection, pinned here so a regression fails in this gate.
  'extension/out/extension.js',
  'extension/backend/server_v2.py',
  'extension/media/app/index.html',
  'extension/themes/tensorviz-dark.json',
  'extension/themes/tensorviz-light.json',
  'extension/media/app/viewer/index.html',
  'extension/media/app/viewer/assets/translations/en.json',
  'extension/media/app/viewer/assets/icons/icon/0_merged.svg',
  'extension/backend/services/ide/code_apply.py',
  'extension/backend/services/ingest/worker.py',
  'extension/backend/requirements-project.txt',
  'extension/backend/requirements-capture.txt',
  'extension/backend/capture_environment.py',
  'extension/backend/services/python_compat.py',
  // Staged backend content identity (execution cache key).
  'extension/backend/backend-identity.json',
];

/** Patterns that must match ZERO entries — the "small .vsix, runtime on first use" contract. */
const FORBIDDEN = [
  /^extension\/agent-plugin\//, // the retired companion must not ship with the VS Code extension
  /^extension\/out\/(?!extension\.js$)/, // only the bundled entry point is used at runtime
  /^extension\/backend\/reports\//, // local developer output
  /^extension\/backend\/(ai|agents)\//,
  /^extension\/backend\/api\/routers\/(ai_.*|graph_explanations)\.py$/,
  /\.(tar\.gz|tgz)$/, // the runtime archive travels inside no artifact
  /(^|\/)pyvenv\.cfg$/, // a venv
  /(^|\/)downloads\//, // a partial or cached download
  /(^|\/)staging-/, // a mid-provision staging directory
  /(^|\/)old-/, // a retired copy from an atomic swap
  /\.vsix$/, // a nested prior artifact
  /(^|\/)out\/test\//, // the vscode test stub must never ship
  /__pycache__\//,
  /(^|\/)\.runtime\//,
  /(^|\/)scripts\//,
  /(^|\/)tests?\//,
  /(^|\/)evals\//,
  /(^|\/)\.env/,
  /(^|\/)requirements-test\.txt$/,
  /(^|\/)node_modules\//,
  /\.(pem|key|p12)$/, // secret-looking names, same rule as the manual inspection
];

let failed = false;
for (const required of REQUIRED) {
  if (!entries.includes(required)) {
    console.error(`verify-package: REQUIRED entry missing: ${required}`);
    failed = true;
  }
}
for (const entry of entries) {
  const offending = FORBIDDEN.find((pattern) => pattern.test(entry));
  if (offending) {
    console.error(`verify-package: FORBIDDEN entry shipped: ${entry} (matches ${offending})`);
    failed = true;
  }
}
if (failed) {
  process.exit(1);
}

const firebaseKey = readFileSync(join(extensionDir, 'src/websiteAccount.ts'), 'utf8').match(/^  apiKey: '(AIza[\w-]+)',$/m)?.[1];
const hostBundle = execFileSync('unzip', ['-p', vsix, 'extension/out/extension.js'], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
const publicConfigLines = firebaseKey ? hostBundle.split('\n').filter(line => line.includes(firebaseKey)) : [];
if (publicConfigLines.length !== 1 || !publicConfigLines[0].includes('// secret-detector:ignore -- public Firebase app identifier;')) {
  throw new Error('The packaged public Firebase identifier is missing its narrow Open VSX false-positive annotation.');
}

for (const file of ['LICENSE', 'THIRD_PARTY_NOTICES.md', 'DEPENDENCY_NOTICES.txt', 'licenses/tldraw-5.0.1.md']) {
  const entry = file === 'LICENSE' ? 'LICENSE.txt' : file;
  const shipped = execFileSync('unzip', ['-p', vsix, `extension/${entry}`], { encoding: 'utf8' });
  if (shipped !== readFileSync(join(extensionDir, file), 'utf8')) throw new Error(`Packaged ${file} differs from the release input`);
}
if (readFileSync(join(extensionDir, 'LICENSE'), 'utf8') !== readFileSync(join(extensionDir, '../LICENSE'), 'utf8')) {
  throw new Error('The packaged project license differs from the repository license.');
}
const tldrawVersion = JSON.parse(readFileSync(join(extensionDir, '../canvas/node_modules/tldraw/package.json'), 'utf8')).version;
if (!entries.includes(`extension/licenses/tldraw-${tldrawVersion}.md`)) {
  throw new Error(`Missing the license for bundled tldraw ${tldrawVersion}; review its new terms before packaging.`);
}
for (const directory of ['extension/media/app/static/js/', 'extension/media/app/viewer/static/js/']) {
  if (!entries.some(entry => entry.startsWith(directory) && entry.endsWith('.LICENSE.txt'))) {
    throw new Error(`Missing extracted frontend license notices in ${directory}`);
  }
}

// Without its renderer key, the released Canvas shows tldraw's "Get a license for production"
// watermark. When a build supplies the key (Release VSIX does), it must be signed and current,
// and both bundles must embed it. The key itself is never printed. `npm run package:release`
// passes --release and refuses a package built without that key.
const rendererKey = releaseRendererKey();
if (rendererKey) {
  const license = await verifyRendererLicense(rendererKey);
  for (const [surface, directory] of [['Canvas', 'extension/media/app/static/js/'], ['viewer', 'extension/media/app/viewer/static/js/']]) {
    const embedded = entries.filter(entry => entry.startsWith(directory) && entry.endsWith('.js')).some(entry =>
      execFileSync('unzip', ['-p', vsix, entry], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 }).includes(rendererKey));
    if (!embedded) throw new Error(`The ${surface} bundle does not embed the renderer license; the released UI would show tldraw's production watermark.`);
  }
  console.log(`verify-package: renderer license (${license.type}, expires ${license.expires}) embedded in Canvas and viewer.`);
} else {
  console.log('verify-package: renderer license not checked; REACT_APP_TLDRAW_LICENSE_KEY is not set for this build.');
}

const bytes = statSync(vsix).size;
console.log(
  `verify-package: ok — ${candidates[0].name}, ${entries.length} entries, ` +
    `${(bytes / 1024 / 1024).toFixed(1)} MB, runtime manifest pinned, no runtime artifacts shipped.`,
);
