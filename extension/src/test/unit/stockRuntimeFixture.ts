/** A stock install staged in a temp dir: an extension carrying runtime-manifest.json (the
 *  REAL manifest, its matching platform re-pinned in the copy to the fabricated archive's
 *  actual checksum — pinning to the fixture bytes is what makes the download verify, and the
 *  copy is validated by the same loader the packaged install uses), a staged backend with the
 *  two requirements files, and a transport that streams that archive. Nothing network, nothing
 *  workspace. Shared by runtime.test.ts (the states) and preflight.test.ts (the consent gate
 *  in front of them). */
import * as assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import * as fs from 'node:fs';
import * as path from 'node:path';

import { ManagedRuntime } from '../../runtime';
import { Transport } from '@tensorviz/graph-host/runtimeAcquire';
import { runtimePlatformKey } from '@tensorviz/graph-host/runtimeManifest';
import { archive, gz } from '../../../../tests/fixtures/tar.cjs';

/** What the real pinned Python's venv/pip/probe need to answer. POSIX only — the tests that
 *  provision with it skip on Windows, the same as the other shell-script interpreters here. */
export const SCRIPTED_PYTHON = [
  '#!/bin/sh',
  'case "$1" in',
  '  -c) echo "3.12.14"; exit 0 ;;',
  '  -m)',
  '    if [ "$2" = "venv" ]; then',
  // A real venv records where its base interpreter lives in pyvenv.cfg and quotes its own
  // location in bin/ scripts — exactly what rewriteVenvSelfPaths must repair before the
  // staging-to-service rename. Writing both here is what makes the lifecycle tests able to
  // fail when the rewrite is missing.
  // It also shims bin/python* the way python -m venv really does on POSIX: bin/python3 is an
  // ABSOLUTE symlink into the base tree (which dangles after the staging rename unless the
  // rewrite re-points its target) and bin/python is a relative hop to it.
  '      mkdir -p "$3/bin";',
  '      BASE="$(cd "$(dirname "$0")/.." && pwd)";',
  '      ln -s "$BASE/bin/python3" "$3/bin/python3";',
  '      ln -s python3 "$3/bin/python";',
  '      printf "home = %s/bin\\n" "$BASE" > "$3/pyvenv.cfg";',
  '      printf "#!%s/bin/python\\n" "$3" > "$3/bin/pip"; chmod 755 "$3/bin/pip"; exit 0',
  '    fi',
  '    if [ "$2" = "pip" ]; then',
  '      case "$6" in */requirements-standalone.txt) exit 0 ;; *) echo "unexpected service dependency manifest" >&2; exit 91 ;; esac',
  '    fi ;;',
  'esac',
  'exit 0',
  '',
].join('\n');

const REAL_MANIFEST = path.resolve(__dirname, '..', '..', '..', 'runtime-manifest.json');

export interface StockFixture {
  runtime: ManagedRuntime;
  extensionPath: string;
  globalStorageDir: string;
  backendDir: string;
  transportCalls: string[];
  /** Where the fabricated archive lives, for tests that need to serve it slowly or tampered. */
  assetPath: string;
}

export function stockDownloadFixture(temp: string, transport?: Transport, remoteName?: string): StockFixture {
  const key = runtimePlatformKey();
  if (!key) {
    assert.fail('this test host has no pinned runtime in the manifest');
  }
  const root = fs.mkdtempSync(path.join(temp, 'stock-'));
  const extensionPath = path.join(root, 'ext');
  const globalStorageDir = path.join(root, 'globalStorage');
  fs.mkdirSync(extensionPath, { recursive: true });
  fs.mkdirSync(globalStorageDir, { recursive: true });

  const tarball = gz(
    archive([
      { name: 'python/', type: '5', mode: 0o755 },
      { name: 'python/bin', type: '5', mode: 0o755 },
      { name: 'python/bin/python3', mode: 0o755, data: Buffer.from(SCRIPTED_PYTHON) },
    ]),
  );
  const sha256 = createHash('sha256').update(tarball).digest('hex');
  const manifest = JSON.parse(fs.readFileSync(REAL_MANIFEST, 'utf8')) as {
    platforms: Record<string, { sha256: string }>;
  };
  manifest.platforms[key] = { ...manifest.platforms[key], sha256 };
  fs.writeFileSync(path.join(extensionPath, 'runtime-manifest.json'), JSON.stringify(manifest));

  const backendDir = path.join(root, 'backend');
  fs.mkdirSync(backendDir, { recursive: true });
  fs.writeFileSync(path.join(backendDir, 'requirements-standalone.txt'), 'websockets>=12\n');
  fs.writeFileSync(path.join(backendDir, 'requirements.txt'), 'torch==2.7.1\n');
  fs.writeFileSync(path.join(backendDir, 'server_v2.py'), '# staged backend fixture\n');

  const assetPath = path.join(root, 'asset.tar.gz');
  fs.writeFileSync(assetPath, tarball);
  const transportCalls: string[] = [];
  const serve: Transport =
    transport ??
    (async (url) => {
      transportCalls.push(url);
      return {
        status: 200,
        headers: { 'content-length': String(tarball.byteLength) },
        stream: fs.createReadStream(assetPath),
      };
    });

  return {
    runtime: new ManagedRuntime({
      globalStorageDir,
      extensionPath,
      remoteName,
      backendDir: () => backendDir,
      transport: serve,
    }),
    extensionPath,
    globalStorageDir,
    backendDir,
    transportCalls,
    assetPath,
  };
}

