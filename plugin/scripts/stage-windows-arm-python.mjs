// CI uses the same reviewed Python pin as runtime provisioning. This avoids
// relying on uv's embedded download catalog or silently selecting emulated x64.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {nativeArchiveTool} from './native-platforms.mjs';

export async function verifiedPythonArchive(response, sha256, maximum) {
  if (!Number.isSafeInteger(maximum) || maximum < 1 || maximum > 134217728) throw new Error('Invalid Python archive size limit');
  if (!/^[a-f0-9]{64}$/.test(sha256)) throw new Error('Invalid Python archive checksum pin');
  if (!response.ok || !response.body) throw new Error(`Python download failed: ${response.status}`);
  const chunks = []; let size = 0;
  for await (const chunk of response.body) {
    size += chunk.length;
    if (size > maximum) throw new Error('Python archive exceeds reviewed size limit');
    chunks.push(Buffer.from(chunk));
  }
  const bytes = Buffer.concat(chunks);
  if (createHash('sha256').update(bytes).digest('hex') !== sha256) throw new Error('Python archive checksum mismatch');
  return bytes;
}

export async function stageWindowsArmPython() {
  if (process.platform !== 'win32' || process.arch !== 'arm64') throw new Error('Native Windows ARM64 staging only');
  const root = fileURLToPath(new URL('../..', import.meta.url));
  const manifest = JSON.parse(await fs.readFile(path.join(root, 'runtime/python/manifest.json'), 'utf8'));
  const pin = manifest.platforms['win32-arm64'];
  if (!/^3\.12\.\d+$/.test(manifest.python) || !/^\d{8}$/.test(manifest.release) || manifest.flavour !== 'install_only_stripped' || pin?.interpreter !== 'python.exe') throw new Error('Unsupported reviewed Python archive contract');
  if (pin?.triple !== 'aarch64-pc-windows-msvc' || !/^[a-f0-9]{64}$/.test(pin.sha256)) throw new Error('Invalid reviewed ARM64 Python pin');
  if (manifest.baseUrl !== 'https://github.com/astral-sh/python-build-standalone/releases/download') throw new Error('Unreviewed Python download source');
  const name = `cpython-${manifest.python}+${manifest.release}-${pin.triple}-${manifest.flavour}.tar.gz`;
  const url = `${manifest.baseUrl}/${manifest.release}/${encodeURIComponent(name)}`;
  const response = await fetch(url, {signal: AbortSignal.timeout(120000)});
  if (!response.url.startsWith('https:')) throw new Error('Python download redirected away from HTTPS');
  const bytes = await verifiedPythonArchive(response, pin.sha256, manifest.maxDownloadBytes);
  const destination = path.join(root, 'plugin/out/native-arm-python');
  await fs.rm(destination, {recursive:true, force:true});
  await fs.mkdir(destination, {recursive:true});
  const archive = path.join(destination, 'verified-python.tar.gz');
  await fs.writeFile(archive, bytes);
  // Only the exact hash-pinned official archive is extracted or executed.
  execFileSync(nativeArchiveTool(), ['-xzf', archive, '-C', destination], {stdio:'inherit'});
  await fs.rm(archive);
  const python = path.join(destination, 'python', pin.interpreter);
  await fs.access(python);
  return python;
}
