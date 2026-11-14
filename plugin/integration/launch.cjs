#!/usr/bin/env node
'use strict';
// Optional portable bootstrap. The existing shell launcher remains the default.
// This file contains no TensorViz core and requires Node.js 22 or newer.
const fs = require('node:fs/promises');
const constants = require('node:fs').constants;
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const zlib = require('node:zlib');
const { spawn } = require('node:child_process');

const MAX_ARCHIVE = 256 * 1024 * 1024;
const MAX_UNPACKED = 768 * 1024 * 1024;
const ABIS = Object.freeze({
  'linux-x64-musl': 'cpython-312-x86_64-linux-musl',
  'darwin-arm64': 'cpython-312-darwin', 'darwin-x64': 'cpython-312-darwin',
  'linux-x64': 'cpython-312-x86_64-linux-gnu', 'linux-arm64': 'cpython-312-aarch64-linux-gnu',
  'win32-x64': 'cp312-win_amd64', 'win32-arm64': 'cp312-win_arm64',
});
const fail = message => { throw new Error(message); };
const digest = bytes => crypto.createHash('sha256').update(bytes).digest('hex');

function httpsURL(value) {
  if (typeof value !== 'string' || /[\s\r\n]/.test(value)) fail('Invalid runtime URL.');
  const url = new URL(value);
  if (url.protocol !== 'https:' || url.username || url.password || url.hash) fail('Runtime downloads require HTTPS without credentials or fragments.');
  return url;
}

function isMusl(report = process.report.getReport()) {
  return !report.header?.glibcVersionRuntime && report.sharedObjects?.some(file => /(?:^|\/)ld-musl-[^/]+\.so\.1$/.test(file));
}
function selectRuntime(lock, platform = process.platform, arch = process.arch, report = platform === process.platform ? process.report.getReport() : undefined) {
  if (platform === 'linux' && report && !report.header?.glibcVersionRuntime && !isMusl(report)) fail('Unknown Linux libc; no runtime was downloaded.');
  const key = `${platform}-${arch}${platform === 'linux' && isMusl(report) ? '-musl' : ''}`;
  if (!Object.hasOwn(ABIS, key)) fail(`Unsupported MCP execution host: ${key}. No runtime was downloaded.`);
  const entry = lock?.formatVersion === 2 ? lock.runtimes?.[key] : lock;
  if (![1, 2].includes(lock?.formatVersion) || entry?.formatVersion !== 1 || entry.platform !== key
      || entry.version !== lock.version || !/^\d+\.\d+\.\d+$/.test(entry.version)
      || entry.pythonABI !== ABIS[key]) fail(`No valid ${key} runtime in this installation. No runtime was downloaded.`);
  if (!/^[a-f0-9]{64}$/.test(entry.sha256) || !Number.isSafeInteger(entry.bytes)
      || entry.bytes < 1 || entry.bytes > MAX_ARCHIVE) fail('Invalid runtime digest or size.');
  httpsURL(entry.url);
  return entry;
}

function checkHost(platform = process.platform, report = process.report.getReport(), arch = process.arch) {
  if (platform === 'linux') {
    if (isMusl(report) && arch === 'x64') return;
    const version = report.header?.glibcVersionRuntime;
    const parts = /^(\d+)\.(\d+)(?:\.\d+)?$/.exec(version ?? '');
    // Lowering this floor requires a symbol audit and real capture on that libc.
    const floor = arch === 'arm64' ? 35 : 31;
    if (!parts || +parts[1] < 2 || (+parts[1] === 2 && +parts[2] < floor)) fail(`Linux ${arch} runtime requires glibc 2.${floor} or newer; musl builds are not included.`);
  }
}

async function readVerified(file, entry) {
  const handle = await fs.open(file, constants.O_RDONLY | (constants.O_NOFOLLOW || 0));
  try {
    const stat = await handle.stat();
    if (!stat.isFile() || stat.size !== entry.bytes) fail('Cached runtime size mismatch. Nothing was executed.');
    const bytes = await handle.readFile();
    if (bytes.length !== entry.bytes || digest(bytes) !== entry.sha256) fail('Runtime checksum mismatch. Nothing was executed.');
    return bytes;
  } finally { await handle.close(); }
}

async function download(entry, request = fetch) {
  let url = httpsURL(entry.url);
  const signal = AbortSignal.timeout(300000);
  for (let redirects = 0; redirects <= 5; redirects++) {
    const response = await request(url, { redirect: 'manual', signal });
    if ([301, 302, 303, 307, 308].includes(response.status)) {
      await response.body?.cancel();
      if (redirects === 5 || !response.headers.get('location')) fail('Too many or invalid runtime redirects.');
      url = httpsURL(new URL(response.headers.get('location'), url).href);
      continue;
    }
    if (!response.ok || !response.body) { await response.body?.cancel(); fail(`Runtime download failed (HTTP ${response.status}).`); }
    const chunks = []; let size = 0;
    for await (const chunk of response.body) {
      size += chunk.length;
      if (size > entry.bytes) { fail('Runtime download exceeded its pinned size.'); }
      chunks.push(chunk);
    }
    const bytes = Buffer.concat(chunks);
    if (size !== entry.bytes || digest(bytes) !== entry.sha256) fail('Runtime download size or checksum mismatch. Nothing was executed.');
    return bytes;
  }
  fail('Runtime download failed.');
}

// Extract only regular files/directories. No external tar, links, devices, drives,
// traversal or alternate Windows data streams can enter the private session.
function safeName(name) {
  const parts = name.replace(/\/$/, '').split('/');
  if (!name || name.startsWith('/') || /[\\:\x00-\x1f]/.test(name)
      || parts.some(part => !part || part === '.' || part === '..' || /[. ]$/.test(part)
        || /^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(part))) fail(`Unsafe runtime archive path: ${JSON.stringify(name)}`);
  return parts.join('/');
}
function tarString(bytes) { return bytes.subarray(0, bytes.indexOf(0) < 0 ? bytes.length : bytes.indexOf(0)).toString('utf8'); }
function tarNumber(bytes) {
  const text = tarString(bytes).trim();
  if (!/^[0-7]*$/.test(text)) fail('Unsupported runtime tar number.');
  const result = parseInt(text || '0', 8);
  if (!Number.isSafeInteger(result)) fail('Invalid runtime tar number.');
  return result;
}
function paxFields(data) {
  let cursor = 0; const result = {};
  while (cursor < data.length) {
    const space = data.indexOf(32, cursor);
    if (space < 0 || !/^\d+$/.test(data.subarray(cursor, space).toString())) fail('Invalid runtime PAX header.');
    const length = Number(data.subarray(cursor, space).toString());
    if (length <= space - cursor + 2 || cursor + length > data.length || data[cursor + length - 1] !== 10) fail('Invalid runtime PAX record.');
    const record = data.subarray(space + 1, cursor + length - 1).toString('utf8');
    const equals = record.indexOf('=');
    if (equals < 1) fail('Invalid runtime PAX field.');
    const key = record.slice(0, equals);
    if (Object.hasOwn(result, key)) fail('Duplicate runtime PAX field.');
    result[key] = record.slice(equals + 1); cursor += length;
  }
  return result;
}
function archiveFiles(archive) {
  const tar = zlib.gunzipSync(archive, { maxOutputLength: MAX_UNPACKED });
  const entries = []; const names = new Map(); let cursor = 0, pending = null, ended = false;
  while (cursor + 512 <= tar.length) {
    const header = tar.subarray(cursor, cursor + 512); cursor += 512;
    if (header.every(value => value === 0)) {
      if (pending || !tar.subarray(cursor).every(value => value === 0)) fail('Malformed runtime tar terminator.');
      ended = true; break;
    }
    const expected = tarNumber(header.subarray(148, 156));
    const sum = header.reduce((sum, value, index) => sum + (index >= 148 && index < 156 ? 32 : value), 0);
    if (sum !== expected) fail('Runtime tar checksum mismatch.');
    const size = tarNumber(header.subarray(124, 136));
    if (size > MAX_UNPACKED || cursor + Math.ceil(size / 512) * 512 > tar.length) fail('Truncated or oversized runtime tar entry.');
    const data = tar.subarray(cursor, cursor + size); cursor += Math.ceil(size / 512) * 512;
    const type = String.fromCharCode(header[156] || 48);
    if (type === 'x' || type === 'L') {
      if (pending) fail('Repeated runtime tar path extension.');
      pending = type === 'x' ? paxFields(data) : { path: tarString(data) };
      continue;
    }
    if (!['0', '5'].includes(type)) fail(`Unsupported runtime tar entry type: ${type}`);
    if (pending?.linkpath || pending?.size || Object.keys(pending ?? {}).some(key => /sparse/i.test(key))) fail('Unsafe runtime PAX metadata.');
    const prefix = tarString(header.subarray(345, 500));
    const name = safeName(pending?.path ?? `${prefix ? prefix + '/' : ''}${tarString(header.subarray(0, 100))}`);
    pending = null;
    const key = name.toLowerCase();
    if (names.has(key)) fail('Duplicate or case-colliding runtime archive path.');
    names.set(key, type); entries.push({ name, directory: type === '5', data });
    if (type === '5' && size) fail('Runtime directory contains data.');
  }
  if (!ended || pending) fail('Missing runtime tar terminator.');
  for (const entry of entries) {
    const parts = entry.name.split('/'); parts.pop();
    while (parts.length) {
      if (names.get(parts.join('/').toLowerCase()) === '0') fail('Runtime file shadows a directory.');
      parts.pop();
    }
  }
  return entries;
}
async function extract(archive, directory) {
  const entries = archiveFiles(archive);
  for (const entry of entries) {
    const target = path.join(directory, ...entry.name.split('/'));
    await fs.mkdir(entry.directory ? target : path.dirname(target), { recursive: true, mode: 0o700 });
    if (!entry.directory) await fs.writeFile(target, entry.data, { flag: 'wx', mode: entry.name === 'tensorviz-host' ? 0o700 : 0o600 });
  }
}

async function main(args = process.argv.slice(2)) {
  if (Number(process.versions.node.split('.')[0]) < 22) fail('The optional portable launcher requires Node.js 22 or newer.');
  const prepare = args[0] === '--prepare-runtime';
  if (prepare && args.length !== 1) fail('--prepare-runtime does not accept additional arguments.');
  const plugin = path.resolve(__dirname, '..');
  const entry = selectRuntime(JSON.parse(await fs.readFile(path.join(plugin, 'runtime-lock.json'), 'utf8')));
  checkHost();
  const base = process.platform === 'win32' ? process.env.LOCALAPPDATA || path.join(os.homedir(), 'AppData', 'Local')
    : process.platform === 'darwin' ? path.join(os.homedir(), 'Library', 'Caches') : process.env.XDG_CACHE_HOME || path.join(os.homedir(), '.cache');
  const cache = process.env.TENSORVIZ_RUNTIME_CACHE_DIR || path.join(base, 'dev.tensorviz', 'runtime');
  if (!path.isAbsolute(cache)) fail('Runtime cache must be an absolute path.');
  await fs.mkdir(cache, { recursive: true, mode: 0o700 });
  const stat = await fs.lstat(cache);
  if (!stat.isDirectory() || stat.isSymbolicLink() || (process.getuid && (stat.uid !== process.getuid() || stat.mode & 0o022))) fail('Runtime cache must be a private, owned directory, not a link.');
  const archive = path.join(cache, `${entry.sha256}.tar.gz`);
  let bytes;
  try { bytes = await readVerified(archive, entry); }
  catch (error) {
    if (error.code !== 'ENOENT') throw error;
    console.error(`TensorViz: downloading runtime ${entry.version} for reuse by later connections.`);
    bytes = await download(entry);
    const temporary = path.join(cache, `.download-${crypto.randomUUID()}`);
    try { await fs.writeFile(temporary, bytes, { flag: 'wx', mode: 0o600 }); await fs.rename(temporary, archive); }
    finally { await fs.rm(temporary, { force: true }); }
  }
  if (prepare) { console.error(`TensorViz: runtime ${entry.version} verified and cached; reconnect the plugin.`); return 0; }
  const session = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-native-'));
  let child, forceKill; const signals = new Map();
  try {
    await extract(bytes, session);
    const host = path.join(session, process.platform === 'win32' ? 'tensorviz-host.exe' : 'tensorviz-host');
    const hostStat = await fs.lstat(host);
    if (!hostStat.isFile() || hostStat.isSymbolicLink()) fail('Runtime archive has no native host.');
    child = spawn(host, args, { stdio: 'inherit', windowsHide: true,
      env: { ...process.env, TORCHINDUCTOR_CACHE_DIR: path.join(session, '.torchinductor') } });
    for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP']) {
      const handler = () => {
        child.kill(signal);
        forceKill ??= setTimeout(() => child.kill('SIGKILL'), 5000).unref();
      };
      signals.set(signal, handler); process.on(signal, handler);
    }
    return await new Promise((resolve, reject) => {
      child.once('error', reject); child.once('exit', (code, signal) => resolve(code ?? (signal === 'SIGINT' ? 130 : 143)));
    });
  } finally {
    for (const [signal, handler] of signals) process.off(signal, handler);
    clearTimeout(forceKill);
    await fs.rm(session, { recursive: true, force: true, maxRetries: 4, retryDelay: 100 });
  }
}
module.exports = { selectRuntime, checkHost, httpsURL, download, archiveFiles, extract, readVerified, main };
if (require.main === module) main().then(code => { process.exitCode = code; }, error => { console.error(`TensorViz: ${error.message}`); process.exitCode = 1; });
