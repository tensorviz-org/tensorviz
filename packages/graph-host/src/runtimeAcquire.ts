/** Secure acquisition of the pinned managed backend runtime for stock VS Code installs.
 *
 * This module is all of the network and archive handling for the zero-setup backend. The
 * rules it enforces come straight from the launch contract:
 *
 *  - HTTPS only, with redirects bounded and confined to the release host and GitHub's own
 *    asset CDN. A redirect anywhere else is refused before a byte is trusted.
 *  - Bytes stream to `<dest>.part`, are counted against a ceiling and hashed as they arrive.
 *    Nothing is ever verified by asking the server what the checksum should be — the SHA-256
 *    is the pinned value from runtime-manifest.json and nothing else can weaken it.
 *  - The archive is inspected entry by entry before a single path is written: one expected
 *    top-level directory, no absolute names, no `..`, no device nodes, no fifos, no
 *    hardlinks, and no symlink whose target leaves the extraction root.
 *  - Nothing downloaded is ever EXECUTED except the verified interpreter itself, with fixed
 *    argument lists (venv creation and pip, step 3).
 *
 *  Both entry points take injectable seams (`transport`, and all state under explicit paths)
 *  so every failure mode is a unit test with a fake transport or a fabricated archive. No
 *  unit test needs the public network. */
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import * as fs from 'node:fs';
import * as https from 'node:https';
import * as path from 'node:path';
import { Readable } from 'node:stream';
import * as zlib from 'node:zlib';

// ---------------------------------------------------------------------------
// Download
// ---------------------------------------------------------------------------

export interface TransportResponse {
  status: number;
  headers: Record<string, string | string[] | undefined>;
  stream: Readable;
}

/** One HTTPS GET. Redirects are NOT followed inside the transport: policy (how many, to
 *  whom) is the downloader's job, so tests see every hop. */
export type Transport = (url: string, signal?: AbortSignal) => Promise<TransportResponse>;

/** Three hops is generous: GitHub does one into its CDN. Anything longer is a loop or a game. */
export const MAX_REDIRECTS = 3;
export const DOWNLOAD_ATTEMPTS = 3;

/** No data for this long means the connection is dead, whatever TCP thinks. */
const IDLE_TIMEOUT_MS = 120_000;

/** user-agent exists because some CDNs 403 a bare Node client string. */
function httpsTransport(url: string, signal?: AbortSignal): Promise<TransportResponse> {
  return new Promise((resolve, reject) => {
    const request = https.get(
      url,
      { signal, headers: { 'user-agent': 'tensorviz', accept: 'application/octet-stream' } },
      (response) =>
        resolve({
          status: response.statusCode ?? 0,
          headers: response.headers,
          stream: response,
        }),
    );
    request.on('error', reject);
    request.setTimeout(IDLE_TIMEOUT_MS, () => request.destroy(Object.assign(new Error('no response'), { code: 'ETIMEDOUT' })));
  });
}

/** Is following `from` to `to` acceptable? HTTPS always; the release host itself; and
 *  GitHub's asset CDNs, which is where a release download legitimately 302s. The checksum
 *  pin is the stronger line — a host elsewhere could serve exactly the pinned bytes and be
 *  accepted only after this layer lets it try. */
function redirectAllowed(from: URL, to: URL): boolean {
  if (to.protocol !== 'https:') {
    return false;
  }
  if (to.host === from.host) {
    return true;
  }
  return to.hostname === 'githubusercontent.com' || to.hostname.endsWith('.githubusercontent.com');
}

export type DownloadResult =
  | { ok: true; path: string; bytes: number }
  | { ok: false; message: string; detail: string; cancelled?: boolean; retryable?: boolean };

export interface DownloadOptions {
  url: string;
  /** The pinned hex digest from the manifest. Compared after streaming; a mismatch deletes
   *  the partial file and never retries around verification. */
  sha256: string;
  maxBytes: number;
  /** Final path. Only ever written by rename from `<destPath>.part` after verification. */
  destPath: string;
  transport?: Transport;
  report?: (message: string) => void;
  isCancelled?: () => boolean;
  signal?: AbortSignal;
  /** Test hook: shrink the idle budget. */
  idleTimeoutMs?: number;
  /** Test hook: avoid waiting through production's bounded retry delays. */
  retryDelayMs?: number;
}

function temporaryNetworkError(error: unknown): boolean {
  const cause = error as { code?: string; message?: string } | undefined;
  return ['ECONNRESET', 'ECONNREFUSED', 'ETIMEDOUT', 'EAI_AGAIN', 'ENOTFOUND',
    'ENETUNREACH', 'EHOSTUNREACH', 'ERR_STREAM_PREMATURE_CLOSE'].includes(cause?.code ?? '')
    || cause?.message === 'aborted';
}

/** Keep transport causes while excluding signed URL queries and credentials. */
export function runtimeDiagnostic(detail: string): string {
  return detail.replace(/https?:\/\/[^\s<>"']+/g, value => {
    try {
      const url = new URL(value);
      url.username = ''; url.password = ''; url.search = ''; url.hash = '';
      return url.toString();
    } catch { return '[invalid URL]'; }
  }).replace(/\bBearer\s+[^\s,;]+/gi, 'Bearer [redacted]').slice(0, 8000);
}

function downloadFailure(message: string, detail: string): Extract<DownloadResult, { ok: false }> {
  return { ok: false, message, detail: runtimeDiagnostic(detail) };
}

/** Download `options.url` to `options.destPath`, or explain exactly why not.
 *
 * Cache layering is the caller's job (ManagedRuntime checks the cache before calling). This
 * function's promise is narrower and stronger: if it answers ok, `destPath` contains exactly
 * the bytes whose sha256 the manifest pins. */
export async function downloadVerified(options: DownloadOptions): Promise<DownloadResult> {
  const controller = new AbortController();
  const cancel = () => controller.abort();
  options.signal?.addEventListener('abort', cancel, { once: true });
  if (options.signal?.aborted) cancel();
  // Older extension callers expose a cancellation predicate instead of a signal.
  const monitor = options.isCancelled ? setInterval(() => { if (options.isCancelled?.()) cancel(); }, 50) : undefined;
  monitor?.unref();
  try {
    if (options.isCancelled?.()) cancel();
    for (let attempt = 1; ; attempt++) {
      const result = await download({ ...options, signal: controller.signal });
      if (result.ok) return result;
      const safe = { ...result, detail: runtimeDiagnostic(result.detail) };
      if (result.cancelled || !result.retryable || attempt === DOWNLOAD_ATTEMPTS) {
        return { ...safe, detail: safe.detail + (attempt > 1 ? `\nDownload attempts: ${attempt}/${DOWNLOAD_ATTEMPTS}.` : '') };
      }
      options.report?.(`Download interrupted; retrying (${attempt + 1}/${DOWNLOAD_ATTEMPTS})…`);
      await new Promise<void>(resolve => {
        const done = () => { clearTimeout(timer); controller.signal.removeEventListener('abort', done); resolve(); };
        const timer = setTimeout(done, options.retryDelayMs ?? 1000 * attempt);
        controller.signal.addEventListener('abort', done, { once: true });
        if (controller.signal.aborted) done();
      });
    }
  }
  finally {
    if (monitor) clearInterval(monitor);
    options.signal?.removeEventListener('abort', cancel);
  }
}

/** Bound connection setup too: a socket timeout alone does not cover stalled DNS. */
function connect(transport: Transport, url: string, signal: AbortSignal, timeoutMs: number): Promise<TransportResponse> {
  return new Promise((resolve, reject) => {
    let settled = false;
    const request = new AbortController();
    const finish = (error?: Error, response?: TransportResponse) => {
      if (settled) { response?.stream.destroy(); return; }
      settled = true; clearTimeout(timer); signal.removeEventListener('abort', abort);
      if (error) { request.abort(); reject(error); } else resolve(response!);
    };
    const abort = () => finish(new Error('cancelled by the user'));
    const timer = setTimeout(() => finish(Object.assign(new Error('connection timed out'), { code: 'ETIMEDOUT' })), timeoutMs);
    signal.addEventListener('abort', abort, { once: true });
    if (signal.aborted) { abort(); return; }
    Promise.resolve().then(() => transport(url, request.signal)).then(response => finish(undefined, response), error => finish(error));
  });
}

async function download(options: DownloadOptions): Promise<DownloadResult> {
  let url: URL;
  try {
    url = new URL(options.url);
  } catch {
    return downloadFailure('The runtime download URL is not a URL.', options.url);
  }
  if (url.protocol !== 'https:') {
    return downloadFailure('The runtime download URL is not HTTPS.', url.toString());
  }
  const transport = options.transport ?? httpsTransport;
  const part = `${options.destPath}.part`;
  const abandon = (): void => {
    fs.rmSync(part, { force: true });
  };
  // The destination directory must exist before the write stream opens: fs.createWriteStream
  // reports a missing directory asynchronously, so a late mkdir races the first bytes.
  fs.mkdirSync(path.dirname(options.destPath), { recursive: true });
  abandon();

  for (let hops = 0; ; hops++) {
    if (options.signal?.aborted) return { ok: false, cancelled: true, message: 'The runtime download was cancelled.', detail: 'cancelled by the user' };
    let response: TransportResponse;
    try {
      options.report?.('Connecting to the pinned Python download…');
      response = await connect(transport, url.toString(), options.signal!, options.idleTimeoutMs ?? IDLE_TIMEOUT_MS);
    } catch (err) {
      if (options.signal?.aborted) return { ok: false, cancelled: true, message: 'The runtime download was cancelled.', detail: 'cancelled by the user' };
      return { ...downloadFailure(
        'The runtime download could not connect.',
        `${url}\n\n${String(err)}`,
      ), retryable: temporaryNetworkError(err) };
    }
    if ([301, 302, 303, 307, 308].includes(response.status)) {
      response.stream.destroy();
      const location = response.headers['location'];
      const target =
        typeof location === 'string' && location ? new URL(location, url) : undefined;
      if (!target) {
        return downloadFailure(
          'The runtime download redirected without a target.',
          `${url} answered HTTP ${response.status} with no usable Location.`,
        );
      }
      if (hops + 1 > MAX_REDIRECTS) {
        return downloadFailure(
          'The runtime download redirected too many times.',
          `${hops + 1} redirects is over the bound of ${MAX_REDIRECTS}.`,
        );
      }
      if (!redirectAllowed(url, target)) {
        return downloadFailure(
          'The runtime download was redirected somewhere untrusted.',
          `Refusing to follow ${url}\n  ->  ${target}`,
        );
      }
      url = target;
      continue;
    }
    if (response.status !== 200) {
      response.stream.destroy();
      return { ...downloadFailure(
        `The runtime download failed (HTTP ${response.status}).`,
        url.toString(),
      ), retryable: [408, 429, 500, 502, 503, 504].includes(response.status) };
    }
    const lengthHeader = response.headers['content-length'];
    const declared =
      typeof lengthHeader === 'string' && /^\d+$/.test(lengthHeader) ? Number(lengthHeader) : undefined;
    if (declared !== undefined && declared > options.maxBytes) {
      response.stream.destroy();
      return downloadFailure(
        'The runtime archive is larger than the pinned ceiling.',
        `${declared} bytes announced; the ceiling is ${options.maxBytes}.`,
      );
    }

    const streamed = await streamToPart(response.stream, part, options, (m) => options.report?.(m));
    if (!streamed.ok) {
      abandon();
      return streamed.cancelled
        ? { ok: false, cancelled: true, message: 'The runtime download was cancelled.', detail: streamed.detail }
        : streamed;
    }
    const got = streamed.sha256;
    if (got !== options.sha256.toLowerCase()) {
      abandon();
      // Both digests are printed — the same diagnostic fetch-runtime.sh gives — because the
      // correct response to this is finding out why the asset changed, never editing the pin.
      return downloadFailure(
        'The runtime archive failed its checksum.',
        `${url}\n\n  expected  ${options.sha256}\n  got       ${got}`,
      );
    }
    // Same-directory rename is atomic on every filesystem we ship on; the final path flips
    // from absent to complete without any intermediate state another process could observe.
    fs.renameSync(part, options.destPath);
    options.report?.('Download verified (sha256 ok).');
    return { ok: true, path: options.destPath, bytes: streamed.bytes };
  }
}

type StreamOutcome =
  | { ok: true; sha256: string; bytes: number }
  | { ok: false; cancelled?: boolean; message: string; detail: string; retryable?: boolean };

function streamToPart(
  stream: Readable,
  part: string,
  options: DownloadOptions,
  report: (message: string) => void,
): Promise<StreamOutcome> {
  return new Promise<StreamOutcome>((resolve) => {
    const hash = createHash('sha256');
    const file = fs.createWriteStream(part);
    let received = 0;
    let lastReported = 0;
    let settled = false;
    let idle: NodeJS.Timeout | undefined;

    const finish = (result: StreamOutcome): void => {
      if (settled) {
        return;
      }
      settled = true;
      options.signal?.removeEventListener('abort', abort);
      if (idle) {
        clearTimeout(idle);
      }
      stream.destroy();
      file.destroy();
      // Resolve only after the write stream has closed. createWriteStream defers open()
      // to the threadpool, so a fast-failing stream can settle before the .part file
      // exists; a synchronous rm would run ahead of the pending open and the orphaned
      // open then leaks an empty .part behind it. After close fires, any open() has
      // completed and cleanup sees the final filesystem state.
      file.once('close', () => resolve(result));
    };
    const abort = () => finish({ ok: false, cancelled: true, message: 'cancelled', detail: 'cancelled by the user' });
    const arm = (): void => {
      if (idle) {
        clearTimeout(idle);
      }
      idle = setTimeout(
        () =>
          finish({
            ok: false,
            message: 'The runtime download stalled.',
            retryable: true,
            detail: `no data for ${Math.round((options.idleTimeoutMs ?? IDLE_TIMEOUT_MS) / 1000)}s`,
          }),
        options.idleTimeoutMs ?? IDLE_TIMEOUT_MS,
      );
      idle.unref?.();
    };
    arm();

    file.on('error', (err) =>
      finish({
        ok: false,
        message: 'The runtime download could not be written to disk.',
        detail: `${part}\n\n${String(err)}`,
      }),
    );
    stream.on('error', (err) =>
      finish({
        ok: false,
        message: 'The runtime download was interrupted.',
        detail: String(err),
        retryable: temporaryNetworkError(err),
      }),
    );
    stream.on('close', () => {
      if (!stream.readableEnded && !settled) finish({ ok: false, retryable: true,
        message: 'The runtime download was interrupted.', detail: 'The connection closed before the archive finished.' });
    });
    stream.on('data', (chunk: Buffer) => {
      if (settled) {
        return;
      }
      if (options.isCancelled?.()) {
        finish({
          ok: false,
          cancelled: true,
          message: 'cancelled',
          detail: 'cancelled by the user',
        });
        return;
      }
      received += chunk.length;
      if (received > options.maxBytes) {
        finish({
          ok: false,
          message: 'The runtime archive exceeded the pinned size ceiling while streaming.',
          detail: `over ${options.maxBytes} bytes`,
        });
        return;
      }
      hash.update(chunk);
      if (!file.write(chunk)) {
        stream.pause();
        file.once('drain', () => stream.resume());
      }
      if (received - lastReported >= 8 * 1024 * 1024) {
        lastReported = received;
        report(`Downloading the tensorViz runtime… ${Math.floor(received / (1024 * 1024))} MB`);
      }
      arm();
    });
    options.signal?.addEventListener('abort', abort, { once: true });
    if (options.signal?.aborted) abort();
    stream.on('end', () => {
      if (settled) {
        return;
      }
      file.end(() =>
        finish({ ok: true, sha256: hash.digest('hex'), bytes: received }),
      );
    });
  });
}

// ---------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------

/** A defensive rejection of an archive member. Everything thrown as TarError is presented
 *  as "the archive did not pass inspection", never as an internal stack. */
class TarError extends Error {}

const BLOCK = 512;

function parseOctal(field: Buffer): number {
  const text = field.toString('latin1').replace(/[\0 ]/g, '');
  return text ? parseInt(text, 8) || 0 : 0;
}

/** File size: octal in ustar, base-256 with the top bit set in the GNU extension. */
function readSize(field: Buffer): number {
  if (field[0] & 0x80) {
    let value = 0n;
    for (let i = 0; i < field.length; i++) {
      value = (value << 8n) | BigInt(i === 0 ? field[0] & 0x7f : field[i]);
    }
    if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new TarError('an entry declares a size this build cannot represent');
    }
    return Number(value);
  }
  return parseOctal(field);
}

function readText(field: Buffer): string {
  const nul = field.indexOf(0);
  return field.toString('utf8', 0, nul === -1 ? field.length : nul);
}

function checkHeader(header: Buffer): void {
  const stored = parseOctal(header.subarray(148, 156));
  let sum = 0;
  for (let i = 0; i < BLOCK; i++) {
    sum += i >= 148 && i < 156 ? 0x20 : header[i];
  }
  if (sum !== stored) {
    throw new TarError('a tar header checksum failed — the archive is corrupt');
  }
}

/** Parse a PAX extended-header payload for the two keys we honour (`path`, `linkpath`).
 *  Records are `"<len> <key>=<value>\n"` with <len> including itself. */
function parsePax(payload: Buffer): { path?: string; linkpath?: string } {
  const out: { path?: string; linkpath?: string } = {};
  let i = 0;
  while (i < payload.length) {
    const space = payload.indexOf(0x20, i);
    if (space === -1) {
      break;
    }
    const length = parseInt(payload.toString('latin1', i, space), 10);
    if (!Number.isInteger(length) || length < 3 || i + length > payload.length) {
      break;
    }
    const record = payload.toString('utf8', space + 1, i + length).replace(/\n$/, '');
    const eq = record.indexOf('=');
    if (eq > 0) {
      const key = record.slice(0, eq);
      const value = record.slice(eq + 1);
      if (key === 'path') {
        out.path = value;
      } else if (key === 'linkpath') {
        out.linkpath = value;
      }
    }
    i += length;
  }
  return out;
}

function hasDriveOrAbsolute(rel: string): boolean {
  return path.posix.isAbsolute(rel) || /^[A-Za-z]:/.test(rel);
}

/** Validate and normalize one member path: must live under the single expected root
 *  directory, never be absolute, never contain `..` or backslashes (a backslash is a path
 *  separator on Windows and an ordinary character elsewhere — the ambiguity is enough to
 *  refuse). Answers the path RELATIVE to the root, '' for the root itself. */
function checkedRel(name: string, entryRoot: string): string {
  if (name.includes('\0')) {
    throw new TarError('an entry name contains NUL');
  }
  const trimmed = name.replace(/\/+$/, '');
  if (trimmed === entryRoot) {
    return '';
  }
  if (!trimmed.startsWith(`${entryRoot}/`)) {
    throw new TarError(`entry "${name}" is not under the expected root "${entryRoot}/"`);
  }
  const rel = trimmed.slice(entryRoot.length + 1);
  if (!rel) {
    return '';
  }
  if (hasDriveOrAbsolute(rel) || hasDriveOrAbsolute(trimmed)) {
    throw new TarError(`entry "${name}" is absolute`);
  }
  if (rel.includes('\\')) {
    throw new TarError(`entry "${name}" contains a backslash`);
  }
  if (rel.split('/').some((segment) => segment === '..')) {
    throw new TarError(`entry "${name}" would escape the extraction root`);
  }
  return rel;
}

/** A symlink is kept only when its target is relative AND resolves inside the root. */
function checkedLinkTarget(memberRel: string, target: string): void {
  if (!target || target.includes('\0') || target.includes('\\')) {
    throw new TarError(`symlink "${memberRel}" has an unacceptable target`);
  }
  if (hasDriveOrAbsolute(target)) {
    throw new TarError(`symlink "${memberRel} -> ${target}" is absolute`);
  }
  const resolved = path.posix.normalize(path.posix.join(path.posix.dirname(memberRel), target));
  if (resolved === '..' || resolved.startsWith('../')) {
    throw new TarError(`symlink "${memberRel} -> ${target}" would escape the extraction root`);
  }
}

export type ExtractResult =
  | { ok: true; entries: number; bytes: number }
  | { ok: false; message: string; detail: string };

export interface ExtractOptions {
  archive: string;
  /** A unique staging directory. Created here, removed here on failure, populated only by
   *  entries that passed inspection. */
  destDir: string;
  /** Where the interpreter must exist after extraction, e.g. `bin/python3`. Its absence is a
   *  failure even when every entry passed inspection — an archive of something else that
   *  happens to be well-formed is still not the runtime. */
  expectedInterpreter: string;
  /** The archive's single allowed top-level directory. Defaults to what the pinned
   *  python-build-standalone assets actually contain. */
  entryRoot?: string;
  /** Extraction ceiling, guarding against an uncompressed bomb: the pinned flavour extracts
   *  to roughly 120 MB. */
  maxTotalBytes?: number;
}

const DEFAULT_MAX_EXTRACT_BYTES = 1024 * 1024 * 1024;

/** Unpack the pinned runtime tarball into `options.destDir`, inspecting every entry first.
 *
 * Inspection and writing are per-member sequential, which makes the security property easy
 * to state: an entry is on disk only if it passed checkedRel/checkedLinkTarget, and the
 * first bad entry aborts the whole archive and removes the staging directory. */
export async function extractPythonRuntime(options: ExtractOptions): Promise<ExtractResult> {
  const entryRoot = options.entryRoot ?? 'python';
  const maxTotal = options.maxTotalBytes ?? DEFAULT_MAX_EXTRACT_BYTES;
  const destDir = options.destDir;
  fs.mkdirSync(destDir, { recursive: true });

  let entries = 0;
  let totalBytes = 0;
  let longName: string | undefined;
  let longLink: string | undefined;
  let paxPath: string | undefined;
  let paxLink: string | undefined;
  let sawEnd = false;

  const writeEntry = (
    name: string,
    typeflag: number,
    mode: number,
    linkname: string,
    content: Buffer,
  ): void => {
    const rel = checkedRel(name, entryRoot);
    const target = path.join(destDir, ...rel.split('/'));
    if (typeflag === 0x35) {
      // '5' — directory.
      fs.mkdirSync(target, { recursive: true });
      if (mode) {
        fs.chmodSync(target, mode & 0o777);
      }
      entries++;
      return;
    }
    if (rel === '' && typeflag !== 0x35) {
      throw new TarError(`entry "${name}" tries to overwrite the root itself`);
    }
    if (typeflag === 0x30 || typeflag === 0x00) {
      // '0'/'\0' — regular file.
      totalBytes += content.length;
      if (totalBytes > maxTotal) {
        throw new TarError(`extraction exceeded the ${maxTotal} byte ceiling`);
      }
      fs.mkdirSync(path.dirname(target), { recursive: true });
      fs.writeFileSync(target, content);
      if (mode) {
        fs.chmodSync(target, mode & 0o777);
      }
      entries++;
      return;
    }
    if (typeflag === 0x32) {
      // '2' — symlink. The target is stored verbatim (relative), only after proving it
      // resolves inside the root; extraction creates no links that leave staging.
      checkedLinkTarget(rel, linkname);
      fs.mkdirSync(path.dirname(target), { recursive: true });
      fs.rmSync(target, { force: true });
      fs.symlinkSync(linkname, target);
      entries++;
      return;
    }
    const types: Record<number, string> = {
      0x31: 'a hard link',
      0x33: 'a character device',
      0x34: 'a block device',
      0x36: 'a fifo',
    };
    throw new TarError(`entry "${name}" is ${types[typeflag] ?? `type ${String.fromCharCode(typeflag)}`}, which this runtime never contains`);
  };

  /** Consume as many complete tar records as `pending` holds; answer the remainder. */
  const consume = (buffer: Buffer): Buffer => {
    let pending = buffer;
    for (;;) {
      if (sawEnd) {
        return Buffer.alloc(0);
      }
      if (pending.length < BLOCK) {
        return pending;
      }
      const header = pending.subarray(0, BLOCK);
      if (header.every((byte) => byte === 0)) {
        // The end marker. Padding to a 10 KiB record may follow; all of it stays unconsumed.
        sawEnd = true;
        return Buffer.alloc(0);
      }
      checkHeader(header);
      const size = readSize(header.subarray(124, 136));
      const whole = BLOCK + Math.ceil(size / BLOCK) * BLOCK;
      if (pending.length < whole) {
        return pending;
      }
      const payload = pending.subarray(BLOCK, BLOCK + size);
      const typeflag = header[156];
      const linkname = readText(header.subarray(157, 257));
      const magic = header.subarray(257, 263).toString('latin1');
      let name = readText(header.subarray(0, 100));
      if (magic.startsWith('ustar')) {
        const prefix = readText(header.subarray(345, 500));
        if (prefix) {
          name = `${prefix}/${name}`;
        }
      }
      pending = pending.subarray(whole);

      if (typeflag === 0x4c) {
        // 'L' — GNU long name; applies to the next member header.
        longName = payload.toString('utf8').replace(/\0+$/, '');
        continue;
      }
      if (typeflag === 0x4b) {
        // 'K' — GNU long link name.
        longLink = payload.toString('utf8').replace(/\0+$/, '');
        continue;
      }
      if (typeflag === 0x78) {
        // 'x' — PAX extended header; only path/linkpath are honoured.
        const pax = parsePax(payload);
        paxPath = pax.path;
        paxLink = pax.linkpath;
        continue;
      }
      if (typeflag === 0x67) {
        // 'g' — a PAX global header, which would apply to every member. We honour no
        // globals, so one we cannot ignore is refused instead of half-applied.
        throw new TarError('the archive uses PAX global headers, which are not accepted');
      }
      const finalName = paxPath ?? longName ?? name;
      const finalLink = paxLink ?? longLink ?? linkname;
      longName = longLink = paxPath = paxLink = undefined;
      writeEntry(finalName, typeflag, parseOctal(header.subarray(100, 108)), finalLink, payload);
    }
  };

  try {
    const inflated = fs.createReadStream(options.archive).pipe(zlib.createGunzip());
    let pending: Buffer = Buffer.alloc(0);
    for await (const chunk of inflated) {
      pending = consume(pending.length ? Buffer.concat([pending, chunk as Buffer]) : (chunk as Buffer));
    }
    if (!sawEnd) {
      throw new TarError(
        pending.length > 0
          ? 'the archive ended mid-entry — it is truncated'
          : 'the archive has no end marker — it is truncated',
      );
    }
  } catch (err) {
    // The staging tree exists to be deletable; nothing in it has been promoted anywhere.
    fs.rmSync(destDir, { recursive: true, force: true });
    const text = err instanceof TarError ? err.message : String(err);
    return {
      ok: false,
      message: 'The runtime archive could not be unpacked safely.',
      detail: `${options.archive}\n\n${text}`,
    };
  }

  const interpreter = path.join(destDir, ...options.expectedInterpreter.split('/'));
  if (!fs.existsSync(interpreter)) {
    fs.rmSync(destDir, { recursive: true, force: true });
    return {
      ok: false,
      message: 'The runtime archive did not contain the expected interpreter.',
      detail: `${options.archive} has no ${options.expectedInterpreter} after extraction.`,
    };
  }
  return { ok: true, entries, bytes: totalBytes };
}

/** Run the freshly extracted interpreter once to prove it is what the manifest pins. Called
 *  before any swap, so a corrupt-but-extractable tree never becomes the live runtime. */
export async function verifyRuntimeInterpreter(
  interpreter: string,
  expectedVersion: string,
  timeoutMs = 30_000,
): Promise<{ ok: true } | { ok: false; why: string }> {
  return new Promise((resolve) => {
    const child = spawn(
      interpreter,
      ['-c', 'import sys; print(".".join(map(str, sys.version_info[:3])))'],
      { stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true },
    );
    let stdout = '';
    let stderr = '';
    let settled = false;
    const finish = (why: string | undefined): void => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      resolve(why ? { ok: false, why } : { ok: true });
    };
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      finish(`no answer within ${timeoutMs / 1000}s`);
    }, timeoutMs);
    timer.unref?.();
    child.stdout?.on('data', (b: Buffer) => (stdout += b.toString()));
    child.stderr?.on('data', (b: Buffer) => (stderr += b.toString()));
    child.on('error', (err) => finish(err.message));
    child.on('close', (code) => {
      if (code !== 0) {
        finish(stderr.trim().split('\n').pop() || `exit ${code}`);
        return;
      }
      const got = stdout.trim();
      finish(
        got === expectedVersion
          ? undefined
          : `the extracted interpreter reports Python ${got}, expected ${expectedVersion}`,
      );
    });
  });
}
