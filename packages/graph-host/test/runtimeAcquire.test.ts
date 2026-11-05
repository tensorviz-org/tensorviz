import * as assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { Readable } from 'node:stream';

import {
  Transport,
  TransportResponse,
  downloadVerified,
  DOWNLOAD_ATTEMPTS,
  extractPythonRuntime,
  verifyRuntimeInterpreter,
} from '@tensorviz/graph-host/runtimeAcquire';
import { archive, gz, Member } from '../../../tests/fixtures/tar.cjs';

let temp: string;

setup(() => {
  temp = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'tensorviz-acquire-'));
});

teardown(() => {
  fs.rmSync(temp, { recursive: true, force: true });
});

/** A small but REAL-shaped runtime tree: root dir, bin/, lib/, an interpreter stubfile, an
 *  in-root symlink (python-build-standalone ships bin/python3 -> python3.12), a GNU long
 *  name, a ustar prefix split, and a PAX path header. */
function goodArchive(): Buffer {
  const longDir = 'python/lib/python3.12/site-packages/' + 'averylongmodulename'.repeat(7);
  const longName = `${longDir}/module.py`;
  const paxTarget = 'python/share/licenses/' + 'some-library-with-a-long-name'.repeat(4) + '/LICENSE';
  // PAX length includes its own digits, the space, the record and the trailing newline;
  // grow it until the arithmetic is self-consistent.
  let paxLen = paxTarget.length + 7;
  while (`${paxLen} path=${paxTarget}\n`.length !== paxLen) {
    paxLen++;
  }
  const paxPayload = Buffer.from(`${paxLen} path=${paxTarget}\n`);
  return archive([
    { name: 'python/', type: '5', mode: 0o755 },
    { name: 'python/bin', type: '5', mode: 0o755 },
    { name: 'python/lib', type: '5', mode: 0o755 },
    { name: 'python/bin/python3.12', mode: 0o755, data: Buffer.from('#!/bin/sh\necho 3.12.14\n') },
    { name: 'python/bin/python3', type: '2', linkname: 'python3.12', mode: 0o777 },
    { name: 'python/lib/libpython.dylib', type: '2', linkname: '../bin/python3.12', mode: 0o777 },
    { name: './LongName', type: 'L', data: Buffer.from(longName) },
    { name: longName.slice(0, 40), mode: 0o644, data: Buffer.from('# long\n') },
    { name: 'module.py', prefix: 'python/lib/prefixsplit', mode: 0o644, data: Buffer.from('# prefix\n') },
    { name: 'PaxHeader', type: 'x', data: paxPayload },
    { name: 'short', mode: 0o644, data: Buffer.from('# pax\n') },
  ]);
}

function writeFixture(plain: Buffer, name = 'runtime.tar.gz'): string {
  const file = path.join(temp, name);
  fs.writeFileSync(file, gz(plain));
  return file;
}

function staging(name: string): string {
  return path.join(temp, name);
}

const EXPECTED = 'bin/python3';

suite('runtimeAcquire.extractPythonRuntime', () => {
  test('extracts a well-formed archive: files, modes, symlinks, long names, pax paths', async function () {
    if (process.platform === 'win32') {
      this.skip(); // symlink creation needs privileges Windows CI should not assume
    }
    const dest = staging('ok');
    const result = await extractPythonRuntime({
      archive: writeFixture(goodArchive()),
      destDir: dest,
      expectedInterpreter: EXPECTED,
    });

    assert.equal(result.ok, true);
    if (!result.ok) return;
    assert.ok(result.entries >= 9);

    const interpreter = path.join(dest, 'bin', 'python3');
    assert.ok(fs.lstatSync(interpreter).isSymbolicLink(), 'the in-root symlink is preserved');
    assert.equal(fs.readlinkSync(interpreter), 'python3.12');
    assert.equal(fs.lstatSync(path.join(dest, 'lib', 'libpython.dylib')).isSymbolicLink(), true);
    const mode = fs.statSync(path.join(dest, 'bin', 'python3.12')).mode;
    assert.equal(mode & 0o111, 0o111, 'the interpreter keeps its executable bit');
    const longDir = path.join(dest, 'lib', 'python3.12', 'site-packages');
    assert.ok(fs.readdirSync(longDir).length === 1);
    assert.ok(fs.existsSync(path.join(dest, 'lib', 'prefixsplit', 'module.py')));
    assert.ok(
      fs.existsSync(path.join(dest, 'share', 'licenses', 'some-library-with-a-long-name'.repeat(4), 'LICENSE')),
      'PAX path= was honoured',
    );
  });

  const refuses: Array<[string, Member[], RegExp]> = [
    ['an absolute path', [{ name: '/etc/passwd', data: Buffer.from('x') }], /expected root/],
    ['a drive-letter path', [{ name: 'python/C:/windows/x', data: Buffer.from('x') }], /absolute/],
    ['a .. traversal', [{ name: 'python/bin/../../evil', data: Buffer.from('x') }], /escape the extraction root/],
    ['a backslash path', [{ name: 'python\\bin\\x', data: Buffer.from('x') }], /expected root/],
    ['an unexpected root', [{ name: 'other/x', data: Buffer.from('x') }], /expected root/],
    ['a device entry', [{ name: 'python/dev/null', type: '3', data: Buffer.alloc(0) }], /character device/],
    ['a fifo entry', [{ name: 'python/tmp/pipe', type: '6', data: Buffer.alloc(0) }], /fifo/],
    ['a hard link', [{ name: 'python/bin/x', type: '1', linkname: 'bin/python3' }], /hard link/],
  ];

  for (const [index, [what, members, pattern]] of refuses.entries()) {
    test(`refuses ${what}`, async () => {
      const dest = staging(`refuse-${index}`);
      const result = await extractPythonRuntime({
        archive: writeFixture(archive(members)),
        destDir: dest,
        expectedInterpreter: EXPECTED,
      });
      assert.equal(result.ok, false, what);
      if (result.ok) return;
      assert.match(result.detail, pattern, what);
      assert.ok(!fs.existsSync(dest), 'nothing from a refused archive remains on disk');
    });
  }

  test('refuses a symlink whose target is absolute', async () => {
    const dest = staging('link-abs');
    const result = await extractPythonRuntime({
      archive: writeFixture(archive([{ name: 'python/bin/x', type: '2', linkname: '/etc/passwd' }])),
      destDir: dest,
      expectedInterpreter: EXPECTED,
    });
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.match(result.detail, /is absolute/);
    assert.ok(!fs.existsSync(dest));
  });

  test('refuses a symlink whose target resolves outside the root', async () => {
    const dest = staging('link-escape');
    const result = await extractPythonRuntime({
      archive: writeFixture(archive([{ name: 'python/lib/x', type: '2', linkname: '../../../etc' }])),
      destDir: dest,
      expectedInterpreter: EXPECTED,
    });
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.match(result.detail, /escape the extraction root/);
    assert.ok(!fs.existsSync(dest));
  });

  test('refuses a truncated archive and cleans staging', async () => {
    const dest = staging('truncated');
    const cut = goodArchive().subarray(0, 1300); // ends inside a member
    const result = await extractPythonRuntime({
      archive: writeFixture(cut),
      destDir: dest,
      expectedInterpreter: EXPECTED,
    });
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.match(result.detail, /truncated/);
    assert.ok(!fs.existsSync(dest));
  });

  test('refuses a corrupt header via its checksum', async () => {
    const broken = Buffer.from(goodArchive());
    broken[200] ^= 0xff; // inside the first header's uid field: changes the sum
    const dest = staging('corrupt');
    const result = await extractPythonRuntime({
      archive: writeFixture(broken),
      destDir: dest,
      expectedInterpreter: EXPECTED,
    });
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.match(result.detail, /checksum/);
    assert.ok(!fs.existsSync(dest));
  });

  test('an archive that is valid but not the runtime is rejected by the interpreter check', async () => {
    const dest = staging('no-interp');
    const result = await extractPythonRuntime({
      archive: writeFixture(archive([{ name: 'python/README.txt', data: Buffer.from('hi\n') }])),
      destDir: dest,
      expectedInterpreter: EXPECTED,
    });
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.match(result.message, /did not contain the expected interpreter/);
    assert.ok(!fs.existsSync(dest));
  });

  test('the extraction ceiling stops an uncompressed bomb', async () => {
    const dest = staging('bomb');
    const result = await extractPythonRuntime({
      archive: writeFixture(goodArchive()),
      destDir: dest,
      expectedInterpreter: EXPECTED,
      maxTotalBytes: 10,
    });
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.match(result.detail, /ceiling/);
    assert.ok(!fs.existsSync(dest));
  });

  test('PAX global headers are refused rather than half-applied', async () => {
    const dest = staging('pax-global');
    const result = await extractPythonRuntime({
      archive: writeFixture(archive([{ name: 'GlobalHead', type: 'g', data: Buffer.from('12 a=b\n') }])),
      destDir: dest,
      expectedInterpreter: EXPECTED,
    });
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.match(result.detail, /global headers/);
  });

  test('the extracted interpreter is run and version-checked before any swap', async function () {
    if (process.platform === 'win32') {
      this.skip();
    }
    const dest = staging('verify');
    const extracted = await extractPythonRuntime({
      archive: writeFixture(goodArchive()),
      destDir: dest,
      expectedInterpreter: 'bin/python3.12',
    });
    assert.equal(extracted.ok, true);

    const good = await verifyRuntimeInterpreter(path.join(dest, 'bin', 'python3.12'), '3.12.14');
    assert.equal(good.ok, true);
    const wrong = await verifyRuntimeInterpreter(path.join(dest, 'bin', 'python3.12'), '3.11.9');
    assert.equal(wrong.ok, false);
    if (wrong.ok) return;
    assert.match(wrong.why, /expected 3\.11\.9/);
  });
});

// ---------------------------------------------------------------------------
// Downloads: a fake transport maps URLs to fabricated responses. No network.
// ---------------------------------------------------------------------------

function fakeTransport(routes: Record<string, (url: string) => TransportResponse>): {
  transport: Transport;
  calls: string[];
} {
  const calls: string[] = [];
  return {
    calls,
    transport: async (url) => {
      calls.push(url);
      // Longest prefix first, so a redirect target that extends a route can't re-match the
      // original shorter prefix and loop.
      const handler = Object.entries(routes)
        .sort(([a], [b]) => b.length - a.length)
        .find(([prefix]) => url.startsWith(prefix));
      if (!handler) {
        return { status: 404, headers: {}, stream: Readable.from([]) };
      }
      return handler[1](url);
    },
  };
}

function body(...chunks: string[]): TransportResponse {
  return {
    status: 200,
    headers: { 'content-length': String(Buffer.concat(chunks.map(Buffer.from)).length) },
    stream: Readable.from(chunks.map(Buffer.from)),
  };
}

function redirect(to: string): TransportResponse {
  return { status: 302, headers: { location: to }, stream: Readable.from([]) };
}

const PINNED = createHash('sha256').update('the runtime bytes').digest('hex');
const GOOD = 'https://github.com/astral-sh/python-build-standalone/releases/download/20260901/asset.tar.gz';

function destPath(name = 'asset.tar.gz'): string {
  return path.join(temp, 'cache', name);
}

suite('runtimeAcquire.downloadVerified', () => {
  test('a correct archive streams to .part, is hashed in flight, and is promoted', async () => {
    const { transport, calls } = fakeTransport({
      [GOOD]: () => {
        const payload = 'the runtime bytes';
        return {
          status: 200,
          headers: { 'content-length': String(payload.length) },
          stream: Readable.from([Buffer.from('the run'), Buffer.from('time bytes')]),
        };
      },
    });
    const dest = destPath();
    const result = await downloadVerified({
      url: GOOD,
      sha256: PINNED,
      maxBytes: 1024,
      destPath: dest,
      transport,
    });

    assert.equal(result.ok, true);
    if (!result.ok) return;
    assert.equal(fs.readFileSync(dest, 'utf8'), 'the runtime bytes');
    assert.ok(!fs.existsSync(`${dest}.part`), 'no partial file survives success');
    assert.equal(calls.length, 1);
  });

  test('reports the total and throttled progress before eight MiB, then the final byte count', async () => {
    const chunk = Buffer.alloc(64 * 1024, 'r');
    const chunks = Array.from({ length: 48 }, () => chunk);
    const payload = Buffer.concat(chunks);
    const progress: string[] = [];
    const result = await downloadVerified({
      url: GOOD, sha256: createHash('sha256').update(payload).digest('hex'),
      maxBytes: payload.length, destPath: destPath(), report: message => progress.push(message),
      transport: async () => ({ status: 200,
        headers: { 'content-length': String(payload.length) }, stream: Readable.from(chunks) }),
    });
    assert.equal(result.ok, true);
    const counts = progress.filter(message => message.startsWith('Downloading the tensorViz runtime…'));
    assert.equal(counts[0], 'Downloading the tensorViz runtime… 0 B / 3.1 MB');
    assert.ok(counts.some(message => /… [12]\.\d MB \/ 3\.1 MB$/.test(message)),
      'the user sees intermediate progress for a download smaller than eight MiB');
    assert.equal(counts.at(-1), 'Downloading the tensorViz runtime… 3.1 MB / 3.1 MB');
    assert.ok(counts.length < chunks.length / 2, 'progress is throttled rather than emitted for every chunk');
    assert.equal(progress.at(-1), 'Download verified (sha256 ok).');
  });

  for (const [label, length] of [
    ['missing', undefined], ['malformed', 'not-a-size'], ['zero', '0'],
    ['negative', '-17'], ['multiple', ['17', '17']],
  ] as [string, string | string[] | undefined][]) {
    test(`reports received bytes without inventing a total for a ${label} Content-Length`, async () => {
      const progress: string[] = [];
      const result = await downloadVerified({ url: GOOD, sha256: PINNED, maxBytes: 1024,
        destPath: destPath(), report: message => progress.push(message),
        transport: async () => ({ status: 200,
          headers: { 'content-length': length },
          stream: Readable.from([Buffer.from('the runtime bytes')]) }),
      });
      assert.equal(result.ok, true);
      assert.deepEqual(progress.filter(message => message.startsWith('Downloading the tensorViz runtime…')), [
        'Downloading the tensorViz runtime… 0 B',
        'Downloading the tensorViz runtime… 17 B',
      ]);
    });
  }

  test('stops displaying an announced total when more bytes actually arrive', async () => {
    const progress: string[] = [];
    const result = await downloadVerified({ url: GOOD, sha256: PINNED, maxBytes: 1024,
      destPath: destPath(), report: message => progress.push(message),
      transport: async () => ({ status: 200, headers: { 'content-length': '10' },
        stream: Readable.from([Buffer.from('the runtime bytes')]) }),
    });
    assert.equal(result.ok, true);
    assert.ok(progress.includes('Downloading the tensorViz runtime… 0 B / 10 B'));
    assert.ok(progress.includes('Downloading the tensorViz runtime… 17 B'));
  });

  test('reports slow transfers while fewer than one MiB have arrived', async function () {
    this.timeout(5000);
    const payload = Buffer.alloc(2048, 'r');
    const progress: { message: string; transferring: boolean }[] = [];
    let transferring = true;
    const result = await downloadVerified({
      url: GOOD, sha256: createHash('sha256').update(payload).digest('hex'),
      maxBytes: payload.length, destPath: destPath(),
      report: message => progress.push({ message, transferring }),
      transport: async () => ({ status: 200, headers: { 'content-length': String(payload.length) },
        stream: Readable.from((async function* () {
          yield payload.subarray(0, 1024);
          await new Promise(resolve => setTimeout(resolve, 1100));
          yield payload.subarray(1024);
          transferring = false;
        })()) }),
    });
    assert.equal(result.ok, true);
    assert.ok(progress.some(item => item.transferring
      && item.message === 'Downloading the tensorViz runtime… 2.0 KB / 2.0 KB'));
  });

  test('a checksum mismatch deletes everything and prints both digests', async () => {
    const { transport, calls } = fakeTransport({ [GOOD]: () => body('swapped bytes') });
    const dest = destPath();
    const result = await downloadVerified({
      url: GOOD, sha256: PINNED, maxBytes: 1024, destPath: dest, transport,
    });
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.match(result.message, /checksum/);
    assert.match(result.detail, /expected/);
    assert.match(result.detail, /got/);
    assert.ok(!fs.existsSync(dest));
    assert.ok(!fs.existsSync(`${dest}.part`));
    assert.equal(calls.length, 1, 'integrity failure must never be retried');
  });

  test('an announced size over the ceiling is refused before a byte is written', async () => {
    const { transport } = fakeTransport({
      [GOOD]: () => ({ status: 200, headers: { 'content-length': '999999' }, stream: Readable.from([]) }),
    });
    const dest = destPath();
    const result = await downloadVerified({
      url: GOOD, sha256: PINNED, maxBytes: 100, destPath: dest, transport,
    });
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.match(result.message, /larger than the pinned ceiling/);
    assert.ok(!fs.existsSync(`${dest}.part`));
  });

  test('a stream that grows past the ceiling mid-flight is refused and cleaned', async () => {
    const { transport } = fakeTransport({
      [GOOD]: () => ({ status: 200, headers: {}, stream: Readable.from([Buffer.alloc(200)]) }),
    });
    const dest = destPath();
    const result = await downloadVerified({
      url: GOOD, sha256: PINNED, maxBytes: 100, destPath: dest, transport,
    });
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.match(result.message, /exceeded the pinned size ceiling/);
    assert.ok(!fs.existsSync(`${dest}.part`));
  });

  test('non-HTTPS URLs are refused without a request', async () => {
    const { transport, calls } = fakeTransport({});
    const result = await downloadVerified({
      url: 'http://example.com/x.tar.gz', sha256: PINNED, maxBytes: 100, destPath: destPath(), transport,
    });
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.match(result.message, /not HTTPS/);
    assert.equal(calls.length, 0);
  });

  test('redirects to the same host are followed; to HTTP or another host, refused', async () => {
    const destA = destPath('a.tar.gz');
    const { transport: t1, calls: c1 } = fakeTransport({
      [GOOD]: () => redirect(`${GOOD}?sig=1`),
      [`${GOOD}?sig=1`]: () => body('the runtime bytes'),
    });
    const okRedirect = await downloadVerified({
      url: GOOD, sha256: PINNED, maxBytes: 1024, destPath: destA, transport: t1,
    });
    assert.equal(okRedirect.ok, true);
    assert.equal(c1.length, 2);

    const { transport: t2 } = fakeTransport({ [GOOD]: () => redirect('http://github.com/asset') });
    const httpHop = await downloadVerified({
      url: GOOD, sha256: PINNED, maxBytes: 1024, destPath: destPath('b.tar.gz'), transport: t2,
    });
    assert.equal(httpHop.ok, false);
    if (httpHop.ok) return;
    assert.match(httpHop.message, /redirected somewhere untrusted/);
  });

  test('a redirect to an untrusted host never sees a second request', async () => {
    const { transport, calls } = fakeTransport({
      [GOOD]: () => redirect('https://evil.example.net/runtime.tar.gz'),
    });
    const result = await downloadVerified({
      url: GOOD, sha256: PINNED, maxBytes: 1024, destPath: destPath(), transport,
    });
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.match(result.message, /redirected somewhere untrusted/);
    assert.equal(calls.length, 1, 'the redirect target was never fetched');
  });

  test('redirect loops are cut at the bound', async () => {
    const { transport, calls } = fakeTransport({ [GOOD]: () => redirect(GOOD) });
    const result = await downloadVerified({
      url: GOOD, sha256: PINNED, maxBytes: 1024, destPath: destPath(), transport,
    });
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.match(result.message, /too many times/);
    assert.equal(calls.length, 4); // the initial request plus three follows
  });

  test('HTTP errors and connection failures are plain language', async () => {
    const gone = await downloadVerified({
      url: GOOD, sha256: PINNED, maxBytes: 1024, destPath: destPath(),
      transport: async () => ({ status: 404, headers: {}, stream: Readable.from([]) }),
    });
    assert.equal(gone.ok, false);
    if (gone.ok) return;
    assert.match(gone.message, /HTTP 404/);

    const offline = await downloadVerified({
      url: GOOD, sha256: PINNED, maxBytes: 1024, destPath: destPath(),
      transport: async () => {
        throw new Error('getaddrinfo ENOTFOUND');
      },
    });
    assert.equal(offline.ok, false);
    if (offline.ok) return;
    assert.match(offline.message, /could not connect/);
    assert.match(offline.detail, /ENOTFOUND/);
  });

  test('cancellation mid-stream removes the partial file', async () => {
    let sent = 0;
    const { transport } = fakeTransport({
      [GOOD]: () => ({
        status: 200,
        headers: {},
        stream: new Readable({
          read() {
            if (sent++ < 3) {
              this.push(Buffer.from('some bytes'));
            } else {
              this.push(null);
            }
          },
        }),
      }),
    });
    let chunks = 0;
    const dest = destPath();
    const result = await downloadVerified({
      url: GOOD, sha256: PINNED, maxBytes: 1024, destPath: dest, transport,
      isCancelled: () => chunks++ >= 1, // cancel from the second chunk on
    });
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.equal(result.cancelled, true);
    assert.ok(!fs.existsSync(`${dest}.part`));
  });

  test('cancellation interrupts a connection with no response and destroys any late response', async () => {
    const controller = new AbortController();
    let answer!: (response: TransportResponse) => void;
    let requestSignal: AbortSignal | undefined;
    const pending = downloadVerified({ url: GOOD, sha256: PINNED, maxBytes: 1024, destPath: destPath(), signal: controller.signal,
      transport: (_url, signal) => { requestSignal = signal; return new Promise(resolve => { answer = resolve; }); } });
    await new Promise(resolve => setImmediate(resolve));
    controller.abort();
    const result = await pending;
    assert.equal(result.ok, false);
    assert.ok(!result.ok && result.cancelled);
    assert.equal(requestSignal?.aborted, true);
    const late = new Readable({ read() {} });
    answer({ status: 200, headers: {}, stream: late });
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(late.destroyed, true);
    assert.ok(!fs.existsSync(destPath() + '.part'));
  });

  test('cancellation interrupts a stream without another byte and removes its partial file', async () => {
    const controller = new AbortController(), dest = destPath();
    const hanging = new Readable({ read() {} });
    const pending = downloadVerified({ url: GOOD, sha256: PINNED, maxBytes: 1024, destPath: dest, signal: controller.signal,
      transport: async () => ({ status: 200, headers: {}, stream: hanging }) });
    await new Promise(resolve => setTimeout(resolve, 10));
    controller.abort();
    const result = await pending;
    assert.ok(!result.ok && result.cancelled);
    assert.equal(hanging.destroyed, true);
    assert.ok(!fs.existsSync(dest + '.part'));
    assert.ok(!fs.existsSync(dest));
  });

  test('a connection without a response reaches its bound', async () => {
    const result = await downloadVerified({ url: GOOD, sha256: PINNED, maxBytes: 1024, destPath: destPath(), idleTimeoutMs: 10, retryDelayMs: 0,
      transport: () => new Promise(() => {}) });
    assert.ok(!result.ok && result.detail.includes('connection timed out'));
  });

  test('a stalled stream is reported instead of hanging the first-run experience', async () => {
    const hanging = new Readable({ read() {} });
    const { transport } = fakeTransport({
      [GOOD]: () => ({ status: 200, headers: {}, stream: hanging }),
    });
    const result = await downloadVerified({
      url: GOOD, sha256: PINNED, maxBytes: 1024, destPath: destPath(), transport,
      idleTimeoutMs: 30,
      retryDelayMs: 0,
    });
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.match(result.message, /stalled/);
  });

  test('temporary connection failure retries then verifies only the complete successful archive', async () => {
    let calls = 0;
    const progress: string[] = [];
    const result = await downloadVerified({ url: GOOD, sha256: PINNED, maxBytes: 1024,
      destPath: destPath(), retryDelayMs: 0, report: message => progress.push(message),
      transport: async () => {
        if (++calls === 1) throw Object.assign(new Error('socket reset'), { code: 'ECONNRESET' });
        return body('the runtime bytes');
      } });
    assert.ok(result.ok);
    assert.equal(calls, 2);
    assert.match(progress.join('\n'), /retrying \(2\/3\)/);
    assert.equal(fs.readFileSync(destPath(), 'utf8'), 'the runtime bytes');
    assert.ok(!fs.existsSync(destPath() + '.part'));
  });

  test('mid-stream abort restarts from clean bytes rather than promoting a partial archive', async () => {
    let calls = 0;
    const result = await downloadVerified({ url: GOOD, sha256: PINNED, maxBytes: 1024,
      destPath: destPath(), retryDelayMs: 0,
      transport: async () => {
        if (++calls > 1) return body('the runtime bytes');
        return { status: 200, headers: {}, stream: new Readable({ read() {
          this.push(Buffer.from('partial bytes'));
          this.destroy(Object.assign(new Error('aborted'), { code: 'ECONNRESET' }));
        } }) };
      } });
    assert.ok(result.ok);
    assert.equal(calls, 2);
    assert.equal(fs.readFileSync(destPath(), 'utf8'), 'the runtime bytes');
    assert.ok(!fs.existsSync(destPath() + '.part'));
  });

  test('temporary HTTP failures exhaust a fixed retry budget with sanitized diagnostics', async () => {
    let calls = 0;
    const result = await downloadVerified({ url: GOOD + '?signed=private', sha256: PINNED, maxBytes: 1024,
      destPath: destPath(), retryDelayMs: 0,
      transport: async () => { calls++; return { status: 503, headers: {}, stream: Readable.from([]) }; } });
    assert.ok(!result.ok);
    if (result.ok) return;
    assert.equal(calls, DOWNLOAD_ATTEMPTS);
    assert.match(result.message, /HTTP 503/);
    assert.match(result.detail, /3\/3/);
    assert.doesNotMatch(result.detail, /signed|private/);
    assert.ok(!fs.existsSync(destPath() + '.part'));
  });

  test('cancelling during retry wait starts no additional connection', async () => {
    let calls = 0;
    const controller = new AbortController();
    const result = await downloadVerified({ url: GOOD, sha256: PINNED, maxBytes: 1024,
      destPath: destPath(), signal: controller.signal,
      report: message => { if (message.includes('retrying')) controller.abort(); },
      transport: async () => { calls++; throw Object.assign(new Error('timeout'), { code: 'ETIMEDOUT' }); } });
    assert.ok(!result.ok && result.cancelled);
    assert.equal(calls, 1);
  });

  test('certificate errors do not retry and preserve their safe cause', async () => {
    let calls = 0;
    const result = await downloadVerified({ url: GOOD, sha256: PINNED, maxBytes: 1024,
      destPath: destPath(), retryDelayMs: 0,
      transport: async () => { calls++; throw Object.assign(new Error('certificate rejected'), { code: 'CERT_HAS_EXPIRED' }); } });
    assert.ok(!result.ok);
    assert.equal(calls, 1);
    if (!result.ok) assert.match(result.detail, /certificate rejected/);
  });
});
