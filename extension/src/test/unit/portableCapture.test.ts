import * as assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as path from 'node:path';
import { preparePortableCapture, PORTABLE_CAPTURE_MAX_BYTES } from '@tensorviz/graph-host/portableCapture';
import { canonicalFixture } from '../../../../tests/fixtures/semantic.cjs';

const fixture = () => JSON.parse(readFileSync(path.join(__dirname, '../../../../tests/fixtures/contracts/portable-capture.json'), 'utf8'));
function prepare(fetchImpl: typeof fetch, signal = new AbortController().signal) {
  return preparePortableCapture(canonicalFixture().capture, { origin: 'http://localhost:9000', token: 'SECRET', signal, fetchImpl });
}

suite('portable preparation boundary', () => {
  test('only posts the retained capture to authenticated local preparation and preserves exact evidence', async () => {
    const value = await prepare(async (url, init) => {
      assert.equal(url, 'http://localhost:9000/api/ide/capture/portable');
      assert.equal(init?.redirect, 'error');
      assert.equal((init?.headers as Record<string, string>)['X-tensorViz-Token'], 'SECRET');
      assert.deepEqual(JSON.parse(init!.body as string), canonicalFixture().capture);
      return new Response(JSON.stringify(fixture()));
    });
    assert.deepEqual(value, fixture());
  });

  test('refuses altered computation or source aliases even when the schema is satisfied', async () => {
    for (const change of [
      (value: any) => { value.capture.operations[0].arguments.args[1].value = '2'; },
      (value: any) => { value.capture.source.files[0].path = 'private/model.py'; },
      (value: any) => { value.capture.source.configDigest = 'a'.repeat(64); },
      (value: any) => { value.capture.id = 'another-capture'; },
    ]) {
      const value = fixture(); change(value);
      await assert.rejects(prepare(async () => new Response(JSON.stringify(value))), /refused/i);
    }
  });

  test('counts streamed UTF-8 bytes, cancels oversize bodies and refuses malformed responses without echoing them', async () => {
    let cancelled = false;
    const stream = new ReadableStream<Uint8Array>({
      start(controller) { controller.enqueue(new Uint8Array(PORTABLE_CAPTURE_MAX_BYTES + 1)); },
      cancel() { cancelled = true; },
    });
    await assert.rejects(prepare(async () => new Response(stream)), /Nothing was truncated/);
    assert.equal(cancelled, true);
    for (const status of [200, 400, 413, 500]) {
      await assert.rejects(prepare(async () => new Response('PRIVATE_ERROR', { status })), error => {
        assert.ok(error instanceof Error && !error.message.includes('PRIVATE_ERROR'));
        return true;
      });
    }
  });

  test('aborted preparation never delivers a late valid result', async () => {
    const controller = new AbortController();
    const result = prepare(async (_url, init) => {
      controller.abort();
      assert.equal(init!.signal!.aborted, true);
      return new Response(JSON.stringify(fixture()));
    }, controller.signal);
    await assert.rejects(result, /cancelled/);
  });
});
