const { test } = require('node:test');
const assert = require('node:assert/strict');
const { Backend } = require(process.env.TENSORVIZ_TEST_BACKEND_ADAPTER || '../../out/host/backend.cjs');

function backend() {
  const instance = new Backend('/unused/backend', '/unused/browser');
  instance.origin = 'http://127.0.0.1:1';
  instance.token = 'test';
  return instance;
}

test('a projected large capture may exceed the former 32 MiB service limit', async t => {
  const text = 'x'.repeat(33 * 1024 * 1024);
  t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({ text })));
  const result = await backend().request('/capture', {}, new AbortController().signal);
  assert.equal(result.text.length, text.length);
});

test('the service cancels a stream beyond its bounded 128 MiB response limit', async t => {
  let cancelled = false;
  const chunk = new Uint8Array(1024 * 1024);
  t.mock.method(globalThis, 'fetch', async () => new Response(new ReadableStream({
    pull(controller) { controller.enqueue(chunk); },
    cancel() { cancelled = true; },
  })));
  await assert.rejects(backend().request('/capture', {}, new AbortController().signal),
    error => error.code === 'RESULT_LIMIT');
  assert.equal(cancelled, true);
});
