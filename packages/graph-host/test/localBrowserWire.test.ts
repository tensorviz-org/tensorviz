import * as assert from 'node:assert/strict';
import { gzipSync } from 'node:zlib';
import { decodeLocalBrowserMessage as decode, encodeLocalBrowserMessage as encode,
  MAX_DECODED_BYTES, MAX_WIRE_BYTES } from '@tensorviz/graph-host/localBrowserWire';

suite('bounded local-browser wire encoding', function () {
  this.timeout(15_000);   // multi-megabyte graphs are compressed and checked byte by byte
  test('decodes the Python-produced cross-language fixture', () => {
    const fixture = require('../../../tests/fixtures/contracts/local-browser-wire.json');
    assert.deepEqual(decode(fixture.wire), fixture.message);
  });
  test('keeps small messages plain and preserves a graph larger than the frame limit', () => {
    const small = { type: 'ack', label: '入力 👋' };
    assert.deepEqual(JSON.parse(encode(small)), small);
    const large = { nodes: Array(1_100_000).fill(small) };
    assert.ok(Buffer.byteLength(JSON.stringify(large)) > 32 * 1024 * 1024, "Full model graphs exceed the former decoded limit");
    const encoded = encode(large);
    assert.ok(Buffer.byteLength(encoded) < MAX_WIRE_BYTES);
    assert.deepEqual(decode(encoded), large);
  });
  test('bounds inflated output as well as UTF-8 frame bytes', () => {
    assert.throws(() => decode(JSON.stringify('界'.repeat(MAX_WIRE_BYTES / 3 + 1))));
    const data = gzipSync(JSON.stringify('x'.repeat(MAX_DECODED_BYTES))).toString('base64');
    assert.throws(() => decode(JSON.stringify({ encoding: 'tensorviz-gzip-json-v1', data })));
    assert.throws(() => encode('x'.repeat(MAX_DECODED_BYTES)));
  });
  test('refuses corrupt, truncated and unknown wrappers', () => {
    for (const message of [
      { encoding: 'unknown', data: '' }, { encoding: 'tensorviz-gzip-json-v1', data: '!' },
      { encoding: 'tensorviz-gzip-json-v1', data: gzipSync('{}').subarray(0, -4).toString('base64') },
    ]) assert.throws(() => decode(JSON.stringify(message)));
  });
});
