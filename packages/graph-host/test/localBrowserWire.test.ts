import * as assert from 'node:assert/strict';
import { randomBytes } from 'node:crypto';
import { gzipSync } from 'node:zlib';
import { LOCAL_BROWSER_PUBLISH_FILE_MAX_LENGTH, LOCAL_BROWSER_PUBLISH_MAX_PACKED_BYTES } from '@tensorviz/contracts/protocol';
import { decodeLocalBrowserMessage as decode, encodeLocalBrowserMessage as encode,
  MAX_DECODED_BYTES, MAX_WIRE_BYTES } from '@tensorviz/graph-host/localBrowserWire';

suite('bounded local-browser wire encoding', () => {
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
  test('carries Publish\'s largest graph file in one frame, though gzip cannot shrink it', () => {
    const file = randomBytes(LOCAL_BROWSER_PUBLISH_MAX_PACKED_BYTES).toString('base64');
    assert.ok(file.length <= LOCAL_BROWSER_PUBLISH_FILE_MAX_LENGTH);
    const request = { type: 'graphPublishRequest', protocolVersion: 1, workspaceSessionId: 's'.repeat(36),
      messageId: 'm'.repeat(36), requestId: 'r'.repeat(36), target: { workspaceFolder: '/ws', file: 'net.py' }, file };
    const encoded = encode(request);
    assert.ok(Buffer.byteLength(encoded) <= MAX_WIRE_BYTES);
    assert.deepEqual(decode(encoded), request);
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
