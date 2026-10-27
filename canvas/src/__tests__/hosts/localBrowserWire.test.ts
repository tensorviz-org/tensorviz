import { gzipSync } from 'node:zlib';
import { decodeLocalBrowserMessage as decode, MAX_DECODED_BYTES, MAX_WIRE_BYTES } from '../../ide/localBrowserWire';

beforeAll(() => {
  Object.assign(global, { DecompressionStream: require('node:stream/web').DecompressionStream });
  global.TextDecoder = require('node:util').TextDecoder;
});

const wrap = (raw: string) => JSON.stringify({ encoding: 'tensorviz-gzip-json-v1', data: gzipSync(raw).toString('base64') });

test('decodes the Python-produced cross-language fixture', async () => {
  const fixture = JSON.parse(require('node:fs').readFileSync(
    require('node:path').resolve(process.cwd(), '../tests/fixtures/contracts/local-browser-wire.json'), 'utf8'));
  expect(await decode(fixture.wire)).toEqual(fixture.message);
});

test('decodes a graph larger than one frame exactly, including Unicode', async () => {
  const message = { nodes: Array(1_100_000).fill({ label: '入力 👋', shape: [1, 3, 64, 64] }) };
  const raw = JSON.stringify(message);
  expect(new TextEncoder().encode(raw).length).toBeGreaterThan(32 * 1024 * 1024);
  const wire = wrap(raw);
  expect(wire.length).toBeLessThan(MAX_WIRE_BYTES);
  expect(await decode(wire)).toEqual(message);
  expect(decode('{"type":"ack"}')).toEqual({ type: 'ack' });
});

test('streaming decode stops at its output bound and rejects corrupt data', async () => {
  await expect(decode(wrap(JSON.stringify('x'.repeat(MAX_DECODED_BYTES))))).rejects.toThrow('Decoded message budget');
  for (const data of ['!', 'eA==', gzipSync('{}').subarray(0, -4).toString('base64')]) {
    await expect(decode(JSON.stringify({ encoding: 'tensorviz-gzip-json-v1', data }))).rejects.toThrow();
  }
  expect(() => decode(JSON.stringify('界'.repeat(MAX_WIRE_BYTES / 3 + 1)))).toThrow('Wire message budget');
  await expect(decode('{"encoding":"unknown","data":""}')).rejects.toThrow('Invalid wire encoding');
});
