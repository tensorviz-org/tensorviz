/** Lossless transport only: protocol, identity and role checks still run after decoding. */
import { gzipSync, gunzipSync } from 'node:zlib';

/** Fits Publish's graph file, gzipped in base64 and up to about 31 MiB on the wire. */
export const MAX_WIRE_BYTES = 32 * 1024 * 1024;
export const MAX_DECODED_BYTES = 96 * 1024 * 1024;
const COMPRESSION_THRESHOLD = 256 * 1024;
const ENCODING = 'tensorviz-gzip-json-v1';

export function encodeLocalBrowserMessage(message: unknown): string {
  const raw = Buffer.from(JSON.stringify(message), 'utf8');
  if (raw.length > MAX_DECODED_BYTES) throw new Error('Decoded message budget exceeded.');
  const text = raw.length < COMPRESSION_THRESHOLD ? raw.toString('utf8') : JSON.stringify({
    encoding: ENCODING, data: gzipSync(raw, { level: 1 }).toString('base64'),
  });
  if (Buffer.byteLength(text, 'utf8') > MAX_WIRE_BYTES) throw new Error('Wire message budget exceeded.');
  return text;
}

export function decodeLocalBrowserMessage(text: string): unknown {
  if (Buffer.byteLength(text, 'utf8') > MAX_WIRE_BYTES) throw new Error('Wire message budget exceeded.');
  const message = JSON.parse(text);
  if (!message || typeof message !== 'object' || !('encoding' in message)) return message;
  if (Object.keys(message).length !== 2 || message.encoding !== ENCODING || typeof message.data !== 'string') {
    throw new Error('Invalid wire encoding.');
  }
  const bytes = Buffer.from(message.data, 'base64');
  if (bytes.toString('base64') !== message.data) throw new Error('Invalid base64 message.');
  return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(
    gunzipSync(bytes, { maxOutputLength: MAX_DECODED_BYTES })));
}
