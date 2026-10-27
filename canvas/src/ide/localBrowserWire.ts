/* GENERATED from packages/graph-host/src/viewWire.ts; npm run protocol:sync. */
/** The browser uses its streaming inflater so a small frame cannot allocate an unbounded document. */
export const MAX_WIRE_BYTES = 16 * 1024 * 1024;
export const MAX_DECODED_BYTES = 96 * 1024 * 1024;
const ENCODING = 'tensorviz-gzip-json-v1';
// CRA's pinned TypeScript DOM library predates this native Chromium API.
declare const DecompressionStream: new (format: 'gzip') => ReadableWritablePair<Uint8Array, Uint8Array>;

export function decodeLocalBrowserMessage(text: string): unknown | Promise<unknown> {
  if (new TextEncoder().encode(text).length > MAX_WIRE_BYTES) throw new Error('Wire message budget exceeded.');
  const message = JSON.parse(text);
  if (!message || typeof message !== 'object' || !('encoding' in message)) return message;
  return decodeCompressed(message);
}

async function decodeCompressed(message: { encoding: unknown; data?: unknown }): Promise<unknown> {
  if (Object.keys(message).length !== 2 || message.encoding !== ENCODING || typeof message.data !== 'string') {
    throw new Error('Invalid wire encoding.');
  }
  const binary = atob(message.data);
  if (btoa(binary) !== message.data) throw new Error('Invalid base64 message.');
  return inflate(Uint8Array.from(binary, char => char.charCodeAt(0)));
}

async function inflate(bytes: Uint8Array): Promise<unknown> {
  const source = new ReadableStream<Uint8Array>({ start(controller) { controller.enqueue(bytes); controller.close(); } });
  const reader = source.pipeThrough(new DecompressionStream('gzip')).getReader();
  const decoder = new TextDecoder('utf-8', { fatal: true });
  const chunks: string[] = [];
  let length = 0;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > MAX_DECODED_BYTES) throw new Error('Decoded message budget exceeded.');
      chunks.push(decoder.decode(value, { stream: true }));
    }
    chunks.push(decoder.decode());
    return JSON.parse(chunks.join(''));
  } finally {
    await reader.cancel();
  }
}
