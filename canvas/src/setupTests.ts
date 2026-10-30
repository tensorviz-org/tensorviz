// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';

// JSDOM omits the browser's UTF-8 encoder used by protocol byte limits.
if (typeof global.TextEncoder === 'undefined') {
  global.TextEncoder = require('node:util').TextEncoder;
}
if (typeof global.TextDecoder === 'undefined') {
  global.TextDecoder = require('node:util').TextDecoder;
}
if (typeof global.structuredClone === 'undefined') {
  const { serialize, deserialize } = require('node:v8');
  global.structuredClone = value => deserialize(serialize(value));
}

if (typeof global.TransformStream === 'undefined') {
  const { TransformStream, ReadableStream, WritableStream } = require('node:stream/web');
  global.TransformStream = TransformStream;
  global.ReadableStream = ReadableStream;
  global.WritableStream = WritableStream;
}
