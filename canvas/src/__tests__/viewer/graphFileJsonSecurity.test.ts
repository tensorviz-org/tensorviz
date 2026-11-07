import { createGraphFile, parseGraphFile } from '../../services/graphFile';
import { nestedGraph } from '../../test/fixtures/graphFileFixture';

const raw = JSON.stringify(createGraphFile(nestedGraph(), { name: 'Security fixture', status: 'static' }));
const withIgnored = (json: string) => `{"ignored":${json},${raw.slice(1)}`;
const originalPrototype = Object.getOwnPropertyDescriptors(Object.prototype);

afterEach(() => {
  jest.restoreAllMocks();
  expect(Object.getOwnPropertyDescriptors(Object.prototype)).toEqual(originalPrototype);
});

it.each(['__proto__', 'constructor', 'prototype'])('rejects forbidden own key %s in ignored nested objects and arrays', key => {
  for (const json of [`{"${key}":null}`, `[{"safe":[{"${key}":{"polluted":true}}]}]`]) {
    expect(() => parseGraphFile(withIgnored(json))).toThrow('Unsafe object key in graph file.');
  }
});

it.each(['__pro\\u0074o__', 'constr\\u0075ctor', 'proto\\u0074ype'])('checks decoded Unicode key names: %s', key => {
  expect(() => parseGraphFile(withIgnored(`[{"${key}":0}]`))).toThrow('Unsafe object key in graph file.');
});

it('rejects keys in document and collaboration branches before field consumption', () => {
  for (const branch of ['document', 'collaboration']) {
    const value = JSON.parse(raw);
    value[branch].ignored = JSON.parse('{"__proto__":{"polluted":true}}');
    expect(() => parseGraphFile(JSON.stringify(value))).toThrow('Unsafe object key in graph file.');
  }
});

it('retains native duplicate-key final-value semantics', () => {
  expect(parseGraphFile(withIgnored('{"discarded":{"constructor":true},"discarded":{}}'))).toEqual(parseGraphFile(raw));
  expect(() => parseGraphFile(withIgnored('{"duplicate":{},"duplicate":{"prototype":0}}'))).toThrow('Unsafe object key in graph file.');
  expect(() => parseGraphFile(withIgnored('{"__proto__":{},"__proto__":null}'))).toThrow('Unsafe object key in graph file.');
});

it('allows forbidden words as ordinary values and safe own-key substrings', () => {
  expect(parseGraphFile(withIgnored('{"safe":["__proto__","constructor","prototype"],"constructorName":0,"__proto__suffix":false}'))).toEqual(parseGraphFile(raw));
});

it.each(['{', '[1,]', '{"ignored":undefined}', '{"a":1} trailing', '"unterminated'])('preserves malformed-JSON errors: %s', json => {
  expect(() => parseGraphFile(json)).toThrow('This file is not valid JSON. Choose a .tensorviz.json graph file.');
});

it.each([1, 1000])('does not install a scalar reviver for %i primitive leaves', count => {
  const json = withIgnored(JSON.stringify(Array.from({ length: count }, (_, i) => i)));
  const parse = jest.spyOn(JSON, 'parse');
  parseGraphFile(json);
  const fileParses = parse.mock.calls.filter(call => call[0] === json);
  expect(fileParses).toHaveLength(1);
  expect(fileParses[0][1]).toBeUndefined();
});

it('retains the legacy reviver fallback for deep but accepted ignored metadata', () => {
  const json = withIgnored('['.repeat(150) + '0' + ']'.repeat(150));
  const parse = jest.spyOn(JSON, 'parse');
  expect(parseGraphFile(json)).toEqual(parseGraphFile(raw));
  const fileParses = parse.mock.calls.filter(call => call[0] === json);
  expect(fileParses).toHaveLength(2);
  expect(typeof fileParses[1][1]).toBe('function');
});

it('keeps deeply nested unsafe keys rejected through the fallback', () => {
  const json = withIgnored('['.repeat(150) + '{"constructor":null}' + ']'.repeat(150));
  expect(() => parseGraphFile(json)).toThrow('Unsafe object key in graph file.');
});

it('keeps the engine-dependent rejection of extremely deep otherwise-valid JSON', () => {
  const json = withIgnored('['.repeat(20000) + '0' + ']'.repeat(20000));
  let legacyAccepts = true;
  try { JSON.parse(json, (_key, item) => item); } catch { legacyAccepts = false; }
  const actual = (() => {
    try { return { value: parseGraphFile(json) }; } catch (error) { return { error }; }
  })();
  const expected = legacyAccepts
    ? { value: parseGraphFile(raw) }
    : { error: new Error('This file is not valid JSON. Choose a .tensorviz.json graph file.') };
  expect(actual).toEqual(expected);
});

it('preserves legacy error order when deep subtrees and unsafe keys coexist', () => {
  const deep = '['.repeat(20000) + '0' + ']'.repeat(20000);
  for (const branches of [`"deep":${deep},"constructor":0`, `"constructor":0,"deep":${deep}`]) {
    const json = withIgnored(`{${branches}}`);
    let expected = '';
    try {
      JSON.parse(json, (key, item) => {
        if (['__proto__', 'constructor', 'prototype'].includes(key)) throw new Error('Unsafe object key in graph file.');
        return item;
      });
    } catch (error) {
      expected = error instanceof Error && error.message === 'Unsafe object key in graph file.'
        ? error.message : 'This file is not valid JSON. Choose a .tensorviz.json graph file.';
    }
    expect(expected).not.toBe('');
    expect(() => parseGraphFile(json)).toThrow(expected);
  }
});
