/**
 * An Input node's shape had three different renderings of the same value, because the
 * backend wrote a LIST while every reader assumed the canonical comma string:
 *
 *   inspector field   [1, 64, 48]
 *   canvas card       16448          <- React concatenates an array's children
 *   Export PyTorch    "inputNode.data.parameters.shape.split is not a function"
 *
 * `operation_resolver._shape_param` now always writes "1,64,48", but graphs saved before
 * that fix still hold arrays, so parsing stays tolerant of both forms.
 */
import {
  DEFAULT_INPUT_SHAPE,
  displayInputShape,
  formatInputShape,
  parseInputShape,
} from '../../utils/inputShape';

describe('parseInputShape', () => {
  it('reads the canonical comma string', () => {
    expect(parseInputShape('1,64,48')).toEqual([1, 64, 48]);
    expect(parseInputShape('1, 3, 224, 224')).toEqual([1, 3, 224, 224]);
  });

  it('reads a list, which is what an AI-created graph used to store', () => {
    expect(parseInputShape([1, 64, 48])).toEqual([1, 64, 48]);
    expect(parseInputShape(['1', '64'])).toEqual([1, 64]);
  });

  it('never returns an empty shape, whatever it is handed', () => {
    // A caller that gets [] would build a zero-dim tensor and fail far from the cause.
    expect(parseInputShape(undefined)).toEqual(DEFAULT_INPUT_SHAPE);
    expect(parseInputShape('')).toEqual(DEFAULT_INPUT_SHAPE);
    expect(parseInputShape('abc')).toEqual(DEFAULT_INPUT_SHAPE);
    expect(parseInputShape([])).toEqual(DEFAULT_INPUT_SHAPE);
    expect(parseInputShape([0, -4])).toEqual(DEFAULT_INPUT_SHAPE);
  });

  it('drops non-positive and fractional dims rather than passing them to torch', () => {
    expect(parseInputShape('1,0,48')).toEqual([1, 48]);
    expect(parseInputShape([1, 2.7, 48])).toEqual([1, 2, 48]);
  });

  it('does not throw on the bracketed string the inspector used to write back', () => {
    // Pressing Apply with "[1, 64, 48]" in the field stored that literal string.
    expect(() => parseInputShape('[1, 64, 48]')).not.toThrow();
    expect(parseInputShape('[1, 64, 48]')).toEqual([64, 48]);
  });
});

describe('formatInputShape', () => {
  it('always produces the canonical stored form', () => {
    expect(formatInputShape([1, 64, 48])).toBe('1,64,48');
    expect(formatInputShape('1, 64, 48')).toBe('1,64,48');
  });

  it('round-trips, so editing the field cannot corrupt the value', () => {
    const stored = formatInputShape([1, 64, 48]);
    expect(formatInputShape(stored)).toBe(stored);
    expect(parseInputShape(stored)).toEqual([1, 64, 48]);
  });
});

describe('displayInputShape', () => {
  it('matches the existing canvas style and never shows a joined array', () => {
    expect(displayInputShape('1,16')).toBe('(1, 16)');
    // The exact bug from the screenshot: [1, 64, 48] rendered as "16448".
    expect(displayInputShape([1, 64, 48])).toBe('(1, 64, 48)');
    expect(displayInputShape([1, 64, 48])).not.toBe('16448');
  });
});

describe('a card never states a shape nobody supplied', () => {
  // The ImageNet default belongs to code generation, which needs *a* tensor. A card
  // is a claim about this model: showing (1, 3, 224, 224) for a character LM, in the
  // same slot and type as the measured numbers, is the one number a reader is most
  // likely to trust — and it was invented. "" is the case that slipped through: not
  // undefined, so a `=== undefined` guard fell straight past it to the default.
  it.each([undefined, null, '', '   ', 'abc', [], [0], ['x']])(
    'displayInputShape(%p) is null, so the card falls back to "Not recorded"',
    (raw) => {
      expect(displayInputShape(raw)).toBeNull();
    },
  );

  it('still reads a real shape, in either stored form', () => {
    expect(displayInputShape('2,32')).toBe('(2, 32)');
    expect(displayInputShape([1, 64, 48])).toBe('(1, 64, 48)');
  });

  it('leaves the generators their default, which is what they are for', () => {
    // Export PyTorch Code has to write a tensor literal; refusing is not an option
    // there, so parseInputShape keeps substituting and says so in its name.
    expect(parseInputShape('')).toEqual(DEFAULT_INPUT_SHAPE);
    expect(parseInputShape('2,32')).toEqual([2, 32]);
  });
});
