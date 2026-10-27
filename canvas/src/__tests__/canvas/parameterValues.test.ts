/**
 * The draft <-> graph value conversion, which had no tests at all because it lived
 * inside BuilderPageV2.tsx (unimportable under the jest transform — it pulls in
 * tldraw). That is how a value-INVERTING bug survived: `formatValue` writes booleans
 * the Python way, 'True'/'False', and `parseParameterInput` only recognised the
 * lowercase form, so a real boolean came back as a string.
 *
 * Why that matters rather than being cosmetic, measured against real torch:
 *   nn.Linear(128, 64, bias=False)   -> no bias,  8192 parameters
 *   nn.Linear(128, 64, bias='False') -> HAS bias, 8256 parameters
 * A non-empty string is truthy in Python, so the string form turns the bias back
 * ON — in the built model and in the exported source — while shape inference
 * reports the graph healthy, because Linear's shape function never reads `bias`.
 *
 * The cases below are the value kinds a sweep of 476 real (key, value) pairs found
 * in the catalogue and in live template instantiations, plus the ones the code
 * import path produces.
 */
import {
  commitParameters,
  formatParamValue,
  formatValue,
  parseParameterInput,
} from '../../components/ui2/parameterValues';

/** What the panel actually does to a value it merely displays and hands back. */
const roundTrip = (key: string, value: unknown) => parseParameterInput(formatParamValue(key, value));

describe('the draft round trip', () => {
  it('keeps a boolean a boolean, in both spellings', () => {
    // The whole bug: 'True' used to come back as the string 'True'.
    expect(roundTrip('bias', true)).toBe(true);
    expect(roundTrip('bias', false)).toBe(false);
    // And the <select>'s own option values, which are lowercase.
    expect(parseParameterInput('true')).toBe(true);
    expect(parseParameterInput('false')).toBe(false);
  });

  it('heals a graph that already stores the Python spelling as a string', () => {
    // 16 parameter entries on this machine's saved graphs hold the string 'True'.
    expect(parseParameterInput('True')).toBe(true);
    expect(parseParameterInput('False')).toBe(false);
  });

  it('keeps null null — `stride=None` is the torch default and every import carries it', () => {
    expect(roundTrip('stride', null)).toBeNull();
  });

  it('keeps an array with a null in it, instead of mangling it to an unparseable string', () => {
    // nn.AdaptiveAvgPool2d((None, 7)) imports as [null, 7]. join() renders null as
    // an empty string, so this used to format to '[, 7]' and commit that text.
    expect(formatParamValue('output_size', [null, 7])).toBe('[null, 7]');
    expect(roundTrip('output_size', [null, 7])).toEqual([null, 7]);
    // A numeric array still reads exactly as it did.
    expect(formatParamValue('kernel_size', [7, 7])).toBe('[7, 7]');
    expect(roundTrip('kernel_size', [7, 7])).toEqual([7, 7]);
  });

  it('keeps a float that String() writes in exponent form', () => {
    // The cliff is at 1e-7: 1e-5 and 1e-6 print as decimals and always survived.
    expect(roundTrip('eps', 1e-8)).toBe(1e-8);
    expect(roundTrip('eps', 1e-5)).toBe(1e-5);
  });

  it('reads a leading-dot decimal, which a number input will happily produce', () => {
    // Left as a string, nn.Dropout(p='.5') raises inside shape inference while
    // Export still succeeds — failing in one place and not the other.
    expect(parseParameterInput('.5')).toBe(0.5);
    expect(parseParameterInput('-.5')).toBe(-0.5);
  });

  it('refuses a number it cannot store, rather than committing Infinity', () => {
    // Infinity would re-format to the string 'Infinity' on the next seed, so the
    // value would change type twice and never settle.
    expect(parseParameterInput('1e999')).toBe('1e999');
  });

  it('leaves ordinary values alone', () => {
    expect(roundTrip('in_features', 512)).toBe(512);
    expect(roundTrip('dim', -1)).toBe(-1);
    expect(roundTrip('mode', 'nearest')).toBe('nearest');
    expect(roundTrip('description', '')).toBe('');
  });

  it('still spells booleans for humans, because the card and the snippet do', () => {
    // formatValue also builds the Layer Code snippet, which must stay real Python.
    expect(formatValue(true)).toBe('True');
    expect(formatValue(false)).toBe('False');
  });
});

describe('commitParameters', () => {
  it('writes back only the key that moved', () => {
    const current = { in_features: 512, out_features: 256, bias: true };
    const baseline = { in_features: '512', out_features: '256', bias: 'True' };
    const draft = { ...baseline, out_features: '321' };

    // The point of the whole exercise: `bias` is not re-derived from 'True' at all,
    // so it cannot be turned into a string by an edit to a different field.
    expect(commitParameters(current, baseline, draft)).toEqual({
      in_features: 512,
      out_features: 321,
      bias: true,
    });
  });

  it('does not retype a key the panel merely displayed', () => {
    // normalized_shape is declared type:'string' and documented comma-separated, so
    // '512' must stay '512'. Re-parsing every key turned it into the number 512.
    const current = { normalized_shape: '512', eps: 1e-5 };
    const baseline = { normalized_shape: '512', eps: '0.00001' };
    const draft = { ...baseline, eps: '0.001' };

    const next = commitParameters(current, baseline, draft);
    expect(next.normalized_shape).toBe('512');
    expect(next.eps).toBe(0.001);
  });

  it('keeps the internal render props the panel deliberately hides', () => {
    // The draft never carries `_`-prefixed keys, and the commit replaces the whole
    // map — so the first keystroke on a template node used to delete the caption,
    // accent and shape annotation the backend authored.
    const current = { p: 0.5, _subtitle: 'flatten patches', _accent: 'var(--cat-attention)' };
    const baseline = { p: '0.5' };
    const draft = { p: '0.2' };

    expect(commitParameters(current, baseline, draft)).toEqual({
      p: 0.2,
      _subtitle: 'flatten patches',
      _accent: 'var(--cat-attention)',
    });
  });

  it('commits a field the node did not carry yet', () => {
    // The panel renders rows from the catalogue, so a draft key can be genuinely new.
    expect(commitParameters({ p: 0.5 }, { p: '0.5' }, { p: '0.5', inplace: 'True' })).toEqual({
      p: 0.5,
      inplace: true,
    });
  });

  it('writes nothing when nothing moved', () => {
    const current = { bias: true, normalized_shape: '512' };
    const baseline = { bias: 'True', normalized_shape: '512' };
    expect(commitParameters(current, baseline, { ...baseline })).toEqual(current);
  });
});
