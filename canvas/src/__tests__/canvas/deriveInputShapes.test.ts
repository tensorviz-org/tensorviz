/**
 * A node's input shape is its first upstream node's output shape.
 *
 * The two inference paths disagreed for a long time and nothing noticed: the
 * Validate button derived this, and "Refresh all shapes" set `outputShape` only.
 * So after a Refresh every node's input shape was blank — the Shape Table's INPUT
 * column with it, and anything keyed on `inputShape` silently fell back to a
 * degraded mode. The Slice range control is keyed on it, which is how the gap
 * finally surfaced: it always showed "Run Refresh shapes…" *after* a Refresh.
 *
 * Both paths call this now, so they cannot drift again.
 */
import { deriveInputShapes } from '../../services/graphTopology';

const edge = (source: string, target: string) => ({ source, target });

describe('deriveInputShapes', () => {
  it('gives a node its upstream output shape', () => {
    const got = deriveInputShapes([edge('a', 'b')], { a: [1, 64, 32, 32] });
    expect(got.get('b')).toEqual([1, 64, 32, 32]);
  });

  it('walks a whole chain', () => {
    const got = deriveInputShapes(
      [edge('a', 'b'), edge('b', 'c')],
      { a: [1, 3, 8, 8], b: [1, 16, 8, 8], c: [1, 16, 8, 8] },
    );
    expect(got.get('b')).toEqual([1, 3, 8, 8]);
    expect(got.get('c')).toEqual([1, 16, 8, 8]);
  });

  it('leaves a source node without an input shape', () => {
    const got = deriveInputShapes([edge('a', 'b')], { a: [1, 4] });
    expect(got.has('a')).toBe(false);
  });

  it('takes the FIRST parent on a merge, matching codegen argument order', () => {
    // A merge has several parents and one input-shape slot. Edge insertion order
    // is what codegen uses for torch.cat's argument order, so this agrees with it.
    const got = deriveInputShapes(
      [edge('x', 'merge'), edge('y', 'merge')],
      { x: [1, 8], y: [1, 99] },
    );
    expect(got.get('merge')).toEqual([1, 8]);
  });

  it('skips an upstream node whose shape is unknown', () => {
    // A declining op (split/chunk, or a reshape with no target) has no shape, and
    // its child must get nothing rather than a stale or wrong one.
    const got = deriveInputShapes([edge('a', 'b')], { a: undefined });
    expect(got.has('b')).toBe(false);
  });

  it('still resolves later parents when the first has no shape', () => {
    const got = deriveInputShapes(
      [edge('nope', 'merge'), edge('yes', 'merge')],
      { yes: [1, 5] },
    );
    expect(got.get('merge')).toEqual([1, 5]);
  });

  it('tolerates no shapes at all', () => {
    expect(deriveInputShapes([edge('a', 'b')], undefined).size).toBe(0);
    expect(deriveInputShapes([], { a: [1] }).size).toBe(0);
  });

  it('ignores a non-array value rather than passing it on', () => {
    const got = deriveInputShapes([edge('a', 'b')], { a: 'nonsense' as any });
    expect(got.has('b')).toBe(false);
  });
});
