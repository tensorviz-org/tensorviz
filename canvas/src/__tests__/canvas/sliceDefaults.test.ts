/**
 * The default Slice has to be applied without asking, so it has to be a slice
 * that cannot break the graph it lands in. These pin both halves of that: the
 * axis it picks, and the fact that it selects all of it.
 */
import { sliceDefaults } from '../../canvas/sliceDefaults';

describe('sliceDefaults', () => {
  it('slices the last axis, whatever the rank', () => {
    expect(sliceDefaults([1, 64, 32, 32]).dim).toBe(-1);
    expect(sliceDefaults([1, 512]).dim).toBe(-1);
  });

  it('selects the whole axis, so applying it automatically changes no shape', () => {
    // The point: out[-1] === in[-1]. A default of length 1 would reshape
    // [1, 512] to [1, 1] and fail the next Linear on sight.
    expect(sliceDefaults([1, 512])).toEqual({ dim: -1, start: 0, length: 512 });
    expect(sliceDefaults([1, 64, 32, 32])).toEqual({ dim: -1, start: 0, length: 32 });
  });

  it('falls back to 1 when the shape is not known yet, rather than guessing a width', () => {
    expect(sliceDefaults(undefined)).toEqual({ dim: -1, start: 0, length: 1 });
    expect(sliceDefaults([])).toEqual({ dim: -1, start: 0, length: 1 });
    // Rank 1 has only the batch axis to slice, which this canvas does not slice.
    expect(sliceDefaults([8])).toEqual({ dim: -1, start: 0, length: 1 });
  });

  it('ignores a non-integer or non-positive extent', () => {
    expect(sliceDefaults([1, 0]).length).toBe(1);
    expect(sliceDefaults([1, NaN]).length).toBe(1);
  });
});
