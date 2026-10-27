/**
 * The two errors the audit watched a beginner hit, plus the ones next to them.
 * Each asserts that the sentence says what to DO, and that the raw torch message
 * survives — an experienced user reads that first, and it is what anyone searches
 * for.
 */
import { explainTorchError } from '../../utils/explainTorchError';

describe('explainTorchError', () => {
  it('explains a matmul mismatch, naming both numbers and the fix', () => {
    const raw = 'mat1 and mat2 shapes cannot be multiplied (1x788544 and 512x256)';
    const out = explainTorchError(raw);
    expect(out).toContain('in_features');
    expect(out).toContain('788,544');
    expect(out).toContain('512');
    expect(out).toContain(raw);
  });

  it('explains a conv fed flattened data', () => {
    const raw =
      'Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [1, 32]';
    const out = explainTorchError(raw);
    expect(out).toContain('image-shaped');
    expect(out).toContain('Flatten');
    expect(out).toContain(raw);
  });

  it('keeps English number grouping when the host defaults to an Indian locale', () => {
    const format = Number.prototype.toLocaleString;
    const spy = jest.spyOn(Number.prototype, 'toLocaleString').mockImplementation(function (
      this: number, locales, options,
    ) {
      return format.call(this, locales ?? 'en-IN', options);
    });
    try {
      const matmul = explainTorchError('mat1 and mat2 shapes cannot be multiplied (1x788544 and 512000x256)');
      expect(matmul).toContain('512,000');
      expect(matmul).toContain('788,544');
      const raw = "shape '[1, 512]' is invalid for input of size 788544";
      expect(explainTorchError(raw)).toContain('788,544 values');
      expect(explainTorchError(raw)).toContain(raw);
    } finally {
      spy.mockRestore();
    }
  });

  it('explains a channel mismatch', () => {
    const raw =
      'Given groups=1, weight of size [64, 3, 3, 3], expected input[1, 16, 32, 32] to have 3 channels, but got 16 channels instead';
    expect(explainTorchError(raw)).toContain('in_channels to 16');
  });

  it('explains a negative dimension', () => {
    expect(explainTorchError('RuntimeError: Trying to create tensor with negative dimension -5'))
      .toContain('positive whole number');
  });

  it('leaves an error it does not recognise completely alone', () => {
    // Guessing at an unfamiliar error would be worse than silence.
    const raw = 'CUDA error: device-side assert triggered';
    expect(explainTorchError(raw)).toBe(raw);
  });

  it('is safe on an empty message', () => {
    expect(explainTorchError('')).toBe('');
  });
});
