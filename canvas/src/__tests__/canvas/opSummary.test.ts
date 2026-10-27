/**
 * The derived line a card shows above its parameter rows.
 *
 * `dim 1 / start 0 / length 16` never says "slice", so the card renders
 * `x[:, 0:16]` instead. The interesting cases are the ones where a naive
 * formatter would produce a confident falsehood: a NEGATIVE dim counts from the
 * right, so the number of leading axes is unknown and only an ellipsis is honest.
 */
import { opSummary } from '../../canvas/opSummary';

describe('slice summaries read like the notation the user typed', () => {
  it('renders the channel axis with the right number of leading colons', () => {
    expect(opSummary('narrow', { dim: 1, start: 0, length: 16 })).toBe('x[:, 0:16]');
  });

  it('counts leading axes for a later dim', () => {
    expect(opSummary('narrow', { dim: 2, start: 8, length: 16 })).toBe('x[:, :, 8:24]');
    expect(opSummary('narrow', { dim: 3, start: 1, length: 2 })).toBe('x[:, :, :, 1:3]');
  });

  it('slices the first axis with no leading colons', () => {
    expect(opSummary('narrow', { dim: 0, start: 0, length: 1 })).toBe('x[0:1]');
  });

  it('uses an ellipsis for a NEGATIVE dim rather than guessing the rank', () => {
    // The rank is not known here. Emitting 'x[:, 0:8]' for dim -1 would be wrong
    // on any tensor that is not rank 2 — exactly the sort of confident guess this
    // whole change exists to remove.
    expect(opSummary('narrow', { dim: -1, start: 4, length: 8 })).toBe('x[..., 4:12]');
    expect(opSummary('narrow', { dim: -2, start: 0, length: 3 })).toBe('x[..., 0:3]');
  });

  it('shows end as start + length, because the node stores length', () => {
    expect(opSummary('narrow', { dim: 1, start: 16, length: 16 })).toBe('x[:, 16:32]');
  });

  it('accepts numeric strings, which is what number inputs produce', () => {
    expect(opSummary('narrow', { dim: '1', start: '0', length: '16' })).toBe('x[:, 0:16]');
  });
});

describe('a summary is withheld rather than guessed', () => {
  it.each([
    ['no params at all', {}],
    ['missing length', { dim: 1, start: 0 }],
    ['blank strings', { dim: '', start: '', length: '' }],
    ['non-numeric', { dim: 'x', start: 0, length: 4 }],
    ['zero length', { dim: 1, start: 0, length: 0 }],
    ['negative start', { dim: 1, start: -2, length: 4 }],
    ['fractional dim', { dim: 1.5, start: 0, length: 4 }],
  ])('returns null for %s', (_label, params) => {
    expect(opSummary('narrow', params as Record<string, unknown>)).toBeNull();
  });

  it('returns null for undefined params', () => {
    expect(opSummary('narrow', undefined)).toBeNull();
  });

  it('returns null for ops that read fine as plain rows', () => {
    expect(opSummary('conv2d', { kernel_size: 3 })).toBeNull();
    expect(opSummary('relu', {})).toBeNull();
    expect(opSummary('linear', { in_features: 512, out_features: 256 })).toBeNull();
  });
});

describe('resampling summaries state the effect, not the arguments', () => {
  it('shows a scale factor with its mode', () => {
    expect(opSummary('interpolate', { scale_factor: 2, mode: 'bilinear' })).toBe('2× bilinear');
  });

  it('shows an explicit target size', () => {
    expect(opSummary('interpolate', { size: '64,64', mode: 'nearest' })).toBe('→ (64, 64) nearest');
    expect(opSummary('interpolate', { size: [128, 128] })).toBe('→ (128, 128)');
  });

  it('prefers size when both are somehow present, matching the editor', () => {
    // The Inspector clears whichever field is hidden, so this should not occur —
    // but a hand-edited or AI-built graph can still carry both.
    expect(opSummary('interpolate', { size: '32,32', scale_factor: 2, mode: 'area' }))
      .toBe('→ (32, 32) area');
  });

  it('covers Upsample too, which takes the same arguments', () => {
    expect(opSummary('upsample', { scale_factor: 4, mode: 'nearest' })).toBe('4× nearest');
  });

  it('withholds when neither size nor scale_factor is set', () => {
    expect(opSummary('interpolate', { mode: 'bilinear' })).toBeNull();
  });
});

describe('reshape states its target', () => {
  it('formats the target shape', () => {
    expect(opSummary('reshape', { shape: '1,64,1024' })).toBe('→ (1, 64, 1024)');
    expect(opSummary('reshape', { shape: '1,-1' })).toBe('→ (1, -1)');
  });

  it('withholds without a target', () => {
    expect(opSummary('reshape', {})).toBeNull();
    expect(opSummary('reshape', { shape: '' })).toBeNull();
  });
});
