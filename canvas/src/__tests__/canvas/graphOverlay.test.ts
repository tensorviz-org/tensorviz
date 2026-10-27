import { overlayScale, formatOverlayValue } from '../../canvas/graphOverlay';
import type { ModelNode } from '../../types';

const node = (id: string, extra: Partial<ModelNode['data']> = {}, hidden = false): ModelNode => ({
  id, type: 'layerNode', position: { x: 0, y: 0 }, hidden,
  data: {
    layerType: { id: 'linear', name: 'Linear', displayName: 'Linear', category: 'linear', icon: 'Box', description: '', parameters: [] },
    parameters: {}, ...extra,
  },
} as ModelNode);

const measured = (shape: number[], logicalBytes: number, dtype = 'float32'): Partial<ModelNode['data']> => ({
  outputShape: shape, shapeEvidence: 'measured', outputTensor: { shape, logicalBytes, dtype },
});

/**
 * Why: a colour has no room for a provenance badge. If an inferred shape could
 * be tinted like a measured one, the overlay would state a memory figure the
 * product has no evidence for — the exact confident-and-wrong the card's
 * measured/inferred split exists to prevent.
 */
describe('overlayScale', () => {
  it('scales measured activation bytes across the graph', () => {
    const nodes = [
      node('a', measured([2, 16, 64], 8192)),
      node('b', measured([2, 16, 256], 32768)),
    ];
    const scale = overlayScale(nodes, 'activation');
    expect(scale.values.get('a')!.value).toBe(2 * 16 * 64 * 4);
    expect(scale.values.get('b')!.value).toBe(2 * 16 * 256 * 4);
    expect(scale.values.get('a')!.heat).toBe(0);
    expect(scale.values.get('b')!.heat).toBe(1);
  });

  it('refuses to colour an inferred or unknown shape', () => {
    const nodes = [
      node('measured', measured([2, 4], 32)),
      node('inferred', { ...measured([2, 4000], 32000), shapeEvidence: 'inferred' }),
      node('stale', { ...measured([2, 4000], 32000), shapeEvidence: 'stale' }),
      node('nothing', {}),
    ];
    const scale = overlayScale(nodes, 'activation');
    expect([...scale.values.keys()]).toEqual(['measured']);
    expect(scale.unmeasured).toBe(3);
  });

  it('reads the parameter count the tracer already recorded', () => {
    const scale = overlayScale([node('head', { parameterCount: 20, parameters: { _params_M: 0 } }),
      node('relu', { parameterCount: 0 }), node('legacy', { parameters: { _params_M: 0.0092 } }),
      node('stale', { parameterCount: 10, shapeEvidence: 'stale' })], 'params');
    expect(Array.from(scale.values.values()).map(v => v.value)).toEqual([20, 0]);
    expect(scale.unmeasured).toBe(2);
  });

  it('leaves an all-equal graph cold rather than inventing an outlier', () => {
    const nodes = [
      node('a', measured([4], 16)),
      node('b', measured([4], 16)),
    ];
    const scale = overlayScale(nodes, 'activation');
    expect([...scale.values.values()].every(entry => entry.heat === 0)).toBe(true);
  });

  it('ignores hidden nodes so a closed module does not set the scale', () => {
    const nodes = [
      node('visible', measured([10], 40)),
      node('inside', measured([100000], 400000), true),
    ];
    expect(overlayScale(nodes, 'activation').max).toBe(40);
  });

  it('is empty and harmless when the metric is off', () => {
    expect(overlayScale([node('a')], 'none').values.size).toBe(0);
  });

  it('keeps scalar, empty and non-float32 footprints exact, including an all-zero scale', () => {
    const scale = overlayScale([node('scalar', measured([], 8, 'float64')),
      node('empty', measured([0, 3], 0, 'int16')), node('bool', measured([3, 4], 12, 'bool'))], 'activation');
    expect(Array.from(scale.values.values()).map(v => v.value)).toEqual([8, 0, 12]);
    const zero = overlayScale([node('zero', measured([0], 0))], 'activation');
    expect(zero.values.get('zero')).toEqual({ id: 'zero', value: 0, heat: 0 });
    expect(zero.unmeasured).toBe(0);
  });

  it.each([
    { outputTensor: undefined },
    { outputTensor: { shape: [3, 4], logicalBytes: 48 } },
    { outputTensor: { shape: [3, 4], dtype: 'float32' } },
    { outputTensor: { shape: [3, 4], dtype: 'float32', logicalBytes: NaN } },
    { outputTensor: { shape: [3, 4], dtype: 'float32', logicalBytes: -1 } },
    { outputShape: [2, 4] },
    { outputShape: [-1, 4] },
    { shapeEvidence: 'unknown' },
    { parameters: { _shape_provenance: 'unreached' } },
    { parameters: { _shape_provenance: 'observed' } },
  ])('does not fill missing, incomplete or mismatched evidence: %j', override => {
    expect(overlayScale([node('a', { ...measured([3, 4], 48), ...override } as any)], 'activation').values.size).toBe(0);
  });
});

describe('formatOverlayValue', () => {
  it('uses units a reader can compare at a glance', () => {
    expect(formatOverlayValue(32768, 'bytes')).toBe('32.0 KB');
    expect(formatOverlayValue(512, 'bytes')).toBe('512 B');
    expect(formatOverlayValue(9200, 'params')).toBe('9.2k');
  });
});
