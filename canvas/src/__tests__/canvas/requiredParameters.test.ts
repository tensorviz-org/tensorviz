/**
 * Layers missing a required constructor parameter must be caught, not exported.
 *
 * Importing a class whose __init__ takes required arguments (`nn.Linear(784,
 * hidden_dim)` with no default) cannot know what hidden_dim is, so those parameters
 * arrive empty. The import still reported success, the card read "Linear #fc2" with
 * no parameters at all, and Export Code then produced `nn.Linear()` — offered with
 * Copy and Download buttons — which raises TypeError the moment it runs.
 */
import { findMissingRequiredParameters, describeMissingParameters } from '../../utils/requiredParameters';

const node = (id: string, typeId: string, parameters: Record<string, any>) => ({
  id,
  data: { layerType: { id: typeId }, parameters },
});

describe('missing required parameters', () => {
  it('flags a layer whose required parameter never got a value', () => {
    const missing = findMissingRequiredParameters([
      node('fc1', 'linear', { in_features: 784 }),
      node('fc2', 'linear', {}),
    ]);

    expect(missing.map((m) => m.nodeId)).toEqual(['fc1', 'fc2']);
    expect(missing[0].missing).toContain('out_features');
    expect(missing[1].missing).toEqual(expect.arrayContaining(['in_features', 'out_features']));
  });

  it('passes a fully specified layer', () => {
    expect(
      findMissingRequiredParameters([node('fc1', 'linear', { in_features: 784, out_features: 128 })]),
    ).toEqual([]);
  });

  it('treats empty string as missing, since a cleared field reads that way', () => {
    const missing = findMissingRequiredParameters([
      node('fc1', 'linear', { in_features: 784, out_features: '' }),
    ]);
    expect(missing[0].missing).toEqual(['out_features']);
  });

  it('ignores optional parameters', () => {
    // Only required ones block an export; a default-bearing parameter is fine empty.
    const missing = findMissingRequiredParameters([
      node('fc1', 'linear', { in_features: 8, out_features: 4 }),
    ]);
    expect(missing).toEqual([]);
  });

  it('ignores the input and output terminals', () => {
    expect(
      findMissingRequiredParameters([node('input', 'input', {}), node('output', 'output', {})]),
    ).toEqual([]);
  });

  it('names the layer and the fields, so a refusal can say what to fix', () => {
    const text = describeMissingParameters(
      findMissingRequiredParameters([node('fc2', 'linear', {})]),
    );
    expect(text).toContain('#fc2');
    expect(text).toContain('in_features');
  });
});
