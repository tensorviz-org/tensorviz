/**
 * A graph with no Input node must say so.
 *
 * The warning loop skips the terminals, so a MISSING Input was never reported.
 * Deleting the Input node and clicking Validate produced one warning — "Linear - No
 * incoming edge from upstream layer" — which names the symptom, not the cause, and
 * nothing said where the shape it validated against came from. A single unconnected
 * Linear on an otherwise blank canvas was likewise reported as a hard shape error
 * the user could not act on.
 *
 * Guide 06 says Validate "pushes a fake tensor through your graph" using the shape
 * set on the Input node, so with no Input there is nothing to push and no way to
 * know what shape was assumed.
 *
 * A behaviour test, not a source-text pin. This used to read an 800-character
 * window of BuilderPageV2.tsx — including the exact spacing of
 * `if (nodes.length > 0 && !terminals.includes('input'))` — because BuilderPageV2
 * pulls in tldraw and cannot be mounted under the CRA jest transform. The rule now
 * lives in src/utils/graphWarnings.ts and this asserts the rows it ANSWERS with.
 *
 * Measured in a browser: a complete template raises no such warning; deleting the
 * Input takes the drawer from "Warnings (1)" to "Warnings (2)", the new row reading
 * "Model - No Input node. ... Add an Input layer and connect it to your first layer."
 */
import { graphWarnings } from '../../utils/graphWarnings';

const node = (id: string, layerId: string, displayName?: string) => ({
  id,
  data: { layerType: { id: layerId, displayName: displayName || layerId } },
});

const chain = (...ids: string[]) =>
  ids.slice(1).map((target, index) => ({ source: ids[index], target }));

describe('a missing Input node is reported', () => {
  it('reports the missing Input once, at graph level, not once per node', () => {
    const nodes = [node('l1', 'linear', 'Linear'), node('l2', 'relu', 'ReLU'), node('out', 'output')];
    const warnings = graphWarnings(nodes, chain('l1', 'l2', 'out'));
    const missing = warnings.filter((w) => w.name === 'Model');
    expect(missing).toHaveLength(1);
  });

  it('says what is missing and what to do about it', () => {
    const warnings = graphWarnings([node('l1', 'linear', 'Linear')], []);
    const missing = warnings.find((w) => w.name === 'Model');
    expect(missing?.message).toContain('No Input node');
    expect(missing?.message).toContain('Add an Input layer');
  });

  it('says the shape results came from a fallback', () => {
    // DE2-04: nothing stated where the shape Validate used had come from.
    const warnings = graphWarnings([node('l1', 'linear', 'Linear')], []);
    expect(warnings.find((w) => w.name === 'Model')?.message).toContain('fallback shape');
  });

  it('stays quiet on an empty canvas', () => {
    // 0 layers is a new model, not a broken one.
    expect(graphWarnings([], [])).toEqual([]);
  });

  it('does not fire when an Input is present', () => {
    const nodes = [node('in', 'input', 'Input'), node('l1', 'linear', 'Linear'), node('out', 'output')];
    const warnings = graphWarnings(nodes, chain('in', 'l1', 'out'));
    expect(warnings).toEqual([]);
  });

  it('recognises the Input whatever case its layer id arrives in', () => {
    // Graphs come from templates, code import and the backend; the id has been
    // seen as 'Input' as well as 'input'.
    const nodes = [node('in', 'Input', 'Input'), node('l1', 'linear', 'Linear')];
    const warnings = graphWarnings(nodes, chain('in', 'l1'));
    expect(warnings.map((w) => w.name)).not.toContain('Model');
  });

  it('leaves the per-node connectivity warnings in place', () => {
    const nodes = [
      node('in', 'input', 'Input'),
      node('l1', 'linear', 'Linear'),   // connected both ways
      node('l2', 'relu', 'ReLU'),       // no outgoing
      node('lone', 'conv2d', 'Conv2d'), // neither
      node('tail', 'linear', 'Tail'),   // no incoming
      node('out', 'output', 'Output'),
    ];
    const edges = [...chain('in', 'l1', 'l2'), { source: 'tail', target: 'out' }];
    const messages = new Map(graphWarnings(nodes, edges).map((w) => [w.name, w.message]));
    expect(messages.get('ReLU')).toBe('No outgoing edge to downstream layer.');
    expect(messages.get('Conv2d')).toBe('Isolated node not connected to graph path.');
    expect(messages.get('Tail')).toBe('No incoming edge from upstream layer.');
    expect(messages.get('Linear')).toBeUndefined();
  });

  it('never warns about the terminals themselves', () => {
    // Input has no incoming edge and Output has no outgoing one by definition;
    // reporting those was the noise that hid the real cause.
    const nodes = [node('in', 'input', 'Input'), node('out', 'output', 'Output')];
    expect(graphWarnings(nodes, chain('in', 'out'))).toEqual([]);
  });

  it('caps the list so a broken import cannot fill the drawer', () => {
    const nodes = Array.from({ length: 40 }, (_, i) => node(`n${i}`, 'linear', `L${i}`));
    expect(graphWarnings(nodes, []).length).toBeLessThanOrEqual(24);
  });
});
