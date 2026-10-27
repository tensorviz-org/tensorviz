/**
 * An Output added from the palette must join the chain.
 *
 * Auto-wiring was gated on `nodeType === 'layerNode'`, so building a model by hand
 * — Input, then Linear, then Output from the palette — left the Output isolated and
 * the warnings tab flagged it. The reported case was the third palette add, but it
 * was never about position: a palette-added Output received no connection at any
 * point in the sequence.
 *
 * An Output is a terminal, so it attaches to the chain's tail and never splices,
 * and it must NOT become the anchor — otherwise the next added layer appends after
 * the Output.
 *
 * ModelBuilder cannot be mounted under the CRA jest transform (importing tldraw
 * breaks jest — see qa_builder_ux.test.ts), so this pins the source contract, as
 * deleteRemovesWholeSelection.test.ts does. Behaviour was measured in a browser:
 * template 5 layers/4 edges -> delete the Output -> 4/3 -> re-add Output from the
 * palette -> 5/4.
 */
import fs from 'fs';
import path from 'path';

const SOURCE = fs.readFileSync(
  path.join(__dirname, '../..', 'components', 'ModelBuilder.tsx'),
  'utf8',
);

const branch = (() => {
  const start = SOURCE.indexOf("if (!options?.skipAutoConnect && nodeType === 'outputNode')");
  if (start === -1) return '';
  // Bound at this branch's own `return;` — past it lies the layerNode path, whose
  // splice logic would otherwise satisfy the assertions below.
  const end = SOURCE.indexOf('return;', start);
  return SOURCE.slice(start, end);
})();

describe('a palette-added Output joins the chain', () => {
  it('has an auto-wiring branch for output nodes at all', () => {
    expect(branch).not.toBe('');
  });

  it('wires from a node that has nothing downstream', () => {
    expect(branch).toContain("!edgesRef.current.some((e) => e.source === n.id)");
    expect(branch).toContain('setEdges');
  });

  it('never wires an Output into another Output', () => {
    expect(branch).toContain("n.type !== 'outputNode'");
  });

  it('appends only — an Output has nothing downstream to splice', () => {
    // The layerNode path removes the anchor's downstream edge and re-points it.
    // Doing that for a terminal would drop an edge and orphan the tail.
    expect(branch).not.toContain('downstream');
    expect(branch).not.toMatch(/eds\.filter/);
  });

  it('does not make the Output the anchor for the next add', () => {
    // setSelectedNodeId(id) here would append the next layer after the Output.
    expect(branch).not.toContain('setSelectedNodeId');
  });

  it('still honours skipAutoConnect, which the AI and import paths rely on', () => {
    expect(branch).toContain('!options?.skipAutoConnect');
  });

  it('leaves the existing layer-node wiring in place', () => {
    expect(SOURCE).toContain("if (!options?.skipAutoConnect && anchor && nodeType === 'layerNode')");
  });
});
