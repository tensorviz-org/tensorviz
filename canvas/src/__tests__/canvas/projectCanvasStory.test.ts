/**
 * The project canvas draws two kinds of edge, and they are not the same claim.
 *
 * **Dataflow** — `feeds`, `trains`, `reads`, `produces` — is the pipeline: the
 * order things actually happen in, and the story a reader opened the graph for.
 * **`builds`** and `calls` say that one definition constructs or calls another,
 * which is true of nearly every pair of definitions in a codebase and says nothing
 * about sequence.
 *
 * Drawn and weighted alike, the result is what the whole project looked like on a
 * real repository: everything joined to everything, no line to follow. Two changes,
 * one idea — the layout straightens the pipeline and lets the rest go slack, and
 * the renderer lights the pipeline and quiets the rest.
 */
import { isFlowEdge, layoutTopLevel } from '../../services/canvasLayout';
import { ModelEdge, ModelNode } from '../../types';

const node = (id: string): ModelNode =>
  ({ id, type: 'layerNode', position: { x: 0, y: 0 }, data: { parameters: {} } } as unknown as ModelNode);

const edge = (source: string, target: string, relationship?: string): ModelEdge =>
  ({ id: `${source}->${target}`, source, target, data: relationship ? { _relationship: relationship } : {} } as unknown as ModelEdge);

describe('the layout knows which edges are the story', () => {
  it('tells a pipeline edge from a construction edge', () => {
    expect(isFlowEdge(edge('a', 'b', 'feeds'))).toBe(true);
    expect(isFlowEdge(edge('a', 'b', 'builds'))).toBe(false);
    expect(isFlowEdge(edge('a', 'b', 'calls'))).toBe(false);
  });

  it('covers every dataflow relationship the contract names', () => {
    // research/contracts.py::DATAFLOW_RELATIONSHIPS. A relationship missing here
    // is a pipeline edge quietly demoted to scenery, which is invisible until
    // someone notices a graph reads worse than it used to.
    for (const relationship of ['reads', 'transforms', 'feeds', 'trains', 'evaluates', 'produces', 'logs', 'derives_from']) {
      expect(isFlowEdge(edge('a', 'b', relationship))).toBe(true);
    }
  });

  it('leaves a per-class graph exactly as it was', () => {
    // The counterweight. Those edges carry no `_relationship` at all, and they are
    // all the same kind of claim, so weighting one of them differently would be
    // inventing a distinction the data does not make.
    expect(isFlowEdge(edge('a', 'b'))).toBe(false);
    expect(isFlowEdge({})).toBe(false);
  });

  it('puts the pipeline on one row and the call graph below it', () => {
    // `data → model → trainer` is the story; `main` builds all three. Ranked
    // together, `data` is pulled to the centroid of its three successors and the
    // three stages land on three different rows.
    const nodes = ['main', 'data', 'model', 'trainer'].map(node);
    const edges = [
      edge('data', 'model', 'feeds'),
      edge('model', 'trainer', 'trains'),
      edge('main', 'data', 'builds'),
      edge('main', 'model', 'builds'),
      edge('main', 'trainer', 'builds'),
    ];
    const laid = layoutTopLevel(nodes, edges);

    const y = (id: string) => laid.get(id)!.y;
    const x = (id: string) => laid.get(id)!.x;
    // The three pipeline stages share a row…
    expect(y('data')).toBe(y('model'));
    expect(y('model')).toBe(y('trainer'));
    // …run left to right in the order they happen…
    expect(x('data')).toBeLessThan(x('model'));
    expect(x('model')).toBeLessThan(x('trainer'));
    // …and what merely builds them is underneath, not in the way.
    expect(y('main')).toBeGreaterThan(y('trainer'));
  });

  it('changes nothing for a graph that has no pipeline at all', () => {
    // The counterweight, and it is a real case: `docs/check_links.py` is five
    // functions joined entirely by `builds`. Splitting a graph with no spine into
    // a spine and a band would put every node in the band.
    const nodes = ['main', 'check', 'a', 'b'].map(node);
    const builds = [edge('main', 'check', 'builds'), edge('check', 'a', 'builds'), edge('check', 'b', 'builds')];
    const plain = [edge('main', 'check'), edge('check', 'a'), edge('check', 'b')];
    const withKinds = layoutTopLevel(nodes, builds);
    const without = layoutTopLevel(nodes, plain);
    for (const id of ['main', 'check', 'a', 'b']) {
      expect(withKinds.get(id)).toEqual(without.get(id));
    }
  });
});
