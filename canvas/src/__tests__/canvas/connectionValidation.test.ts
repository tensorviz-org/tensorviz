/**
 * The structural rules drag-to-connect enforces.
 *
 * Each case is a graph a user can really build on the canvas, not a synthetic
 * shape: the point is that a gesture which would corrupt the model is refused
 * BEFORE it reaches setEdges, because nothing downstream re-checks. Codegen and
 * the shape engine both assume a DAG — a cycle committed here hangs the
 * topological walk rather than reporting an error.
 *
 * Deliberately no tldraw import: connectivity.ts is pure so these can run under
 * the CRA jest transform, which cannot load tldraw's ESM-only dependency.
 */
import {
  ConnectableNode,
  anchorReach,
  hasPort,
  sourceEndpoint,
  targetEndpoint,
  validateConnection,
} from '../../canvas/connectivity';

const layer = (id: string): ConnectableNode => ({ id, kind: 'layer' });
const input = (id: string): ConnectableNode => ({ id, kind: 'input' });
const output = (id: string): ConnectableNode => ({ id, kind: 'output' });
const moduleFrame = (id: string, entry?: string, exit?: string): ConnectableNode => ({
  id,
  kind: 'module',
  moduleEntryNodeId: entry,
  moduleExitNodeId: exit,
});

const edge = (source: string, target: string) => ({ source, target });

/** What a real drag does: resolve the anchor, walk once, then test a candidate. */
const dragFrom = (
  from: ConnectableNode,
  side: 'source' | 'target',
  candidate: ConnectableNode,
  edges: Array<{ source: string; target: string }>,
) =>
  validateConnection(
    { node: from, side, reach: anchorReach(edges, from, side) },
    candidate,
    edges,
  );

describe('drag-to-connect refuses the connections that would corrupt the graph', () => {
  it('refuses a self-loop', () => {
    // Dragging a layer's own output back onto its own input. `conv` would then be
    // its own parent, which the shape engine cannot order.
    const conv = layer('conv');
    expect(dragFrom(conv, 'source', conv, [])).toEqual({ ok: false, reason: 'self-loop' });
  });

  it('refuses a duplicate of an edge that already exists in that direction', () => {
    const edges = [edge('conv', 'relu')];
    expect(dragFrom(layer('conv'), 'source', layer('relu'), edges)).toEqual({
      ok: false,
      reason: 'duplicate',
    });
  });

  it('still allows a SECOND parent into the same layer — a merge is legal', () => {
    // The duplicate rule is about the ordered pair, not about fan-in. Rejecting
    // this would make residual and concat graphs unbuildable.
    const edges = [edge('conv', 'add')];
    expect(dragFrom(layer('skip'), 'source', layer('add'), edges)).toEqual({
      ok: true,
      source: 'skip',
      target: 'add',
    });
  });

  it('refuses a cycle through an existing path', () => {
    // a -> b -> c already exists; wiring c back into a closes the loop. The
    // reverse-BFS guard has to see it through b, not just as a direct pair.
    const edges = [edge('a', 'b'), edge('b', 'c')];
    expect(dragFrom(layer('c'), 'source', layer('a'), edges)).toEqual({
      ok: false,
      reason: 'cycle',
    });
  });

  it('refuses a cycle when the drag runs backwards, from a target port', () => {
    // Same graph, opposite gesture: dragging a's INPUT port onto c's output is
    // the same illegal edge, so the forward-BFS branch must catch it too.
    const edges = [edge('a', 'b'), edge('b', 'c')];
    expect(dragFrom(layer('a'), 'target', layer('c'), edges)).toEqual({
      ok: false,
      reason: 'cycle',
    });
  });

  it('refuses an Input node as a target', () => {
    // An Input is the graph's entry point; nothing may feed it. Enforced by the
    // card rendering no target port at all.
    expect(hasPort(input('in-1'), 'target')).toBe(false);
    expect(targetEndpoint(input('in-1'))).toBeNull();
    expect(dragFrom(layer('conv'), 'source', input('in-1'), [])).toEqual({
      ok: false,
      reason: 'missing-port',
    });
  });

  it('refuses an Output node as a source', () => {
    expect(hasPort(output('out-1'), 'source')).toBe(false);
    expect(sourceEndpoint(output('out-1'))).toBeNull();
    expect(dragFrom(layer('conv'), 'target', output('out-1'), [])).toEqual({
      ok: false,
      reason: 'missing-port',
    });
  });

  it('refuses a collapsed module frame that cannot name both its entry and exit', () => {
    // Both ids are optional on ModelNodeData. Without them there is no child to
    // attach to, and attaching to the FRAME would leave an edge the reconciler
    // deletes on the next collapse/expand.
    const halfWired = moduleFrame('block-1', 'block-1__conv', undefined);
    expect(hasPort(halfWired, 'target')).toBe(false);
    expect(hasPort(halfWired, 'source')).toBe(false);
    expect(dragFrom(layer('conv'), 'source', halfWired, [])).toEqual({
      ok: false,
      reason: 'unresolved-module',
    });
  });

  it('resolves a fully-routed module frame to its entry and exit children', () => {
    // The committed edge must name a child, never the frame, so the frame itself
    // stays edge-free.
    const block = moduleFrame('block-1', 'block-1__conv', 'block-1__relu');
    expect(dragFrom(layer('stem'), 'source', block, [])).toEqual({
      ok: true,
      source: 'stem',
      target: 'block-1__conv',
    });
    expect(dragFrom(layer('head'), 'target', block, [])).toEqual({
      ok: true,
      source: 'block-1__relu',
      target: 'head',
    });
  });

  it('refuses wiring a module frame back into itself through its own internals', () => {
    // exit -> entry is not a self-loop on ids, but the module's internal chain
    // already runs entry -> exit, so it closes a cycle.
    const block = moduleFrame('block-1', 'block-1__conv', 'block-1__relu');
    const edges = [edge('block-1__conv', 'block-1__relu')];
    expect(dragFrom(block, 'source', block, edges)).toEqual({ ok: false, reason: 'cycle' });
  });

  it('accepts the ordinary forward connection', () => {
    expect(dragFrom(input('in-1'), 'source', layer('conv'), [])).toEqual({
      ok: true,
      source: 'in-1',
      target: 'conv',
    });
    expect(dragFrom(layer('conv'), 'source', output('out-1'), [])).toEqual({
      ok: true,
      source: 'conv',
      target: 'out-1',
    });
  });
});
