import * as assert from 'node:assert/strict';

import { mergeSavedPositions } from '../../projection';

/** A projection document as the tracer emits it: positions for every node. */
const doc = (positions: Record<string, { x: number; y: number }>) => ({
  graph: { nodes: [] },
  view: { node_positions: positions },
});

suite('projection.mergeSavedPositions', () => {
  test('a saved layout is applied, and reported as restored', () => {
    // `restored` is the whole point: the canvas draws a restored layout exactly as
    // the user left it, and arranges anything else. Both halves are this one flag.
    const merged = mergeSavedPositions(
      doc({ l1: { x: 0, y: 0 }, l2: { x: 0, y: 0 } }),
      { graph_id: 'g1', node_positions: { l1: { x: 640, y: 220 } } },
      'g1',
    );

    assert.equal(merged.restored, true);
    const view = (merged.document as { view: { node_positions: Record<string, unknown> } }).view;
    assert.deepEqual(view.node_positions.l1, { x: 640, y: 220 });
  });

  test('no sidecar is not a restored layout', () => {
    // The common case, and the one the UX turns on: a graph opened for the first
    // time carries tracer positions, which are an emission order and not a layout.
    const merged = mergeSavedPositions(doc({ l1: { x: 12, y: 4 } }), undefined, 'g1');
    assert.equal(merged.restored, false);
  });

  test('a sidecar for a different graph is ignored', () => {
    // The file or the class was renamed. Applying that layout would place cards
    // by ids that mean something else now.
    const merged = mergeSavedPositions(
      doc({ l1: { x: 0, y: 0 } }),
      { graph_id: 'other', node_positions: { l1: { x: 900, y: 100 } } },
      'g1',
    );

    assert.equal(merged.restored, false);
    const view = (merged.document as { view: { node_positions: Record<string, unknown> } }).view;
    assert.deepEqual(view.node_positions.l1, { x: 0, y: 0 });
  });

  test('a sidecar whose every id has gone is not a restored layout', () => {
    // The subtle one. The class was rewritten under the sidecar, so nothing it
    // names is on the canvas: what the user is looking at is a graph nobody has
    // arranged, and reporting it as restored would open it in emission order
    // forever, with no way back but the layout button.
    const merged = mergeSavedPositions(
      doc({ fresh: { x: 3, y: 3 } }),
      { graph_id: 'g1', node_positions: { gone: { x: 500, y: 500 } } },
      'g1',
    );

    assert.equal(merged.restored, false);
    const view = (merged.document as { view: { node_positions: Record<string, unknown> } }).view;
    assert.equal('gone' in view.node_positions, false);
  });
});
