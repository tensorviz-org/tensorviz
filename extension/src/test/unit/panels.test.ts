import * as assert from 'node:assert/strict';

import {
  GRAPH_PANEL,
  PANELS,
  RECORD_KINDS,
  RUNS_PANEL,
  isRecordKind,
  panelsFor,
  primaryPanelFor,
} from '../../panels';

/**
 * The registry is one lookup and two entries, and both halves of that are the test.
 *
 * A third registrant is a milestone, not a commit (task.md M9 pins the count): the
 * abstraction exists so a data-pipeline or eval view is an ADDITION, and the way an
 * abstraction like that rots is by quietly accumulating entries nobody chose.
 */
suite('panels', () => {
  test('exactly two registrants: the graph panel and the Runs view', () => {
    assert.equal(PANELS.length, 2);
    assert.deepEqual(
      PANELS.map((panel) => panel.id),
      [GRAPH_PANEL.id, RUNS_PANEL.id],
    );
    assert.equal(RUNS_PANEL.view, 'tensorviz.runs');
    assert.equal(RUNS_PANEL.title, 'tensorViz Runs');
  });

  test('the canvas takes the kinds whose record carries a model graph', () => {
    // `train_step` included on purpose and not by association: its graph IS the
    // model's, node for node, because the backward is measured and not drawn
    // (backend/tests/capture/test_trace_record.py asserts the two are identical).
    assert.equal(GRAPH_PANEL.handles('model'), true);
    assert.equal(GRAPH_PANEL.handles('train_step'), true);
    // A pipeline's graph is source lines with no class to parse, so
    // `/api/ide/project` has nothing to merge it with; drawing it on the class
    // canvas would be a graph the code panel beside it could not follow.
    assert.equal(GRAPH_PANEL.handles('data'), false);
  });

  test('the Runs view shows every kind, including one no panel can open', () => {
    // A run nothing can open is still a run that happened. Hiding it would make the
    // list disagree with the directory it is a list of.
    for (const kind of RECORD_KINDS) {
      assert.ok(RUNS_PANEL.handles(kind), kind);
      assert.ok(
        panelsFor(kind).includes(RUNS_PANEL),
        `${kind} is not listed anywhere`,
      );
    }
    assert.deepEqual(panelsFor('data').map((p) => p.id), [RUNS_PANEL.id]);
    assert.equal(primaryPanelFor('model')?.id, GRAPH_PANEL.id);
    assert.equal(primaryPanelFor('data')?.id, RUNS_PANEL.id);
  });

  test('a kind this build does not know is not a kind', () => {
    // `isRecordKind` is what stops an `eval` record written by a newer build from
    // becoming a blank row with no icon and no panel.
    assert.equal(isRecordKind('train_step'), true);
    assert.equal(isRecordKind('eval'), false);
    assert.equal(isRecordKind(undefined), false);
  });
});
